"""Coupled bifurcation graph implicit step, snapshot rebuild and exact adjoint.

The coupled graph owns its factor and refinement workspaces. It shares native
main-channel geometry, commit and snapshot code with :mod:`hydronet.phys.implicit_inertia`,
and registers both ``"cuda"`` and ``"triton"`` backends.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import torch
from hydroforge.kernels import (
    BackendRegistry,
    CudaCall,
    CudaFill,
    CudaKernel,
    CudaSource,
    TritonProgram,
    launch_triton_kernel,
)

from hydronet.config import (
    ADJOINT_NOT_CONVERGED,
    FACTOR_LAYOUT,
    GRAPH_CLOSED_ELEVATION,
    GRAPH_COUPLING_FULL_PHASE,
    GRAPH_DRY_DEPTH,
    GRAPH_LINEAR_ITERATIONS,
    GRAPH_LINEAR_RESTARTS,
    GRAPH_LINEAR_TOLERANCE,
    GRAPH_SLOPE_LIMIT,
    GRAPH_SOLVE_THREADS,
    NEWTON_ROUND_LIMIT,
)
from hydronet.phys.implicit_inertia import (
    _COMMIT,
    _SOURCE,
    TREE_THREADS,
    UNIT_THREADS,
    _cdiv,
)
from hydronet.phys.specs import (
    INERTIA_GRAPH_ADJOINT,
    INERTIA_GRAPH_REBUILD,
    INERTIA_GRAPH_STEP,
)

_GRAPH_SOURCE = CudaSource(
    path=Path(__file__).with_name("cuda") / "bifurcation.cu",
    include_root=Path(__file__).with_name("cuda"),
    toolkit_headers=True,
    options=(
        *_SOURCE.options,
        f"-DGRAPH_THREADS={GRAPH_SOLVE_THREADS}",
        f"-DGRAPH_LINEAR_RESTARTS={GRAPH_LINEAR_RESTARTS}",
        f"-DGRAPH_LINEAR_ITERATIONS={GRAPH_LINEAR_ITERATIONS}",
        f"-DGRAPH_COUPLING_FULL_PHASE={GRAPH_COUPLING_FULL_PHASE}",
        f"-DGRAPH_LINEAR_TOLERANCE={GRAPH_LINEAR_TOLERANCE}f",
        f"-DGRAPH_SLOPE_LIMIT={GRAPH_SLOPE_LIMIT}f",
        f"-DGRAPH_CLOSED_ELEVATION={GRAPH_CLOSED_ELEVATION}f",
        f"-DGRAPH_DRY_DEPTH={GRAPH_DRY_DEPTH}f",
        *(f"-DFACTOR_{name.upper()}={i}" for i, name in enumerate(FACTOR_LAYOUT)),
    ),
)


def _graph_solve(kernel):
    """A cooperative graph kernel: at most one block per multiprocessor."""
    return CudaCall(
        kernel=kernel,
        grid="multiprocessors",
        cap_grid=True,
        cooperative=True,
        threads=GRAPH_SOLVE_THREADS,
    )


def _triton_graph_step():
    return TritonProgram(_prepare_graph_step)


def _triton_graph_rebuild():
    return TritonProgram(_prepare_graph_rebuild)


def _triton_graph_adjoint():
    return TritonProgram(_prepare_graph_adjoint)


solve_inertia_graph_step = BackendRegistry(
    INERTIA_GRAPH_STEP,
    {
        "cuda": CudaKernel(
            _GRAPH_SOURCE,
            steps=(
                CudaFill(buffer="grid_control_ptr"),
                CudaCall(
                    kernel="graph_solve_pure_trees",
                    size="num_pure_trees",
                    grid="blocks",
                    threads=TREE_THREADS,
                ),
                _graph_solve("graph_solve"),
                _COMMIT,
                CudaCall(
                    kernel="graph_commit",
                    size=("num_catchments", "num_bifurcation_values"),
                    extent="max",
                    threads=UNIT_THREADS,
                ),
            ),
        ),
        "triton": _triton_graph_step,
    },
)
# The converged graph snapshot again, at full coupling, with the pathway Jacobians.
solve_inertia_graph_rebuild = BackendRegistry(
    INERTIA_GRAPH_REBUILD,
    {
        "cuda": CudaKernel(
            _GRAPH_SOURCE,
            steps=(
                CudaCall(kernel="graph_rebuild_stage", threads=UNIT_THREADS),
                CudaCall(
                    kernel="graph_rebuild_flux",
                    size=("num_catchments", "num_bifurcation_paths"),
                    extent="max",
                    threads=UNIT_THREADS,
                ),
                CudaCall(kernel="graph_rebuild_residual", threads=UNIT_THREADS),
                CudaCall(
                    kernel="graph_rebuild_snapshot",
                    size=("num_catchments", "num_bifurcation_values"),
                    extent="max",
                    threads=UNIT_THREADS,
                ),
            ),
        ),
        "triton": _triton_graph_rebuild,
    },
)
solve_inertia_graph_adjoint = BackendRegistry(
    INERTIA_GRAPH_ADJOINT,
    {
        "cuda": CudaKernel(
            _GRAPH_SOURCE,
            steps=(
                CudaFill(buffer="grid_control_ptr"),
                _graph_solve("graph_adjoint"),
            ),
        ),
        "triton": _triton_graph_adjoint,
    },
)


# --------------------------------------------------------------------------- #
# Host helpers: sparse-LU + 1-solve fast path + BiCGStab closures
# --------------------------------------------------------------------------- #


def _bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<i", bits))[0]


def _unpack_factor_plan(plan_tensor: torch.Tensor, rounds: int):
    """Return zero-copy GPU slices of the packed factor_layout and CPU round counts."""
    n_names = len(FACTOR_LAYOUT)
    offsets = plan_tensor[:n_names].cpu().tolist()
    total = int(plan_tensor.numel())
    slices = {}
    for i, name in enumerate(FACTOR_LAYOUT):
        beg = offsets[i]
        end = offsets[i + 1] if i + 1 < n_names else total
        slices[name] = plan_tensor[beg:end]
    ro = slices["round_offset"].cpu().tolist()
    rh = slices["round_heavy"].cpu().tolist()
    round_info = []
    for r in range(rounds):
        rb, re, hv = ro[r], ro[r + 1], rh[r]
        split = rb + hv
        light = re - split
        round_info.append((rb, split, re, hv, light))
    return slices, tuple(round_info)


def _build_lu_launchers(round_info):
    from hydronet.phys.triton import bifurcation as tk

    num_heavy, num_light = [], []
    fwd_heavy, fwd_light = [], []
    bwd_heavy, bwd_light = [], []
    for _, _, _, hv, light in round_info:
        num_heavy.append(
            launch_triton_kernel(tk.factor_numeric_heavy_kernel, (hv,), physics=True)
            if hv > 0
            else None
        )
        num_light.append(
            launch_triton_kernel(
                tk.factor_numeric_light_kernel, (_cdiv(light, 32),), physics=True
            )
            if light > 0
            else None
        )
        fwd_heavy.append(
            launch_triton_kernel(tk.factor_solve_fwd_heavy_kernel, (hv,), physics=True)
            if hv > 0
            else None
        )
        fwd_light.append(
            launch_triton_kernel(
                tk.factor_solve_fwd_light_kernel, (_cdiv(light, 32),), physics=True
            )
            if light > 0
            else None
        )
        bwd_heavy.append(
            launch_triton_kernel(tk.factor_solve_bwd_heavy_kernel, (hv,), physics=True)
            if hv > 0
            else None
        )
        bwd_light.append(
            launch_triton_kernel(
                tk.factor_solve_bwd_light_kernel, (_cdiv(light, 32),), physics=True
            )
            if light > 0
            else None
        )
    return num_heavy, num_light, fwd_heavy, fwd_light, bwd_heavy, bwd_light


def _run_factor_numeric(
    slices, round_info, rounds, num_heavy, num_light, u_comp, ti, factor, all_active
):
    sl = slices
    for r in range(rounds - 1, -1, -1):
        rb, split, re_idx, hv, light = round_info[r]
        if hv > 0:
            num_heavy[r](
                rb,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                sl["urow_ptr"],
                sl["urow_entry"],
                sl["lcol_ptr"],
                sl["lcol_entry"],
                sl["product_ptr"],
                sl["product_lower"],
                sl["product_upper"],
                factor,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )
        if light > 0:
            num_light[r](
                split,
                re_idx,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                sl["urow_ptr"],
                sl["urow_entry"],
                sl["lcol_ptr"],
                sl["lcol_entry"],
                sl["product_ptr"],
                sl["product_lower"],
                sl["product_upper"],
                factor,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )


def _run_factor_solve(
    slices,
    round_info,
    rounds,
    fwd_heavy,
    fwd_light,
    bwd_heavy,
    bwd_light,
    u_comp,
    ti,
    factor,
    scaling,
    y,
    x,
    c,
    out,
    trans,
    all_active,
):
    sl = slices
    fptr = sl["ucol_ptr"] if trans else sl["lrow_ptr"]
    fentry = sl["ucol_entry"] if trans else sl["lrow_entry"]
    findex = sl["ucol_index"] if trans else sl["lrow_index"]
    bptr = sl["lcol_ptr"] if trans else sl["urow_ptr"]
    bentry = sl["lcol_entry"] if trans else sl["urow_entry"]
    bindex = sl["lcol_index"] if trans else sl["urow_index"]

    for r in range(rounds - 1, -1, -1):
        rb, split, re_idx, hv, light = round_info[r]
        if hv > 0:
            fwd_heavy[r](
                rb,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                fptr,
                fentry,
                findex,
                factor,
                c,
                y,
                TRANS=trans,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )
        if light > 0:
            fwd_light[r](
                split,
                re_idx,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                fptr,
                fentry,
                findex,
                factor,
                c,
                y,
                TRANS=trans,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )
    for r in range(rounds):
        rb, split, re_idx, hv, light = round_info[r]
        if hv > 0:
            bwd_heavy[r](
                rb,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                bptr,
                bentry,
                bindex,
                factor,
                scaling,
                y,
                x,
                out,
                TRANS=trans,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )
        if light > 0:
            bwd_light[r](
                split,
                re_idx,
                sl["path_unit"],
                sl["path_offset"],
                sl["order"],
                u_comp,
                ti,
                sl["diag"],
                bptr,
                bentry,
                bindex,
                factor,
                scaling,
                y,
                x,
                out,
                TRANS=trans,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=32,
                num_warps=1,
            )


def _make_coupled_linear_solver(arguments: dict):
    """Coupled-component sparse-LU + 1-solve fast path + BiCGStab closure."""
    from hydronet.phys.triton import bifurcation as tk

    a = arguments
    device = a["storage_ptr"].device
    ngu = int(a["num_graph_units"])
    ngc = int(a["num_graph_comps"])
    entries = int(a["num_core_factor_entries"])
    rounds = int(a["num_core_factor_rounds"])
    all_active = ngc == 1
    slices, round_info = _unpack_factor_plan(a["core_factor_layout_ptr"], rounds)

    blk = 256
    grid_u = (_cdiv(max(ngu, 1), blk),)
    n_blocks = grid_u[0]
    norm_buf = torch.zeros(1, dtype=torch.int32, device=device)

    k_mat_init = launch_triton_kernel(
        tk.coupled_matrix_and_init_kernel, grid_u, physics=True
    )
    k_assemble = (
        launch_triton_kernel(
            tk.factor_assemble_kernel, (_cdiv(entries, blk),), physics=True
        )
        if entries > 0
        else None
    )
    k_copy_dh = launch_triton_kernel(tk.coupled_copy_dh_kernel, grid_u, physics=True)
    k_norm = launch_triton_kernel(tk.coupled_norm_kernel, grid_u, physics=True)
    k_dot = launch_triton_kernel(tk.coupled_dot_kernel, grid_u, physics=True)
    k_dot_pair = launch_triton_kernel(tk.coupled_dot_pair_kernel, grid_u, physics=True)
    k_init_restart = launch_triton_kernel(
        tk.coupled_bicg_init_kernel, grid_u, physics=True
    )
    k_upd_p = launch_triton_kernel(tk.coupled_bicg_upd_p_kernel, grid_u, physics=True)
    k_upd_s = launch_triton_kernel(tk.coupled_bicg_upd_s_kernel, grid_u, physics=True)
    k_upd_r = launch_triton_kernel(tk.coupled_bicg_upd_r_kernel, grid_u, physics=True)
    k_accum_sol = launch_triton_kernel(
        tk.coupled_bicg_accum_kernel, grid_u, physics=True
    )
    k_res_diff = launch_triton_kernel(tk.coupled_res_diff_kernel, grid_u, physics=True)
    k_matvec = launch_triton_kernel(tk.coupled_matvec_kernel, grid_u, physics=True)
    num_heavy, num_light, fwd_heavy, fwd_light, bwd_heavy, bwd_light = (
        _build_lu_launchers(round_info)
    )

    gunit = a["graph_unit_ptr"]
    u_comp = a["unit_comp_ptr"]
    ti = a["tree_control_ptr"]
    tf = a["tree_scalars_ptr"]

    def _read_norm() -> float:
        return _bits_to_f32(int(norm_buf.item()))

    def _norm(x: torch.Tensor) -> float:
        norm_buf.zero_()
        k_norm(
            gunit, u_comp, ti, x, norm_buf, ngu, ALL_ACTIVE=all_active, BLOCK_SIZE=blk
        )
        return _read_norm()

    def _dot(va: torch.Tensor, vb: torch.Tensor) -> float:
        red = a["graph_reduction_ptr"].view(-1)
        k_dot(
            gunit, u_comp, ti, va, vb, red, ngu, ALL_ACTIVE=all_active, BLOCK_SIZE=blk
        )
        return float(np.float32(red[:n_blocks].sum(dtype=torch.float32).item()))

    def _dot_pair(t: torch.Tensor, s: torch.Tensor) -> tuple[float, float]:
        red = a["graph_reduction_ptr"].view(-1)
        k_dot_pair(
            gunit, u_comp, ti, t, s, red, ngu, ALL_ACTIVE=all_active, BLOCK_SIZE=blk
        )
        pair = (
            red[: 2 * n_blocks]
            .view(n_blocks, 2)
            .sum(dim=0, dtype=torch.float32)
            .tolist()
        )
        return float(np.float32(pair[0])), float(np.float32(pair[1]))

    def _factor_solve(c: torch.Tensor, out: torch.Tensor):
        _run_factor_solve(
            slices,
            round_info,
            rounds,
            fwd_heavy,
            fwd_light,
            bwd_heavy,
            bwd_light,
            u_comp,
            ti,
            a["graph_factor_ptr"],
            a["graph_scaling_ptr"],
            a["graph_lu_forward_ptr"],
            a["graph_lu_backward_ptr"],
            c,
            out,
            False,
            all_active,
        )

    def _matvec(dt: float, x: torch.Tensor, out: torch.Tensor, compensated: bool):
        k_matvec(
            dt,
            gunit,
            u_comp,
            ti,
            tf,
            x,
            a["graph_solution_low_ptr"],
            out,
            a["graph_bound_ptr"],
            a["graph_scaling_ptr"],
            a["trial_area_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["downstream_idx_ptr"],
            a["downstream_edge_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["bifurcation_catchment_idx_ptr"],
            a["bifurcation_downstream_idx_ptr"],
            a["graph_path_sum_ptr"],
            ngu,
            ALL_ACTIVE=all_active,
            COMPENSATED=compensated,
            BLOCK_SIZE=blk,
            num_warps=8,
        )

    def solve(dt: float) -> tuple[int, int, float]:
        if ngu == 0:
            return 0, 0, 0.0
        sl = slices
        norm_buf.zero_()
        k_mat_init(
            dt,
            gunit,
            u_comp,
            ti,
            tf,
            a["trial_area_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["graph_path_sum_ptr"],
            a["graph_bound_ptr"],
            a["graph_scaling_ptr"],
            a["graph_rhs_ptr"],
            a["newton_step_ptr"],
            a["graph_solution_low_ptr"],
            norm_buf,
            ngu,
            ALL_ACTIVE=all_active,
            BLOCK_SIZE=blk,
        )
        rhs_norm = _read_norm()
        if not (rhs_norm > 0.0):
            return 0, 0, 0.0
        if not math.isfinite(rhs_norm):
            return 2, 0, float("inf")

        r_vec = a["graph_r_ptr"]
        rhat = a["graph_rhat_ptr"]
        p_vec = a["graph_p_ptr"]
        v_vec = a["graph_v_ptr"]
        s_vec = a["graph_s_ptr"]
        t_vec = a["graph_t_ptr"]
        delta = a["graph_delta_ptr"]
        tmp = a["graph_temporary_ptr"]
        dh = a["newton_step_ptr"]
        sol_low = a["graph_solution_low_ptr"]
        rhs = a["graph_rhs_ptr"]
        tol = float(GRAPH_LINEAR_TOLERANCE)

        if k_assemble is not None:
            k_assemble(
                dt,
                sl["source_ptr"],
                sl["source_code"],
                a["core_entry_comp_ptr"],
                ti,
                a["downstream_edge_ptr"],
                a["downstream_idx_ptr"],
                a["bifurcation_catchment_idx_ptr"],
                a["bifurcation_downstream_idx_ptr"],
                a["graph_bound_ptr"],
                a["graph_scaling_ptr"],
                a["outflow_own_slope_ptr"],
                a["outflow_down_slope_ptr"],
                a["graph_path_sum_ptr"],
                a["graph_factor_ptr"],
                entries,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=blk,
            )
            _run_factor_numeric(
                slices,
                round_info,
                rounds,
                num_heavy,
                num_light,
                u_comp,
                ti,
                a["graph_factor_ptr"],
                all_active,
            )
            _factor_solve(rhs, delta)

        solves_used = 1
        k_copy_dh(
            gunit,
            u_comp,
            ti,
            delta,
            dh,
            sol_low,
            ngu,
            ALL_ACTIVE=all_active,
            BLOCK_SIZE=blk,
        )
        _matvec(dt, dh, tmp, True)
        norm_buf.zero_()
        k_res_diff(
            gunit,
            u_comp,
            ti,
            rhs,
            tmp,
            norm_buf,
            ngu,
            ALL_ACTIVE=all_active,
            BLOCK_SIZE=blk,
        )
        error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
        if math.isfinite(error) and error < tol:
            return 0, solves_used, error

        for _ in range(GRAPH_LINEAR_RESTARTS):
            initial = _norm(tmp)
            if not (initial > 0.0) or not math.isfinite(initial):
                break
            k_init_restart(
                gunit,
                u_comp,
                ti,
                tmp,
                initial,
                r_vec,
                rhat,
                p_vec,
                v_vec,
                delta,
                ngu,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=blk,
            )
            rho0 = np.float32(1.0)
            alpha = np.float32(1.0)
            omega = np.float32(1.0)
            scale_rel = float(np.float32(initial) / np.float32(rhs_norm))
            for _ in range(GRAPH_LINEAR_ITERATIONS):
                rho = np.float32(_dot(rhat, r_vec))
                if not math.isfinite(float(rho)) or abs(float(rho)) < 1e-30:
                    break
                beta = np.float32((rho / rho0) * (alpha / omega))
                k_upd_p(
                    gunit,
                    u_comp,
                    ti,
                    r_vec,
                    p_vec,
                    v_vec,
                    float(beta),
                    float(omega),
                    ngu,
                    ALL_ACTIVE=all_active,
                    BLOCK_SIZE=blk,
                )
                _factor_solve(p_vec, tmp)
                solves_used += 1
                _matvec(dt, tmp, v_vec, False)
                den = np.float32(_dot(rhat, v_vec))
                if not math.isfinite(float(den)) or abs(float(den)) < 1e-30:
                    break
                alpha = np.float32(rho / den)
                norm_buf.zero_()
                k_upd_s(
                    gunit,
                    u_comp,
                    ti,
                    r_vec,
                    v_vec,
                    tmp,
                    s_vec,
                    delta,
                    float(alpha),
                    norm_buf,
                    ngu,
                    ALL_ACTIVE=all_active,
                    BLOCK_SIZE=blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                _factor_solve(s_vec, tmp)
                solves_used += 1
                _matvec(dt, tmp, t_vec, False)
                tt_f, ts_f = _dot_pair(t_vec, s_vec)
                if not (tt_f > 1e-30) or not math.isfinite(tt_f):
                    break
                omega = np.float32(np.float32(ts_f) / np.float32(tt_f))
                if not math.isfinite(float(omega)) or abs(float(omega)) < 1e-20:
                    break
                norm_buf.zero_()
                k_upd_r(
                    gunit,
                    u_comp,
                    ti,
                    s_vec,
                    t_vec,
                    tmp,
                    r_vec,
                    delta,
                    float(omega),
                    norm_buf,
                    ngu,
                    ALL_ACTIVE=all_active,
                    BLOCK_SIZE=blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                rho0 = rho

            k_accum_sol(
                gunit,
                u_comp,
                ti,
                dh,
                sol_low,
                delta,
                initial,
                ngu,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=blk,
            )
            _matvec(dt, dh, tmp, True)
            norm_buf.zero_()
            k_res_diff(
                gunit,
                u_comp,
                ti,
                rhs,
                tmp,
                norm_buf,
                ngu,
                ALL_ACTIVE=all_active,
                BLOCK_SIZE=blk,
            )
            error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
            if math.isfinite(error) and error < tol:
                return 0, solves_used, error
            if not math.isfinite(error):
                break
        return (1 if math.isfinite(error) else 2), solves_used, error

    return solve


def _make_adjoint_linear_solver(arguments: dict):
    """Full-map transpose sparse-LU + 1-solve fast path + BiCGStab closure."""
    from hydronet.phys.triton import bifurcation as tk

    a = arguments
    device = a["storage_ptr"].device
    n = int(a["num_catchments"])
    entries = int(a["num_factor_entries"])
    rounds = int(a["num_factor_rounds"])
    slices, round_info = _unpack_factor_plan(a["factor_layout_ptr"], rounds)

    blk = 256
    grid_n = (_cdiv(n, blk),)
    n_blocks = grid_n[0]
    norm_buf = torch.zeros(1, dtype=torch.int32, device=device)

    k_matrix_init = launch_triton_kernel(
        tk.graph_matrix_and_init_kernel, grid_n, physics=True
    )
    k_assemble = (
        launch_triton_kernel(
            tk.factor_assemble_kernel, (_cdiv(entries, blk),), physics=True
        )
        if entries > 0
        else None
    )
    k_norm = launch_triton_kernel(tk.vec_norm_kernel, grid_n, physics=True)
    k_dot = launch_triton_kernel(tk.vec_dot_kernel, grid_n, physics=True)
    k_dot_pair = launch_triton_kernel(tk.vec_dot_pair_kernel, grid_n, physics=True)
    k_init_restart = launch_triton_kernel(
        tk.bicg_init_restart_kernel, grid_n, physics=True
    )
    k_upd_p = launch_triton_kernel(tk.bicg_update_p_kernel, grid_n, physics=True)
    k_upd_s = launch_triton_kernel(tk.bicg_update_s_delta_kernel, grid_n, physics=True)
    k_upd_r = launch_triton_kernel(tk.bicg_update_r_delta_kernel, grid_n, physics=True)
    k_accum_sol = launch_triton_kernel(
        tk.bicg_accum_solution_kernel, grid_n, physics=True
    )
    k_res_diff = launch_triton_kernel(
        tk.bicg_residual_diff_kernel, grid_n, physics=True
    )
    k_scale_div = launch_triton_kernel(tk.scale_div_kernel, grid_n, physics=True)
    k_matvec = launch_triton_kernel(tk.graph_matvec_kernel, grid_n, physics=True)
    num_heavy, num_light, fwd_heavy, fwd_light, bwd_heavy, bwd_light = (
        _build_lu_launchers(round_info)
    )

    dummy_i32 = a["unit_comp_ptr"]

    def _read_norm() -> float:
        return _bits_to_f32(int(norm_buf.item()))

    def _norm(x: torch.Tensor) -> float:
        norm_buf.zero_()
        k_norm(x, norm_buf, n, BLOCK_SIZE=blk)
        return _read_norm()

    def _dot(va: torch.Tensor, vb: torch.Tensor) -> float:
        red = a["graph_reduction_ptr"].view(-1)
        k_dot(va, vb, red, n, BLOCK_SIZE=blk)
        return float(np.float32(red[:n_blocks].sum(dtype=torch.float32).item()))

    def _dot_pair(t: torch.Tensor, s: torch.Tensor) -> tuple[float, float]:
        red = a["graph_reduction_ptr"].view(-1)
        k_dot_pair(t, s, red, n, BLOCK_SIZE=blk)
        pair = (
            red[: 2 * n_blocks]
            .view(n_blocks, 2)
            .sum(dim=0, dtype=torch.float32)
            .tolist()
        )
        return float(np.float32(pair[0])), float(np.float32(pair[1]))

    def _precondition(inp: torch.Tensor, out: torch.Tensor):
        pre = a["graph_precondition_rhs_ptr"]
        k_scale_div(inp, a["graph_scaling_ptr"], pre, n, BLOCK_SIZE=blk)
        _run_factor_solve(
            slices,
            round_info,
            rounds,
            fwd_heavy,
            fwd_light,
            bwd_heavy,
            bwd_light,
            dummy_i32,
            dummy_i32,
            a["graph_factor_ptr"],
            a["graph_scaling_ptr"],
            a["graph_lu_forward_ptr"],
            a["graph_lu_backward_ptr"],
            pre,
            out,
            True,
            True,
        )

    def _matvec(dt: float, x: torch.Tensor, out: torch.Tensor, compensated: bool):
        k_matvec(
            dt,
            x,
            a["graph_solution_low_ptr"],
            out,
            a["graph_bound_ptr"],
            a["graph_scaling_ptr"],
            a["trial_area_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["downstream_idx_ptr"],
            a["downstream_edge_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["bifurcation_catchment_idx_ptr"],
            a["bifurcation_downstream_idx_ptr"],
            a["graph_path_sum_ptr"],
            n,
            TRANS=True,
            COMPENSATED=compensated,
            BLOCK_SIZE=blk,
            num_warps=8,
        )

    def solve(dt: float) -> tuple[int, int, float]:
        sl = slices
        norm_buf.zero_()
        k_matrix_init(
            dt,
            a["trial_area_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["graph_path_sum_ptr"],
            a["graph_bound_ptr"],
            a["graph_scaling_ptr"],
            a["graph_rhs_ptr"],
            a["newton_step_ptr"],
            a["graph_solution_low_ptr"],
            a["graph_temporary_ptr"],
            norm_buf,
            n,
            BLOCK_SIZE=blk,
        )
        rhs_norm = _read_norm()
        if not (rhs_norm > 0.0):
            return 0, 0, 0.0
        if not math.isfinite(rhs_norm):
            return 2, 0, float("inf")

        if k_assemble is not None:
            k_assemble(
                dt,
                sl["source_ptr"],
                sl["source_code"],
                dummy_i32,
                dummy_i32,
                a["downstream_edge_ptr"],
                a["downstream_idx_ptr"],
                a["bifurcation_catchment_idx_ptr"],
                a["bifurcation_downstream_idx_ptr"],
                a["graph_bound_ptr"],
                a["graph_scaling_ptr"],
                a["outflow_own_slope_ptr"],
                a["outflow_down_slope_ptr"],
                a["graph_path_sum_ptr"],
                a["graph_factor_ptr"],
                entries,
                ALL_ACTIVE=True,
                BLOCK_SIZE=blk,
            )
            _run_factor_numeric(
                slices,
                round_info,
                rounds,
                num_heavy,
                num_light,
                dummy_i32,
                dummy_i32,
                a["graph_factor_ptr"],
                True,
            )

        r_vec = a["graph_r_ptr"]
        rhat = a["graph_rhat_ptr"]
        p_vec = a["graph_p_ptr"]
        v_vec = a["graph_v_ptr"]
        s_vec = a["graph_s_ptr"]
        t_vec = a["graph_t_ptr"]
        delta = a["graph_delta_ptr"]
        tmp = a["graph_temporary_ptr"]
        dh = a["newton_step_ptr"]
        sol_low = a["graph_solution_low_ptr"]
        rhs = a["graph_rhs_ptr"]
        tol = float(GRAPH_LINEAR_TOLERANCE)

        _precondition(rhs, delta)
        solves_used = 1
        dh.copy_(delta)
        sol_low.zero_()
        _matvec(dt, dh, tmp, True)
        norm_buf.zero_()
        k_res_diff(rhs, tmp, norm_buf, n, BLOCK_SIZE=blk)
        error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
        if math.isfinite(error) and error < tol:
            return 0, solves_used, error

        for _ in range(GRAPH_LINEAR_RESTARTS):
            initial = _norm(tmp)
            if not (initial > 0.0) or not math.isfinite(initial):
                return 2, solves_used, error
            k_init_restart(
                tmp, initial, r_vec, rhat, p_vec, v_vec, delta, n, BLOCK_SIZE=blk
            )
            rho0 = np.float32(1.0)
            alpha = np.float32(1.0)
            omega = np.float32(1.0)
            scale_rel = float(np.float32(initial) / np.float32(rhs_norm))
            for _ in range(GRAPH_LINEAR_ITERATIONS):
                rho = np.float32(_dot(rhat, r_vec))
                if not math.isfinite(float(rho)) or abs(float(rho)) < 1e-30:
                    break
                beta = np.float32((rho / rho0) * (alpha / omega))
                k_upd_p(
                    r_vec, p_vec, v_vec, float(beta), float(omega), n, BLOCK_SIZE=blk
                )
                _precondition(p_vec, tmp)
                solves_used += 1
                _matvec(dt, tmp, v_vec, False)
                den = np.float32(_dot(rhat, v_vec))
                if not math.isfinite(float(den)) or abs(float(den)) < 1e-30:
                    break
                alpha = np.float32(rho / den)
                norm_buf.zero_()
                k_upd_s(
                    r_vec,
                    v_vec,
                    tmp,
                    s_vec,
                    delta,
                    float(alpha),
                    norm_buf,
                    n,
                    BLOCK_SIZE=blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                _precondition(s_vec, tmp)
                solves_used += 1
                _matvec(dt, tmp, t_vec, False)
                tt_f, ts_f = _dot_pair(t_vec, s_vec)
                if not (tt_f > 1e-30) or not math.isfinite(tt_f):
                    break
                omega = np.float32(np.float32(ts_f) / np.float32(tt_f))
                if not math.isfinite(float(omega)) or abs(float(omega)) < 1e-20:
                    break
                norm_buf.zero_()
                k_upd_r(
                    s_vec,
                    t_vec,
                    tmp,
                    r_vec,
                    delta,
                    float(omega),
                    norm_buf,
                    n,
                    BLOCK_SIZE=blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                rho0 = rho

            k_accum_sol(dh, sol_low, delta, initial, n, BLOCK_SIZE=blk)
            _matvec(dt, dh, tmp, True)
            norm_buf.zero_()
            k_res_diff(rhs, tmp, norm_buf, n, BLOCK_SIZE=blk)
            error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
            if math.isfinite(error) and error < tol:
                return 0, solves_used, error
            if not math.isfinite(error):
                return 2, solves_used, error
        return 1, solves_used, error

    return solve


# --------------------------------------------------------------------------- #
# TritonProgram builders (graph step, graph rebuild, graph adjoint)
# --------------------------------------------------------------------------- #


def _prepare_graph_step(arguments, buffer_dtypes):
    from hydronet.phys.triton import bifurcation as tk
    from hydronet.phys.triton import implicit_inertia as fk

    del buffer_dtypes
    a = arguments
    device = a["storage_ptr"].device
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    npt = int(a["num_pure_trees"])
    ngc = int(a["num_graph_comps"])
    ngu = int(a["num_graph_units"])
    nct = int(a["num_coupled_trees"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    n_vals = int(a["num_bifurcation_values"])
    eps = float(a["slope_regularization"])
    tol = float(a["newton_tolerance"])
    maxit = int(a["newton_max_iterations"])

    blk = 256
    grid_n = (_cdiv(n, blk),)
    pure_launch = (
        launch_triton_kernel(fk.inertia_solve_trees_kernel, (npt,), physics=True)
        if npt > 0
        else None
    )

    coupled_plan = None
    if ngc > 0 and ngu > 0:
        grid_u = (_cdiv(ngu, blk),)
        grid_m = (_cdiv(max(m_paths, 1), blk),)
        grid_c = (_cdiv(ngc, 32),)
        c_trees_long = a["coupled_trees_ptr"].long()
        c_rep_for_tree = a["comp_rep_tree_ptr"].long()[
            a["coupled_tree_comp_ptr"].long()
        ]
        need_meta_copy = nct > ngc
        coupled_plan = dict(
            linear_solve=_make_coupled_linear_solver(a),
            start_u=launch_triton_kernel(
                tk.coupled_start_units_kernel, grid_u, physics=True
            ),
            start_c=launch_triton_kernel(
                tk.coupled_start_comps_kernel, grid_c, physics=True
            ),
            stage=launch_triton_kernel(tk.coupled_stage_kernel, grid_u, physics=True),
            mflux=launch_triton_kernel(
                tk.coupled_main_flux_kernel, grid_u, physics=True
            ),
            bflux=(
                launch_triton_kernel(tk.coupled_bif_flux_kernel, grid_m, physics=True)
                if m_paths > 0
                else None
            ),
            res=launch_triton_kernel(tk.coupled_residual_kernel, grid_u, physics=True),
            ctrl=launch_triton_kernel(tk.coupled_control_kernel, grid_c, physics=True),
            apply=launch_triton_kernel(tk.coupled_apply_kernel, grid_u, physics=True),
            prep_rhs=launch_triton_kernel(
                tk.coupled_prep_rhs_kernel, grid_u, physics=True
            ),
            trial=launch_triton_kernel(tk.coupled_trial_kernel, grid_u, physics=True),
            c_trees_long=c_trees_long,
            c_rep_for_tree=c_rep_for_tree,
            need_meta_copy=need_meta_copy,
            single_comp=(ngc == 1),
        )

    k_commit = launch_triton_kernel(fk.inertia_commit_kernel, grid_n, physics=True)
    k_gcommit_p = (
        launch_triton_kernel(
            tk.graph_commit_paths_kernel, (_cdiv(n_vals, blk),), physics=True
        )
        if n_vals > 0
        else None
    )
    k_gcommit_i = (
        launch_triton_kernel(tk.graph_commit_inflow_kernel, grid_n, physics=True)
        if m_paths > 0
        else None
    )

    save_surface = a["converged_surface_ptr"] is not None
    pair_ptr = a["converged_surface_ptr"] if save_surface else a["storage_ptr"]
    ctl = a["grid_control_ptr"]
    lin_rec = a["linear_record_ptr"]
    meta = a["newton_record_ptr"]

    def launch():
        ctl.zero_()
        # 1. Pure-forest basins: one CTA per tree
        if pure_launch is not None:
            pure_launch(
                a["storage_ptr"],
                a["runoff_ptr"],
                a["outflow_ptr"],
                a["river_manning_ptr"],
                a["flood_manning_ptr"],
                a["river_length_ptr"],
                a["river_width_ptr"],
                a["river_height_ptr"],
                a["catchment_area_ptr"],
                a["flood_depth_table_ptr"],
                a["flood_bed_length_ptr"],
                a["catchment_elevation_ptr"],
                a["downstream_distance_ptr"],
                a["time_step_ptr"],
                a["downstream_idx_ptr"],
                a["downstream_edge_ptr"],
                a["upstream_offset_ptr"],
                a["upstream_index_ptr"],
                a["pure_block_tree_ptr"],
                a["tree_unit_ptr"],
                a["level_offset_ptr"],
                a["tree_level_ptr"],
                a["newton_offset_ptr"],
                a["trial_offset_ptr"],
                a["newton_reference_ptr"],
                a["trial_reference_ptr"],
                a["trial_storage_ptr"],
                a["trial_area_ptr"],
                a["channel_conveyance_ptr"],
                a["flood_conveyance_ptr"],
                a["channel_conveyance_slope_ptr"],
                a["flood_conveyance_slope_ptr"],
                a["trial_outflow_ptr"],
                a["outflow_own_slope_ptr"],
                a["outflow_down_slope_ptr"],
                a["residual_ptr"],
                a["balance_weight_ptr"],
                a["accepted_weight_ptr"],
                a["pivot_ptr"],
                a["pivot_rhs_ptr"],
                a["newton_step_ptr"],
                a["coupling_ptr"],
                a["phase_trace_ptr"],
                meta,
                a["newton_status_ptr"],
                eps,
                tol,
                maxit,
                n,
                levels,
                BLOCK_SIZE=256,
                num_warps=8,
            )

        # 2. Coupled bifurcation components
        if coupled_plan is None:
            lin_rec.zero_()
        else:
            cp = coupled_plan
            dt = float(a["time_step_ptr"].item())
            cp["start_u"](
                a["graph_unit_ptr"],
                a["storage_ptr"],
                a["river_manning_ptr"],
                a["flood_manning_ptr"],
                a["river_length_ptr"],
                a["river_width_ptr"],
                a["river_height_ptr"],
                a["catchment_area_ptr"],
                a["flood_depth_table_ptr"],
                a["newton_reference_ptr"],
                a["newton_offset_ptr"],
                a["trial_reference_ptr"],
                a["trial_offset_ptr"],
                a["graph_bound_ptr"],
                a["newton_status_ptr"],
                ngu,
                levels,
                BLOCK_SIZE=blk,
            )
            cp["start_c"](
                a["tree_control_ptr"],
                a["tree_scalars_ptr"],
                a["tree_merit_ptr"],
                a["tree_line_ptr"],
                dt,
                ngc,
                BLOCK_SIZE=32,
                num_warps=1,
            )
            total_solves = 0
            last_lin = (0.0, 0.0, 0.0)
            converged_loop = False

            for _ in range(1 << 22):
                cp["stage"](
                    a["graph_unit_ptr"],
                    a["unit_comp_ptr"],
                    a["tree_control_ptr"],
                    a["trial_reference_ptr"],
                    a["trial_offset_ptr"],
                    a["river_length_ptr"],
                    a["river_width_ptr"],
                    a["river_height_ptr"],
                    a["catchment_area_ptr"],
                    a["flood_depth_table_ptr"],
                    a["flood_bed_length_ptr"],
                    a["trial_storage_ptr"],
                    a["trial_area_ptr"],
                    a["channel_conveyance_ptr"],
                    a["flood_conveyance_ptr"],
                    a["channel_conveyance_slope_ptr"],
                    a["flood_conveyance_slope_ptr"],
                    ngu,
                    levels,
                    BLOCK_SIZE=blk,
                    num_warps=8,
                )
                cp["mflux"](
                    dt,
                    eps,
                    a["graph_unit_ptr"],
                    a["unit_comp_ptr"],
                    a["tree_control_ptr"],
                    a["outflow_ptr"],
                    a["river_manning_ptr"],
                    a["flood_manning_ptr"],
                    a["river_length_ptr"],
                    a["river_width_ptr"],
                    a["river_height_ptr"],
                    a["catchment_area_ptr"],
                    a["flood_depth_table_ptr"],
                    a["flood_bed_length_ptr"],
                    a["catchment_elevation_ptr"],
                    a["downstream_distance_ptr"],
                    a["downstream_idx_ptr"],
                    a["downstream_edge_ptr"],
                    a["trial_reference_ptr"],
                    a["trial_offset_ptr"],
                    a["trial_storage_ptr"],
                    a["trial_area_ptr"],
                    a["channel_conveyance_ptr"],
                    a["flood_conveyance_ptr"],
                    a["channel_conveyance_slope_ptr"],
                    a["flood_conveyance_slope_ptr"],
                    a["trial_outflow_ptr"],
                    a["outflow_own_slope_ptr"],
                    a["outflow_down_slope_ptr"],
                    ngu,
                    levels,
                    BLOCK_SIZE=blk,
                    num_warps=8,
                )
                if cp["bflux"] is not None:
                    cp["bflux"](
                        dt,
                        a["path_comp_ptr"],
                        a["tree_control_ptr"],
                        a["bifurcation_catchment_idx_ptr"],
                        a["bifurcation_downstream_idx_ptr"],
                        a["catchment_elevation_ptr"],
                        a["river_height_ptr"],
                        a["trial_reference_ptr"],
                        a["trial_offset_ptr"],
                        a["bifurcation_width_ptr"],
                        a["bifurcation_length_ptr"],
                        a["bifurcation_elevation_ptr"],
                        a["bifurcation_manning_ptr"],
                        a["bifurcation_outflow_ptr"],
                        a["graph_path_values_ptr"],
                        a["graph_path_sum_ptr"],
                        a["newton_status_ptr"],
                        m_paths,
                        blevels,
                        BLOCK_SIZE=blk,
                        num_warps=8,
                    )
                cp["res"](
                    dt,
                    a["graph_unit_ptr"],
                    a["unit_comp_ptr"],
                    a["tree_control_ptr"],
                    a["storage_ptr"],
                    a["runoff_ptr"],
                    a["trial_storage_ptr"],
                    a["trial_outflow_ptr"],
                    a["upstream_offset_ptr"],
                    a["upstream_index_ptr"],
                    a["incident_offset_ptr"],
                    a["incident_path_ptr"],
                    a["graph_path_sum_ptr"],
                    a["residual_ptr"],
                    a["balance_weight_ptr"],
                    a["graph_rhs_ptr"],
                    a["graph_balance_scale_ptr"],
                    a["tree_merit_ptr"],
                    a["tree_line_ptr"],
                    ngu,
                    SINGLE_COMP=cp["single_comp"],
                    BLOCK_SIZE=blk,
                    num_warps=8,
                )
                ctl[5:7].zero_()
                cp["ctrl"](
                    a["tree_control_ptr"],
                    a["tree_scalars_ptr"],
                    a["tree_merit_ptr"],
                    a["tree_line_ptr"],
                    a["comp_rep_tree_ptr"],
                    ctl,
                    meta,
                    a["newton_status_ptr"],
                    tol,
                    maxit,
                    ngc,
                    BLOCK_SIZE=32,
                    num_warps=1,
                )
                cp["apply"](
                    a["graph_unit_ptr"],
                    a["unit_comp_ptr"],
                    a["tree_control_ptr"],
                    a["storage_ptr"],
                    a["river_length_ptr"],
                    a["river_width_ptr"],
                    a["river_height_ptr"],
                    a["catchment_area_ptr"],
                    a["flood_depth_table_ptr"],
                    a["newton_reference_ptr"],
                    a["newton_offset_ptr"],
                    a["trial_reference_ptr"],
                    a["trial_offset_ptr"],
                    a["balance_weight_ptr"],
                    a["accepted_weight_ptr"],
                    a["phase_trace_ptr"],
                    ngu,
                    n,
                    levels,
                    BLOCK_SIZE=blk,
                )

                active_c, solving_c = ctl[5:7].tolist()
                if active_c == 0:
                    if cp["need_meta_copy"]:
                        meta[cp["c_trees_long"]] = meta[cp["c_rep_for_tree"]]
                    lin_rec.copy_(
                        torch.tensor(
                            [
                                last_lin[0],
                                last_lin[1],
                                last_lin[2],
                                float(total_solves),
                            ],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    converged_loop = True
                    break

                if solving_c > 0:
                    cp["prep_rhs"](
                        a["graph_unit_ptr"],
                        a["unit_comp_ptr"],
                        a["tree_control_ptr"],
                        a["newton_reference_ptr"],
                        a["newton_offset_ptr"],
                        a["river_height_ptr"],
                        a["residual_ptr"],
                        a["graph_rhs_ptr"],
                        a["graph_balance_scale_ptr"],
                        a["graph_bound_ptr"],
                        ngu,
                        ALL_ACTIVE=cp["single_comp"],
                        BLOCK_SIZE=blk,
                    )
                    res_code, iters, err = cp["linear_solve"](dt)
                    total_solves += iters
                    last_lin = (float(res_code), float(iters), float(err))

                cp["trial"](
                    a["graph_unit_ptr"],
                    a["unit_comp_ptr"],
                    a["tree_control_ptr"],
                    a["tree_scalars_ptr"],
                    a["river_length_ptr"],
                    a["river_width_ptr"],
                    a["river_height_ptr"],
                    a["catchment_area_ptr"],
                    a["flood_depth_table_ptr"],
                    a["newton_reference_ptr"],
                    a["newton_offset_ptr"],
                    a["trial_reference_ptr"],
                    a["trial_offset_ptr"],
                    a["newton_step_ptr"],
                    a["trial_area_ptr"],
                    a["pivot_ptr"],
                    a["pivot_rhs_ptr"],
                    a["coupling_ptr"],
                    ngu,
                    levels,
                    BLOCK_SIZE=blk,
                )

            if not converged_loop:
                ctl[4] = int(NEWTON_ROUND_LIMIT)

        # 3. Commit main-channel & bifurcation state
        k_commit(
            a["trial_storage_ptr"],
            a["trial_outflow_ptr"],
            a["newton_reference_ptr"],
            a["newton_offset_ptr"],
            a["downstream_edge_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            ctl,
            a["storage_ptr"],
            a["outflow_ptr"],
            a["water_surface_ptr"],
            a["inflow_ptr"],
            a["newton_status_ptr"],
            pair_ptr,
            n,
            SAVE_SURFACE=save_surface,
            BLOCK_SIZE=blk,
        )
        if k_gcommit_p is not None:
            k_gcommit_p(
                a["graph_path_values_ptr"],
                a["bifurcation_outflow_ptr"],
                n_vals,
                BLOCK_SIZE=blk,
            )
        if k_gcommit_i is not None:
            k_gcommit_i(
                a["incident_offset_ptr"],
                a["incident_path_ptr"],
                a["graph_path_values_ptr"],
                a["inflow_ptr"],
                n,
                blevels,
                BLOCK_SIZE=blk,
            )

    return launch


def _prepare_graph_rebuild(arguments, buffer_dtypes):
    from hydronet.phys.triton import bifurcation as tk

    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    eps = float(a["slope_regularization"])

    blk = 256
    grid_n = (_cdiv(n, blk),)

    k_stage = launch_triton_kernel(tk.graph_rebuild_stage_kernel, grid_n, physics=True)
    k_mflux = launch_triton_kernel(tk.graph_main_flux_kernel, grid_n, physics=True)
    k_bflux = (
        launch_triton_kernel(
            tk.graph_bif_flux_kernel, (_cdiv(m_paths, blk),), physics=True
        )
        if m_paths > 0
        else None
    )
    k_res = launch_triton_kernel(tk.graph_residual_kernel, grid_n, physics=True)
    k_snap = launch_triton_kernel(
        tk.graph_rebuild_snapshot_kernel, grid_n, physics=True
    )

    def launch():
        dt = float(a["time_step_ptr"].item())
        k_stage(
            a["converged_surface_ptr"],
            a["river_length_ptr"],
            a["river_width_ptr"],
            a["river_height_ptr"],
            a["catchment_area_ptr"],
            a["flood_depth_table_ptr"],
            a["flood_bed_length_ptr"],
            a["newton_reference_ptr"],
            a["newton_offset_ptr"],
            a["trial_reference_ptr"],
            a["trial_offset_ptr"],
            a["trial_storage_ptr"],
            a["trial_area_ptr"],
            a["channel_conveyance_ptr"],
            a["flood_conveyance_ptr"],
            a["channel_conveyance_slope_ptr"],
            a["flood_conveyance_slope_ptr"],
            n,
            levels,
            BLOCK_SIZE=blk,
        )
        k_mflux(
            dt,
            eps,
            a["outflow_ptr"],
            a["river_manning_ptr"],
            a["flood_manning_ptr"],
            a["river_length_ptr"],
            a["river_width_ptr"],
            a["river_height_ptr"],
            a["catchment_area_ptr"],
            a["flood_depth_table_ptr"],
            a["flood_bed_length_ptr"],
            a["catchment_elevation_ptr"],
            a["downstream_distance_ptr"],
            a["downstream_idx_ptr"],
            a["downstream_edge_ptr"],
            a["trial_reference_ptr"],
            a["trial_offset_ptr"],
            a["trial_storage_ptr"],
            a["trial_area_ptr"],
            a["channel_conveyance_ptr"],
            a["flood_conveyance_ptr"],
            a["channel_conveyance_slope_ptr"],
            a["flood_conveyance_slope_ptr"],
            a["trial_outflow_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            n,
            levels,
            BLOCK_SIZE=blk,
        )
        if k_bflux is not None:
            k_bflux(
                dt,
                1.0,
                a["bifurcation_catchment_idx_ptr"],
                a["bifurcation_downstream_idx_ptr"],
                a["catchment_elevation_ptr"],
                a["river_height_ptr"],
                a["trial_reference_ptr"],
                a["trial_offset_ptr"],
                a["bifurcation_width_ptr"],
                a["bifurcation_length_ptr"],
                a["bifurcation_elevation_ptr"],
                a["bifurcation_manning_ptr"],
                a["bifurcation_outflow_ptr"],
                a["graph_path_values_ptr"],
                a["graph_path_sum_ptr"],
                a["newton_status_ptr"],
                m_paths,
                blevels,
                BLOCK_SIZE=blk,
            )
        k_res(
            dt,
            a["storage_ptr"],
            a["runoff_ptr"],
            a["trial_storage_ptr"],
            a["trial_outflow_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["graph_path_sum_ptr"],
            a["residual_ptr"],
            a["balance_weight_ptr"],
            a["graph_rhs_ptr"],
            n,
            BLOCK_SIZE=blk,
        )
        k_snap(
            dt,
            eps,
            a["outflow_ptr"],
            a["river_manning_ptr"],
            a["flood_manning_ptr"],
            a["river_length_ptr"],
            a["river_width_ptr"],
            a["river_height_ptr"],
            a["catchment_area_ptr"],
            a["flood_depth_table_ptr"],
            a["flood_bed_length_ptr"],
            a["catchment_elevation_ptr"],
            a["downstream_distance_ptr"],
            a["downstream_idx_ptr"],
            a["downstream_edge_ptr"],
            a["newton_reference_ptr"],
            a["newton_offset_ptr"],
            a["trial_storage_ptr"],
            a["trial_area_ptr"],
            a["trial_outflow_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["residual_ptr"],
            a["jacobian_snapshot_ptr"],
            n,
            levels,
            BLOCK_SIZE=blk,
        )
        if m_paths > 0:
            a["bifurcation_jacobian_ptr"].view(-1).copy_(
                a["graph_path_values_ptr"].view(-1)
            )

    return launch


def _prepare_graph_adjoint(arguments, buffer_dtypes):
    from hydronet.phys.triton import bifurcation as tk

    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    blk = 256
    grid_n = (_cdiv(n, blk),)

    linear_solve = _make_adjoint_linear_solver(a)
    k_prep_c = launch_triton_kernel(
        tk.graph_adjoint_prep_cells_kernel, grid_n, physics=True
    )
    k_prep_p = (
        launch_triton_kernel(
            tk.graph_adjoint_prep_paths_kernel, (_cdiv(m_paths, blk),), physics=True
        )
        if m_paths > 0
        else None
    )
    k_s1 = launch_triton_kernel(tk.graph_adjoint_step1_kernel, grid_n, physics=True)
    k_s2 = launch_triton_kernel(tk.graph_adjoint_step2_kernel, grid_n, physics=True)
    k_s3 = (
        launch_triton_kernel(
            tk.graph_adjoint_step3_kernel, (_cdiv(m_paths, blk),), physics=True
        )
        if m_paths > 0
        else None
    )
    k_report = launch_triton_kernel(tk.report_single_error_kernel, (1,), physics=True)
    lin_rec = a["linear_record_ptr"]

    def launch():
        dt = float(a["jacobian_snapshot_ptr"][0, 16].item())
        k_prep_c(
            a["jacobian_snapshot_ptr"],
            a["bifurcation_jacobian_ptr"],
            a["grad_outflow_ptr"],
            a["grad_storage_ptr"],
            a["grad_water_surface_ptr"],
            a["grad_bifurcation_outflow_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["outflow_own_slope_ptr"],
            a["outflow_down_slope_ptr"],
            a["trial_area_ptr"],
            a["graph_bound_ptr"],
            a["graph_rhs_ptr"],
            n,
            blevels,
            BLOCK_SIZE=blk,
        )
        if k_prep_p is not None:
            k_prep_p(
                a["bifurcation_jacobian_ptr"],
                a["graph_path_sum_ptr"],
                m_paths,
                blevels,
                BLOCK_SIZE=blk,
            )
        res_code, iters, err = linear_solve(dt)
        lin_rec[:3].copy_(
            torch.tensor(
                [float(res_code), float(iters), float(err)],
                dtype=torch.float32,
                device=lin_rec.device,
            )
        )
        if res_code != 0:
            k_report(a["newton_status_ptr"], int(ADJOINT_NOT_CONVERGED), 0, num_warps=1)

        k_s1(
            dt,
            a["jacobian_snapshot_ptr"],
            a["newton_step_ptr"],
            a["graph_solution_low_ptr"],
            a["downstream_edge_ptr"],
            a["downstream_idx_ptr"],
            a["grad_outflow_ptr"],
            a["grad_storage_ptr"],
            a["graph_temporary_ptr"],
            a["grad_runoff_ptr"],
            a["grad_river_width_ptr"],
            a["grad_river_manning_ptr"],
            a["grad_flood_manning_ptr"],
            n,
            BLOCK_SIZE=blk,
        )
        k_s2(
            dt,
            a["jacobian_snapshot_ptr"],
            a["bifurcation_jacobian_ptr"],
            a["newton_step_ptr"],
            a["graph_solution_low_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["incident_offset_ptr"],
            a["incident_path_ptr"],
            a["bifurcation_catchment_idx_ptr"],
            a["bifurcation_downstream_idx_ptr"],
            a["graph_temporary_ptr"],
            a["grad_storage_ptr"],
            a["grad_bifurcation_outflow_ptr"],
            a["grad_river_height_ptr"],
            n,
            blevels,
            BLOCK_SIZE=blk,
        )
        if k_s3 is not None:
            k_s3(
                dt,
                a["jacobian_snapshot_ptr"],
                a["bifurcation_jacobian_ptr"],
                a["newton_step_ptr"],
                a["graph_solution_low_ptr"],
                a["bifurcation_catchment_idx_ptr"],
                a["bifurcation_downstream_idx_ptr"],
                a["grad_bifurcation_outflow_ptr"],
                a["grad_bifurcation_manning_ptr"],
                m_paths,
                blevels,
                BLOCK_SIZE=blk,
            )

    return launch
