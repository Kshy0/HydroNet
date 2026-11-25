"""FP32 implicit local-inertia forest step, snapshot rebuild and exact adjoint.

The nonlinear unknown is the end-of-step water surface. Channel and floodplain
Manning roughness are inputs, fixed during Newton. The adjoint solves the
transpose of the converged continuity Jacobian; it does not unroll Newton, the
line search or the damping.

Step: one block per tree below ``options.grid_tree_size`` catchments, and one
persistent cooperative launch with one block per multiprocessor for the larger
trees (HydroForge checks that every block is resident; grid barriers are
``cooperative_groups`` grid syncs); then the commit of storage, outflow, surface
and inflow, and, when the model is differentiable, of the converged surface.
Rebuild (in the reverse step, which has the step's time step): the converged
Jacobian snapshot, evaluated again at that surface from the step's inputs (the
evaluation Newton ended with, bit for bit, without Newton). Adjoint: one launch
per heavy-path round in each direction, then the local parameter adjoints. Both
``"cuda"`` and ``"triton"`` backends are registered for every forest kernel.
"""

from __future__ import annotations

from pathlib import Path

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
    GRID_SOLVE_THREADS,
    NEWTON_DAMPED_GROWTH,
    NEWTON_DAMPED_SIGMA,
    NEWTON_DAMPED_STEPS,
    NEWTON_ROUND_LIMIT,
    PARALLEL_PATH_MAX,
    PARALLEL_PATH_MIN,
    PARALLEL_PATH_TOLERANCE_MIN,
    TREE_INTEGERS,
    TREE_SCALARS,
)
from hydronet.phys.specs import INERTIA_ADJOINT, INERTIA_REBUILD, INERTIA_STEP

_SOURCE = CudaSource(
    path=Path(__file__).with_name("cuda") / "implicit_inertia.cu",
    options=(
        f"-DSOLVE_THREADS={GRID_SOLVE_THREADS}",
        f"-DPARALLEL_PATH_MIN={PARALLEL_PATH_MIN}",
        f"-DPARALLEL_PATH_MAX={PARALLEL_PATH_MAX}",
        f"-DPARALLEL_PATH_TOLERANCE_MIN={PARALLEL_PATH_TOLERANCE_MIN}f",
        f"-DNEWTON_DAMPED_STEPS={NEWTON_DAMPED_STEPS}",
        f"-DNEWTON_DAMPED_SIGMA={NEWTON_DAMPED_SIGMA!r}f",
        f"-DNEWTON_DAMPED_GROWTH={NEWTON_DAMPED_GROWTH!r}f",
        f"-DTREE_INTEGERS={TREE_INTEGERS}",
        f"-DTREE_SCALARS={TREE_SCALARS}",
    ),
    toolkit_headers=True,
)
SOLVE_THREADS = GRID_SOLVE_THREADS
TREE_THREADS = 1024
UNIT_THREADS = 256
PATH_THREADS = 256

# Every kernel takes its spec's values by name (HYDROFORGE_KERNEL_PARAMETERS).
_COMMIT = CudaCall(kernel="inertia_commit", threads=UNIT_THREADS)


def _triton_step():
    return TritonProgram(_prepare_inertia_step)


def _triton_rebuild():
    return TritonProgram(_prepare_inertia_rebuild)


def _triton_adjoint():
    return TritonProgram(_prepare_inertia_adjoint)


def _metal_step():
    from hydronet.phys.metal import metal_inertia_step

    return metal_inertia_step


def _metal_rebuild():
    from hydronet.phys.metal import metal_inertia_rebuild

    return metal_inertia_rebuild


def _metal_adjoint():
    from hydronet.phys.metal import metal_inertia_adjoint

    return metal_inertia_adjoint


solve_inertia_step = BackendRegistry(
    INERTIA_STEP,
    {
        "cuda": CudaKernel(
            _SOURCE,
            steps=(
                CudaFill(buffer="grid_control_ptr"),
                # Trees below grid_tree_size: one block each.
                CudaCall(
                    kernel="inertia_solve_trees",
                    size="num_block_trees",
                    grid="blocks",
                    threads=TREE_THREADS,
                ),
                # Larger trees: one persistent cooperative block per multiprocessor.
                CudaCall(
                    kernel="inertia_solve",
                    size="num_grid_trees",
                    grid="multiprocessors",
                    cooperative=True,
                    threads=SOLVE_THREADS,
                ),
                _COMMIT,
            ),
        ),
        "triton": _triton_step,
        "metal": _metal_step,
    },
)
# The converged snapshot again: stage, flux and residual at the converged surface.
solve_inertia_rebuild = BackendRegistry(
    INERTIA_REBUILD,
    {
        "cuda": CudaKernel(
            _SOURCE,
            steps=(
                CudaCall(kernel="inertia_rebuild_stage", threads=UNIT_THREADS),
                CudaCall(kernel="inertia_rebuild_flux", threads=UNIT_THREADS),
                CudaCall(kernel="inertia_rebuild_residual", threads=UNIT_THREADS),
                CudaCall(kernel="inertia_rebuild_snapshot", threads=UNIT_THREADS),
            ),
        ),
        "triton": _triton_rebuild,
        "metal": _metal_rebuild,
    },
)
# Heavy-path rounds upwards, then downwards, then the local parameter adjoints.
solve_inertia_adjoint = BackendRegistry(
    INERTIA_ADJOINT,
    {
        "cuda": CudaKernel(
            _SOURCE,
            steps=(
                CudaCall(
                    kernel="inertia_adjoint_up",
                    size="max_round_paths",
                    threads=PATH_THREADS,
                    rounds="num_path_rounds",
                    reverse_rounds=True,
                ),
                CudaCall(
                    kernel="inertia_adjoint_down",
                    size="max_round_paths",
                    threads=PATH_THREADS,
                    rounds="num_path_rounds",
                ),
                CudaCall(kernel="inertia_adjoint_finish", threads=UNIT_THREADS),
            ),
        ),
        "triton": _triton_adjoint,
        "metal": _metal_adjoint,
    },
)


# --------------------------------------------------------------------------- #
# TritonProgram builders
# --------------------------------------------------------------------------- #


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_pcr_block(length: int) -> int:
    if length <= 32:
        return 32
    if length <= 256:
        return 256
    return 1024


def _pcr_warps(block: int) -> int:
    return {32: 1, 256: 8, 1024: 32}[block]


def _prepare_inertia_step(arguments, buffer_dtypes):
    from hydronet.phys.triton import implicit_inertia as tk

    del buffer_dtypes
    device = arguments["storage_ptr"].device
    n = int(arguments["num_catchments"])
    levels = int(arguments["num_flood_levels"])
    ngu = int(arguments["num_grid_units"])
    ngt = int(arguments["num_grid_trees"])
    nbt = int(arguments["num_block_trees"])
    rounds = int(arguments["num_path_rounds"])
    eps = float(arguments["slope_regularization"])
    tol = float(arguments["newton_tolerance"])
    maxit = int(arguments["newton_max_iterations"])

    # Sort block trees descending by size into one launch so the largest trees start
    # immediately and tiny trees backfill SMs without serial tail stalls.
    tree_launch = None
    btree_sorted = None
    if nbt > 0:
        btree_cpu = arguments["block_tree_ptr"].cpu()
        tsize_cpu = arguments["tree_size_ptr"].cpu()
        order = torch.argsort(tsize_cpu[btree_cpu.long()], descending=True, stable=True)
        btree_sorted = (
            btree_cpu[order].to(device=device, dtype=torch.int32).contiguous()
        )
        tree_launch = launch_triton_kernel(
            tk.inertia_solve_trees_kernel, (nbt,), physics=True
        )

    # Prepare grid-solved large trees (if any).
    grid_plan = None
    if ngt > 0 and ngu > 0:
        pptr_cpu = arguments["path_offset_ptr"].cpu()
        gpath_cpu = arguments["grid_path_ptr"].cpu()
        grr_cpu = arguments["grid_round_offset_ptr"].cpu().tolist()
        use_pcr = tol >= PARALLEL_PATH_TOLERANCE_MIN

        round_plans = []
        max_pcr_nodes = 0
        for r in range(rounds):
            rb, re = grr_cpu[r], grr_cpu[r + 1]
            r_paths = gpath_cpu[rb:re]
            if not use_pcr:
                p_dev = r_paths.to(device=device, dtype=torch.int32).contiguous()
                n_p = int(p_dev.numel())
                elim_s = (
                    launch_triton_kernel(
                        tk.grid_eliminate_serial_kernel, (_cdiv(n_p, 32),), physics=True
                    )
                    if n_p
                    else None
                )
                sub_s = (
                    launch_triton_kernel(
                        tk.grid_substitute_serial_kernel,
                        (_cdiv(n_p, 32),),
                        physics=True,
                    )
                    if n_p
                    else None
                )
                round_plans.append(((p_dev, n_p, elim_s, sub_s), (), None))
            else:
                lens = (
                    pptr_cpu[r_paths.long() + 1] - pptr_cpu[r_paths.long()]
                ).tolist()
                serial_list = []
                pcr_buckets: dict[int, list[tuple[int, int]]] = {}
                all_pcr_ids = []
                for p_id, plen in zip(r_paths.tolist(), lens):
                    if PARALLEL_PATH_MIN <= plen <= PARALLEL_PATH_MAX:
                        pcr_buckets.setdefault(_next_pcr_block(plen), []).append(
                            (p_id, plen)
                        )
                        all_pcr_ids.append(p_id)
                    else:
                        serial_list.append(p_id)
                if serial_list:
                    sp_dev = torch.tensor(serial_list, dtype=torch.int32, device=device)
                    n_sp = len(serial_list)
                    elim_s = launch_triton_kernel(
                        tk.grid_eliminate_serial_kernel,
                        (_cdiv(n_sp, 32),),
                        physics=True,
                    )
                    sub_s = launch_triton_kernel(
                        tk.grid_substitute_serial_kernel,
                        (_cdiv(n_sp, 32),),
                        physics=True,
                    )
                    serial_entry = (sp_dev, n_sp, elim_s, sub_s)
                else:
                    serial_entry = (None, 0, None, None)

                offset = 0
                bucket_entries = []
                for blk in sorted(pcr_buckets):
                    items = pcr_buckets[blk]
                    p_ids = [x[0] for x in items]
                    slots = []
                    for _, plen in items:
                        slots.append(offset)
                        offset += plen
                    bp_dev = torch.tensor(p_ids, dtype=torch.int32, device=device)
                    bs_dev = torch.tensor(slots, dtype=torch.int32, device=device)
                    el_l = launch_triton_kernel(
                        tk.grid_eliminate_pcr_kernel, (len(p_ids),), physics=True
                    )
                    bucket_entries.append((bp_dev, bs_dev, blk, _pcr_warps(blk), el_l))
                if all_pcr_ids:
                    all_pcr_dev = torch.tensor(
                        all_pcr_ids, dtype=torch.int32, device=device
                    )
                    sub_pcr_l = launch_triton_kernel(
                        tk.grid_substitute_pcr_kernel, (len(all_pcr_ids),), physics=True
                    )
                    sub_pcr_entry = (all_pcr_dev, sub_pcr_l)
                else:
                    sub_pcr_entry = None
                max_pcr_nodes = max(max_pcr_nodes, offset)
                round_plans.append((serial_entry, tuple(bucket_entries), sub_pcr_entry))

        pcr_scratch = (
            torch.empty(10 * max_pcr_nodes, dtype=torch.float32, device=device)
            if max_pcr_nodes > 0
            else None
        )
        single_tree = ngt == 1
        g_blk = 256
        g_grid = (_cdiv(ngu, g_blk),)
        t_grid = (_cdiv(ngt, 32),)
        grid_plan = dict(
            start_u=launch_triton_kernel(
                tk.grid_start_units_kernel, g_grid, physics=True
            ),
            start_t=launch_triton_kernel(
                tk.grid_start_trees_kernel, t_grid, physics=True
            ),
            stage=launch_triton_kernel(tk.grid_stage_kernel, g_grid, physics=True),
            flux=launch_triton_kernel(tk.grid_flux_kernel, g_grid, physics=True),
            residual=launch_triton_kernel(
                tk.grid_residual_kernel, g_grid, physics=True
            ),
            control=launch_triton_kernel(tk.grid_control_kernel, t_grid, physics=True),
            apply=launch_triton_kernel(tk.grid_apply_kernel, g_grid, physics=True),
            trial=launch_triton_kernel(tk.grid_trial_kernel, g_grid, physics=True),
            rounds=tuple(round_plans),
            pcr_scratch=pcr_scratch,
            pcr_stride=max_pcr_nodes,
            single_tree=single_tree,
            g_blk=g_blk,
        )

    save_surface = arguments["converged_surface_ptr"] is not None
    pair_ptr = (
        arguments["converged_surface_ptr"] if save_surface else arguments["storage_ptr"]
    )
    commit_launch = launch_triton_kernel(
        tk.inertia_commit_kernel, (_cdiv(n, 256),), physics=True
    )

    # Local bindings for hot launch loop
    a = arguments
    ctl = a["grid_control_ptr"]

    def launch():
        ctl.zero_()
        # 1. Small/medium trees: one CTA per tree (BLOCK_SIZE=256, 8 warps = 1 unit/thread)
        if tree_launch is not None:
            tree_launch(
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
                btree_sorted,
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
                a["newton_record_ptr"],
                a["newton_status_ptr"],
                eps,
                tol,
                maxit,
                n,
                levels,
                BLOCK_SIZE=256,
                num_warps=8,
            )

        # 2. Large grid-solved trees
        if grid_plan is not None:
            gp = grid_plan
            gb = gp["g_blk"]
            gp["start_u"](
                a["grid_unit_ptr"],
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
                a["newton_status_ptr"],
                ngu,
                levels,
                BLOCK_SIZE=gb,
                num_warps=8,
            )
            gp["start_t"](
                a["grid_tree_ptr"],
                a["tree_control_ptr"],
                a["tree_scalars_ptr"],
                a["tree_merit_ptr"],
                a["tree_line_ptr"],
                a["time_step_ptr"],
                ngt,
                BLOCK_SIZE=32,
                num_warps=1,
            )
            finished = False
            for _ in range(1 << 22):
                gp["stage"](
                    a["grid_unit_ptr"],
                    a["tree_index_ptr"],
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
                    BLOCK_SIZE=gb,
                    num_warps=8,
                )
                gp["flux"](
                    a["grid_unit_ptr"],
                    a["tree_index_ptr"],
                    a["tree_control_ptr"],
                    a["tree_scalars_ptr"],
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
                    eps,
                    ngu,
                    levels,
                    BLOCK_SIZE=gb,
                    num_warps=8,
                )
                gp["residual"](
                    a["grid_unit_ptr"],
                    a["grid_tree_ptr"],
                    a["tree_index_ptr"],
                    a["tree_control_ptr"],
                    a["tree_scalars_ptr"],
                    a["storage_ptr"],
                    a["runoff_ptr"],
                    a["trial_storage_ptr"],
                    a["trial_outflow_ptr"],
                    a["upstream_offset_ptr"],
                    a["upstream_index_ptr"],
                    a["residual_ptr"],
                    a["balance_weight_ptr"],
                    a["accepted_weight_ptr"],
                    a["tree_merit_ptr"],
                    a["tree_line_ptr"],
                    ngu,
                    SINGLE_TREE=gp["single_tree"],
                    BLOCK_SIZE=gb,
                    num_warps=8,
                )
                ctl[:4].zero_()
                gp["control"](
                    a["grid_tree_ptr"],
                    a["tree_control_ptr"],
                    a["tree_scalars_ptr"],
                    a["tree_merit_ptr"],
                    a["tree_line_ptr"],
                    ctl,
                    a["newton_record_ptr"],
                    a["newton_status_ptr"],
                    a["time_step_ptr"],
                    tol,
                    maxit,
                    ngt,
                    BLOCK_SIZE=32,
                    num_warps=1,
                )
                gp["apply"](
                    a["grid_unit_ptr"],
                    a["tree_index_ptr"],
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
                    n,
                    ngu,
                    levels,
                    BLOCK_SIZE=gb,
                    num_warps=8,
                )
                ctl_vals = ctl[:3].tolist()
                n_active, n_solving = ctl_vals[0], ctl_vals[2]
                if n_active == 0:
                    finished = True
                    break
                if n_solving > 0:
                    for r in range(rounds - 1, -1, -1):
                        (sp_dev, n_sp, elim_s, _), buckets, _ = gp["rounds"][r]
                        if n_sp > 0:
                            elim_s(
                                sp_dev,
                                n_sp,
                                a["path_unit_ptr"],
                                a["path_offset_ptr"],
                                a["tree_index_ptr"],
                                a["tree_control_ptr"],
                                a["tree_scalars_ptr"],
                                a["river_height_ptr"],
                                a["downstream_edge_ptr"],
                                a["upstream_offset_ptr"],
                                a["upstream_index_ptr"],
                                a["newton_reference_ptr"],
                                a["newton_offset_ptr"],
                                a["residual_ptr"],
                                a["trial_area_ptr"],
                                a["outflow_own_slope_ptr"],
                                a["outflow_down_slope_ptr"],
                                a["pivot_ptr"],
                                a["pivot_rhs_ptr"],
                                a["coupling_ptr"],
                                BLOCK_SIZE=32,
                                num_warps=1,
                            )
                        for bp_dev, bs_dev, blk, warps, el_l in buckets:
                            el_l(
                                bp_dev,
                                bs_dev,
                                gp["pcr_scratch"],
                                gp["pcr_stride"],
                                a["path_unit_ptr"],
                                a["path_offset_ptr"],
                                a["tree_index_ptr"],
                                a["tree_control_ptr"],
                                a["tree_scalars_ptr"],
                                a["river_height_ptr"],
                                a["downstream_edge_ptr"],
                                a["upstream_offset_ptr"],
                                a["upstream_index_ptr"],
                                a["newton_reference_ptr"],
                                a["newton_offset_ptr"],
                                a["residual_ptr"],
                                a["trial_area_ptr"],
                                a["outflow_own_slope_ptr"],
                                a["outflow_down_slope_ptr"],
                                a["pivot_ptr"],
                                a["pivot_rhs_ptr"],
                                a["coupling_ptr"],
                                BLOCK_SIZE=blk,
                                num_warps=warps,
                            )
                    for r in range(rounds):
                        (sp_dev, n_sp, _, sub_s), _, sub_pcr_entry = gp["rounds"][r]
                        if n_sp > 0:
                            sub_s(
                                sp_dev,
                                n_sp,
                                a["path_unit_ptr"],
                                a["path_offset_ptr"],
                                a["tree_index_ptr"],
                                a["tree_control_ptr"],
                                a["downstream_idx_ptr"],
                                a["downstream_edge_ptr"],
                                a["pivot_ptr"],
                                a["pivot_rhs_ptr"],
                                a["coupling_ptr"],
                                a["newton_step_ptr"],
                                BLOCK_SIZE=32,
                                num_warps=1,
                            )
                        if sub_pcr_entry is not None:
                            all_pcr_dev, sb_l = sub_pcr_entry
                            sb_l(
                                all_pcr_dev,
                                a["path_unit_ptr"],
                                a["path_offset_ptr"],
                                a["tree_index_ptr"],
                                a["tree_control_ptr"],
                                a["downstream_idx_ptr"],
                                a["downstream_edge_ptr"],
                                a["pivot_rhs_ptr"],
                                a["coupling_ptr"],
                                a["newton_step_ptr"],
                                BLOCK_SIZE=64,
                                num_warps=2,
                            )
                gp["trial"](
                    a["grid_unit_ptr"],
                    a["tree_index_ptr"],
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
                    BLOCK_SIZE=gb,
                    num_warps=8,
                )
            if not finished:
                ctl[4] = NEWTON_ROUND_LIMIT

        # 3. Commit
        commit_launch(
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
            BLOCK_SIZE=256,
            num_warps=8,
        )

    return launch


def _prepare_inertia_rebuild(arguments, buffer_dtypes):
    from hydronet.phys.triton import implicit_inertia as tk

    del buffer_dtypes
    n = int(arguments["num_catchments"])
    levels = int(arguments["num_flood_levels"])
    eps = float(arguments["slope_regularization"])
    grid = (_cdiv(n, 256),)
    stage_l = launch_triton_kernel(tk.rebuild_stage_kernel, grid, physics=True)
    flux_l = launch_triton_kernel(tk.rebuild_flux_kernel, grid, physics=True)
    res_l = launch_triton_kernel(tk.rebuild_residual_kernel, grid, physics=True)
    snap_l = launch_triton_kernel(tk.rebuild_snapshot_kernel, grid, physics=True)
    a = arguments

    def launch():
        stage_l(
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
            BLOCK_SIZE=256,
        )
        flux_l(
            a["time_step_ptr"],
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
            BLOCK_SIZE=256,
        )
        res_l(
            a["time_step_ptr"],
            a["storage_ptr"],
            a["runoff_ptr"],
            a["trial_storage_ptr"],
            a["trial_outflow_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["residual_ptr"],
            a["balance_weight_ptr"],
            n,
            BLOCK_SIZE=256,
        )
        snap_l(
            a["time_step_ptr"],
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
            BLOCK_SIZE=256,
        )

    return launch


def _prepare_inertia_adjoint(arguments, buffer_dtypes):
    from hydronet.phys.triton import implicit_inertia as tk

    del buffer_dtypes
    n = int(arguments["num_catchments"])
    rounds = int(arguments["num_path_rounds"])
    rptr = arguments["round_offset_ptr"].cpu().tolist()
    up_launches = []
    down_launches = []
    for r in range(rounds):
        rb, re = rptr[r], rptr[r + 1]
        cnt = re - rb
        if cnt > 0:
            grid = (_cdiv(cnt, 32),)
            up_launches.append(
                (rb, re, launch_triton_kernel(tk.adjoint_up_kernel, grid, physics=True))
            )
            down_launches.append(
                (
                    rb,
                    re,
                    launch_triton_kernel(tk.adjoint_down_kernel, grid, physics=True),
                )
            )
        else:
            up_launches.append((rb, re, None))
            down_launches.append((rb, re, None))
    finish_l = launch_triton_kernel(
        tk.adjoint_finish_kernel, (_cdiv(n, 256),), physics=True
    )
    a = arguments

    def launch():
        for r in range(rounds - 1, -1, -1):
            rb, re, up_l = up_launches[r]
            if up_l is not None:
                up_l(
                    a["jacobian_snapshot_ptr"],
                    a["grad_outflow_ptr"],
                    a["grad_storage_ptr"],
                    a["grad_water_surface_ptr"],
                    a["upstream_offset_ptr"],
                    a["upstream_index_ptr"],
                    a["path_unit_ptr"],
                    a["path_offset_ptr"],
                    a["pivot_ptr"],
                    a["pivot_rhs_ptr"],
                    rb,
                    re,
                    BLOCK_SIZE=32,
                    num_warps=1,
                )
        for r in range(rounds):
            rb, re, dn_l = down_launches[r]
            if dn_l is not None:
                dn_l(
                    a["jacobian_snapshot_ptr"],
                    a["grad_outflow_ptr"],
                    a["downstream_idx_ptr"],
                    a["downstream_edge_ptr"],
                    a["path_unit_ptr"],
                    a["path_offset_ptr"],
                    a["pivot_ptr"],
                    a["pivot_rhs_ptr"],
                    a["adjoint_continuity_ptr"],
                    a["adjoint_flux_ptr"],
                    a["grad_runoff_ptr"],
                    rb,
                    re,
                    BLOCK_SIZE=32,
                    num_warps=1,
                )
        finish_l(
            a["jacobian_snapshot_ptr"],
            a["grad_storage_ptr"],
            a["upstream_offset_ptr"],
            a["upstream_index_ptr"],
            a["adjoint_continuity_ptr"],
            a["adjoint_flux_ptr"],
            a["grad_river_width_ptr"],
            a["grad_river_height_ptr"],
            a["grad_river_manning_ptr"],
            a["grad_flood_manning_ptr"],
            n,
            BLOCK_SIZE=256,
        )

    return launch
