# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton forest solver, snapshot rebuild and heavy-path adjoint."""

from __future__ import annotations

import triton
import triton.language as tl
from hydroforge.kernels.triton_errors import report_error

from hydronet.phys.triton.common import (
    CODE_INVALID_ROUGHNESS,
    CODE_INVALID_STATE,
    CODE_NEWTON_NOT_CONVERGED,
    CODE_NEWTON_ROUND_LIMIT,
    CTL_ACTIVE,
    CTL_ERROR,
    CTL_SOLVING,
    DAMPED_PHASE,
    IDX_BEST,
    IDX_BEST_ST,
    IDX_BT,
    IDX_COMMIT,
    IDX_DSTEPS,
    IDX_DT,
    IDX_FEVALS,
    IDX_INITIAL,
    IDX_LAM,
    IDX_MERIT,
    IDX_MODE,
    IDX_PHASE,
    IDX_RESET,
    IDX_SAVE,
    IDX_SIGMA,
    IDX_SOLVE,
    IDX_ST,
    IDX_STEPS,
    IDX_SUM_FE,
    IDX_SUM_G,
    IDX_TOTALBT,
    MAX_ROUNDS,
    MODE_DONE,
    MODE_EVAL,
    MODE_TRIAL,
    RESTART_PHASE,
    TF,
    TI,
    eval_flux_vec_inline,
    invert_storage_inline,
    isfinite_f32_inline,
    newton_control_step_inline,
    pair_add_inline,
    phase_offset_value_inline,
    snapshot_unit_vec_inline,
    stage_pair_inline,
    terrain_coordinate_inline,
    terrain_inverse_inline,
    trial_pair_inline,
)

# --------------------------------------------------------------------------- #
# Shared helper: block-per-tree Newton solve (one CTA per tree)
# --------------------------------------------------------------------------- #


@triton.jit
def inertia_solve_trees_kernel(
    S0_ptr,
    runoff_ptr,
    Q0_ptr,
    nc_ptr,
    nf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    elev_ptr,
    dist_ptr,
    dt0_ptr,
    down_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    btree_ptr,
    tunit_ptr,
    loff_ptr,
    tlev_ptr,
    h_ptr,
    ht_ptr,
    ref_ptr,
    tref_ptr,
    S_ptr,
    area_ptr,
    kc_ptr,
    kf_ptr,
    dkc_ptr,
    dkf_ptr,
    Q_ptr,
    qi_ptr,
    qd_ptr,
    R_ptr,
    wb_ptr,
    wa_ptr,
    ap_ptr,
    rp_ptr,
    dh_ptr,
    b_ptr,
    trace_ptr,
    meta_ptr,
    status_ptr,
    eps,
    tol,
    maxit,
    n,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    t = tl.load(btree_ptr + pid)
    lev_begin = tl.load(tlev_ptr + t)
    lev_end = tl.load(tlev_ptr + t + 1)
    first = tl.load(loff_ptr + lev_begin)
    last = tl.load(loff_ptr + lev_end)
    dt0 = tl.load(dt0_ptr)
    lanes = tl.arange(0, BLOCK_SIZE)

    # Start units of this tree
    off = first
    while off < last:
        k = off + lanes
        m = k < last
        i = tl.load(tunit_ptr + k, mask=m, other=0)
        s0 = tl.load(S0_ptr + i, mask=m, other=0.0)
        length = tl.load(L_ptr + i, mask=m, other=1.0)
        w = tl.load(W_ptr + i, mask=m, other=1.0)
        hh = tl.load(H_ptr + i, mask=m, other=0.0)
        Ai = tl.load(A_ptr + i, mask=m, other=1.0)
        r0 = invert_storage_inline(i, s0, length, w, hh, Ai, tab_ptr, m, levels)
        z0 = tl.zeros_like(r0)
        tl.store(ref_ptr + i, r0, mask=m)
        tl.store(h_ptr + i, z0, mask=m)
        tl.store(tref_ptr + i, r0, mask=m)
        tl.store(ht_ptr + i, z0, mask=m)
        nc_i = tl.load(nc_ptr + i, mask=m, other=1.0)
        nf_i = tl.load(nf_ptr + i, mask=m, other=1.0)
        bad_n = m & (
            ~(
                (nc_i > 0.0)
                & isfinite_f32_inline(nc_i)
                & (nf_i > 0.0)
                & isfinite_f32_inline(nf_i)
            )
        )
        if tl.max(bad_n.to(tl.int32), axis=0) != 0:
            report_error(status_ptr, bad_n, CODE_INVALID_ROUGHNESS, i)
        off += BLOCK_SIZE
    tl.debug_barrier()

    # Tree scalar controller state (uniform across the CTA)
    mode = tl.full((), 1, tl.int32)  # MODE_EVAL
    phase = tl.full((), -1, tl.int32)
    steps = tl.full((), 0, tl.int32)
    bt = tl.full((), 0, tl.int32)
    st = tl.full((), 0, tl.int32)
    fevals = tl.full((), 0, tl.int32)
    totalbt = tl.full((), 0, tl.int32)
    sum_g = tl.full((), 0, tl.int32)
    sum_fe = tl.full((), 0, tl.int32)
    dsteps = tl.full((), 0, tl.int32)
    best_st = tl.full((), 2, tl.int32)
    dt = dt0 + tl.full((), 0.0, tl.float32)
    lam = tl.full((), 0.0, tl.float32)
    merit = tl.full((), 0.0, tl.float32)
    best = tl.full((), 1e30, tl.float32)
    initial = tl.full((), 0.0, tl.float32)
    sigma = tl.full((), 0.0, tl.float32)

    round_idx = tl.full((), 0, tl.int32)
    inf_val = tl.full((), float("inf"), tl.float32)

    while (round_idx < MAX_ROUNDS) & (mode != MODE_DONE):
        # 1. Stage evaluation
        off = first
        while off < last:
            k = off + lanes
            m = k < last
            i = tl.load(tunit_ptr + k, mask=m, other=0)
            tr = tl.load(tref_ptr + i, mask=m, other=0.0)
            ht = tl.load(ht_ptr + i, mask=m, other=0.0)
            length = tl.load(L_ptr + i, mask=m, other=1.0)
            w = tl.load(W_ptr + i, mask=m, other=1.0)
            hh = tl.load(H_ptr + i, mask=m, other=0.0)
            Ai = tl.load(A_ptr + i, mask=m, other=1.0)
            s_v, a_v, kc_v, kf_v, dkc_v, dkf_v = stage_pair_inline(
                i, tr, ht, length, w, hh, Ai, tab_ptr, bed_ptr, m, levels
            )
            tl.store(S_ptr + i, s_v, mask=m)
            tl.store(area_ptr + i, a_v, mask=m)
            tl.store(kc_ptr + i, kc_v, mask=m)
            tl.store(kf_ptr + i, kf_v, mask=m)
            tl.store(dkc_ptr + i, dkc_v, mask=m)
            tl.store(dkf_ptr + i, dkf_v, mask=m)
            off += BLOCK_SIZE
        tl.debug_barrier()

        # 2. Flux evaluation
        off = first
        while off < last:
            k = off + lanes
            m = k < last
            i = tl.load(tunit_ptr + k, mask=m, other=0)
            q_v, qi_v, qd_v = eval_flux_vec_inline(
                i,
                dt,
                eps,
                m,
                Q0_ptr,
                nc_ptr,
                nf_ptr,
                L_ptr,
                W_ptr,
                H_ptr,
                A_ptr,
                tab_ptr,
                bed_ptr,
                elev_ptr,
                dist_ptr,
                down_ptr,
                edge_ptr,
                tref_ptr,
                ht_ptr,
                S_ptr,
                area_ptr,
                kc_ptr,
                kf_ptr,
                dkc_ptr,
                dkf_ptr,
                levels,
            )
            tl.store(Q_ptr + i, q_v, mask=m)
            tl.store(qi_ptr + i, qi_v, mask=m)
            tl.store(qd_ptr + i, qd_v, mask=m)
            off += BLOCK_SIZE
        tl.debug_barrier()

        # 3. Residual evaluation + block-wide max reduction
        weighted = (phase == RESTART_PHASE) | (phase == DAMPED_PHASE)
        m_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        l_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        off = first
        while off < last:
            k = off + lanes
            m = k < last
            i = tl.load(tunit_ptr + k, mask=m, other=0)
            e = tl.load(cp_ptr + i, mask=m, other=0)
            e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
            e_act = m & (e < e_end)
            inflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            ainflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            while tl.max(e_act.to(tl.int32), axis=0) != 0:
                kid = tl.load(kids_ptr + e, mask=e_act, other=0)
                qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
                inflow_q = tl.where(e_act, inflow_q + qk, inflow_q)
                ainflow_q = tl.where(e_act, ainflow_q + tl.abs(qk), ainflow_q)
                e += 1
                e_act = m & (e < e_end)
            Qi = tl.load(Q_ptr + i, mask=m, other=0.0)
            Si = tl.load(S_ptr + i, mask=m, other=0.0)
            S0i = tl.load(S0_ptr + i, mask=m, other=0.0)
            ri = tl.load(runoff_ptr + i, mask=m, other=0.0)
            r = (Si - S0i) - dt * (ri + inflow_q - Qi)
            scale = S0i + dt * (tl.abs(ri) + ainflow_q + tl.abs(Qi)) + 1000.0
            fin = isfinite_f32_inline(r) & isfinite_f32_inline(Qi)
            score = tl.where(fin, tl.div_rn(tl.abs(r), scale), inf_val)
            wb = tl.div_rn(tl.full((), 1.0, tl.float32), scale)
            tl.store(R_ptr + i, r, mask=m)
            tl.store(wb_ptr + i, wb, mask=m)
            m_vec = tl.where(m, tl.maximum(m_vec, score), m_vec)
            if weighted:
                wai = tl.load(wa_ptr + i, mask=m, other=0.0)
                lk = tl.where(isfinite_f32_inline(score), tl.abs(r) * wai, inf_val)
                l_vec = tl.where(m, tl.maximum(l_vec, lk), l_vec)
            off += BLOCK_SIZE
        mt = tl.max(m_vec, axis=0)
        line_val = tl.max(l_vec, axis=0)

        # 4. Scalar Newton controller
        (
            mode,
            phase,
            steps,
            bt,
            st,
            fevals,
            totalbt,
            sum_g,
            sum_fe,
            dsteps,
            best_st,
            dt,
            lam,
            merit,
            best,
            initial,
            sigma,
            commit,
            save,
            reset,
            solve,
            finished_now,
        ) = newton_control_step_inline(
            mode,
            phase,
            steps,
            bt,
            st,
            fevals,
            totalbt,
            sum_g,
            sum_fe,
            dsteps,
            best_st,
            dt,
            lam,
            merit,
            best,
            initial,
            sigma,
            mt,
            line_val,
            dt0,
            tol,
            maxit,
            False,
        )

        # 5. Apply unit (commit / wa / save / reset)
        need_wa = (solve != 0) & ((phase == RESTART_PHASE) | (phase == DAMPED_PHASE))
        need_apply = (commit != 0) | need_wa | (save != 0) | (reset != 0)
        if need_apply:
            off = first
            while off < last:
                k = off + lanes
                m = k < last
                i = tl.load(tunit_ptr + k, mask=m, other=0)
                ref_v = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                h_v = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
                if commit != 0:
                    ref_v = tl.load(tref_ptr + i, mask=m, other=0.0)
                    h_v = tl.load(ht_ptr + i, mask=m, other=0.0)
                    tl.store(ref_ptr + i, ref_v, mask=m)
                    tl.store(h_ptr + i, h_v, mask=m)
                else:
                    if (save & 3) != 0:
                        ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
                        h_v = tl.load(h_ptr + i, mask=m, other=0.0)
                if need_wa:
                    wb_v = tl.load(wb_ptr + i, mask=m, other=0.0)
                    tl.store(wa_ptr + i, wb_v, mask=m)
                if (save & 1) != 0:
                    tl.store(trace_ptr + i, ref_v + h_v, mask=m)
                    tl.store(trace_ptr + n + i, ref_v, mask=m)
                    tl.store(trace_ptr + 2 * n + i, h_v, mask=m)
                if (save & 2) != 0:
                    tl.store(trace_ptr + n + i, ref_v, mask=m)
                    tl.store(trace_ptr + 2 * n + i, h_v, mask=m)
                if (save & 4) != 0:
                    ref_v = tl.load(trace_ptr + n + i, mask=m, other=0.0)
                    h_v = tl.load(trace_ptr + 2 * n + i, mask=m, other=0.0)
                    tl.store(ref_ptr + i, ref_v, mask=m)
                    tl.store(h_ptr + i, h_v, mask=m)
                if reset == 1:
                    s0 = tl.load(S0_ptr + i, mask=m, other=0.0)
                    length = tl.load(L_ptr + i, mask=m, other=1.0)
                    w = tl.load(W_ptr + i, mask=m, other=1.0)
                    hh = tl.load(H_ptr + i, mask=m, other=0.0)
                    Ai = tl.load(A_ptr + i, mask=m, other=1.0)
                    r0 = invert_storage_inline(
                        i, s0, length, w, hh, Ai, tab_ptr, m, levels
                    )
                    tl.store(ref_ptr + i, r0, mask=m)
                    tl.store(h_ptr + i, tl.zeros_like(r0), mask=m)
                if reset == 2:
                    hh = tl.load(H_ptr + i, mask=m, other=0.0)
                    tr0 = tl.load(trace_ptr + i, mask=m, other=0.0)
                    r_off = tl.maximum(-hh, tr0 + phase_offset_value_inline(phase - 13))
                    tl.store(ref_ptr + i, r_off, mask=m)
                    tl.store(h_ptr + i, tl.zeros_like(r_off), mask=m)
                off += BLOCK_SIZE

        if mode != MODE_DONE:
            tl.debug_barrier()

            # 6. Level-by-level tree elimination & substitution
            if solve != 0:
                lv = lev_begin
                while lv < lev_end:
                    lb = tl.load(loff_ptr + lv)
                    le = tl.load(loff_ptr + lv + 1)
                    off = lb
                    while off < le:
                        k = off + lanes
                        m = k < le
                        i = tl.load(tunit_ptr + k, mask=m, other=0)
                        ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
                        h_v = tl.load(h_ptr + i, mask=m, other=0.0)
                        hh = tl.load(H_ptr + i, mask=m, other=0.0)
                        R_v = tl.load(R_ptr + i, mask=m, other=0.0)
                        dry = (~(((ref_v + hh) + h_v) > 0.0)) & (R_v > 0.0)
                        hd = tl.load(edge_ptr + i, mask=m, other=False)
                        area_v = tl.load(area_ptr + i, mask=m, other=0.0)
                        qi_v = tl.load(qi_ptr + i, mask=m, other=0.0)
                        qd_v = tl.load(qd_ptr + i, mask=m, other=0.0)
                        a = area_v + dt * qi_v
                        r = -R_v
                        b_val = tl.where(dry | (~hd), 0.0, dt * qd_v)
                        damp_d = area_v + dt * tl.abs(qi_v)
                        e = tl.load(cp_ptr + i, mask=m, other=0)
                        e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
                        e_act = m & (e < e_end)
                        while tl.max(e_act.to(tl.int32), axis=0) != 0:
                            j = tl.load(kids_ptr + e, mask=e_act, other=0)
                            qdj = tl.load(qd_ptr + j, mask=e_act, other=0.0)
                            qij = tl.load(qi_ptr + j, mask=e_act, other=0.0)
                            bj = tl.load(b_ptr + j, mask=e_act, other=0.0)
                            apj = tl.load(ap_ptr + j, mask=e_act, other=1.0)
                            rpj = tl.load(rp_ptr + j, mask=e_act, other=0.0)
                            apj_safe = tl.where(e_act, apj, 1.0)
                            a = tl.where(e_act, a - dt * qdj, a)
                            c = -dt * qij
                            a = tl.where(e_act, a - tl.div_rn(c * bj, apj_safe), a)
                            r = tl.where(e_act, r - tl.div_rn(c * rpj, apj_safe), r)
                            if sigma > 0.0:
                                damp_d = tl.where(
                                    e_act, damp_d + dt * tl.abs(qdj), damp_d
                                )
                            e += 1
                            e_act = m & (e < e_end)
                        if sigma > 0.0:
                            a = a + sigma * damp_d
                        tl.store(b_ptr + i, b_val, mask=m)
                        tl.store(ap_ptr + i, tl.where(dry, 1.0, a), mask=m)
                        tl.store(rp_ptr + i, tl.where(dry, 0.0, r), mask=m)
                        off += BLOCK_SIZE
                    tl.debug_barrier()
                    lv += 1

                lv = lev_end - 1
                while lv >= lev_begin:
                    lb = tl.load(loff_ptr + lv)
                    le = tl.load(loff_ptr + lv + 1)
                    off = lb
                    while off < le:
                        k = off + lanes
                        m = k < le
                        i = tl.load(tunit_ptr + k, mask=m, other=0)
                        hd = tl.load(edge_ptr + i, mask=m, other=False)
                        down_i = tl.load(down_ptr + i, mask=m, other=0)
                        d_safe = tl.where(hd & m, down_i, 0)
                        dh_d = tl.load(dh_ptr + d_safe, mask=m & hd, other=0.0)
                        rp_v = tl.load(rp_ptr + i, mask=m, other=0.0)
                        b_v = tl.load(b_ptr + i, mask=m, other=0.0)
                        ap_v = tl.load(ap_ptr + i, mask=m, other=1.0)
                        ap_safe = tl.where(m, ap_v, 1.0)
                        dh_i = tl.div_rn(rp_v - tl.where(hd, b_v * dh_d, 0.0), ap_safe)
                        tl.store(dh_ptr + i, dh_i, mask=m)
                        off += BLOCK_SIZE
                    tl.debug_barrier()
                    lv -= 1

            # 7. Trial unit
            off = first
            while off < last:
                k = off + lanes
                m = k < last
                i = tl.load(tunit_ptr + k, mask=m, other=0)
                ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
                h_v = tl.load(h_ptr + i, mask=m, other=0.0)
                if mode == MODE_TRIAL:
                    hh = tl.load(H_ptr + i, mask=m, other=0.0)
                    direction = tl.load(dh_ptr + i, mask=m, other=0.0)
                    rn, xn = trial_pair_inline(ref_v, h_v, lam * direction, hh)
                    crossed = m & (((ref_v + h_v) > 0.0) != ((rn + xn) > 0.0))
                    if tl.max(crossed.to(tl.int32), axis=0) != 0:
                        length = tl.load(L_ptr + i, mask=m, other=1.0)
                        w = tl.load(W_ptr + i, mask=m, other=1.0)
                        Ai = tl.load(A_ptr + i, mask=m, other=1.0)
                        if solve != 0:
                            v_hi, v_lo = terrain_coordinate_inline(
                                i, ref_v, h_v, length, w, Ai, tab_ptr, crossed, levels
                            )
                            area_v = tl.load(area_ptr + i, mask=m, other=0.0)
                            ratio = tl.div_rn(area_v, length * w)
                            tl.store(ap_ptr + i, v_hi, mask=crossed)
                            tl.store(rp_ptr + i, v_lo, mask=crossed)
                            tl.store(b_ptr + i, ratio, mask=crossed)
                        else:
                            v_hi = tl.load(ap_ptr + i, mask=crossed, other=0.0)
                            v_lo = tl.load(rp_ptr + i, mask=crossed, other=0.0)
                            ratio = tl.load(b_ptr + i, mask=crossed, other=1.0)
                        retract = crossed & (
                            (bt >= 2) | ((phase == DAMPED_PHASE) & (direction > 0.0))
                        )
                        if tl.max(retract.to(tl.int32), axis=0) != 0:
                            t_hi, t_lo = pair_add_inline(
                                v_hi,
                                v_lo,
                                lam * (ratio * direction),
                                tl.zeros_like(v_hi),
                            )
                            rh, xh = terrain_inverse_inline(
                                i,
                                t_hi,
                                t_lo,
                                length,
                                w,
                                hh,
                                Ai,
                                tab_ptr,
                                retract,
                                levels,
                            )
                            rn = tl.where(retract, rh, rn)
                            xn = tl.where(retract, xh, xn)
                    tl.store(tref_ptr + i, rn, mask=m)
                    tl.store(ht_ptr + i, xn, mask=m)
                else:
                    tl.store(tref_ptr + i, ref_v, mask=m)
                    tl.store(ht_ptr + i, h_v, mask=m)
                off += BLOCK_SIZE
            tl.debug_barrier()

        round_idx += 1

    # Write newton_record and report failures (lane 0)
    lane0 = lanes == 0
    if mode == MODE_DONE:
        base_m = t.to(tl.int64) * 16
        z_f = tl.full((), 0.0, tl.float32)
        tl.store(meta_ptr + base_m + 0 + lanes * 0, merit + lanes * 0.0, mask=lane0)
        tl.store(
            meta_ptr + base_m + 1 + lanes * 0,
            sum_g.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(
            meta_ptr + base_m + 2 + lanes * 0,
            st.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(
            meta_ptr + base_m + 3 + lanes * 0,
            totalbt.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(
            meta_ptr + base_m + 4 + lanes * 0,
            sum_g.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(meta_ptr + base_m + 5 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(
            meta_ptr + base_m + 6 + lanes * 0,
            sum_fe.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(meta_ptr + base_m + 7 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(meta_ptr + base_m + 8 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(meta_ptr + base_m + 9 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(meta_ptr + base_m + 10 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(meta_ptr + base_m + 11 + lanes * 0, z_f + lanes * 0.0, mask=lane0)
        tl.store(
            meta_ptr + base_m + 12 + lanes * 0,
            dsteps.to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        p_code = tl.where(phase < 0, 1, phase + 2).to(tl.float32)
        tl.store(meta_ptr + base_m + 13 + lanes * 0, p_code + lanes * 0.0, mask=lane0)
        tl.store(
            meta_ptr + base_m + 14 + lanes * 0,
            (phase >= 0).to(tl.float32) + lanes * 0.0,
            mask=lane0,
        )
        tl.store(meta_ptr + base_m + 15 + lanes * 0, initial + lanes * 0.0, mask=lane0)
        if st != 0:
            report_error(status_ptr, lane0, CODE_NEWTON_NOT_CONVERGED, t + lanes * 0)
    else:
        report_error(status_ptr, lane0, CODE_NEWTON_ROUND_LIMIT, t + lanes * 0)


# --------------------------------------------------------------------------- #
# Grid-solved large trees: separate fast Triton kernels
# --------------------------------------------------------------------------- #


@triton.jit
def grid_start_units_kernel(
    gunit_ptr,
    S0_ptr,
    nc_ptr,
    nf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    ref_ptr,
    h_ptr,
    tref_ptr,
    ht_ptr,
    status_ptr,
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    s0 = tl.load(S0_ptr + i, mask=m, other=0.0)
    length = tl.load(L_ptr + i, mask=m, other=1.0)
    w = tl.load(W_ptr + i, mask=m, other=1.0)
    hh = tl.load(H_ptr + i, mask=m, other=0.0)
    Ai = tl.load(A_ptr + i, mask=m, other=1.0)
    r0 = invert_storage_inline(i, s0, length, w, hh, Ai, tab_ptr, m, levels)
    z0 = tl.zeros_like(r0)
    tl.store(ref_ptr + i, r0, mask=m)
    tl.store(h_ptr + i, z0, mask=m)
    tl.store(tref_ptr + i, r0, mask=m)
    tl.store(ht_ptr + i, z0, mask=m)
    nc_i = tl.load(nc_ptr + i, mask=m, other=1.0)
    nf_i = tl.load(nf_ptr + i, mask=m, other=1.0)
    bad_n = m & (
        ~(
            (nc_i > 0.0)
            & isfinite_f32_inline(nc_i)
            & (nf_i > 0.0)
            & isfinite_f32_inline(nf_i)
        )
    )
    if tl.max(bad_n.to(tl.int32), axis=0) != 0:
        report_error(status_ptr, bad_n, CODE_INVALID_ROUGHNESS, i)


@triton.jit
def grid_start_trees_kernel(
    gtree_ptr,
    ti_ptr,
    tf_ptr,
    tm_ptr,
    tline_ptr,
    dt0_ptr,
    ngt,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngt
    t = tl.load(gtree_ptr + idx, mask=m, other=0)
    dt0 = tl.load(dt0_ptr)
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bt_i = t.to(tl.int64) * TI
    bt_f = t.to(tl.int64) * TF
    for c in range(TI):
        tl.store(ti_ptr + bt_i + c, zi, mask=m)
    for c in range(TF):
        tl.store(tf_ptr + bt_f + c, zf, mask=m)
    tl.store(ti_ptr + bt_i + IDX_MODE, zi + MODE_EVAL, mask=m)
    tl.store(ti_ptr + bt_i + IDX_PHASE, zi - 1, mask=m)
    tl.store(ti_ptr + bt_i + IDX_BEST_ST, zi + 2, mask=m)
    tl.store(tf_ptr + bt_f + IDX_DT, zf + dt0, mask=m)
    tl.store(tf_ptr + bt_f + IDX_BEST, zf + 1e30, mask=m)
    tl.store(tm_ptr + t, zi, mask=m)
    tl.store(tline_ptr + t, zi, mask=m)


@triton.jit
def grid_stage_kernel(
    gunit_ptr,
    tree_ptr,
    ti_ptr,
    tref_ptr,
    ht_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    S_ptr,
    area_ptr,
    kc_ptr,
    kf_ptr,
    dkc_ptr,
    dkf_ptr,
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    t = tl.load(tree_ptr + i, mask=m, other=0)
    mode = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_MODE, mask=m, other=0)
    act = m & (mode != MODE_DONE)
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    tr = tl.load(tref_ptr + i, mask=act, other=0.0)
    ht = tl.load(ht_ptr + i, mask=act, other=0.0)
    length = tl.load(L_ptr + i, mask=act, other=1.0)
    w = tl.load(W_ptr + i, mask=act, other=1.0)
    hh = tl.load(H_ptr + i, mask=act, other=0.0)
    Ai = tl.load(A_ptr + i, mask=act, other=1.0)
    s_v, a_v, kc_v, kf_v, dkc_v, dkf_v = stage_pair_inline(
        i, tr, ht, length, w, hh, Ai, tab_ptr, bed_ptr, act, levels
    )
    tl.store(S_ptr + i, s_v, mask=act)
    tl.store(area_ptr + i, a_v, mask=act)
    tl.store(kc_ptr + i, kc_v, mask=act)
    tl.store(kf_ptr + i, kf_v, mask=act)
    tl.store(dkc_ptr + i, dkc_v, mask=act)
    tl.store(dkf_ptr + i, dkf_v, mask=act)


@triton.jit
def grid_flux_kernel(
    gunit_ptr,
    tree_ptr,
    ti_ptr,
    tf_ptr,
    Q0_ptr,
    nc_ptr,
    nf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    elev_ptr,
    dist_ptr,
    down_ptr,
    edge_ptr,
    tref_ptr,
    ht_ptr,
    S_ptr,
    area_ptr,
    kc_ptr,
    kf_ptr,
    dkc_ptr,
    dkf_ptr,
    Q_ptr,
    qi_ptr,
    qd_ptr,
    eps,
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    t = tl.load(tree_ptr + i, mask=m, other=0)
    mode = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_MODE, mask=m, other=0)
    act = m & (mode != MODE_DONE)
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    dt = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_DT, mask=act, other=0.0)
    q_v, qi_v, qd_v = eval_flux_vec_inline(
        i,
        dt,
        eps,
        act,
        Q0_ptr,
        nc_ptr,
        nf_ptr,
        L_ptr,
        W_ptr,
        H_ptr,
        A_ptr,
        tab_ptr,
        bed_ptr,
        elev_ptr,
        dist_ptr,
        down_ptr,
        edge_ptr,
        tref_ptr,
        ht_ptr,
        S_ptr,
        area_ptr,
        kc_ptr,
        kf_ptr,
        dkc_ptr,
        dkf_ptr,
        levels,
    )
    tl.store(Q_ptr + i, q_v, mask=act)
    tl.store(qi_ptr + i, qi_v, mask=act)
    tl.store(qd_ptr + i, qd_v, mask=act)


@triton.jit
def grid_residual_kernel(
    gunit_ptr,
    gtree_ptr,
    tree_ptr,
    ti_ptr,
    tf_ptr,
    S0_ptr,
    runoff_ptr,
    S_ptr,
    Q_ptr,
    cp_ptr,
    kids_ptr,
    R_ptr,
    wb_ptr,
    wa_ptr,
    tm_ptr,
    tline_ptr,
    ngu,
    SINGLE_TREE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if SINGLE_TREE:
        t0 = tl.load(gtree_ptr)
        mode0 = tl.load(ti_ptr + t0.to(tl.int64) * TI + IDX_MODE)
        if mode0 == MODE_DONE:
            return
        act = m
        dt = tl.load(tf_ptr + t0.to(tl.int64) * TF + IDX_DT) + tl.zeros(
            (BLOCK_SIZE,), dtype=tl.float32
        )
        phase0 = tl.load(ti_ptr + t0.to(tl.int64) * TI + IDX_PHASE)
        weighted = (phase0 == RESTART_PHASE) | (phase0 == DAMPED_PHASE)
        t = t0 + tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    else:
        t = tl.load(tree_ptr + i, mask=m, other=0)
        mode = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_MODE, mask=m, other=0)
        act = m & (mode != MODE_DONE)
        if tl.max(act.to(tl.int32), axis=0) == 0:
            return
        dt = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_DT, mask=act, other=0.0)
        phase = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_PHASE, mask=act, other=0)
        weighted = act & ((phase == RESTART_PHASE) | (phase == DAMPED_PHASE))

    e = tl.load(cp_ptr + i, mask=act, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
    e_act = act & (e < e_end)
    inflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    ainflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        kid = tl.load(kids_ptr + e, mask=e_act, other=0)
        qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
        inflow_q = tl.where(e_act, inflow_q + qk, inflow_q)
        ainflow_q = tl.where(e_act, ainflow_q + tl.abs(qk), ainflow_q)
        e += 1
        e_act = act & (e < e_end)
    Qi = tl.load(Q_ptr + i, mask=act, other=0.0)
    Si = tl.load(S_ptr + i, mask=act, other=0.0)
    S0i = tl.load(S0_ptr + i, mask=act, other=0.0)
    ri = tl.load(runoff_ptr + i, mask=act, other=0.0)
    r = (Si - S0i) - dt * (ri + inflow_q - Qi)
    scale = S0i + dt * (tl.abs(ri) + ainflow_q + tl.abs(Qi)) + 1000.0
    inf_val = tl.full((), float("inf"), tl.float32)
    fin = isfinite_f32_inline(r) & isfinite_f32_inline(Qi)
    score = tl.where(fin, tl.div_rn(tl.abs(r), scale), inf_val)
    wb = tl.div_rn(tl.full((), 1.0, tl.float32), scale)
    tl.store(R_ptr + i, r, mask=act)
    tl.store(wb_ptr + i, wb, mask=act)

    if SINGLE_TREE:
        bm = tl.max(tl.where(act, score, 0.0), axis=0)
        lanes = tl.arange(0, BLOCK_SIZE)
        lane0 = lanes == 0
        bm_i = tl.cast(tl.abs(bm), tl.int32, bitcast=True)
        tl.atomic_max(
            tm_ptr + t0 + lanes * 0,
            bm_i + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )
        if weighted:
            wai = tl.load(wa_ptr + i, mask=act, other=0.0)
            lk = tl.where(isfinite_f32_inline(score), tl.abs(r) * wai, inf_val)
            bl = tl.max(tl.where(act, lk, 0.0), axis=0)
            bl_i = tl.cast(tl.abs(bl), tl.int32, bitcast=True)
            tl.atomic_max(
                tline_ptr + t0 + lanes * 0,
                bl_i + lanes * 0,
                mask=lane0,
                sem="relaxed",
                scope="gpu",
            )
    else:
        sc_i = tl.cast(tl.abs(score), tl.int32, bitcast=True)
        tl.atomic_max(tm_ptr + t, sc_i, mask=act, sem="relaxed", scope="gpu")
        if tl.max(weighted.to(tl.int32), axis=0) != 0:
            wai = tl.load(wa_ptr + i, mask=weighted, other=0.0)
            lk = tl.where(isfinite_f32_inline(score), tl.abs(r) * wai, inf_val)
            lk_i = tl.cast(tl.abs(lk), tl.int32, bitcast=True)
            tl.atomic_max(
                tline_ptr + t, lk_i, mask=weighted, sem="relaxed", scope="gpu"
            )


@triton.jit
def grid_control_kernel(
    gtree_ptr,
    ti_ptr,
    tf_ptr,
    tm_ptr,
    tline_ptr,
    ctl_ptr,
    meta_ptr,
    status_ptr,
    dt0_ptr,
    tol,
    maxit,
    ngt,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngt
    t = tl.load(gtree_ptr + idx, mask=m, other=0)
    dt0 = tl.load(dt0_ptr)
    bt_i = t.to(tl.int64) * TI
    bt_f = t.to(tl.int64) * TF

    mode = tl.load(ti_ptr + bt_i + IDX_MODE, mask=m, other=MODE_DONE)
    phase = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=m, other=0)
    steps = tl.load(ti_ptr + bt_i + IDX_STEPS, mask=m, other=0)
    bt = tl.load(ti_ptr + bt_i + IDX_BT, mask=m, other=0)
    st = tl.load(ti_ptr + bt_i + IDX_ST, mask=m, other=0)
    fevals = tl.load(ti_ptr + bt_i + IDX_FEVALS, mask=m, other=0)
    totalbt = tl.load(ti_ptr + bt_i + IDX_TOTALBT, mask=m, other=0)
    sum_g = tl.load(ti_ptr + bt_i + IDX_SUM_G, mask=m, other=0)
    sum_fe = tl.load(ti_ptr + bt_i + IDX_SUM_FE, mask=m, other=0)
    dsteps = tl.load(ti_ptr + bt_i + IDX_DSTEPS, mask=m, other=0)
    best_st = tl.load(ti_ptr + bt_i + IDX_BEST_ST, mask=m, other=2)

    dt = tl.load(tf_ptr + bt_f + IDX_DT, mask=m, other=0.0)
    lam = tl.load(tf_ptr + bt_f + IDX_LAM, mask=m, other=0.0)
    merit = tl.load(tf_ptr + bt_f + IDX_MERIT, mask=m, other=0.0)
    best = tl.load(tf_ptr + bt_f + IDX_BEST, mask=m, other=1e30)
    initial = tl.load(tf_ptr + bt_f + IDX_INITIAL, mask=m, other=0.0)
    sigma = tl.load(tf_ptr + bt_f + IDX_SIGMA, mask=m, other=0.0)

    tm_bits = tl.load(tm_ptr + t, mask=m, other=0)
    tline_bits = tl.load(tline_ptr + t, mask=m, other=0)
    mt = tl.cast(tm_bits, tl.float32, bitcast=True)
    line_v = tl.cast(tline_bits, tl.float32, bitcast=True)
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    tl.store(tm_ptr + t, zi, mask=m)
    tl.store(tline_ptr + t, zi, mask=m)

    (
        mode,
        phase,
        steps,
        bt,
        st,
        fevals,
        totalbt,
        sum_g,
        sum_fe,
        dsteps,
        best_st,
        dt,
        lam,
        merit,
        best,
        initial,
        sigma,
        commit,
        save,
        reset,
        solve,
        finished_now,
    ) = newton_control_step_inline(
        mode,
        phase,
        steps,
        bt,
        st,
        fevals,
        totalbt,
        sum_g,
        sum_fe,
        dsteps,
        best_st,
        dt,
        lam,
        merit,
        best,
        initial,
        sigma,
        mt,
        line_v,
        dt0,
        tol,
        maxit,
        False,
    )

    tl.store(ti_ptr + bt_i + IDX_MODE, mode, mask=m)
    tl.store(ti_ptr + bt_i + IDX_PHASE, phase, mask=m)
    tl.store(ti_ptr + bt_i + IDX_STEPS, steps, mask=m)
    tl.store(ti_ptr + bt_i + IDX_BT, bt, mask=m)
    tl.store(ti_ptr + bt_i + IDX_ST, st, mask=m)
    tl.store(ti_ptr + bt_i + IDX_FEVALS, fevals, mask=m)
    tl.store(ti_ptr + bt_i + IDX_TOTALBT, totalbt, mask=m)
    tl.store(ti_ptr + bt_i + IDX_SUM_G, sum_g, mask=m)
    tl.store(ti_ptr + bt_i + IDX_SUM_FE, sum_fe, mask=m)
    tl.store(ti_ptr + bt_i + IDX_DSTEPS, dsteps, mask=m)
    tl.store(ti_ptr + bt_i + IDX_BEST_ST, best_st, mask=m)
    tl.store(ti_ptr + bt_i + IDX_SOLVE, solve, mask=m)
    tl.store(ti_ptr + bt_i + IDX_COMMIT, commit, mask=m)
    tl.store(ti_ptr + bt_i + IDX_SAVE, save, mask=m)
    tl.store(ti_ptr + bt_i + IDX_RESET, reset, mask=m)

    tl.store(tf_ptr + bt_f + IDX_DT, dt, mask=m)
    tl.store(tf_ptr + bt_f + IDX_LAM, lam, mask=m)
    tl.store(tf_ptr + bt_f + IDX_MERIT, merit, mask=m)
    tl.store(tf_ptr + bt_f + IDX_BEST, best, mask=m)
    tl.store(tf_ptr + bt_f + IDX_INITIAL, initial, mask=m)
    tl.store(tf_ptr + bt_f + IDX_SIGMA, sigma, mask=m)

    fin_m = m & (finished_now != 0)
    if tl.max(fin_m.to(tl.int32), axis=0) != 0:
        base_m = t.to(tl.int64) * 16
        zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        tl.store(meta_ptr + base_m + 0, merit, mask=fin_m)
        tl.store(meta_ptr + base_m + 1, sum_g.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 2, st.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 3, totalbt.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 4, sum_g.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 5, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 6, sum_fe.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 7, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 8, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 9, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 10, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 11, zf, mask=fin_m)
        tl.store(meta_ptr + base_m + 12, dsteps.to(tl.float32), mask=fin_m)
        tl.store(
            meta_ptr + base_m + 13,
            tl.where(phase < 0, 1, phase + 2).to(tl.float32),
            mask=fin_m,
        )
        tl.store(meta_ptr + base_m + 14, (phase >= 0).to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + base_m + 15, initial, mask=fin_m)
        bad_t = fin_m & (st != 0)
        if tl.max(bad_t.to(tl.int32), axis=0) != 0:
            report_error(status_ptr, bad_t, CODE_NEWTON_NOT_CONVERGED, t)

    n_act = tl.sum((m & (mode != MODE_DONE)).to(tl.int32), axis=0)
    n_sol = tl.sum((m & (solve != 0)).to(tl.int32), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    lane0 = lanes == 0
    if n_act > 0:
        tl.atomic_add(
            ctl_ptr + CTL_ACTIVE + lanes * 0,
            n_act + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )
    if n_sol > 0:
        tl.atomic_add(
            ctl_ptr + CTL_SOLVING + lanes * 0,
            n_sol + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def grid_apply_kernel(
    gunit_ptr,
    tree_ptr,
    ti_ptr,
    S0_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    ref_ptr,
    h_ptr,
    tref_ptr,
    ht_ptr,
    wb_ptr,
    wa_ptr,
    trace_ptr,
    n,
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    t = tl.load(tree_ptr + i, mask=m, other=0)
    bt = t.to(tl.int64) * TI
    commit = tl.load(ti_ptr + bt + IDX_COMMIT, mask=m, other=0)
    solve = tl.load(ti_ptr + bt + IDX_SOLVE, mask=m, other=0)
    phase = tl.load(ti_ptr + bt + IDX_PHASE, mask=m, other=0)
    save = tl.load(ti_ptr + bt + IDX_SAVE, mask=m, other=0)
    reset = tl.load(ti_ptr + bt + IDX_RESET, mask=m, other=0)

    cm = m & (commit != 0)
    ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
    h_v = tl.load(h_ptr + i, mask=m, other=0.0)
    if tl.max(cm.to(tl.int32), axis=0) != 0:
        ref_v = tl.where(cm, tl.load(tref_ptr + i, mask=cm, other=0.0), ref_v)
        h_v = tl.where(cm, tl.load(ht_ptr + i, mask=cm, other=0.0), h_v)
        tl.store(ref_ptr + i, ref_v, mask=cm)
        tl.store(h_ptr + i, h_v, mask=cm)

    wam = m & (solve != 0) & ((phase == RESTART_PHASE) | (phase == DAMPED_PHASE))
    if tl.max(wam.to(tl.int32), axis=0) != 0:
        tl.store(wa_ptr + i, tl.load(wb_ptr + i, mask=wam, other=0.0), mask=wam)

    s1 = m & ((save & 1) != 0)
    if tl.max(s1.to(tl.int32), axis=0) != 0:
        tl.store(trace_ptr + i, ref_v + h_v, mask=s1)
        tl.store(trace_ptr + n + i, ref_v, mask=s1)
        tl.store(trace_ptr + 2 * n + i, h_v, mask=s1)
    s2 = m & ((save & 2) != 0)
    if tl.max(s2.to(tl.int32), axis=0) != 0:
        tl.store(trace_ptr + n + i, ref_v, mask=s2)
        tl.store(trace_ptr + 2 * n + i, h_v, mask=s2)
    s4 = m & ((save & 4) != 0)
    if tl.max(s4.to(tl.int32), axis=0) != 0:
        ref_v = tl.where(s4, tl.load(trace_ptr + n + i, mask=s4, other=0.0), ref_v)
        h_v = tl.where(s4, tl.load(trace_ptr + 2 * n + i, mask=s4, other=0.0), h_v)
        tl.store(ref_ptr + i, ref_v, mask=s4)
        tl.store(h_ptr + i, h_v, mask=s4)

    r1 = m & (reset == 1)
    if tl.max(r1.to(tl.int32), axis=0) != 0:
        s0 = tl.load(S0_ptr + i, mask=r1, other=0.0)
        length = tl.load(L_ptr + i, mask=r1, other=1.0)
        w = tl.load(W_ptr + i, mask=r1, other=1.0)
        hh = tl.load(H_ptr + i, mask=r1, other=0.0)
        Ai = tl.load(A_ptr + i, mask=r1, other=1.0)
        r0 = invert_storage_inline(i, s0, length, w, hh, Ai, tab_ptr, r1, levels)
        tl.store(ref_ptr + i, r0, mask=r1)
        tl.store(h_ptr + i, tl.zeros_like(r0), mask=r1)
    r2 = m & (reset == 2)
    if tl.max(r2.to(tl.int32), axis=0) != 0:
        hh = tl.load(H_ptr + i, mask=r2, other=0.0)
        tr0 = tl.load(trace_ptr + i, mask=r2, other=0.0)
        r_off = tl.maximum(-hh, tr0 + phase_offset_value_inline(phase - 13))
        tl.store(ref_ptr + i, r_off, mask=r2)
        tl.store(h_ptr + i, tl.zeros_like(r_off), mask=r2)


@triton.jit
def grid_trial_kernel(
    gunit_ptr,
    tree_ptr,
    ti_ptr,
    tf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    ref_ptr,
    h_ptr,
    tref_ptr,
    ht_ptr,
    dh_ptr,
    area_ptr,
    ap_ptr,
    rp_ptr,
    b_ptr,
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    t = tl.load(tree_ptr + i, mask=m, other=0)
    bt_i = t.to(tl.int64) * TI
    bt_f = t.to(tl.int64) * TF
    mode = tl.load(ti_ptr + bt_i + IDX_MODE, mask=m, other=MODE_DONE)
    act = m & (mode != MODE_DONE)
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    ref_v = tl.load(ref_ptr + i, mask=act, other=0.0)
    h_v = tl.load(h_ptr + i, mask=act, other=0.0)
    is_trial = act & (mode == MODE_TRIAL)
    rn = ref_v
    xn = h_v
    if tl.max(is_trial.to(tl.int32), axis=0) != 0:
        hh = tl.load(H_ptr + i, mask=is_trial, other=0.0)
        lam = tl.load(tf_ptr + bt_f + IDX_LAM, mask=is_trial, other=0.0)
        direction = tl.load(dh_ptr + i, mask=is_trial, other=0.0)
        rn_t, xn_t = trial_pair_inline(ref_v, h_v, lam * direction, hh)
        crossed = is_trial & (((ref_v + h_v) > 0.0) != ((rn_t + xn_t) > 0.0))
        if tl.max(crossed.to(tl.int32), axis=0) != 0:
            length = tl.load(L_ptr + i, mask=crossed, other=1.0)
            w = tl.load(W_ptr + i, mask=crossed, other=1.0)
            Ai = tl.load(A_ptr + i, mask=crossed, other=1.0)
            sol = tl.load(ti_ptr + bt_i + IDX_SOLVE, mask=crossed, other=0)
            do_cache = crossed & (sol != 0)
            v_hi_c, v_lo_c = terrain_coordinate_inline(
                i, ref_v, h_v, length, w, Ai, tab_ptr, do_cache, levels
            )
            ratio_c = tl.div_rn(
                tl.load(area_ptr + i, mask=do_cache, other=0.0), length * w
            )
            tl.store(ap_ptr + i, v_hi_c, mask=do_cache)
            tl.store(rp_ptr + i, v_lo_c, mask=do_cache)
            tl.store(b_ptr + i, ratio_c, mask=do_cache)
            v_hi = tl.where(
                do_cache,
                v_hi_c,
                tl.load(ap_ptr + i, mask=crossed & (~do_cache), other=0.0),
            )
            v_lo = tl.where(
                do_cache,
                v_lo_c,
                tl.load(rp_ptr + i, mask=crossed & (~do_cache), other=0.0),
            )
            ratio = tl.where(
                do_cache,
                ratio_c,
                tl.load(b_ptr + i, mask=crossed & (~do_cache), other=1.0),
            )
            bt_v = tl.load(ti_ptr + bt_i + IDX_BT, mask=crossed, other=0)
            phase_v = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=crossed, other=0)
            retract = crossed & (
                (bt_v >= 2) | ((phase_v == DAMPED_PHASE) & (direction > 0.0))
            )
            if tl.max(retract.to(tl.int32), axis=0) != 0:
                t_hi, t_lo = pair_add_inline(
                    v_hi, v_lo, lam * (ratio * direction), tl.zeros_like(v_hi)
                )
                rh, xh = terrain_inverse_inline(
                    i, t_hi, t_lo, length, w, hh, Ai, tab_ptr, retract, levels
                )
                rn_t = tl.where(retract, rh, rn_t)
                xn_t = tl.where(retract, xh, xn_t)
        rn = tl.where(is_trial, rn_t, rn)
        xn = tl.where(is_trial, xn_t, xn)
    tl.store(tref_ptr + i, rn, mask=act)
    tl.store(ht_ptr + i, xn, mask=act)


# --------------------------------------------------------------------------- #
# Grid heavy-path elimination & substitution (serial + parallel cyclic reduction)
# --------------------------------------------------------------------------- #


@triton.jit
def grid_eliminate_serial_kernel(
    paths_ptr,
    n_paths,
    pnodes_ptr,
    pptr_ptr,
    tree_ptr,
    ti_ptr,
    tf_ptr,
    H_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    ref_ptr,
    h_ptr,
    R_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    ap_ptr,
    rp_ptr,
    b_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = idx < n_paths
    p = tl.load(paths_ptr + idx, mask=valid, other=0)
    beg = tl.load(pptr_ptr + p, mask=valid, other=0)
    end = tl.load(pptr_ptr + p + 1, mask=valid, other=0)
    first_node = tl.load(pnodes_ptr + beg, mask=valid, other=0)
    t = tl.load(tree_ptr + first_node, mask=valid, other=0)
    sol = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_SOLVE, mask=valid, other=0)
    dt = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_DT, mask=valid, other=0.0)
    sigma = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_SIGMA, mask=valid, other=0.0)
    valid = valid & (sol != 0)
    k = beg
    act = valid & (k < end)
    prev = tl.full((BLOCK_SIZE,), -1, tl.int32)
    pa = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    pr = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    pb = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    pqi = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    pqd = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    while tl.max(act.to(tl.int32), axis=0) != 0:
        i = tl.load(pnodes_ptr + k, mask=act, other=0)
        ref_v = tl.load(ref_ptr + i, mask=act, other=0.0)
        h_v = tl.load(h_ptr + i, mask=act, other=0.0)
        hh = tl.load(H_ptr + i, mask=act, other=0.0)
        R_v = tl.load(R_ptr + i, mask=act, other=0.0)
        dry = (~(((ref_v + hh) + h_v) > 0.0)) & (R_v > 0.0)
        qi_v = tl.load(qi_ptr + i, mask=act, other=0.0)
        qd_v = tl.load(qd_ptr + i, mask=act, other=0.0)
        area_v = tl.load(area_ptr + i, mask=act, other=0.0)
        hd = tl.load(edge_ptr + i, mask=act, other=False)
        a = area_v + dt * qi_v
        r = -R_v
        b_val = tl.where(dry | (~hd), 0.0, dt * qd_v)
        damp_d = area_v + dt * tl.abs(qi_v)
        e = tl.load(cp_ptr + i, mask=act, other=0)
        e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
        e_act = act & (e < e_end)
        while tl.max(e_act.to(tl.int32), axis=0) != 0:
            j = tl.load(kids_ptr + e, mask=e_act, other=0)
            carried = j == prev
            qj = tl.where(carried, pqd, tl.load(qd_ptr + j, mask=e_act, other=0.0))
            qij = tl.where(carried, pqi, tl.load(qi_ptr + j, mask=e_act, other=0.0))
            aj = tl.where(carried, pa, tl.load(ap_ptr + j, mask=e_act, other=1.0))
            bj = tl.where(carried, pb, tl.load(b_ptr + j, mask=e_act, other=0.0))
            rj = tl.where(carried, pr, tl.load(rp_ptr + j, mask=e_act, other=0.0))
            aj_safe = tl.where(e_act, aj, 1.0)
            a = tl.where(e_act, a - dt * qj, a)
            c = -dt * qij
            a = tl.where(e_act, a - tl.div_rn(c * bj, aj_safe), a)
            r = tl.where(e_act, r - tl.div_rn(c * rj, aj_safe), r)
            damp_d = tl.where(e_act, damp_d + dt * tl.abs(qj), damp_d)
            e += 1
            e_act = act & (e < e_end)
        a = tl.where(sigma > 0.0, a + sigma * damp_d, a)
        na = tl.where(dry, 1.0, a)
        nr = tl.where(dry, 0.0, r)
        tl.store(ap_ptr + i, na, mask=act)
        tl.store(rp_ptr + i, nr, mask=act)
        tl.store(b_ptr + i, b_val, mask=act)
        pa = tl.where(act, na, pa)
        pr = tl.where(act, nr, pr)
        pb = tl.where(act, b_val, pb)
        pqi = tl.where(act, qi_v, pqi)
        pqd = tl.where(act, qd_v, pqd)
        prev = tl.where(act, i, prev)
        k += 1
        act = valid & (k < end)


@triton.jit
def grid_substitute_serial_kernel(
    paths_ptr,
    n_paths,
    pnodes_ptr,
    pptr_ptr,
    tree_ptr,
    ti_ptr,
    down_ptr,
    edge_ptr,
    ap_ptr,
    rp_ptr,
    b_ptr,
    dh_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = idx < n_paths
    p = tl.load(paths_ptr + idx, mask=valid, other=0)
    beg = tl.load(pptr_ptr + p, mask=valid, other=0)
    end = tl.load(pptr_ptr + p + 1, mask=valid, other=0)
    first_node = tl.load(pnodes_ptr + beg, mask=valid, other=0)
    t = tl.load(tree_ptr + first_node, mask=valid, other=0)
    sol = tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_SOLVE, mask=valid, other=0)
    valid = valid & (sol != 0) & (beg < end)
    out_node = tl.load(pnodes_ptr + end - 1, mask=valid, other=0)
    out_edge = tl.load(edge_ptr + out_node, mask=valid, other=False)
    out_down = tl.load(down_ptr + out_node, mask=valid & out_edge, other=0)
    parent = tl.where(
        out_edge, tl.load(dh_ptr + out_down, mask=valid & out_edge, other=0.0), 0.0
    )
    k = end - 1
    act = valid & (k >= beg)
    while tl.max(act.to(tl.int32), axis=0) != 0:
        i = tl.load(pnodes_ptr + k, mask=act, other=0)
        hd = tl.load(edge_ptr + i, mask=act, other=False)
        rp_v = tl.load(rp_ptr + i, mask=act, other=0.0)
        b_v = tl.load(b_ptr + i, mask=act, other=0.0)
        ap_v = tl.load(ap_ptr + i, mask=act, other=1.0)
        ap_safe = tl.where(act, ap_v, 1.0)
        val = tl.div_rn(rp_v - tl.where(hd, b_v * parent, 0.0), ap_safe)
        tl.store(dh_ptr + i, val, mask=act)
        parent = tl.where(act, val, parent)
        k -= 1
        act = valid & (k >= beg)


@triton.jit
def grid_eliminate_pcr_kernel(
    paths_ptr,
    slot_ptr,
    scratch_ptr,
    slot_stride,
    pnodes_ptr,
    pptr_ptr,
    tree_ptr,
    ti_ptr,
    tf_ptr,
    H_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    ref_ptr,
    h_ptr,
    R_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    ap_ptr,
    rp_ptr,
    b_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Parallel cyclic reduction for one heavy path of length <= BLOCK_SIZE."""
    pid = tl.program_id(0)
    p = tl.load(paths_ptr + pid)
    beg = tl.load(pptr_ptr + p)
    end = tl.load(pptr_ptr + p + 1)
    length = end - beg
    t = tl.load(tree_ptr + tl.load(pnodes_ptr + beg))
    if tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_SOLVE) == 0:
        return
    dt = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_DT)
    sigma = tl.load(tf_ptr + t.to(tl.int64) * TF + IDX_SIGMA)
    slot = tl.load(slot_ptr + pid)

    k = tl.arange(0, BLOCK_SIZE)
    m = k < length
    i = tl.load(pnodes_ptr + beg + k, mask=m, other=0)
    heavy = tl.where(
        k > 0, tl.load(pnodes_ptr + beg + k - 1, mask=m & (k > 0), other=0), -1
    )
    ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
    h_v = tl.load(h_ptr + i, mask=m, other=0.0)
    hh = tl.load(H_ptr + i, mask=m, other=0.0)
    R_v = tl.load(R_ptr + i, mask=m, other=0.0)
    dry = (~(((ref_v + hh) + h_v) > 0.0)) & (R_v > 0.0)
    area_v = tl.load(area_ptr + i, mask=m, other=0.0)
    qi_v = tl.load(qi_ptr + i, mask=m, other=0.0)
    qd_v = tl.load(qd_ptr + i, mask=m, other=0.0)
    hd = tl.load(edge_ptr + i, mask=m, other=False)

    diag = area_v + dt * qi_v
    rhs = -R_v
    lower = tl.where(
        heavy >= 0,
        -dt * tl.load(qi_ptr + tl.maximum(heavy, 0), mask=m & (heavy >= 0), other=0.0),
        0.0,
    )
    upper_full = tl.where(dry | (~hd), 0.0, dt * qd_v)
    damp_d = area_v + dt * tl.abs(qi_v)

    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        qdj = tl.load(qd_ptr + j, mask=e_act, other=0.0)
        diag = tl.where(e_act, diag - dt * qdj, diag)
        if sigma > 0.0:
            damp_d = tl.where(e_act, damp_d + dt * tl.abs(qdj), damp_d)
        light = e_act & (j != heavy)
        if tl.max(light.to(tl.int32), axis=0) != 0:
            c = -dt * tl.load(qi_ptr + j, mask=light, other=0.0)
            apj = tl.load(ap_ptr + j, mask=light, other=1.0)
            bj = tl.load(b_ptr + j, mask=light, other=0.0)
            rpj = tl.load(rp_ptr + j, mask=light, other=0.0)
            apj_safe = tl.where(light, apj, 1.0)
            diag = tl.where(light, diag - tl.div_rn(c * bj, apj_safe), diag)
            rhs = tl.where(light, rhs - tl.div_rn(c * rpj, apj_safe), rhs)
        e += 1
        e_act = m & (e < e_end)
    if sigma > 0.0:
        diag = diag + sigma * damp_d

    a = tl.where(dry, 0.0, lower)
    d = tl.where(dry, 1.0, diag)
    c_u = tl.where(k + 1 < length, upper_full, 0.0)
    r = tl.where(dry, 0.0, rhs)
    b_bnd = tl.where(k + 1 == length, upper_full, 0.0)

    # Double-buffered scratch in L1/L2: 2 * 5 * slot_stride
    s64 = slot_stride.to(tl.int64)
    base0 = slot.to(tl.int64)
    cur_off = tl.full((), 0, tl.int64)
    nxt_off = 5 * s64

    tl.store(scratch_ptr + base0 + 0 * s64 + k, a, mask=m)
    tl.store(scratch_ptr + base0 + 1 * s64 + k, d, mask=m)
    tl.store(scratch_ptr + base0 + 2 * s64 + k, c_u, mask=m)
    tl.store(scratch_ptr + base0 + 3 * s64 + k, r, mask=m)
    tl.store(scratch_ptr + base0 + 4 * s64 + k, b_bnd, mask=m)
    tl.debug_barrier()

    dist = tl.full((), 1, tl.int32)
    while dist < length:
        has_l = m & (k >= dist)
        kl = tl.where(has_l, k - dist, 0)
        dl = tl.load(
            scratch_ptr + cur_off + base0 + 1 * s64 + kl, mask=has_l, other=1.0
        )
        al = tl.load(
            scratch_ptr + cur_off + base0 + 0 * s64 + kl, mask=has_l, other=0.0
        )
        cl = tl.load(
            scratch_ptr + cur_off + base0 + 2 * s64 + kl, mask=has_l, other=0.0
        )
        rl = tl.load(
            scratch_ptr + cur_off + base0 + 3 * s64 + kl, mask=has_l, other=0.0
        )
        bl = tl.load(
            scratch_ptr + cur_off + base0 + 4 * s64 + kl, mask=has_l, other=0.0
        )

        has_r = m & (k + dist < length)
        kr = tl.where(has_r, k + dist, 0)
        dr = tl.load(
            scratch_ptr + cur_off + base0 + 1 * s64 + kr, mask=has_r, other=1.0
        )
        ar = tl.load(
            scratch_ptr + cur_off + base0 + 0 * s64 + kr, mask=has_r, other=0.0
        )
        cr = tl.load(
            scratch_ptr + cur_off + base0 + 2 * s64 + kr, mask=has_r, other=0.0
        )
        rr = tl.load(
            scratch_ptr + cur_off + base0 + 3 * s64 + kr, mask=has_r, other=0.0
        )
        br = tl.load(
            scratch_ptr + cur_off + base0 + 4 * s64 + kr, mask=has_r, other=0.0
        )

        fl = tl.where(has_l, tl.div_rn(a, dl), 0.0)
        fr = tl.where(has_r, tl.div_rn(c_u, dr), 0.0)
        d = (d - fl * cl) - fr * ar
        r = (r - fl * rl) - fr * rr
        b_bnd = (b_bnd - fl * bl) - fr * br
        a = -fl * al
        c_u = -fr * cr

        tl.store(scratch_ptr + nxt_off + base0 + 0 * s64 + k, a, mask=m)
        tl.store(scratch_ptr + nxt_off + base0 + 1 * s64 + k, d, mask=m)
        tl.store(scratch_ptr + nxt_off + base0 + 2 * s64 + k, c_u, mask=m)
        tl.store(scratch_ptr + nxt_off + base0 + 3 * s64 + k, r, mask=m)
        tl.store(scratch_ptr + nxt_off + base0 + 4 * s64 + k, b_bnd, mask=m)
        tl.debug_barrier()

        tmp = cur_off
        cur_off = nxt_off
        nxt_off = tmp
        dist = dist * 2

    d_safe = tl.where(m, d, 1.0)
    tl.store(ap_ptr + i, tl.full((BLOCK_SIZE,), 1.0, tl.float32), mask=m)
    tl.store(rp_ptr + i, tl.div_rn(r, d_safe), mask=m)
    tl.store(b_ptr + i, tl.div_rn(b_bnd, d_safe), mask=m)


@triton.jit
def grid_substitute_pcr_kernel(
    paths_ptr,
    pnodes_ptr,
    pptr_ptr,
    tree_ptr,
    ti_ptr,
    down_ptr,
    edge_ptr,
    rp_ptr,
    b_ptr,
    dh_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    p = tl.load(paths_ptr + pid)
    beg = tl.load(pptr_ptr + p)
    end = tl.load(pptr_ptr + p + 1)
    length = end - beg
    t = tl.load(tree_ptr + tl.load(pnodes_ptr + beg))
    if tl.load(ti_ptr + t.to(tl.int64) * TI + IDX_SOLVE) == 0:
        return
    out_node = tl.load(pnodes_ptr + end - 1)
    out_edge = tl.load(edge_ptr + out_node)
    parent = tl.where(
        out_edge,
        tl.load(dh_ptr + tl.where(out_edge, tl.load(down_ptr + out_node), 0)),
        0.0,
    )
    lanes = tl.arange(0, BLOCK_SIZE)
    off = tl.full((), 0, tl.int32)
    while off < length:
        k = off + lanes
        m = k < length
        i = tl.load(pnodes_ptr + beg + k, mask=m, other=0)
        rp_v = tl.load(rp_ptr + i, mask=m, other=0.0)
        b_v = tl.load(b_ptr + i, mask=m, other=0.0)
        tl.store(dh_ptr + i, rp_v - b_v * parent, mask=m)
        off += BLOCK_SIZE


# --------------------------------------------------------------------------- #
# Commit kernel (shared by forest and coupled graph)
# --------------------------------------------------------------------------- #


@triton.jit
def inertia_commit_kernel(
    S_ptr,
    Q_ptr,
    ref_ptr,
    h_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    ctl_ptr,
    storage_ptr,
    outflow_ptr,
    surface_ptr,
    inflow_ptr,
    status_ptr,
    pair_ptr,
    n,
    SAVE_SURFACE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    lane0 = i == 0
    if tl.program_id(0) == 0:
        err = tl.load(ctl_ptr + CTL_ERROR)
        if err != 0:
            report_error(status_ptr, lane0, err + i * 0, i * 0)
    ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
    h_v = tl.load(h_ptr + i, mask=m, other=0.0)
    if SAVE_SURFACE:
        tl.store(pair_ptr + 2 * i.to(tl.int64), ref_v, mask=m)
        tl.store(pair_ptr + 2 * i.to(tl.int64) + 1, h_v, mask=m)
    q = tl.load(Q_ptr + i, mask=m, other=0.0)
    s = tl.load(S_ptr + i, mask=m, other=0.0)
    ed = tl.load(edge_ptr + i, mask=m, other=False)
    inflow = tl.where(ed, tl.maximum(-q, 0.0), 0.0)
    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        kid = tl.load(kids_ptr + e, mask=e_act, other=0)
        qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
        inflow = tl.where(e_act, inflow + tl.maximum(qk, 0.0), inflow)
        e += 1
        e_act = m & (e < e_end)
    bad = m & (~(isfinite_f32_inline(q) & (s >= 0.0) & isfinite_f32_inline(s)))
    if tl.max(bad.to(tl.int32), axis=0) != 0:
        report_error(status_ptr, bad, CODE_INVALID_STATE, i)
    tl.store(storage_ptr + i, s, mask=m)
    tl.store(outflow_ptr + i, q, mask=m)
    tl.store(surface_ptr + i, ref_v + h_v, mask=m)
    tl.store(inflow_ptr + i, inflow, mask=m)


# --------------------------------------------------------------------------- #
# Rebuild kernels (forest)
# --------------------------------------------------------------------------- #


@triton.jit
def rebuild_stage_kernel(
    pair_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    ref_ptr,
    h_ptr,
    tref_ptr,
    ht_ptr,
    S_ptr,
    area_ptr,
    kc_ptr,
    kf_ptr,
    dkc_ptr,
    dkf_ptr,
    n,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    r = tl.load(pair_ptr + 2 * i.to(tl.int64), mask=m, other=0.0)
    x = tl.load(pair_ptr + 2 * i.to(tl.int64) + 1, mask=m, other=0.0)
    tl.store(ref_ptr + i, r, mask=m)
    tl.store(h_ptr + i, x, mask=m)
    tl.store(tref_ptr + i, r, mask=m)
    tl.store(ht_ptr + i, x, mask=m)
    length = tl.load(L_ptr + i, mask=m, other=1.0)
    w = tl.load(W_ptr + i, mask=m, other=1.0)
    hh = tl.load(H_ptr + i, mask=m, other=0.0)
    Ai = tl.load(A_ptr + i, mask=m, other=1.0)
    s_v, a_v, kc_v, kf_v, dkc_v, dkf_v = stage_pair_inline(
        i, r, x, length, w, hh, Ai, tab_ptr, bed_ptr, m, levels
    )
    tl.store(S_ptr + i, s_v, mask=m)
    tl.store(area_ptr + i, a_v, mask=m)
    tl.store(kc_ptr + i, kc_v, mask=m)
    tl.store(kf_ptr + i, kf_v, mask=m)
    tl.store(dkc_ptr + i, dkc_v, mask=m)
    tl.store(dkf_ptr + i, dkf_v, mask=m)


@triton.jit
def rebuild_flux_kernel(
    dt0_ptr,
    eps,
    Q0_ptr,
    nc_ptr,
    nf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    elev_ptr,
    dist_ptr,
    down_ptr,
    edge_ptr,
    tref_ptr,
    ht_ptr,
    S_ptr,
    area_ptr,
    kc_ptr,
    kf_ptr,
    dkc_ptr,
    dkf_ptr,
    Q_ptr,
    qi_ptr,
    qd_ptr,
    n,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    dt = tl.load(dt0_ptr)
    q_v, qi_v, qd_v = eval_flux_vec_inline(
        i,
        dt,
        eps,
        m,
        Q0_ptr,
        nc_ptr,
        nf_ptr,
        L_ptr,
        W_ptr,
        H_ptr,
        A_ptr,
        tab_ptr,
        bed_ptr,
        elev_ptr,
        dist_ptr,
        down_ptr,
        edge_ptr,
        tref_ptr,
        ht_ptr,
        S_ptr,
        area_ptr,
        kc_ptr,
        kf_ptr,
        dkc_ptr,
        dkf_ptr,
        levels,
    )
    tl.store(Q_ptr + i, q_v, mask=m)
    tl.store(qi_ptr + i, qi_v, mask=m)
    tl.store(qd_ptr + i, qd_v, mask=m)


@triton.jit
def rebuild_residual_kernel(
    dt0_ptr,
    S0_ptr,
    runoff_ptr,
    S_ptr,
    Q_ptr,
    cp_ptr,
    kids_ptr,
    R_ptr,
    wb_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    dt = tl.load(dt0_ptr)
    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    inflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    ainflow_q = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        kid = tl.load(kids_ptr + e, mask=e_act, other=0)
        qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
        inflow_q = tl.where(e_act, inflow_q + qk, inflow_q)
        ainflow_q = tl.where(e_act, ainflow_q + tl.abs(qk), ainflow_q)
        e += 1
        e_act = m & (e < e_end)
    Qi = tl.load(Q_ptr + i, mask=m, other=0.0)
    Si = tl.load(S_ptr + i, mask=m, other=0.0)
    S0i = tl.load(S0_ptr + i, mask=m, other=0.0)
    ri = tl.load(runoff_ptr + i, mask=m, other=0.0)
    r = (Si - S0i) - dt * (ri + inflow_q - Qi)
    scale = S0i + dt * (tl.abs(ri) + ainflow_q + tl.abs(Qi)) + 1000.0
    tl.store(R_ptr + i, r, mask=m)
    tl.store(wb_ptr + i, tl.div_rn(tl.full((), 1.0, tl.float32), scale), mask=m)


@triton.jit
def rebuild_snapshot_kernel(
    dt0_ptr,
    eps,
    Q0_ptr,
    nc_ptr,
    nf_ptr,
    L_ptr,
    W_ptr,
    H_ptr,
    A_ptr,
    tab_ptr,
    bed_ptr,
    elev_ptr,
    dist_ptr,
    down_ptr,
    edge_ptr,
    ref_ptr,
    h_ptr,
    S_ptr,
    area_ptr,
    Q_ptr,
    qi_ptr,
    qd_ptr,
    R_ptr,
    saved_ptr,
    n,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    dt = tl.load(dt0_ptr)
    snapshot_unit_vec_inline(
        i,
        dt,
        eps,
        m,
        Q0_ptr,
        nc_ptr,
        nf_ptr,
        L_ptr,
        W_ptr,
        H_ptr,
        A_ptr,
        tab_ptr,
        bed_ptr,
        elev_ptr,
        dist_ptr,
        down_ptr,
        edge_ptr,
        ref_ptr,
        h_ptr,
        S_ptr,
        area_ptr,
        Q_ptr,
        qi_ptr,
        qd_ptr,
        R_ptr,
        saved_ptr,
        levels,
    )


# --------------------------------------------------------------------------- #
# Forest adjoint kernels (heavy-path rounds up/down + finish)
# --------------------------------------------------------------------------- #


@triton.jit
def adjoint_up_kernel(
    saved_ptr,
    gQ_ptr,
    gS_ptr,
    gh_ptr,
    cp_ptr,
    kids_ptr,
    pnodes_ptr,
    pptr_ptr,
    ap_ptr,
    rp_ptr,
    round_start,
    round_end,
    BLOCK_SIZE: tl.constexpr,
):
    p = round_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = p < round_end
    beg = tl.load(pptr_ptr + p, mask=valid, other=0)
    end = tl.load(pptr_ptr + p + 1, mask=valid, other=0)
    first_node = tl.load(pnodes_ptr + beg, mask=valid, other=0)
    dt = tl.load(saved_ptr + first_node.to(tl.int64) * 17 + 16, mask=valid, other=0.0)
    k = beg
    act = valid & (k < end)
    while tl.max(act.to(tl.int32), axis=0) != 0:
        i = tl.load(pnodes_ptr + k, mask=act, other=0)
        b_i = i.to(tl.int64) * 17
        j2 = tl.load(saved_ptr + b_i + 2, mask=act, other=0.0)
        j4 = tl.load(saved_ptr + b_i + 4, mask=act, other=0.0)
        bi = tl.load(saved_ptr + b_i + 5, mask=act, other=0.0) > 0.5
        a = tl.where(bi, 1.0, j4 + dt * j2)
        gSi = tl.load(gS_ptr + i, mask=act, other=0.0)
        gQi = tl.load(gQ_ptr + i, mask=act, other=0.0)
        ghi = tl.load(gh_ptr + i, mask=act, other=0.0)
        r = j4 * gSi + j2 * gQi + ghi
        e = tl.load(cp_ptr + i, mask=act, other=0)
        e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
        e_act = act & (e < e_end)
        while tl.max(e_act.to(tl.int32), axis=0) != 0:
            j = tl.load(kids_ptr + e, mask=e_act, other=0)
            b_j = j.to(tl.int64) * 17
            jj2 = tl.load(saved_ptr + b_j + 2, mask=e_act, other=0.0)
            jj3 = tl.load(saved_ptr + b_j + 3, mask=e_act, other=0.0)
            bj = tl.load(saved_ptr + b_j + 5, mask=e_act, other=0.0) > 0.5
            gQj = tl.load(gQ_ptr + j, mask=e_act, other=0.0)
            apj = tl.load(ap_ptr + j, mask=e_act, other=1.0)
            rpj = tl.load(rp_ptr + j, mask=e_act, other=0.0)
            apj_safe = tl.where(e_act, apj, 1.0)
            r = tl.where(e_act, r + jj3 * gQj, r)
            a = tl.where(e_act & (~bi), a - dt * jj3, a)
            upper = tl.where(bj, 0.0, dt * jj3)
            lower = tl.where(bi, 0.0, -dt * jj2)
            a = tl.where(e_act, a - tl.div_rn(upper * lower, apj_safe), a)
            r = tl.where(e_act, r - tl.div_rn(upper * rpj, apj_safe), r)
            e += 1
            e_act = act & (e < e_end)
        tl.store(ap_ptr + i, a, mask=act)
        tl.store(rp_ptr + i, r, mask=act)
        k += 1
        act = valid & (k < end)


@triton.jit
def adjoint_down_kernel(
    saved_ptr,
    gQ_ptr,
    down_ptr,
    edge_ptr,
    pnodes_ptr,
    pptr_ptr,
    ap_ptr,
    rp_ptr,
    lam_ptr,
    ce_ptr,
    g_runoff_ptr,
    round_start,
    round_end,
    BLOCK_SIZE: tl.constexpr,
):
    p = round_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = p < round_end
    beg = tl.load(pptr_ptr + p, mask=valid, other=0)
    end = tl.load(pptr_ptr + p + 1, mask=valid, other=0)
    first_node = tl.load(pnodes_ptr + beg, mask=valid, other=0)
    dt = tl.load(saved_ptr + first_node.to(tl.int64) * 17 + 16, mask=valid, other=0.0)
    k = end - 1
    act = valid & (k >= beg)
    while tl.max(act.to(tl.int32), axis=0) != 0:
        i = tl.load(pnodes_ptr + k, mask=act, other=0)
        b_i = i.to(tl.int64) * 17
        ed = tl.load(edge_ptr + i, mask=act, other=False)
        d = tl.where(ed, tl.load(down_ptr + i, mask=act & ed, other=0), 0)
        bd = (
            tl.load(saved_ptr + d.to(tl.int64) * 17 + 5, mask=act & ed, other=0.0) > 0.5
        )
        j2 = tl.load(saved_ptr + b_i + 2, mask=act, other=0.0)
        bi = tl.load(saved_ptr + b_i + 5, mask=act, other=0.0) > 0.5
        j12 = tl.load(saved_ptr + b_i + 12, mask=act, other=0.0)
        lam_d = tl.load(lam_ptr + d, mask=act & ed, other=0.0)
        lower = tl.where(ed & (~bd), -dt * j2, 0.0)
        rp_v = tl.load(rp_ptr + i, mask=act, other=0.0)
        ap_v = tl.load(ap_ptr + i, mask=act, other=1.0)
        ap_safe = tl.where(act, ap_v, 1.0)
        v = tl.div_rn(rp_v - tl.where(ed, lower * lam_d, 0.0), ap_safe)
        tl.store(lam_ptr + i, v, mask=act)
        li = tl.where(bi, 0.0, v)
        ld_ = tl.where(ed & (~bd), lam_d, 0.0)
        gQi = tl.load(gQ_ptr + i, mask=act, other=0.0)
        ce = gQi - dt * (li - ld_)
        tl.store(ce_ptr + i, ce, mask=act)
        tl.store(g_runoff_ptr + i, dt * li, mask=act)
        tl.store(gQ_ptr + i, ce * j12, mask=act)
        k -= 1
        act = valid & (k >= beg)


@triton.jit
def adjoint_finish_kernel(
    saved_ptr,
    gS_ptr,
    cp_ptr,
    kids_ptr,
    lam_ptr,
    ce_ptr,
    gw_ptr,
    gh_ptr,
    gnc_ptr,
    gnf_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    b_i = i.to(tl.int64) * 17
    bound = tl.load(saved_ptr + b_i + 5, mask=m, other=0.0) > 0.5
    lam_i = tl.load(lam_ptr + i, mask=m, other=0.0)
    li = tl.where(bound, 0.0, lam_i)
    gSi = tl.load(gS_ptr + i, mask=m, other=0.0)
    cei = tl.load(ce_ptr + i, mask=m, other=0.0)
    j7 = tl.load(saved_ptr + b_i + 7, mask=m, other=0.0)
    j8 = tl.load(saved_ptr + b_i + 8, mask=m, other=0.0)
    j9 = tl.load(saved_ptr + b_i + 9, mask=m, other=0.0)
    j10 = tl.load(saved_ptr + b_i + 10, mask=m, other=0.0)
    j13 = tl.load(saved_ptr + b_i + 13, mask=m, other=0.0)
    j14 = tl.load(saved_ptr + b_i + 14, mask=m, other=0.0)

    gw = tl.load(gw_ptr + i, mask=m, other=0.0) + (gSi - li) * j7 + cei * j9
    tl.store(gw_ptr + i, gw, mask=m)
    v = (gSi - li) * j8 + cei * j10
    tl.store(gS_ptr + i, li, mask=m)
    v = tl.where(bound, v - lam_i, v)
    cn = cei * j13
    fn = cei * j14
    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        cej = tl.load(ce_ptr + j, mask=e_act, other=0.0)
        jj11 = tl.load(saved_ptr + j.to(tl.int64) * 17 + 11, mask=e_act, other=0.0)
        v = tl.where(e_act, v + cej * jj11, v)
        e += 1
        e_act = m & (e < e_end)
    tl.store(gh_ptr + i, tl.load(gh_ptr + i, mask=m, other=0.0) + v, mask=m)
    tl.store(gnc_ptr + i, tl.load(gnc_ptr + i, mask=m, other=0.0) + cn, mask=m)
    tl.store(gnf_ptr + i, tl.load(gnf_ptr + i, mask=m, other=0.0) + fn, mask=m)
