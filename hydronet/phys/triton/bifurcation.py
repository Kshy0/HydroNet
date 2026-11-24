# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Triton coupled bifurcation graph solver, rebuild and implicit transpose adjoint."""

from __future__ import annotations

import triton
import triton.language as tl
from hydroforge.kernels.triton_errors import report_error
from hydroforge.kernels.triton_math import cbrt

from hydronet.config import (
    GRAPH_COUPLING_FULL_PHASE,
)
from hydronet.phys.triton.common import (
    CLOSED_ELEV_C,
    CODE_INVALID_ROUGHNESS,
    CODE_NEWTON_NOT_CONVERGED,
    DAMPED_GROWTH_C,
    DAMPED_PHASE,
    DAMPED_SIGMA_C,
    DAMPED_STEPS_C,
    DRY_DEPTH_C,
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
    LINE_SEARCH,
    MODE_DONE,
    MODE_EVAL,
    MODE_FINAL,
    MODE_RESTORE,
    MODE_TRIAL,
    RESTART_PHASE,
    SLOPE_LIMIT_C,
    TF,
    TI,
    eval_flux_vec_inline,
    graph_product_inline,
    graph_scale_inline,
    head_pair_inline,
    invert_storage_inline,
    isfinite_f32_inline,
    pair_add_inline,
    pair_diff_inline,
    phase_offset_value_inline,
    snapshot_unit_vec_inline,
    stage_pair_inline,
    terrain_coordinate_inline,
    terrain_inverse_inline,
    trial_pair_inline,
)

COUPLING_FULL_C = tl.constexpr(GRAPH_COUPLING_FULL_PHASE)


# --------------------------------------------------------------------------- #
# Coupled-component & full-graph physics kernels
# --------------------------------------------------------------------------- #


@triton.jit
def comp_coupling_weight_inline(phase):
    cw = tl.full((), 1.0, tl.float32) + phase.to(tl.float32) * 0.0
    cw = tl.where(phase == 0, 0.0, cw)
    cw = tl.where(phase == 1, 0.015625, cw)
    cw = tl.where(phase == 2, 0.03125, cw)
    cw = tl.where(phase == 3, 0.0625, cw)
    cw = tl.where(phase == 4, 0.125, cw)
    cw = tl.where(phase == 5, 0.25, cw)
    cw = tl.where(phase == 6, 0.5, cw)
    return cw


@triton.jit
def coupled_start_units_kernel(
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
    bound_ptr,
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
    tl.store(bound_ptr + i, z0, mask=m)
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
def coupled_start_comps_kernel(
    ti_ptr,
    tf_ptr,
    tm_ptr,
    tline_ptr,
    dt0,
    ngc,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = c < ngc
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bt_i = c.to(tl.int64) * TI
    bt_f = c.to(tl.int64) * TF
    for k in range(TI):
        tl.store(ti_ptr + bt_i + k, zi, mask=m)
    for k in range(TF):
        tl.store(tf_ptr + bt_f + k, zf, mask=m)
    tl.store(ti_ptr + bt_i + IDX_MODE, zi + MODE_EVAL, mask=m)
    tl.store(ti_ptr + bt_i + IDX_PHASE, zi - 1, mask=m)
    tl.store(ti_ptr + bt_i + IDX_BEST_ST, zi + 2, mask=m)
    tl.store(tf_ptr + bt_f + IDX_DT, zf + dt0, mask=m)
    tl.store(tf_ptr + bt_f + IDX_BEST, zf + 1e30, mask=m)
    tl.store(tm_ptr + c, zi, mask=m)
    tl.store(tline_ptr + c, zi, mask=m)


@triton.jit
def coupled_stage_kernel(
    gunit_ptr,
    u_comp_ptr,
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
    c = tl.load(u_comp_ptr + i, mask=m, other=0)
    act = m & (
        tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_MODE, mask=m, other=MODE_DONE)
        != MODE_DONE
    )
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
def coupled_main_flux_kernel(
    dt,
    eps,
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
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
    ngu,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    c = tl.load(u_comp_ptr + i, mask=m, other=0)
    act = m & (
        tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_MODE, mask=m, other=MODE_DONE)
        != MODE_DONE
    )
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
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
def graph_main_flux_kernel(
    dt,
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
def eval_bif_flux_vec_inline(
    p,
    mask,
    dt,
    cw,
    bsrc_ptr,
    bdst_ptr,
    elev_ptr,
    H_ptr,
    tref_ptr,
    ht_ptr,
    bw_ptr,
    blen_ptr,
    belv_ptr,
    bn_ptr,
    qb0_ptr,
    path_val_ptr,
    path_sum_ptr,
    status_ptr,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    u = tl.load(bsrc_ptr + p, mask=mask, other=0)
    d = tl.load(bdst_ptr + p, mask=mask, other=0)
    ou = tl.load(tref_ptr + u, mask=mask, other=0.0)
    xu = tl.load(ht_ptr + u, mask=mask, other=0.0)
    od = tl.load(tref_ptr + d, mask=mask, other=0.0)
    xd = tl.load(ht_ptr + d, mask=mask, other=0.0)
    eu = tl.load(elev_ptr + u, mask=mask, other=0.0)
    ed = tl.load(elev_ptr + d, mask=mask, other=0.0)
    plen = tl.load(blen_ptr + p, mask=mask, other=1.0)
    head = head_pair_inline(eu, ed, ou, od, xu, xd)
    raw = tl.div_rn(head, plen)
    high_is_u = head >= 0.0
    eh = tl.where(high_is_u, eu, ed)
    oh = tl.where(high_is_u, ou, od)
    xh = tl.where(high_is_u, xu, xd)
    slope = tl.minimum(tl.maximum(raw, -SLOPE_LIMIT_C), SLOPE_LIMIT_C)
    ds = tl.where(
        tl.abs(raw) < SLOPE_LIMIT_C, tl.div_rn(tl.full((), 1.0, tl.float32), plen), 0.0
    )

    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    one = tl.full((), 1.0, tl.float32) + zf
    ten_thirds = tl.div_rn(tl.full((), 10.0, tl.float32), tl.full((), 3.0, tl.float32))
    net = zf
    di_sum = zf
    dd_sum = zf

    for lvl in range(blevels):
        k = p * blevels + lvl
        el_k = tl.load(belv_ptr + k, mask=mask, other=0.0)
        n_k = tl.load(bn_ptr + k, mask=mask, other=1.0)
        w_k = tl.load(bw_ptr + k, mask=mask, other=0.0)
        prev_k = tl.load(qb0_ptr + k, mask=mask, other=0.0)
        bad_n = (
            mask
            & (w_k > 0.0)
            & (el_k < CLOSED_ELEV_C)
            & (~((n_k > 0.0) & isfinite_f32_inline(n_k)))
        )
        if tl.max(bad_n.to(tl.int32), axis=0) != 0:
            report_error(status_ptr, bad_n, CODE_INVALID_ROUGHNESS, k)

        dh_hi, dh_lo = pair_diff_inline(eh, el_k)
        h_hi, h_lo = pair_add_inline(dh_hi, dh_lo, oh, xh)
        depth = tl.maximum(h_hi + h_lo, 0.0)
        active = (
            mask
            & (depth > DRY_DEPTH_C)
            & (w_k > 0.0)
            & (n_k > 0.0)
            & isfinite_f32_inline(n_k)
        )

        a = 9.8 * dt * w_k * depth
        v = prev_k + a * slope
        donor_is_u = v >= 0.0
        donor = tl.where(donor_is_u, u, d)
        e_don = tl.where(donor_is_u, eu, ed)
        o_don = tl.where(donor_is_u, ou, od)
        x_don = tl.where(donor_is_u, xu, xd)
        H_don = tl.load(H_ptr + donor, mask=mask, other=0.0)
        bank_bed = e_don - H_don
        below_bed = bank_bed > el_k
        dd_bed = tl.maximum((o_don + H_don) + x_don, 0.0)
        f1h, f1l = pair_diff_inline(e_don, el_k)
        f2h, f2l = pair_add_inline(f1h, f1l, o_don, x_don)
        dd_el = tl.maximum(f2h + f2l, 0.0)
        donor_depth = tl.where(below_bed, dd_bed, dd_el)

        flow = active & (donor_depth > DRY_DEPTH_C)
        dd_safe = tl.where(flow, donor_depth, one)
        dep_safe = tl.where(flow, depth, one)
        n_safe = tl.where(flow, n_k, one)
        w_safe = tl.where(flow, w_k, one)

        K = tl.div_rn(tl.div_rn(w_safe * dd_safe * dd_safe, cbrt(dd_safe)), n_safe)
        b = tl.div_rn(9.8 * dt * w_safe * dep_safe, K * K)
        den = tl.sqrt_rn(1.0 + 4.0 * b * tl.abs(v))
        q = tl.div_rn(2.0 * v, 1.0 + den)
        fric = b * q * tl.abs(q)
        high_part = tl.div_rn(a * slope - fric, dep_safe)
        donor_part = tl.div_rn(ten_thirds * fric, dd_safe)
        qi = tl.div_rn(
            tl.where(high_is_u, high_part, zf)
            + tl.where(donor_is_u, donor_part, zf)
            + a * ds,
            den,
        )
        qd = tl.div_rn(
            tl.where(~high_is_u, high_part, zf)
            + tl.where(~donor_is_u, donor_part, zf)
            - a * ds,
            den,
        )
        qp = tl.div_rn(one, den)
        qn = tl.div_rn(-2.0 * fric, n_safe * den)
        h_val = tl.where(below_bed, tl.div_rn(donor_part, den), zf)
        hu = tl.where(donor_is_u, h_val, zf)
        hd = tl.where(~donor_is_u, h_val, zf)

        q = tl.where(flow, q * cw, zf)
        qi = tl.where(flow, qi * cw, zf)
        qd = tl.where(flow, qd * cw, zf)
        qp = tl.where(flow, qp * cw, zf)
        qn = tl.where(flow, qn * cw, zf)
        hu = tl.where(flow, hu * cw, zf)
        hd = tl.where(flow, hd * cw, zf)

        bk = k.to(tl.int64) * 7
        tl.store(path_val_ptr + bk + 0, q, mask=mask)
        tl.store(path_val_ptr + bk + 1, qi, mask=mask)
        tl.store(path_val_ptr + bk + 2, qd, mask=mask)
        tl.store(path_val_ptr + bk + 3, qp, mask=mask)
        tl.store(path_val_ptr + bk + 4, qn, mask=mask)
        tl.store(path_val_ptr + bk + 5, hu, mask=mask)
        tl.store(path_val_ptr + bk + 6, hd, mask=mask)
        net = net + q
        di_sum = di_sum + qi
        dd_sum = dd_sum + qd

    bp = p.to(tl.int64) * 3
    tl.store(path_sum_ptr + bp + 0, net, mask=mask)
    tl.store(path_sum_ptr + bp + 1, di_sum, mask=mask)
    tl.store(path_sum_ptr + bp + 2, dd_sum, mask=mask)


@triton.jit
def coupled_bif_flux_kernel(
    dt,
    p_comp_ptr,
    ti_ptr,
    bsrc_ptr,
    bdst_ptr,
    elev_ptr,
    H_ptr,
    tref_ptr,
    ht_ptr,
    bw_ptr,
    blen_ptr,
    belv_ptr,
    bn_ptr,
    qb0_ptr,
    path_val_ptr,
    path_sum_ptr,
    status_ptr,
    m_paths,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = p < m_paths
    c = tl.load(p_comp_ptr + p, mask=m, other=0)
    bt_i = c.to(tl.int64) * TI
    act = m & (tl.load(ti_ptr + bt_i + IDX_MODE, mask=m, other=MODE_DONE) != MODE_DONE)
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    phase = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=act, other=-1)
    cw = comp_coupling_weight_inline(phase)
    eval_bif_flux_vec_inline(
        p,
        act,
        dt,
        cw,
        bsrc_ptr,
        bdst_ptr,
        elev_ptr,
        H_ptr,
        tref_ptr,
        ht_ptr,
        bw_ptr,
        blen_ptr,
        belv_ptr,
        bn_ptr,
        qb0_ptr,
        path_val_ptr,
        path_sum_ptr,
        status_ptr,
        blevels,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def graph_bif_flux_kernel(
    dt,
    coupling_weight,
    bsrc_ptr,
    bdst_ptr,
    elev_ptr,
    H_ptr,
    tref_ptr,
    ht_ptr,
    bw_ptr,
    blen_ptr,
    belv_ptr,
    bn_ptr,
    qb0_ptr,
    path_val_ptr,
    path_sum_ptr,
    status_ptr,
    m_paths,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = p < m_paths
    cw = coupling_weight + tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    eval_bif_flux_vec_inline(
        p,
        mask,
        dt,
        cw,
        bsrc_ptr,
        bdst_ptr,
        elev_ptr,
        H_ptr,
        tref_ptr,
        ht_ptr,
        bw_ptr,
        blen_ptr,
        belv_ptr,
        bn_ptr,
        qb0_ptr,
        path_val_ptr,
        path_sum_ptr,
        status_ptr,
        blevels,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def coupled_residual_kernel(
    dt,
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    S0_ptr,
    runoff_ptr,
    S_ptr,
    Q_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    path_sum_ptr,
    R_ptr,
    wb_ptr,
    rhs_ptr,
    balance_scale_ptr,
    tm_ptr,
    tline_ptr,
    ngu,
    SINGLE_COMP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if SINGLE_COMP:
        c = 0
        bt_i = 0
        mode = tl.load(ti_ptr + IDX_MODE)
    else:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        bt_i = c.to(tl.int64) * TI
        mode = tl.load(ti_ptr + bt_i + IDX_MODE, mask=m, other=MODE_DONE)
    act = m & (mode != MODE_DONE)
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return

    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    in_q = zf
    ain_q = zf
    e = tl.load(cp_ptr + i, mask=act, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
    e_act = act & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        kid = tl.load(kids_ptr + e, mask=e_act, other=0)
        qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
        in_q = tl.where(e_act, in_q + qk, in_q)
        ain_q = tl.where(e_act, ain_q + tl.abs(qk), ain_q)
        e += 1
        e_act = act & (e < e_end)

    out_b = zf
    aout_b = zf
    be = tl.load(boff_ptr + i, mask=act, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=act, other=0)
    be_act = act & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        qb = tl.load(path_sum_ptr + p.to(tl.int64) * 3, mask=be_act, other=0.0)
        sgn = tl.where(lnk > 0, 1.0, -1.0)
        out_b = tl.where(be_act, out_b + sgn * qb, out_b)
        aout_b = tl.where(be_act, aout_b + tl.abs(qb), aout_b)
        be += 1
        be_act = act & (be < be_end)

    q = tl.load(Q_ptr + i, mask=act, other=0.0)
    S_i = tl.load(S_ptr + i, mask=act, other=0.0)
    S0_i = tl.load(S0_ptr + i, mask=act, other=0.0)
    r_in = tl.load(runoff_ptr + i, mask=act, other=0.0)
    r = (S_i - S0_i) - dt * (r_in + in_q - q - out_b)
    scale = S0_i + dt * (tl.abs(r_in) + ain_q + tl.abs(q) + aout_b) + 1000.0
    inv_scale = tl.div_rn(tl.full((), 1.0, tl.float32), scale)
    tl.store(rhs_ptr + i, inv_scale, mask=act)
    tl.store(R_ptr + i, r, mask=act)
    tl.store(wb_ptr + i, inv_scale, mask=act)

    inf_val = tl.full((), float("inf"), tl.float32)
    fin = isfinite_f32_inline(r) & isfinite_f32_inline(q)
    score = tl.where(fin, tl.div_rn(tl.abs(r), scale), inf_val)
    if SINGLE_COMP:
        phase = tl.load(ti_ptr + IDX_PHASE)
    else:
        phase = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=act, other=-1)

    if SINGLE_COMP:
        bm = tl.max(tl.where(act, score, 0.0), axis=0)
        lanes = tl.arange(0, BLOCK_SIZE)
        lane0 = lanes == 0
        bm_i = tl.cast(tl.abs(bm), tl.int32, bitcast=True)
        tl.atomic_max(
            tm_ptr + lanes * 0,
            bm_i + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )
        if (mode == MODE_TRIAL) & (phase != -1):
            bs = tl.load(balance_scale_ptr + i, mask=act, other=0.0)
            lv = tl.where(fin, tl.abs(r) * bs, inf_val)
            bl = tl.max(tl.where(act, lv, 0.0), axis=0)
            bl_i = tl.cast(tl.abs(bl), tl.int32, bitcast=True)
            tl.atomic_max(
                tline_ptr + lanes * 0,
                bl_i + lanes * 0,
                mask=lane0,
                sem="relaxed",
                scope="gpu",
            )
    else:
        sc_i = tl.cast(tl.abs(score), tl.int32, bitcast=True)
        tl.atomic_max(tm_ptr + c, sc_i, mask=act, sem="relaxed", scope="gpu")
        need_line = act & (mode == MODE_TRIAL) & (phase != -1)
        if tl.max(need_line.to(tl.int32), axis=0) != 0:
            bs = tl.load(balance_scale_ptr + i, mask=need_line, other=0.0)
            lv = tl.where(fin, tl.abs(r) * bs, inf_val)
            lv_i = tl.cast(tl.abs(lv), tl.int32, bitcast=True)
            tl.atomic_max(
                tline_ptr + c, lv_i, mask=need_line, sem="relaxed", scope="gpu"
            )


@triton.jit
def graph_residual_kernel(
    dt,
    S0_ptr,
    runoff_ptr,
    S_ptr,
    Q_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    path_sum_ptr,
    R_ptr,
    wb_ptr,
    rhs_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    in_q = zf
    ain_q = zf
    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        kid = tl.load(kids_ptr + e, mask=e_act, other=0)
        qk = tl.load(Q_ptr + kid, mask=e_act, other=0.0)
        in_q = tl.where(e_act, in_q + qk, in_q)
        ain_q = tl.where(e_act, ain_q + tl.abs(qk), ain_q)
        e += 1
        e_act = m & (e < e_end)

    out_b = zf
    aout_b = zf
    be = tl.load(boff_ptr + i, mask=m, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
    be_act = m & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        qb = tl.load(path_sum_ptr + p.to(tl.int64) * 3, mask=be_act, other=0.0)
        sgn = tl.where(lnk > 0, 1.0, -1.0)
        out_b = tl.where(be_act, out_b + sgn * qb, out_b)
        aout_b = tl.where(be_act, aout_b + tl.abs(qb), aout_b)
        be += 1
        be_act = m & (be < be_end)

    q = tl.load(Q_ptr + i, mask=m, other=0.0)
    S_i = tl.load(S_ptr + i, mask=m, other=0.0)
    S0_i = tl.load(S0_ptr + i, mask=m, other=0.0)
    r_in = tl.load(runoff_ptr + i, mask=m, other=0.0)
    r = (S_i - S0_i) - dt * (r_in + in_q - q - out_b)
    scale = S0_i + dt * (tl.abs(r_in) + ain_q + tl.abs(q) + aout_b) + 1000.0
    inv_scale = tl.div_rn(tl.full((), 1.0, tl.float32), scale)
    tl.store(rhs_ptr + i, inv_scale, mask=m)
    tl.store(R_ptr + i, r, mask=m)
    tl.store(wb_ptr + i, inv_scale, mask=m)


@triton.jit
def coupled_control_kernel(
    ti_ptr,
    tf_ptr,
    tm_ptr,
    tline_ptr,
    comp_rep_ptr,
    ctl_ptr,
    meta_ptr,
    status_ptr,
    tol,
    maxit,
    ngc,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = c < ngc
    bt_i = c.to(tl.int64) * TI
    bt_f = c.to(tl.int64) * TF

    mt_i = tl.load(tm_ptr + c, mask=m, other=0)
    ln_i = tl.load(tline_ptr + c, mask=m, other=0)
    mt = tl.cast(mt_i, tl.float32, bitcast=True)
    line = tl.cast(ln_i, tl.float32, bitcast=True)
    zi = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(tm_ptr + c, zi, mask=m)
    tl.store(tline_ptr + c, zi, mask=m)

    mode = tl.load(ti_ptr + bt_i + IDX_MODE, mask=m, other=MODE_DONE)
    phase = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=m, other=-1)
    steps = tl.load(ti_ptr + bt_i + IDX_STEPS, mask=m, other=0)
    bt = tl.load(ti_ptr + bt_i + IDX_BT, mask=m, other=0)
    st = tl.load(ti_ptr + bt_i + IDX_ST, mask=m, other=0)
    fevals = tl.load(ti_ptr + bt_i + IDX_FEVALS, mask=m, other=0)
    totalbt = tl.load(ti_ptr + bt_i + IDX_TOTALBT, mask=m, other=0)
    sum_g = tl.load(ti_ptr + bt_i + IDX_SUM_G, mask=m, other=0)
    sum_fe = tl.load(ti_ptr + bt_i + IDX_SUM_FE, mask=m, other=0)
    dsteps = tl.load(ti_ptr + bt_i + IDX_DSTEPS, mask=m, other=0)
    best_st = tl.load(ti_ptr + bt_i + IDX_BEST_ST, mask=m, other=2)

    lam = tl.load(tf_ptr + bt_f + IDX_LAM, mask=m, other=0.0)
    merit = tl.load(tf_ptr + bt_f + IDX_MERIT, mask=m, other=0.0)
    best = tl.load(tf_ptr + bt_f + IDX_BEST, mask=m, other=1e30)
    initial = tl.load(tf_ptr + bt_f + IDX_INITIAL, mask=m, other=0.0)
    sigma = tl.load(tf_ptr + bt_f + IDX_SIGMA, mask=m, other=0.0)

    commit = zi
    save = zi
    reset = zi
    solve = zi
    finished_now = zi

    active = m & (mode != MODE_DONE)

    # 1. mode == FINAL
    is_final = active & (mode == MODE_FINAL)
    merit = tl.where(is_final, mt, merit)
    st = tl.where(is_final, tl.where(mt < tol, zi, best_st), st)
    mode = tl.where(is_final, zi + MODE_DONE, mode)
    finished_now = tl.where(is_final, zi + 1, finished_now)

    # 2. DAMPED_PHASE (mode != RESTORE)
    is_damped = active & (~is_final) & (phase == DAMPED_PHASE) & (mode != MODE_RESTORE)
    fevals = tl.where(is_damped, fevals + 1, fevals)
    d_trial = is_damped & (mode == MODE_TRIAL)
    d_fin_mt = isfinite_f32_inline(mt)
    d_bad = d_trial & (~d_fin_mt)
    d_good = d_trial & d_fin_mt
    sigma = tl.where(d_bad, tl.maximum(sigma * 10.0, 1e-3), sigma)
    new_tbt = totalbt + 1
    totalbt = tl.where(d_bad, new_tbt, totalbt)
    d_giveup = d_bad & (new_tbt >= 40)
    st = tl.where(d_giveup, zi + 2, st)
    mode = tl.where(d_bad, tl.where(d_giveup, zi + MODE_RESTORE, zi + MODE_EVAL), mode)

    old_d = merit
    commit = tl.where(d_good, zi + 1, commit)
    merit = tl.where(d_good, mt, merit)
    steps = tl.where(d_good, steps + 1, steps)
    dsteps = tl.where(d_good, dsteps + 1, dsteps)
    old_safe = tl.where(d_good, old_d, 1.0)
    sig_new = sigma * tl.minimum(tl.div_rn(line, old_safe), DAMPED_GROWTH_C)
    sig_new = tl.where((sig_new == 0.0) & (line >= old_d), 1e-3, sig_new)
    sig_new = tl.where(sig_new < 1e-7, 0.0, sig_new)
    sigma = tl.where(d_good, sig_new, sigma)

    d_eval = is_damped & (~d_trial)
    merit = tl.where(d_eval, mt, merit)
    d_check = is_damped & (~d_bad)
    d_conv = d_check & (~(mt >= tol))
    d_maxed = d_check & (~d_conv) & (steps >= DAMPED_STEPS_C)
    d_finish = d_conv | d_maxed
    d_step = d_check & (~d_finish)
    st = tl.where(d_conv, zi, tl.where(d_maxed, zi + 1, st))
    sigma = tl.where(d_finish, zf, sigma)
    solve = tl.where(d_step, zi + 1, solve)
    mode = tl.where(d_step, zi + MODE_TRIAL, mode)
    lam = tl.where(d_step, zf + 1.0, lam)
    bt = tl.where(d_step, zi, bt)

    # 3. Standard Newton / line search / restore
    is_std = active & (~is_final) & (~is_damped)
    s_eval = is_std & (mode == MODE_EVAL)
    s_trial = is_std & (mode == MODE_TRIAL)
    s_restore = is_std & (~s_eval) & (~s_trial)
    fixed_w = phase != -1
    step_cap = tl.where((~fixed_w) & (maxit > 36), 36, maxit)

    initial = tl.where(s_eval & (steps == 0) & (phase == -1), mt, initial)
    merit = tl.where(s_eval, mt, merit)
    fevals = tl.where(s_eval, fevals + 1, fevals)
    newton_done = s_eval & (~(mt >= tol))

    fevals = tl.where(s_trial, fevals + 1, fevals)
    totalbt = tl.where(s_trial, totalbt + 1, totalbt)
    old_s = merit
    cmp_v = tl.where(fixed_w, line, mt)
    accepted = (
        s_trial
        & isfinite_f32_inline(mt)
        & ((mt < tol) | (cmp_v <= (1.0 - 0.01 * lam) * old_s))
    )
    rejected = s_trial & (~accepted)

    commit = tl.where(accepted, zi + 1, commit)
    merit = tl.where(accepted, mt, merit)
    steps = tl.where(accepted, steps + 1, steps)
    newton_done = newton_done | (accepted & (~((steps < step_cap) & (mt >= tol))))

    lam = tl.where(rejected, lam * 0.5, lam)
    new_bt = bt + 1
    bt = tl.where(rejected, new_bt, bt)
    bt_fail = rejected & (~(new_bt < LINE_SEARCH))
    mode = tl.where(bt_fail, zi + MODE_RESTORE, mode)

    merit = tl.where(s_restore, mt, merit)
    fevals = tl.where(s_restore, fevals + 2, fevals)
    st = tl.where(s_restore, tl.where(mt < tol, zi, zi + 2), st)

    s_cont = (s_eval | accepted) & (~newton_done)
    solve = tl.where(s_cont, zi + 1, solve)
    mode = tl.where(s_cont, zi + MODE_TRIAL, mode)
    lam = tl.where(s_cont, zf + 1.0, lam)
    bt = tl.where(s_cont, zi, bt)

    s_done = (s_eval | accepted) & newton_done
    fevals = tl.where(s_done, fevals + 1, fevals)
    st = tl.where(s_done, tl.where(merit < tol, zi, zi + 1), st)

    # 4. finish_graph_comp transitions
    fp = d_finish | s_restore | s_done
    init0 = initial
    sum_g = tl.where(fp, sum_g + steps, sum_g)
    sum_fe = tl.where(fp, sum_fe + fevals, sum_fe)
    initial = tl.where(fp & (phase == -1), merit, initial)
    sigma = tl.where(fp & (phase == DAMPED_PHASE), zf, sigma)

    go_restart = fp & (st != 0) & (phase == -1)
    keep_surf = isfinite_f32_inline(merit) & (merit < init0)
    go_damped = fp & (~go_restart) & (st != 0) & (phase == RESTART_PHASE)
    go_cont0 = fp & (~go_restart) & (~go_damped) & (phase == DAMPED_PHASE) & (st != 0)
    go_done_early = (
        fp
        & (~go_restart)
        & (~go_damped)
        & (~go_cont0)
        & ((phase == DAMPED_PHASE) | (phase == RESTART_PHASE) | (phase == -1))
    )

    rem = fp & (~go_restart) & (~go_damped) & (~go_cont0) & (~go_done_early)
    go_next_cont = rem & (phase >= 0) & (phase < COUPLING_FULL_C)
    go_p13 = rem & (~go_next_cont) & (phase == COUPLING_FULL_C) & (st != 0)
    in_p13p = rem & (~go_next_cont) & (~go_p13) & (phase >= 13)
    better = in_p13p & (merit < best)
    best = tl.where(better, merit, best)
    best_st = tl.where(better, st, best_st)
    save = tl.where(better, save | 2, save)
    go_p13_next = in_p13p & (st != 0) & (phase < 20)
    go_p20_final = in_p13p & (st != 0) & (~(phase < 20))
    go_done_other = rem & (~go_next_cont) & (~go_p13) & (~go_p13_next) & (~go_p20_final)

    best = tl.where(go_p13, merit, best)
    best_st = tl.where(go_p13, st, best_st)
    save = tl.where(go_p13, save | 1, save)

    sp_any = go_restart | go_cont0 | go_next_cont | go_p13 | go_p13_next
    target_phase = tl.where(
        go_restart,
        zi + RESTART_PHASE,
        tl.where(
            go_cont0,
            zi,
            tl.where(go_p13, zi + 13, phase + 1),
        ),
    )
    target_reset = tl.where(
        go_restart,
        tl.where(keep_surf, zi, zi + 1),
        tl.where(go_cont0, zi + 1, tl.where(go_p13 | go_p13_next, zi + 2, zi)),
    )
    mode = tl.where(sp_any, zi + MODE_EVAL, mode)
    phase = tl.where(sp_any, target_phase, phase)
    st = tl.where(sp_any, zi, st)
    steps = tl.where(sp_any, zi, steps)
    fevals = tl.where(sp_any, zi, fevals)
    totalbt = tl.where(sp_any, zi, totalbt)
    reset = tl.where(sp_any, target_reset, reset)
    bt = tl.where(go_restart, zi, bt)

    phase = tl.where(go_damped, zi + DAMPED_PHASE, phase)
    mode = tl.where(go_damped, zi + MODE_EVAL, mode)
    st = tl.where(go_damped, zi, st)
    steps = tl.where(go_damped, zi, steps)
    fevals = tl.where(go_damped, zi, fevals)
    totalbt = tl.where(go_damped, zi, totalbt)
    bt = tl.where(go_damped, zi, bt)
    sigma = tl.where(go_damped, zf + DAMPED_SIGMA_C, sigma)

    save = tl.where(go_p20_final, save | 4, save)
    mode = tl.where(go_p20_final, zi + MODE_FINAL, mode)

    go_done = go_done_early | go_done_other
    mode = tl.where(go_done, zi + MODE_DONE, mode)
    finished_now = tl.where(go_done, zi + 1, finished_now)

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

    tl.store(tf_ptr + bt_f + IDX_LAM, lam, mask=m)
    tl.store(tf_ptr + bt_f + IDX_MERIT, merit, mask=m)
    tl.store(tf_ptr + bt_f + IDX_BEST, best, mask=m)
    tl.store(tf_ptr + bt_f + IDX_INITIAL, initial, mask=m)
    tl.store(tf_ptr + bt_f + IDX_SIGMA, sigma, mask=m)

    fin_m = m & (finished_now != 0)
    if tl.max(fin_m.to(tl.int32), axis=0) != 0:
        rep = tl.load(comp_rep_ptr + c, mask=fin_m, other=0)
        bm = rep.to(tl.int64) * 16
        tl.store(meta_ptr + bm + 0, merit, mask=fin_m)
        tl.store(meta_ptr + bm + 1, sum_g.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 2, st.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 3, totalbt.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 4, sum_g.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 5, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 6, sum_fe.to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 7, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 8, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 9, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 10, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 11, zf, mask=fin_m)
        tl.store(meta_ptr + bm + 12, dsteps.to(tl.float32), mask=fin_m)
        p_code = tl.where(phase < 0, 1, phase + 2).to(tl.float32)
        tl.store(meta_ptr + bm + 13, p_code, mask=fin_m)
        tl.store(meta_ptr + bm + 14, (phase >= 0).to(tl.float32), mask=fin_m)
        tl.store(meta_ptr + bm + 15, initial, mask=fin_m)
        bad_fin = fin_m & (st != 0)
        if tl.max(bad_fin.to(tl.int32), axis=0) != 0:
            report_error(status_ptr, bad_fin, CODE_NEWTON_NOT_CONVERGED, rep)

    n_act = tl.sum((m & (mode != MODE_DONE)).to(tl.int32), axis=0)
    n_sol = tl.sum((m & (solve != 0)).to(tl.int32), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    lane0 = lanes == 0
    if n_act > 0:
        tl.atomic_add(
            ctl_ptr + 5 + lanes * 0,
            n_act + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )
    if n_sol > 0:
        tl.atomic_add(
            ctl_ptr + 6 + lanes * 0,
            n_sol + lanes * 0,
            mask=lane0,
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def coupled_apply_kernel(
    gunit_ptr,
    u_comp_ptr,
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
    ngu,
    n,
    levels,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    c = tl.load(u_comp_ptr + i, mask=m, other=0)
    bt_i = c.to(tl.int64) * TI
    commit = tl.load(ti_ptr + bt_i + IDX_COMMIT, mask=m, other=0)
    save = tl.load(ti_ptr + bt_i + IDX_SAVE, mask=m, other=0)
    reset = tl.load(ti_ptr + bt_i + IDX_RESET, mask=m, other=0)
    solve = tl.load(ti_ptr + bt_i + IDX_SOLVE, mask=m, other=0)
    phase = tl.load(ti_ptr + bt_i + IDX_PHASE, mask=m, other=0)

    if tl.max((m & ((commit | save | reset | solve) != 0)).to(tl.int32), axis=0) == 0:
        return

    ref_v = tl.load(ref_ptr + i, mask=m, other=0.0)
    h_v = tl.load(h_ptr + i, mask=m, other=0.0)
    cm = m & (commit != 0)
    if tl.max(cm.to(tl.int32), axis=0) != 0:
        ref_v = tl.where(cm, tl.load(tref_ptr + i, mask=cm, other=0.0), ref_v)
        h_v = tl.where(cm, tl.load(ht_ptr + i, mask=cm, other=0.0), h_v)
        tl.store(ref_ptr + i, ref_v, mask=cm)
        tl.store(h_ptr + i, h_v, mask=cm)

    wm = m & (solve != 0) & ((phase == RESTART_PHASE) | (phase == DAMPED_PHASE))
    if tl.max(wm.to(tl.int32), axis=0) != 0:
        tl.store(wa_ptr + i, tl.load(wb_ptr + i, mask=wm, other=0.0), mask=wm)

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
def coupled_trial_kernel(
    gunit_ptr,
    u_comp_ptr,
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
    c = tl.load(u_comp_ptr + i, mask=m, other=0)
    bt_i = c.to(tl.int64) * TI
    bt_f = c.to(tl.int64) * TF
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


@triton.jit
def coupled_prep_rhs_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    ref_ptr,
    h_ptr,
    H_ptr,
    R_ptr,
    rhs_ptr,
    balance_scale_ptr,
    bound_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if ALL_ACTIVE:
        act = m
    else:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        act = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    bs = tl.load(rhs_ptr + i, mask=act, other=0.0)
    tl.store(balance_scale_ptr + i, bs, mask=act)
    ref_v = tl.load(ref_ptr + i, mask=act, other=0.0)
    h_v = tl.load(h_ptr + i, mask=act, other=0.0)
    hh = tl.load(H_ptr + i, mask=act, other=0.0)
    R_v = tl.load(R_ptr + i, mask=act, other=0.0)
    bound = (~(((ref_v + hh) + h_v) > 0.0)) & (R_v > 0.0)
    tl.store(rhs_ptr + i, tl.where(bound, 0.0, -R_v), mask=act)
    tl.store(bound_ptr + i, tl.where(bound, 1.0, 0.0), mask=act)


# --------------------------------------------------------------------------- #
# Equilibration, static-pattern LU, matvec & BiCGStab kernels
# --------------------------------------------------------------------------- #


@triton.jit
def graph_damping_inline(
    i,
    mask,
    dt,
    u_comp_ptr,
    tf_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    path_sum_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    c = tl.load(u_comp_ptr + i, mask=mask, other=-1)
    has_c = mask & (c >= 0)
    sigma = tl.load(tf_ptr + c.to(tl.int64) * TF + IDX_SIGMA, mask=has_c, other=0.0)
    need = has_c & (sigma > 0.0)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    if tl.max(need.to(tl.int32), axis=0) == 0:
        return zf
    d = tl.load(area_ptr + i, mask=need, other=0.0) + dt * tl.abs(
        tl.load(qi_ptr + i, mask=need, other=0.0)
    )
    e = tl.load(cp_ptr + i, mask=need, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=need, other=0)
    e_act = need & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        d = tl.where(
            e_act, d + dt * tl.abs(tl.load(qd_ptr + j, mask=e_act, other=0.0)), d
        )
        e += 1
        e_act = need & (e < e_end)
    be = tl.load(boff_ptr + i, mask=need, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=need, other=0)
    be_act = need & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        bp = p.to(tl.int64) * 3
        term = tl.load(
            path_sum_ptr + bp + tl.where(lnk > 0, 1, 2), mask=be_act, other=0.0
        )
        d = tl.where(be_act, d + dt * tl.abs(term), d)
        be += 1
        be_act = need & (be < be_end)
    return tl.where(need, sigma * d, zf)


@triton.jit
def coupled_matrix_and_init_kernel(
    dt,
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    tf_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    path_sum_ptr,
    bound_ptr,
    scaling_ptr,
    rhs_ptr,
    dh_ptr,
    sol_low_ptr,
    norm_i32_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if ALL_ACTIVE:
        act = m
    else:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        act = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    if tl.max(act.to(tl.int32), axis=0) == 0:
        return
    bound = tl.load(bound_ptr + i, mask=act, other=0.0) > 0.5
    damp = graph_damping_inline(
        i,
        act & (~bound),
        dt,
        u_comp_ptr,
        tf_ptr,
        area_ptr,
        qi_ptr,
        qd_ptr,
        cp_ptr,
        kids_ptr,
        boff_ptr,
        blink_ptr,
        path_sum_ptr,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    diag = (
        tl.load(area_ptr + i, mask=act, other=0.0)
        + dt * tl.load(qi_ptr + i, mask=act, other=0.0)
        + damp
    )
    e = tl.load(cp_ptr + i, mask=act, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
    e_act = act & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        diag = tl.where(
            e_act, diag - dt * tl.load(qd_ptr + j, mask=e_act, other=0.0), diag
        )
        e += 1
        e_act = act & (e < e_end)
    be = tl.load(boff_ptr + i, mask=act, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=act, other=0)
    be_act = act & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        bp = p.to(tl.int64) * 3
        term = tl.where(
            lnk > 0,
            tl.load(path_sum_ptr + bp + 1, mask=be_act, other=0.0),
            -tl.load(path_sum_ptr + bp + 2, mask=be_act, other=0.0),
        )
        diag = tl.where(be_act, diag + dt * term, diag)
        be += 1
        be_act = act & (be < be_end)
    diag_safe = tl.where(act & (~bound), diag, 1.0)
    sc = tl.where(bound, 1.0, tl.div_rn(tl.full((), 1.0, tl.float32), diag_safe))
    tl.store(scaling_ptr + i, sc, mask=act)
    rhs_v = tl.load(rhs_ptr + i, mask=act, other=0.0) * sc
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(rhs_ptr + i, rhs_v, mask=act)
    tl.store(dh_ptr + i, zf, mask=act)
    tl.store(sol_low_ptr + i, zf, mask=act)
    av = tl.abs(rhs_v)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(act, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def graph_matrix_and_init_kernel(
    dt,
    area_ptr,
    qi_ptr,
    qd_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    path_sum_ptr,
    bound_ptr,
    scaling_ptr,
    rhs_ptr,
    dh_ptr,
    sol_low_ptr,
    tmp_ptr,
    norm_i32_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    bound = tl.load(bound_ptr + i, mask=m, other=0.0) > 0.5
    diag = tl.load(area_ptr + i, mask=m, other=0.0) + dt * tl.load(
        qi_ptr + i, mask=m, other=0.0
    )
    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        diag = tl.where(
            e_act, diag - dt * tl.load(qd_ptr + j, mask=e_act, other=0.0), diag
        )
        e += 1
        e_act = m & (e < e_end)
    be = tl.load(boff_ptr + i, mask=m, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
    be_act = m & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        bp = p.to(tl.int64) * 3
        term = tl.where(
            lnk > 0,
            tl.load(path_sum_ptr + bp + 1, mask=be_act, other=0.0),
            -tl.load(path_sum_ptr + bp + 2, mask=be_act, other=0.0),
        )
        diag = tl.where(be_act, diag + dt * term, diag)
        be += 1
        be_act = m & (be < be_end)
    diag_safe = tl.where(m & (~bound), diag, 1.0)
    sc = tl.where(bound, 1.0, tl.div_rn(tl.full((), 1.0, tl.float32), diag_safe))
    tl.store(scaling_ptr + i, sc, mask=m)
    rhs_v = tl.load(rhs_ptr + i, mask=m, other=0.0) * sc
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(rhs_ptr + i, rhs_v, mask=m)
    tl.store(dh_ptr + i, zf, mask=m)
    tl.store(sol_low_ptr + i, zf, mask=m)
    tl.store(tmp_ptr + i, rhs_v, mask=m)
    av = tl.abs(rhs_v)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def factor_assemble_kernel(
    dt,
    source_ptr,
    source_code,
    ce_comp_ptr,
    ti_ptr,
    edge_ptr,
    down_ptr,
    bsrc_ptr,
    bdst_ptr,
    bound_ptr,
    scaling_ptr,
    qi_ptr,
    qd_ptr,
    path_sum_ptr,
    factor_ptr,
    entries,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    e = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = e < entries
    if not ALL_ACTIVE:
        comp = tl.load(ce_comp_ptr + e, mask=m, other=-1)
        has_c = m & (comp >= 0)
        m = has_c & (
            tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE, mask=has_c, other=0)
            != 0
        )
        if tl.max(m.to(tl.int32), axis=0) == 0:
            return
    c = tl.load(source_ptr + e, mask=m, other=0)
    c_end = tl.load(source_ptr + e + 1, mask=m, other=0)
    act = m & (c < c_end)
    val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    while tl.max(act.to(tl.int32), axis=0) != 0:
        code = tl.load(source_code + c, mask=act, other=0)
        kind = code & 7
        k = code >> 3
        k0 = kind == 0
        k1 = act & (kind == 1)
        k2 = act & (kind == 2)
        k3 = act & (kind == 3)
        k4 = act & (kind >= 4)

        val = tl.where(act & k0, val + 1.0, val)
        if tl.max(k1.to(tl.int32), axis=0) != 0:
            ed = tl.load(edge_ptr + k, mask=k1, other=False)
            bnd = tl.load(bound_ptr + k, mask=k1, other=0.0) > 0.5
            ok1 = k1 & ed & (~bnd)
            val = tl.where(
                ok1,
                val
                + dt
                * tl.load(qd_ptr + k, mask=ok1, other=0.0)
                * tl.load(scaling_ptr + k, mask=ok1, other=0.0),
                val,
            )
        if tl.max(k2.to(tl.int32), axis=0) != 0:
            d = tl.load(down_ptr + k, mask=k2, other=0)
            bnd = tl.load(bound_ptr + d, mask=k2, other=0.0) > 0.5
            ok2 = k2 & (~bnd)
            val = tl.where(
                ok2,
                val
                - dt
                * tl.load(qi_ptr + k, mask=ok2, other=0.0)
                * tl.load(scaling_ptr + d, mask=ok2, other=0.0),
                val,
            )
        if tl.max(k3.to(tl.int32), axis=0) != 0:
            u = tl.load(bsrc_ptr + k, mask=k3, other=0)
            bnd = tl.load(bound_ptr + u, mask=k3, other=0.0) > 0.5
            ok3 = k3 & (~bnd)
            val = tl.where(
                ok3,
                val
                + dt
                * tl.load(path_sum_ptr + k.to(tl.int64) * 3 + 2, mask=ok3, other=0.0)
                * tl.load(scaling_ptr + u, mask=ok3, other=0.0),
                val,
            )
        if tl.max(k4.to(tl.int32), axis=0) != 0:
            d = tl.load(bdst_ptr + k, mask=k4, other=0)
            bnd = tl.load(bound_ptr + d, mask=k4, other=0.0) > 0.5
            ok4 = k4 & (~bnd)
            val = tl.where(
                ok4,
                val
                - dt
                * tl.load(path_sum_ptr + k.to(tl.int64) * 3 + 1, mask=ok4, other=0.0)
                * tl.load(scaling_ptr + d, mask=ok4, other=0.0),
                val,
            )
        c += 1
        act = m & (c < c_end)
    tl.store(factor_ptr + e, val, mask=m)


@triton.jit
def factor_numeric_heavy_kernel(
    round_start,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    urow_ptr,
    urow_entry_ptr,
    lcol_ptr,
    lcol_entry_ptr,
    prod_ptr,
    prod_lo_ptr,
    prod_up_ptr,
    factor_ptr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = round_start + tl.program_id(0)
    t_beg = tl.load(path_off_ptr + p)
    t_end = tl.load(path_off_ptr + p + 1)
    if not ALL_ACTIVE:
        if t_beg < t_end:
            first_node = tl.load(order_ptr + tl.load(path_unit_ptr + t_beg))
            comp = tl.load(u_comp_ptr + first_node)
            if comp < 0:
                return
            if tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE) == 0:
                return
    lanes = tl.arange(0, BLOCK_SIZE)
    t = t_beg
    while t < t_end:
        k = tl.load(path_unit_ptr + t)
        u_beg = tl.load(urow_ptr + k)
        u_end = tl.load(urow_ptr + k + 1)
        off = u_beg
        while off < u_end:
            q = off + lanes
            qm = q < u_end
            e = tl.load(urow_entry_ptr + q, mask=qm, other=0)
            val = tl.load(factor_ptr + e, mask=qm, other=0.0)
            m_idx = tl.load(prod_ptr + e, mask=qm, other=0)
            m_end = tl.load(prod_ptr + e + 1, mask=qm, other=0)
            m_act = qm & (m_idx < m_end)
            while tl.max(m_act.to(tl.int32), axis=0) != 0:
                lo = tl.load(prod_lo_ptr + m_idx, mask=m_act, other=0)
                up = tl.load(prod_up_ptr + m_idx, mask=m_act, other=0)
                val = tl.where(
                    m_act,
                    val
                    - tl.load(factor_ptr + lo, mask=m_act, other=0.0)
                    * tl.load(factor_ptr + up, mask=m_act, other=0.0),
                    val,
                )
                m_idx += 1
                m_act = qm & (m_idx < m_end)
            tl.store(factor_ptr + e, val, mask=qm)
            off += BLOCK_SIZE
        tl.debug_barrier()

        pivot = tl.load(factor_ptr + tl.load(diag_ptr + k))
        l_beg = tl.load(lcol_ptr + k)
        l_end = tl.load(lcol_ptr + k + 1)
        off = l_beg
        while off < l_end:
            q = off + lanes
            qm = q < l_end
            e = tl.load(lcol_entry_ptr + q, mask=qm, other=0)
            val = tl.load(factor_ptr + e, mask=qm, other=0.0)
            m_idx = tl.load(prod_ptr + e, mask=qm, other=0)
            m_end = tl.load(prod_ptr + e + 1, mask=qm, other=0)
            m_act = qm & (m_idx < m_end)
            while tl.max(m_act.to(tl.int32), axis=0) != 0:
                lo = tl.load(prod_lo_ptr + m_idx, mask=m_act, other=0)
                up = tl.load(prod_up_ptr + m_idx, mask=m_act, other=0)
                val = tl.where(
                    m_act,
                    val
                    - tl.load(factor_ptr + lo, mask=m_act, other=0.0)
                    * tl.load(factor_ptr + up, mask=m_act, other=0.0),
                    val,
                )
                m_idx += 1
                m_act = qm & (m_idx < m_end)
            tl.store(factor_ptr + e, tl.div_rn(val, pivot), mask=qm)
            off += BLOCK_SIZE
        tl.debug_barrier()
        t += 1


@triton.jit
def factor_numeric_light_kernel(
    split_start,
    round_end,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    urow_ptr,
    urow_entry_ptr,
    lcol_ptr,
    lcol_entry_ptr,
    prod_ptr,
    prod_lo_ptr,
    prod_up_ptr,
    factor_ptr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = split_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = p < round_end
    t = tl.load(path_off_ptr + p, mask=valid, other=0)
    t_end = tl.load(path_off_ptr + p + 1, mask=valid, other=0)
    t_act = valid & (t < t_end)
    if not ALL_ACTIVE:
        k0 = tl.load(path_unit_ptr + t, mask=t_act, other=0)
        n0 = tl.load(order_ptr + k0, mask=t_act, other=0)
        comp = tl.load(u_comp_ptr + n0, mask=t_act, other=-1)
        has_c = t_act & (comp >= 0)
        t_act = has_c & (
            tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE, mask=has_c, other=0)
            != 0
        )
        valid = t_act
    while tl.max(t_act.to(tl.int32), axis=0) != 0:
        k = tl.load(path_unit_ptr + t, mask=t_act, other=0)
        q = tl.load(urow_ptr + k, mask=t_act, other=0)
        q_end = tl.load(urow_ptr + k + 1, mask=t_act, other=0)
        q_act = t_act & (q < q_end)
        while tl.max(q_act.to(tl.int32), axis=0) != 0:
            e = tl.load(urow_entry_ptr + q, mask=q_act, other=0)
            val = tl.load(factor_ptr + e, mask=q_act, other=0.0)
            m_idx = tl.load(prod_ptr + e, mask=q_act, other=0)
            m_end = tl.load(prod_ptr + e + 1, mask=q_act, other=0)
            m_act = q_act & (m_idx < m_end)
            while tl.max(m_act.to(tl.int32), axis=0) != 0:
                lo = tl.load(prod_lo_ptr + m_idx, mask=m_act, other=0)
                up = tl.load(prod_up_ptr + m_idx, mask=m_act, other=0)
                val = tl.where(
                    m_act,
                    val
                    - tl.load(factor_ptr + lo, mask=m_act, other=0.0)
                    * tl.load(factor_ptr + up, mask=m_act, other=0.0),
                    val,
                )
                m_idx += 1
                m_act = q_act & (m_idx < m_end)
            tl.store(factor_ptr + e, val, mask=q_act)
            q += 1
            q_act = t_act & (q < q_end)

        dk = tl.load(diag_ptr + k, mask=t_act, other=0)
        pivot = tl.load(factor_ptr + dk, mask=t_act, other=1.0)
        pivot_safe = tl.where(t_act, pivot, 1.0)
        q = tl.load(lcol_ptr + k, mask=t_act, other=0)
        q_end = tl.load(lcol_ptr + k + 1, mask=t_act, other=0)
        q_act = t_act & (q < q_end)
        while tl.max(q_act.to(tl.int32), axis=0) != 0:
            e = tl.load(lcol_entry_ptr + q, mask=q_act, other=0)
            val = tl.load(factor_ptr + e, mask=q_act, other=0.0)
            m_idx = tl.load(prod_ptr + e, mask=q_act, other=0)
            m_end = tl.load(prod_ptr + e + 1, mask=q_act, other=0)
            m_act = q_act & (m_idx < m_end)
            while tl.max(m_act.to(tl.int32), axis=0) != 0:
                lo = tl.load(prod_lo_ptr + m_idx, mask=m_act, other=0)
                up = tl.load(prod_up_ptr + m_idx, mask=m_act, other=0)
                val = tl.where(
                    m_act,
                    val
                    - tl.load(factor_ptr + lo, mask=m_act, other=0.0)
                    * tl.load(factor_ptr + up, mask=m_act, other=0.0),
                    val,
                )
                m_idx += 1
                m_act = q_act & (m_idx < m_end)
            tl.store(factor_ptr + e, tl.div_rn(val, pivot_safe), mask=q_act)
            q += 1
            q_act = t_act & (q < q_end)
        t += 1
        t_act = valid & (t < t_end)


@triton.jit
def factor_solve_fwd_heavy_kernel(
    round_start,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    fptr_ptr,
    fentry_ptr,
    findex_ptr,
    factor_ptr,
    c_ptr,
    y_ptr,
    TRANS: tl.constexpr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = round_start + tl.program_id(0)
    t_beg = tl.load(path_off_ptr + p)
    t_end = tl.load(path_off_ptr + p + 1)
    if not ALL_ACTIVE:
        if t_beg < t_end:
            first_node = tl.load(order_ptr + tl.load(path_unit_ptr + t_beg))
            comp = tl.load(u_comp_ptr + first_node)
            if comp < 0:
                return
            if tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE) == 0:
                return
    lanes = tl.arange(0, BLOCK_SIZE)
    lane0 = lanes == 0
    t = t_beg
    while t < t_end:
        k = tl.load(path_unit_ptr + t)
        qb = tl.load(fptr_ptr + k)
        qe = tl.load(fptr_ptr + k + 1)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        off = qb
        while off < qe:
            q = off + lanes
            qm = q < qe
            ent = tl.load(fentry_ptr + q, mask=qm, other=0)
            col = tl.load(findex_ptr + q, mask=qm, other=0)
            acc = acc + tl.where(
                qm,
                tl.load(factor_ptr + ent, mask=qm, other=0.0)
                * tl.load(y_ptr + col, mask=qm, other=0.0),
                0.0,
            )
            off += BLOCK_SIZE
        val = tl.load(c_ptr + tl.load(order_ptr + k)) - tl.sum(acc, axis=0)
        if TRANS:
            val = tl.div_rn(val, tl.load(factor_ptr + tl.load(diag_ptr + k)))
        tl.store(y_ptr + k + lanes * 0, val + lanes * 0.0, mask=lane0)
        tl.debug_barrier()
        t += 1


@triton.jit
def factor_solve_fwd_light_kernel(
    split_start,
    round_end,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    fptr_ptr,
    fentry_ptr,
    findex_ptr,
    factor_ptr,
    c_ptr,
    y_ptr,
    TRANS: tl.constexpr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = split_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = p < round_end
    t = tl.load(path_off_ptr + p, mask=valid, other=0)
    t_end = tl.load(path_off_ptr + p + 1, mask=valid, other=0)
    t_act = valid & (t < t_end)
    if not ALL_ACTIVE:
        k0 = tl.load(path_unit_ptr + t, mask=t_act, other=0)
        n0 = tl.load(order_ptr + k0, mask=t_act, other=0)
        comp = tl.load(u_comp_ptr + n0, mask=t_act, other=-1)
        has_c = t_act & (comp >= 0)
        t_act = has_c & (
            tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE, mask=has_c, other=0)
            != 0
        )
        valid = t_act
    while tl.max(t_act.to(tl.int32), axis=0) != 0:
        k = tl.load(path_unit_ptr + t, mask=t_act, other=0)
        ord_k = tl.load(order_ptr + k, mask=t_act, other=0)
        val = tl.load(c_ptr + ord_k, mask=t_act, other=0.0)
        q = tl.load(fptr_ptr + k, mask=t_act, other=0)
        q_end = tl.load(fptr_ptr + k + 1, mask=t_act, other=0)
        q_act = t_act & (q < q_end)
        while tl.max(q_act.to(tl.int32), axis=0) != 0:
            ent = tl.load(fentry_ptr + q, mask=q_act, other=0)
            col = tl.load(findex_ptr + q, mask=q_act, other=0)
            val = tl.where(
                q_act,
                val
                - tl.load(factor_ptr + ent, mask=q_act, other=0.0)
                * tl.load(y_ptr + col, mask=q_act, other=0.0),
                val,
            )
            q += 1
            q_act = t_act & (q < q_end)
        if TRANS:
            dk = tl.load(diag_ptr + k, mask=t_act, other=0)
            piv = tl.load(factor_ptr + dk, mask=t_act, other=1.0)
            val = tl.div_rn(val, tl.where(t_act, piv, 1.0))
        tl.store(y_ptr + k, val, mask=t_act)
        t += 1
        t_act = valid & (t < t_end)


@triton.jit
def factor_solve_bwd_heavy_kernel(
    round_start,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    bptr_ptr,
    bentry_ptr,
    bindex_ptr,
    factor_ptr,
    scaling_ptr,
    y_ptr,
    x_ptr,
    out_ptr,
    TRANS: tl.constexpr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = round_start + tl.program_id(0)
    t_beg = tl.load(path_off_ptr + p)
    t_end = tl.load(path_off_ptr + p + 1)
    if not ALL_ACTIVE:
        if t_beg < t_end:
            first_node = tl.load(order_ptr + tl.load(path_unit_ptr + t_beg))
            comp = tl.load(u_comp_ptr + first_node)
            if comp < 0:
                return
            if tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE) == 0:
                return
    lanes = tl.arange(0, BLOCK_SIZE)
    lane0 = lanes == 0
    t = t_end - 1
    while t >= t_beg:
        k = tl.load(path_unit_ptr + t)
        qb = tl.load(bptr_ptr + k)
        qe = tl.load(bptr_ptr + k + 1)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        off = qb
        while off < qe:
            q = off + lanes
            qm = q < qe
            col = tl.load(bindex_ptr + q, mask=qm, other=-1)
            use = qm & (col != k)
            ent = tl.load(bentry_ptr + q, mask=use, other=0)
            acc = acc + tl.where(
                use,
                tl.load(factor_ptr + ent, mask=use, other=0.0)
                * tl.load(x_ptr + col, mask=use, other=0.0),
                0.0,
            )
            off += BLOCK_SIZE
        val = tl.load(y_ptr + k) - tl.sum(acc, axis=0)
        if not TRANS:
            val = tl.div_rn(val, tl.load(factor_ptr + tl.load(diag_ptr + k)))
        ord_k = tl.load(order_ptr + k)
        tl.store(x_ptr + k + lanes * 0, val + lanes * 0.0, mask=lane0)
        if TRANS:
            ov = val * tl.load(scaling_ptr + ord_k)
        else:
            ov = val
        tl.store(out_ptr + ord_k + lanes * 0, ov + lanes * 0.0, mask=lane0)
        tl.debug_barrier()
        t -= 1


@triton.jit
def factor_solve_bwd_light_kernel(
    split_start,
    round_end,
    path_unit_ptr,
    path_off_ptr,
    order_ptr,
    u_comp_ptr,
    ti_ptr,
    diag_ptr,
    bptr_ptr,
    bentry_ptr,
    bindex_ptr,
    factor_ptr,
    scaling_ptr,
    y_ptr,
    x_ptr,
    out_ptr,
    TRANS: tl.constexpr,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    p = split_start + tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = p < round_end
    t_beg = tl.load(path_off_ptr + p, mask=valid, other=0)
    t_end = tl.load(path_off_ptr + p + 1, mask=valid, other=0)
    if not ALL_ACTIVE:
        has_p = valid & (t_beg < t_end)
        k0 = tl.load(path_unit_ptr + t_beg, mask=has_p, other=0)
        n0 = tl.load(order_ptr + k0, mask=has_p, other=0)
        comp = tl.load(u_comp_ptr + n0, mask=has_p, other=-1)
        has_c = has_p & (comp >= 0)
        valid = has_c & (
            tl.load(ti_ptr + comp.to(tl.int64) * TI + IDX_SOLVE, mask=has_c, other=0)
            != 0
        )
    t = t_end - 1
    t_act = valid & (t >= t_beg)
    while tl.max(t_act.to(tl.int32), axis=0) != 0:
        k = tl.load(path_unit_ptr + t, mask=t_act, other=0)
        val = tl.load(y_ptr + k, mask=t_act, other=0.0)
        q = tl.load(bptr_ptr + k, mask=t_act, other=0)
        q_end = tl.load(bptr_ptr + k + 1, mask=t_act, other=0)
        q_act = t_act & (q < q_end)
        while tl.max(q_act.to(tl.int32), axis=0) != 0:
            col = tl.load(bindex_ptr + q, mask=q_act, other=-1)
            use = q_act & (col != k)
            ent = tl.load(bentry_ptr + q, mask=use, other=0)
            val = tl.where(
                use,
                val
                - tl.load(factor_ptr + ent, mask=use, other=0.0)
                * tl.load(x_ptr + col, mask=use, other=0.0),
                val,
            )
            q += 1
            q_act = t_act & (q < q_end)
        if not TRANS:
            dk = tl.load(diag_ptr + k, mask=t_act, other=0)
            piv = tl.load(factor_ptr + dk, mask=t_act, other=1.0)
            val = tl.div_rn(val, tl.where(t_act, piv, 1.0))
        ord_k = tl.load(order_ptr + k, mask=t_act, other=0)
        tl.store(x_ptr + k, val, mask=t_act)
        if TRANS:
            ov = val * tl.load(scaling_ptr + ord_k, mask=t_act, other=0.0)
        else:
            ov = val
        tl.store(out_ptr + ord_k, ov, mask=t_act)
        t -= 1
        t_act = valid & (t >= t_beg)


@triton.jit
def coupled_matvec_kernel(
    dt,
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    tf_ptr,
    x_ptr,
    sol_low_ptr,
    out_ptr,
    bound_ptr,
    scaling_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    down_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    bsrc_ptr,
    bdst_ptr,
    path_sum_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    COMPENSATED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if ALL_ACTIVE:
        act_row = m
    else:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        act_row = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    if tl.max(act_row.to(tl.int32), axis=0) == 0:
        return

    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bound = tl.load(bound_ptr + i, mask=act_row, other=0.0) > 0.5
    xhi = tl.load(x_ptr + i, mask=act_row, other=0.0)
    xlo = tl.load(sol_low_ptr + i, mask=act_row, other=0.0) if COMPENSATED else zf
    ai = tl.load(area_ptr + i, mask=act_row, other=0.0)
    sh, sl = graph_scale_inline(xhi, xlo, ai)
    vh = tl.where(bound, xhi, sh)
    vl = tl.where(bound, xlo, sl)

    act = act_row & (~bound)
    damp = graph_damping_inline(
        i,
        act,
        dt,
        u_comp_ptr,
        tf_ptr,
        area_ptr,
        qi_ptr,
        qd_ptr,
        cp_ptr,
        kids_ptr,
        boff_ptr,
        blink_ptr,
        path_sum_ptr,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    has_damp = act & (damp > 0.0)
    if tl.max(has_damp.to(tl.int32), axis=0) != 0:
        dmh, dml = graph_scale_inline(xhi, xlo, damp)
        nvh, nvl = pair_add_inline(vh, vl, dmh, dml)
        vh = tl.where(has_damp, nvh, vh)
        vl = tl.where(has_damp, nvl, vl)

    fh, fl = graph_scale_inline(xhi, xlo, tl.load(qi_ptr + i, mask=act, other=0.0))
    ed = tl.load(edge_ptr + i, mask=act, other=False)
    d = tl.where(ed, tl.load(down_ptr + i, mask=act & ed, other=0), 0)
    d_m = act & ed
    xd_h = tl.load(x_ptr + d, mask=d_m, other=0.0)
    xd_l = tl.load(sol_low_ptr + d, mask=d_m, other=0.0) if COMPENSATED else zf
    dfh, dfl = graph_scale_inline(xd_h, xd_l, tl.load(qd_ptr + i, mask=d_m, other=0.0))
    nfh, nfl = pair_add_inline(fh, fl, dfh, dfl)
    fh = tl.where(d_m, nfh, fh)
    fl = tl.where(d_m, nfl, fl)
    t2h, t2l = graph_scale_inline(fh, fl, dt)
    nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
    vh = tl.where(act, nvh, vh)
    vl = tl.where(act, nvl, vl)

    e = tl.load(cp_ptr + i, mask=act, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
    e_act = act & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        xj_h = tl.load(x_ptr + j, mask=e_act, other=0.0)
        xj_l = tl.load(sol_low_ptr + j, mask=e_act, other=0.0) if COMPENSATED else zf
        p1h, p1l = graph_scale_inline(
            xj_h, xj_l, tl.load(qi_ptr + j, mask=e_act, other=0.0)
        )
        p2h, p2l = graph_scale_inline(
            xhi, xlo, tl.load(qd_ptr + j, mask=e_act, other=0.0)
        )
        fh_, fl_ = pair_add_inline(p1h, p1l, p2h, p2l)
        t2h, t2l = graph_scale_inline(fh_, fl_, -dt)
        nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
        vh = tl.where(e_act, nvh, vh)
        vl = tl.where(e_act, nvl, vl)
        e += 1
        e_act = act & (e < e_end)

    be = tl.load(boff_ptr + i, mask=act, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=act, other=0)
    be_act = act & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        u = tl.load(bsrc_ptr + p, mask=be_act, other=0)
        d_ = tl.load(bdst_ptr + p, mask=be_act, other=0)
        xu_h = tl.load(x_ptr + u, mask=be_act, other=0.0)
        xu_l = tl.load(sol_low_ptr + u, mask=be_act, other=0.0) if COMPENSATED else zf
        xd_h = tl.load(x_ptr + d_, mask=be_act, other=0.0)
        xd_l = tl.load(sol_low_ptr + d_, mask=be_act, other=0.0) if COMPENSATED else zf
        bp = p.to(tl.int64) * 3
        p1h, p1l = graph_scale_inline(
            xu_h, xu_l, tl.load(path_sum_ptr + bp + 1, mask=be_act, other=0.0)
        )
        p2h, p2l = graph_scale_inline(
            xd_h, xd_l, tl.load(path_sum_ptr + bp + 2, mask=be_act, other=0.0)
        )
        fh_, fl_ = pair_add_inline(p1h, p1l, p2h, p2l)
        t2h, t2l = graph_scale_inline(fh_, fl_, tl.where(lnk > 0, dt, -dt))
        nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
        vh = tl.where(be_act, nvh, vh)
        vl = tl.where(be_act, nvl, vl)
        be += 1
        be_act = act & (be < be_end)

    sc = tl.load(scaling_ptr + i, mask=act_row, other=0.0)
    tl.store(out_ptr + i, (vh + vl) * sc, mask=act_row)


@triton.jit
def graph_matvec_kernel(
    dt,
    x_ptr,
    sol_low_ptr,
    out_ptr,
    bound_ptr,
    scaling_ptr,
    area_ptr,
    qi_ptr,
    qd_ptr,
    down_ptr,
    edge_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    bsrc_ptr,
    bdst_ptr,
    path_sum_ptr,
    n,
    TRANS: tl.constexpr,
    COMPENSATED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    bound = tl.load(bound_ptr + i, mask=m, other=0.0) > 0.5
    xhi = tl.load(x_ptr + i, mask=m, other=0.0)
    xlo = tl.load(sol_low_ptr + i, mask=m, other=0.0) if COMPENSATED else zf
    mi_h = tl.where(bound, zf, xhi)
    mi_l = tl.where(bound, zf, xlo)
    ai = tl.load(area_ptr + i, mask=m, other=0.0)
    sh, sl = graph_scale_inline(xhi, xlo, ai)
    vh = tl.where(bound, xhi, sh)
    vl = tl.where(bound, xlo, sl)

    if TRANS:
        ed = tl.load(edge_ptr + i, mask=m, other=False)
        d = tl.where(ed, tl.load(down_ptr + i, mask=m & ed, other=0), 0)
        d_ok = m & ed & (tl.load(bound_ptr + d, mask=m & ed, other=1.0) < 0.5)
        md_h = tl.where(d_ok, tl.load(x_ptr + d, mask=d_ok, other=0.0), zf)
        md_l = (
            tl.where(d_ok, tl.load(sol_low_ptr + d, mask=d_ok, other=0.0), zf)
            if COMPENSATED
            else zf
        )
        dh_, dl_ = pair_add_inline(mi_h, mi_l, -md_h, -md_l)
        t1h, t1l = graph_scale_inline(dh_, dl_, tl.load(qi_ptr + i, mask=m, other=0.0))
        t2h, t2l = graph_scale_inline(t1h, t1l, dt)
        vh, vl = pair_add_inline(vh, vl, t2h, t2l)

        e = tl.load(cp_ptr + i, mask=m, other=0)
        e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
        e_act = m & (e < e_end)
        while tl.max(e_act.to(tl.int32), axis=0) != 0:
            j = tl.load(kids_ptr + e, mask=e_act, other=0)
            jb = tl.load(bound_ptr + j, mask=e_act, other=0.0) > 0.5
            mj_h = tl.where(
                e_act & (~jb), tl.load(x_ptr + j, mask=e_act & (~jb), other=0.0), zf
            )
            mj_l = (
                tl.where(
                    e_act & (~jb),
                    tl.load(sol_low_ptr + j, mask=e_act & (~jb), other=0.0),
                    zf,
                )
                if COMPENSATED
                else zf
            )
            dh_, dl_ = pair_add_inline(mj_h, mj_l, -mi_h, -mi_l)
            t1h, t1l = graph_scale_inline(
                dh_, dl_, tl.load(qd_ptr + j, mask=e_act, other=0.0)
            )
            t2h, t2l = graph_scale_inline(t1h, t1l, dt)
            nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
            vh = tl.where(e_act, nvh, vh)
            vl = tl.where(e_act, nvl, vl)
            e += 1
            e_act = m & (e < e_end)

        be = tl.load(boff_ptr + i, mask=m, other=0)
        be_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
        be_act = m & (be < be_end)
        while tl.max(be_act.to(tl.int32), axis=0) != 0:
            lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
            p = tl.abs(lnk) - 1
            u = tl.load(bsrc_ptr + p, mask=be_act, other=0)
            d_ = tl.load(bdst_ptr + p, mask=be_act, other=0)
            ub = tl.load(bound_ptr + u, mask=be_act, other=0.0) > 0.5
            db = tl.load(bound_ptr + d_, mask=be_act, other=0.0) > 0.5
            mu_h = tl.where(
                be_act & (~ub), tl.load(x_ptr + u, mask=be_act & (~ub), other=0.0), zf
            )
            mu_l = (
                tl.where(
                    be_act & (~ub),
                    tl.load(sol_low_ptr + u, mask=be_act & (~ub), other=0.0),
                    zf,
                )
                if COMPENSATED
                else zf
            )
            md_h = tl.where(
                be_act & (~db), tl.load(x_ptr + d_, mask=be_act & (~db), other=0.0), zf
            )
            md_l = (
                tl.where(
                    be_act & (~db),
                    tl.load(sol_low_ptr + d_, mask=be_act & (~db), other=0.0),
                    zf,
                )
                if COMPENSATED
                else zf
            )
            dh_, dl_ = pair_add_inline(mu_h, mu_l, -md_h, -md_l)
            bp = p.to(tl.int64) * 3
            coef = tl.load(
                path_sum_ptr + bp + tl.where(lnk > 0, 1, 2), mask=be_act, other=0.0
            )
            t1h, t1l = graph_scale_inline(dh_, dl_, coef)
            t2h, t2l = graph_scale_inline(t1h, t1l, dt)
            nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
            vh = tl.where(be_act, nvh, vh)
            vl = tl.where(be_act, nvl, vl)
            be += 1
            be_act = m & (be < be_end)
    else:
        act = m & (~bound)
        fh, fl = graph_scale_inline(xhi, xlo, tl.load(qi_ptr + i, mask=act, other=0.0))
        ed = tl.load(edge_ptr + i, mask=act, other=False)
        d = tl.where(ed, tl.load(down_ptr + i, mask=act & ed, other=0), 0)
        d_m = act & ed
        xd_h = tl.load(x_ptr + d, mask=d_m, other=0.0)
        xd_l = tl.load(sol_low_ptr + d, mask=d_m, other=0.0) if COMPENSATED else zf
        dfh, dfl = graph_scale_inline(
            xd_h, xd_l, tl.load(qd_ptr + i, mask=d_m, other=0.0)
        )
        nfh, nfl = pair_add_inline(fh, fl, dfh, dfl)
        fh = tl.where(d_m, nfh, fh)
        fl = tl.where(d_m, nfl, fl)
        t2h, t2l = graph_scale_inline(fh, fl, dt)
        nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
        vh = tl.where(act, nvh, vh)
        vl = tl.where(act, nvl, vl)

        e = tl.load(cp_ptr + i, mask=act, other=0)
        e_end = tl.load(cp_ptr + i + 1, mask=act, other=0)
        e_act = act & (e < e_end)
        while tl.max(e_act.to(tl.int32), axis=0) != 0:
            j = tl.load(kids_ptr + e, mask=e_act, other=0)
            xj_h = tl.load(x_ptr + j, mask=e_act, other=0.0)
            xj_l = (
                tl.load(sol_low_ptr + j, mask=e_act, other=0.0) if COMPENSATED else zf
            )
            p1h, p1l = graph_scale_inline(
                xj_h, xj_l, tl.load(qi_ptr + j, mask=e_act, other=0.0)
            )
            p2h, p2l = graph_scale_inline(
                xhi, xlo, tl.load(qd_ptr + j, mask=e_act, other=0.0)
            )
            fh_, fl_ = pair_add_inline(p1h, p1l, p2h, p2l)
            t2h, t2l = graph_scale_inline(fh_, fl_, -dt)
            nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
            vh = tl.where(e_act, nvh, vh)
            vl = tl.where(e_act, nvl, vl)
            e += 1
            e_act = act & (e < e_end)

        be = tl.load(boff_ptr + i, mask=act, other=0)
        be_end = tl.load(boff_ptr + i + 1, mask=act, other=0)
        be_act = act & (be < be_end)
        while tl.max(be_act.to(tl.int32), axis=0) != 0:
            lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
            p = tl.abs(lnk) - 1
            u = tl.load(bsrc_ptr + p, mask=be_act, other=0)
            d_ = tl.load(bdst_ptr + p, mask=be_act, other=0)
            xu_h = tl.load(x_ptr + u, mask=be_act, other=0.0)
            xu_l = (
                tl.load(sol_low_ptr + u, mask=be_act, other=0.0) if COMPENSATED else zf
            )
            xd_h = tl.load(x_ptr + d_, mask=be_act, other=0.0)
            xd_l = (
                tl.load(sol_low_ptr + d_, mask=be_act, other=0.0) if COMPENSATED else zf
            )
            bp = p.to(tl.int64) * 3
            p1h, p1l = graph_scale_inline(
                xu_h, xu_l, tl.load(path_sum_ptr + bp + 1, mask=be_act, other=0.0)
            )
            p2h, p2l = graph_scale_inline(
                xd_h, xd_l, tl.load(path_sum_ptr + bp + 2, mask=be_act, other=0.0)
            )
            fh_, fl_ = pair_add_inline(p1h, p1l, p2h, p2l)
            t2h, t2l = graph_scale_inline(fh_, fl_, tl.where(lnk > 0, dt, -dt))
            nvh, nvl = pair_add_inline(vh, vl, t2h, t2l)
            vh = tl.where(be_act, nvh, vh)
            vl = tl.where(be_act, nvl, vl)
            be += 1
            be_act = act & (be < be_end)

    sc = tl.load(scaling_ptr + i, mask=m, other=0.0)
    tl.store(out_ptr + i, (vh + vl) * sc, mask=m)


# Indexed coupled BiCGStab & reduction kernels (active g_unit rows only)
@triton.jit
def coupled_copy_dh_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    delta_ptr,
    dh_ptr,
    sol_low_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    tl.store(dh_ptr + i, tl.load(delta_ptr + i, mask=m, other=0.0), mask=m)
    tl.store(sol_low_ptr + i, tl.zeros((BLOCK_SIZE,), dtype=tl.float32), mask=m)


@triton.jit
def coupled_norm_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    a_ptr,
    norm_i32_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    v = tl.abs(tl.load(a_ptr + i, mask=m, other=0.0))
    inf_val = tl.full((), float("inf"), tl.float32)
    v = tl.where(isfinite_f32_inline(v), v, inf_val)
    bm = tl.max(tl.where(m, v, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def coupled_dot_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    a_ptr,
    b_ptr,
    red_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    s = tl.sum(
        tl.where(
            m,
            tl.load(a_ptr + i, mask=m, other=0.0)
            * tl.load(b_ptr + i, mask=m, other=0.0),
            0.0,
        ),
        axis=0,
    )
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.store(red_ptr + pid + lanes * 0, s + lanes * 0.0, mask=lanes == 0)


@triton.jit
def coupled_dot_pair_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    t_ptr,
    s_ptr,
    red_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    tv = tl.load(t_ptr + i, mask=m, other=0.0)
    sv = tl.load(s_ptr + i, mask=m, other=0.0)
    tt = tl.sum(tl.where(m, tv * tv, 0.0), axis=0)
    ts = tl.sum(tl.where(m, tv * sv, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.store(red_ptr + 2 * pid + lanes * 0, tt + lanes * 0.0, mask=lanes == 0)
    tl.store(red_ptr + 2 * pid + 1 + lanes * 0, ts + lanes * 0.0, mask=lanes == 0)


@triton.jit
def coupled_bicg_init_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    tmp_ptr,
    initial,
    r_ptr,
    rhat_ptr,
    p_ptr,
    v_ptr,
    delta_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    r = tl.div_rn(tl.load(tmp_ptr + i, mask=m, other=0.0), initial)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(r_ptr + i, r, mask=m)
    tl.store(rhat_ptr + i, r, mask=m)
    tl.store(p_ptr + i, zf, mask=m)
    tl.store(v_ptr + i, zf, mask=m)
    tl.store(delta_ptr + i, zf, mask=m)


@triton.jit
def coupled_bicg_upd_p_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    r_ptr,
    p_ptr,
    v_ptr,
    beta,
    omega,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    val = tl.load(r_ptr + i, mask=m, other=0.0) + beta * (
        tl.load(p_ptr + i, mask=m, other=0.0)
        - omega * tl.load(v_ptr + i, mask=m, other=0.0)
    )
    tl.store(p_ptr + i, val, mask=m)


@triton.jit
def coupled_bicg_upd_s_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    r_ptr,
    v_ptr,
    tmp_ptr,
    s_ptr,
    delta_ptr,
    alpha,
    norm_i32_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    sv = tl.load(r_ptr + i, mask=m, other=0.0) - alpha * tl.load(
        v_ptr + i, mask=m, other=0.0
    )
    dv = tl.load(delta_ptr + i, mask=m, other=0.0) + alpha * tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(s_ptr + i, sv, mask=m)
    tl.store(delta_ptr + i, dv, mask=m)
    av = tl.abs(sv)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def coupled_bicg_upd_r_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    s_ptr,
    t_ptr,
    tmp_ptr,
    r_ptr,
    delta_ptr,
    omega,
    norm_i32_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    rv = tl.load(s_ptr + i, mask=m, other=0.0) - omega * tl.load(
        t_ptr + i, mask=m, other=0.0
    )
    dv = tl.load(delta_ptr + i, mask=m, other=0.0) + omega * tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(r_ptr + i, rv, mask=m)
    tl.store(delta_ptr + i, dv, mask=m)
    av = tl.abs(rv)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def coupled_bicg_accum_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    dh_ptr,
    sol_low_ptr,
    delta_ptr,
    initial,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    ph, pl = graph_product_inline(initial, tl.load(delta_ptr + i, mask=m, other=0.0))
    xh, xl = pair_add_inline(
        tl.load(dh_ptr + i, mask=m, other=0.0),
        tl.load(sol_low_ptr + i, mask=m, other=0.0),
        ph,
        pl,
    )
    tl.store(dh_ptr + i, xh, mask=m)
    tl.store(sol_low_ptr + i, xl, mask=m)


@triton.jit
def coupled_res_diff_kernel(
    gunit_ptr,
    u_comp_ptr,
    ti_ptr,
    rhs_ptr,
    tmp_ptr,
    norm_i32_ptr,
    ngu,
    ALL_ACTIVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = idx < ngu
    i = tl.load(gunit_ptr + idx, mask=m, other=0)
    if not ALL_ACTIVE:
        c = tl.load(u_comp_ptr + i, mask=m, other=0)
        m = m & (
            tl.load(ti_ptr + c.to(tl.int64) * TI + IDX_SOLVE, mask=m, other=0) != 0
        )
    diff = tl.load(rhs_ptr + i, mask=m, other=0.0) - tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(tmp_ptr + i, diff, mask=m)
    av = tl.abs(diff)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


# Full-domain vector helpers (used by transpose adjoint)
@triton.jit
def vec_norm_kernel(a_ptr, norm_i32_ptr, n, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    v = tl.abs(tl.load(a_ptr + i, mask=m, other=0.0))
    inf_val = tl.full((), float("inf"), tl.float32)
    v = tl.where(isfinite_f32_inline(v), v, inf_val)
    bm = tl.max(tl.where(m, v, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def vec_dot_kernel(a_ptr, b_ptr, red_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    s = tl.sum(
        tl.where(
            m,
            tl.load(a_ptr + i, mask=m, other=0.0)
            * tl.load(b_ptr + i, mask=m, other=0.0),
            0.0,
        ),
        axis=0,
    )
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.store(red_ptr + pid + lanes * 0, s + lanes * 0.0, mask=lanes == 0)


@triton.jit
def vec_dot_pair_kernel(t_ptr, s_ptr, red_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    tv = tl.load(t_ptr + i, mask=m, other=0.0)
    sv = tl.load(s_ptr + i, mask=m, other=0.0)
    tt = tl.sum(tl.where(m, tv * tv, 0.0), axis=0)
    ts = tl.sum(tl.where(m, tv * sv, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.store(red_ptr + 2 * pid + lanes * 0, tt + lanes * 0.0, mask=lanes == 0)
    tl.store(red_ptr + 2 * pid + 1 + lanes * 0, ts + lanes * 0.0, mask=lanes == 0)


@triton.jit
def bicg_init_restart_kernel(
    tmp_ptr,
    initial,
    r_ptr,
    rhat_ptr,
    p_ptr,
    v_ptr,
    delta_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    r = tl.div_rn(tl.load(tmp_ptr + i, mask=m, other=0.0), initial)
    zf = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(r_ptr + i, r, mask=m)
    tl.store(rhat_ptr + i, r, mask=m)
    tl.store(p_ptr + i, zf, mask=m)
    tl.store(v_ptr + i, zf, mask=m)
    tl.store(delta_ptr + i, zf, mask=m)


@triton.jit
def bicg_update_p_kernel(r_ptr, p_ptr, v_ptr, beta, omega, n, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    val = tl.load(r_ptr + i, mask=m, other=0.0) + beta * (
        tl.load(p_ptr + i, mask=m, other=0.0)
        - omega * tl.load(v_ptr + i, mask=m, other=0.0)
    )
    tl.store(p_ptr + i, val, mask=m)


@triton.jit
def bicg_update_s_delta_kernel(
    r_ptr,
    v_ptr,
    tmp_ptr,
    s_ptr,
    delta_ptr,
    alpha,
    norm_i32_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    sv = tl.load(r_ptr + i, mask=m, other=0.0) - alpha * tl.load(
        v_ptr + i, mask=m, other=0.0
    )
    dv = tl.load(delta_ptr + i, mask=m, other=0.0) + alpha * tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(s_ptr + i, sv, mask=m)
    tl.store(delta_ptr + i, dv, mask=m)
    av = tl.abs(sv)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def bicg_update_r_delta_kernel(
    s_ptr,
    t_ptr,
    tmp_ptr,
    r_ptr,
    delta_ptr,
    omega,
    norm_i32_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    rv = tl.load(s_ptr + i, mask=m, other=0.0) - omega * tl.load(
        t_ptr + i, mask=m, other=0.0
    )
    dv = tl.load(delta_ptr + i, mask=m, other=0.0) + omega * tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(r_ptr + i, rv, mask=m)
    tl.store(delta_ptr + i, dv, mask=m)
    av = tl.abs(rv)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def bicg_accum_solution_kernel(
    dh_ptr, sol_low_ptr, delta_ptr, initial, n, BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    ph, pl = graph_product_inline(initial, tl.load(delta_ptr + i, mask=m, other=0.0))
    xh, xl = pair_add_inline(
        tl.load(dh_ptr + i, mask=m, other=0.0),
        tl.load(sol_low_ptr + i, mask=m, other=0.0),
        ph,
        pl,
    )
    tl.store(dh_ptr + i, xh, mask=m)
    tl.store(sol_low_ptr + i, xl, mask=m)


@triton.jit
def bicg_residual_diff_kernel(
    rhs_ptr, tmp_ptr, norm_i32_ptr, n, BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    diff = tl.load(rhs_ptr + i, mask=m, other=0.0) - tl.load(
        tmp_ptr + i, mask=m, other=0.0
    )
    tl.store(tmp_ptr + i, diff, mask=m)
    av = tl.abs(diff)
    inf_val = tl.full((), float("inf"), tl.float32)
    av = tl.where(isfinite_f32_inline(av), av, inf_val)
    bm = tl.max(tl.where(m, av, 0.0), axis=0)
    lanes = tl.arange(0, BLOCK_SIZE)
    tl.atomic_max(
        norm_i32_ptr + lanes * 0,
        tl.cast(tl.abs(bm), tl.int32, bitcast=True) + lanes * 0,
        mask=lanes == 0,
        sem="relaxed",
        scope="gpu",
    )


@triton.jit
def scale_div_kernel(in_ptr, scaling_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    tl.store(
        out_ptr + i,
        tl.div_rn(
            tl.load(in_ptr + i, mask=m, other=0.0),
            tl.load(scaling_ptr + i, mask=m, other=1.0),
        ),
        mask=m,
    )


@triton.jit
def report_single_error_kernel(status_ptr, code, info):
    z = tl.arange(0, 1)
    report_error(status_ptr, z == 0, code + z * 0, info + z * 0)


# --------------------------------------------------------------------------- #
# Graph commit, rebuild and adjoint kernels
# --------------------------------------------------------------------------- #


@triton.jit
def graph_commit_paths_kernel(path_val_ptr, qb_ptr, n_vals, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n_vals
    tl.store(
        qb_ptr + i,
        tl.load(path_val_ptr + i.to(tl.int64) * 7, mask=m, other=0.0),
        mask=m,
    )


@triton.jit
def graph_commit_inflow_kernel(
    boff_ptr, blink_ptr, path_val_ptr, inflow_ptr, n, blevels, BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    received = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    e = tl.load(boff_ptr + i, mask=m, other=0)
    e_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + e, mask=e_act, other=0)
        p = tl.abs(lnk) - 1
        for k in range(blevels):
            q = tl.load(
                path_val_ptr + (p.to(tl.int64) * blevels + k) * 7, mask=e_act, other=0.0
            )
            received = tl.where(
                e_act, received + tl.maximum(tl.where(lnk > 0, -q, q), 0.0), received
            )
        e += 1
        e_act = m & (e < e_end)
    tl.store(
        inflow_ptr + i, tl.load(inflow_ptr + i, mask=m, other=0.0) + received, mask=m
    )


@triton.jit
def graph_rebuild_stage_kernel(
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
def graph_rebuild_snapshot_kernel(
    dt,
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


@triton.jit
def graph_adjoint_prep_cells_kernel(
    saved_ptr,
    bsaved_ptr,
    gQ_ptr,
    gS_ptr,
    gh_ptr,
    gQB_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    qi_ptr,
    qd_ptr,
    area_ptr,
    bound_ptr,
    rhs_ptr,
    n,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    bi = i.to(tl.int64) * 17
    j2 = tl.load(saved_ptr + bi + 2, mask=m, other=0.0)
    j3 = tl.load(saved_ptr + bi + 3, mask=m, other=0.0)
    j4 = tl.load(saved_ptr + bi + 4, mask=m, other=0.0)
    j5 = tl.load(saved_ptr + bi + 5, mask=m, other=0.0)
    tl.store(qi_ptr + i, j2, mask=m)
    tl.store(qd_ptr + i, j3, mask=m)
    tl.store(area_ptr + i, j4, mask=m)
    tl.store(bound_ptr + i, j5, mask=m)

    p1h, p1l = graph_product_inline(j4, tl.load(gS_ptr + i, mask=m, other=0.0))
    p2h, p2l = graph_product_inline(j2, tl.load(gQ_ptr + i, mask=m, other=0.0))
    vh, vl = pair_add_inline(p1h, p1l, p2h, p2l)
    vh, vl = pair_add_inline(
        vh,
        vl,
        tl.load(gh_ptr + i, mask=m, other=0.0),
        tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
    )

    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        ph, pl = graph_product_inline(
            tl.load(saved_ptr + j.to(tl.int64) * 17 + 3, mask=e_act, other=0.0),
            tl.load(gQ_ptr + j, mask=e_act, other=0.0),
        )
        nvh, nvl = pair_add_inline(vh, vl, ph, pl)
        vh = tl.where(e_act, nvh, vh)
        vl = tl.where(e_act, nvl, vl)
        e += 1
        e_act = m & (e < e_end)

    be = tl.load(boff_ptr + i, mask=m, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
    be_act = m & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        col_off = tl.where(lnk > 0, 1, 2)
        for lvl in range(blevels):
            k = p * blevels + lvl
            ph, pl = graph_product_inline(
                tl.load(
                    bsaved_ptr + k.to(tl.int64) * 7 + col_off, mask=be_act, other=0.0
                ),
                tl.load(gQB_ptr + k, mask=be_act, other=0.0),
            )
            nvh, nvl = pair_add_inline(vh, vl, ph, pl)
            vh = tl.where(be_act, nvh, vh)
            vl = tl.where(be_act, nvl, vl)
        be += 1
        be_act = m & (be < be_end)

    tl.store(rhs_ptr + i, vh + vl, mask=m)


@triton.jit
def graph_adjoint_prep_paths_kernel(
    bsaved_ptr, path_sum_ptr, m_paths, blevels, BLOCK_SIZE: tl.constexpr
):
    p = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = p < m_paths
    a = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    b = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for lvl in range(blevels):
        k = p * blevels + lvl
        bk = k.to(tl.int64) * 7
        a = a + tl.load(bsaved_ptr + bk + 1, mask=m, other=0.0)
        b = b + tl.load(bsaved_ptr + bk + 2, mask=m, other=0.0)
    bp = p.to(tl.int64) * 3
    tl.store(path_sum_ptr + bp + 1, a, mask=m)
    tl.store(path_sum_ptr + bp + 2, b, mask=m)


@triton.jit
def graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, idx, mask):
    bnd = tl.load(saved_ptr + idx.to(tl.int64) * 17 + 5, mask=mask, other=0.0) > 0.5
    zf = idx.to(tl.float32) * 0.0
    lh = tl.where(
        mask & (~bnd), tl.load(dh_ptr + idx, mask=mask & (~bnd), other=0.0), zf
    )
    ll = tl.where(
        mask & (~bnd), tl.load(sol_low_ptr + idx, mask=mask & (~bnd), other=0.0), zf
    )
    return lh, ll


@triton.jit
def graph_flux_adj_inline(uh, ul, dh_, dl_, obj, dt):
    diff_h, diff_l = pair_add_inline(uh, ul, -dh_, -dl_)
    sh, sl = graph_scale_inline(diff_h, diff_l, -dt)
    vh, vl = pair_add_inline(obj, tl.zeros_like(obj), sh, sl)
    return vh + vl


@triton.jit
def graph_adjoint_step1_kernel(
    dt,
    saved_ptr,
    dh_ptr,
    sol_low_ptr,
    edge_ptr,
    down_ptr,
    gQ_ptr,
    gS_ptr,
    tmp_ptr,
    gr_ptr,
    gw_ptr,
    gnc_ptr,
    gnf_ptr,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    lh, ll = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, i, m)
    ed = tl.load(edge_ptr + i, mask=m, other=False)
    d = tl.where(ed, tl.load(down_ptr + i, mask=m & ed, other=0), 0)
    dlh, dll = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, d, m & ed)
    gQi = tl.load(gQ_ptr + i, mask=m, other=0.0)
    ce = graph_flux_adj_inline(lh, ll, dlh, dll, gQi, dt)
    tl.store(tmp_ptr + i, ce, mask=m)
    rgh, rgl = graph_scale_inline(lh, ll, dt)
    tl.store(gr_ptr + i, rgh + rgl, mask=m)
    bi = i.to(tl.int64) * 17
    tl.store(gQ_ptr + i, ce * tl.load(saved_ptr + bi + 12, mask=m, other=0.0), mask=m)
    diff_h, diff_l = pair_add_inline(
        tl.load(gS_ptr + i, mask=m, other=0.0),
        tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
        -lh,
        -ll,
    )
    diff = diff_h + diff_l
    tl.store(
        gw_ptr + i,
        tl.load(gw_ptr + i, mask=m, other=0.0)
        + diff * tl.load(saved_ptr + bi + 7, mask=m, other=0.0)
        + ce * tl.load(saved_ptr + bi + 9, mask=m, other=0.0),
        mask=m,
    )
    tl.store(
        gnc_ptr + i,
        tl.load(gnc_ptr + i, mask=m, other=0.0)
        + ce * tl.load(saved_ptr + bi + 13, mask=m, other=0.0),
        mask=m,
    )
    tl.store(
        gnf_ptr + i,
        tl.load(gnf_ptr + i, mask=m, other=0.0)
        + ce * tl.load(saved_ptr + bi + 14, mask=m, other=0.0),
        mask=m,
    )


@triton.jit
def graph_adjoint_step2_kernel(
    dt,
    saved_ptr,
    bsaved_ptr,
    dh_ptr,
    sol_low_ptr,
    cp_ptr,
    kids_ptr,
    boff_ptr,
    blink_ptr,
    bsrc_ptr,
    bdst_ptr,
    tmp_ptr,
    gS_ptr,
    gQB_ptr,
    gheight_ptr,
    n,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = i < n
    lh, ll = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, i, m)
    diff_h, diff_l = pair_add_inline(
        tl.load(gS_ptr + i, mask=m, other=0.0),
        tl.zeros((BLOCK_SIZE,), dtype=tl.float32),
        -lh,
        -ll,
    )
    bi = i.to(tl.int64) * 17
    height = (diff_h + diff_l) * tl.load(
        saved_ptr + bi + 8, mask=m, other=0.0
    ) + tl.load(tmp_ptr + i, mask=m, other=0.0) * tl.load(
        saved_ptr + bi + 10, mask=m, other=0.0
    )
    bnd = tl.load(saved_ptr + bi + 5, mask=m, other=0.0) > 0.5
    height = tl.where(
        bnd,
        height
        - (
            tl.load(dh_ptr + i, mask=m, other=0.0)
            + tl.load(sol_low_ptr + i, mask=m, other=0.0)
        ),
        height,
    )

    e = tl.load(cp_ptr + i, mask=m, other=0)
    e_end = tl.load(cp_ptr + i + 1, mask=m, other=0)
    e_act = m & (e < e_end)
    while tl.max(e_act.to(tl.int32), axis=0) != 0:
        j = tl.load(kids_ptr + e, mask=e_act, other=0)
        height = tl.where(
            e_act,
            height
            + tl.load(tmp_ptr + j, mask=e_act, other=0.0)
            * tl.load(saved_ptr + j.to(tl.int64) * 17 + 11, mask=e_act, other=0.0),
            height,
        )
        e += 1
        e_act = m & (e < e_end)

    be = tl.load(boff_ptr + i, mask=m, other=0)
    be_end = tl.load(boff_ptr + i + 1, mask=m, other=0)
    be_act = m & (be < be_end)
    while tl.max(be_act.to(tl.int32), axis=0) != 0:
        lnk = tl.load(blink_ptr + be, mask=be_act, other=0)
        p = tl.abs(lnk) - 1
        u = tl.load(bsrc_ptr + p, mask=be_act, other=0)
        d_ = tl.load(bdst_ptr + p, mask=be_act, other=0)
        ulh, ull = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, u, be_act)
        dlh, dll = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, d_, be_act)
        col_off = tl.where(lnk > 0, 5, 6)
        for lvl in range(blevels):
            k = p * blevels + lvl
            ce = graph_flux_adj_inline(
                ulh, ull, dlh, dll, tl.load(gQB_ptr + k, mask=be_act, other=0.0), dt
            )
            height = tl.where(
                be_act,
                height
                + ce
                * tl.load(
                    bsaved_ptr + k.to(tl.int64) * 7 + col_off, mask=be_act, other=0.0
                ),
                height,
            )
        be += 1
        be_act = m & (be < be_end)

    tl.store(
        gheight_ptr + i, tl.load(gheight_ptr + i, mask=m, other=0.0) + height, mask=m
    )
    tl.store(gS_ptr + i, lh + ll, mask=m)


@triton.jit
def graph_adjoint_step3_kernel(
    dt,
    saved_ptr,
    bsaved_ptr,
    dh_ptr,
    sol_low_ptr,
    bsrc_ptr,
    bdst_ptr,
    gQB_ptr,
    gnb_ptr,
    m_paths,
    blevels,
    BLOCK_SIZE: tl.constexpr,
):
    p = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    m = p < m_paths
    u = tl.load(bsrc_ptr + p, mask=m, other=0)
    d_ = tl.load(bdst_ptr + p, mask=m, other=0)
    ulh, ull = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, u, m)
    dlh, dll = graph_lambda_inline(saved_ptr, dh_ptr, sol_low_ptr, d_, m)
    for lvl in range(blevels):
        k = p * blevels + lvl
        bk = k.to(tl.int64) * 7
        ce = graph_flux_adj_inline(
            ulh, ull, dlh, dll, tl.load(gQB_ptr + k, mask=m, other=0.0), dt
        )
        tl.store(
            gQB_ptr + k, ce * tl.load(bsaved_ptr + bk + 3, mask=m, other=0.0), mask=m
        )
        tl.store(
            gnb_ptr + k,
            tl.load(gnb_ptr + k, mask=m, other=0.0)
            + ce * tl.load(bsaved_ptr + bk + 4, mask=m, other=0.0),
            mask=m,
        )
