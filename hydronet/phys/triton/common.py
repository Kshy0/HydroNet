# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared Triton FP32 math, error-free pairs, geometry, flux, retraction and Newton control."""

from __future__ import annotations

import triton
import triton.language as tl
from hydroforge.kernels.triton_math import cbrt, libdevice

from hydronet.config import (
    ADJOINT_NOT_CONVERGED,
    GRAPH_CLOSED_ELEVATION,
    GRAPH_DRY_DEPTH,
    GRAPH_SLOPE_LIMIT,
    INVALID_ROUGHNESS,
    INVALID_STATE,
    NEWTON_DAMPED_GROWTH,
    NEWTON_DAMPED_SIGMA,
    NEWTON_DAMPED_STEPS,
    NEWTON_NOT_CONVERGED,
    NEWTON_ROUND_LIMIT,
    TREE_INTEGERS,
    TREE_SCALARS,
)

# Status codes (mirrored as constexpr for Triton kernels).
CODE_NEWTON_NOT_CONVERGED = tl.constexpr(NEWTON_NOT_CONVERGED)
CODE_NEWTON_ROUND_LIMIT = tl.constexpr(NEWTON_ROUND_LIMIT)
CODE_INVALID_ROUGHNESS = tl.constexpr(INVALID_ROUGHNESS)
CODE_INVALID_STATE = tl.constexpr(INVALID_STATE)
CODE_ADJOINT_NOT_CONVERGED = tl.constexpr(ADJOINT_NOT_CONVERGED)

# Newton controller constants matching implicit_inertia.cu.
LINE_SEARCH = tl.constexpr(16)
MAX_ROUNDS = tl.constexpr(1 << 22)

MODE_DONE = tl.constexpr(0)
MODE_EVAL = tl.constexpr(1)
MODE_TRIAL = tl.constexpr(2)
MODE_RESTORE = tl.constexpr(3)
MODE_FINAL = tl.constexpr(4)

RESTART_PHASE = tl.constexpr(21)
DAMPED_PHASE = tl.constexpr(22)

DAMPED_STEPS_C = tl.constexpr(NEWTON_DAMPED_STEPS)
DAMPED_SIGMA_C = tl.constexpr(float(NEWTON_DAMPED_SIGMA))
DAMPED_GROWTH_C = tl.constexpr(float(NEWTON_DAMPED_GROWTH))

# Tree control integer indices (TI = 15).
IDX_MODE = tl.constexpr(0)
IDX_PHASE = tl.constexpr(1)
IDX_STEPS = tl.constexpr(2)
IDX_BT = tl.constexpr(3)
IDX_ST = tl.constexpr(4)
IDX_FEVALS = tl.constexpr(5)
IDX_TOTALBT = tl.constexpr(6)
IDX_SUM_G = tl.constexpr(7)
IDX_SUM_FE = tl.constexpr(8)
IDX_DSTEPS = tl.constexpr(9)
IDX_BEST_ST = tl.constexpr(10)
IDX_SOLVE = tl.constexpr(11)
IDX_COMMIT = tl.constexpr(12)
IDX_SAVE = tl.constexpr(13)
IDX_RESET = tl.constexpr(14)
TI = tl.constexpr(TREE_INTEGERS)

# Tree control float indices (TF = 6).
IDX_DT = tl.constexpr(0)
IDX_LAM = tl.constexpr(1)
IDX_MERIT = tl.constexpr(2)
IDX_BEST = tl.constexpr(3)
IDX_INITIAL = tl.constexpr(4)
IDX_SIGMA = tl.constexpr(5)
TF = tl.constexpr(TREE_SCALARS)

# Grid control indices (5 int32s).
CTL_ACTIVE = tl.constexpr(0)
CTL_SOLVING = tl.constexpr(2)
CTL_ERROR = tl.constexpr(4)

# Graph constants.
SLOPE_LIMIT_C = tl.constexpr(float(GRAPH_SLOPE_LIMIT))
CLOSED_ELEV_C = tl.constexpr(float(GRAPH_CLOSED_ELEVATION))
DRY_DEPTH_C = tl.constexpr(float(GRAPH_DRY_DEPTH))


# --------------------------------------------------------------------------- #
# IEEE FP32 math & error-free pairs
# --------------------------------------------------------------------------- #


@triton.jit
def isfinite_f32_inline(x):
    inf = tl.full((), float("inf"), tl.float32)
    return (x == x) & (tl.abs(x) < inf)


@triton.jit
def p23_inline(a):
    return cbrt(a * a)


@triton.jit
def phi_inline(j, e):
    return j * libdevice.rsqrt(tl.sqrt_rn(j * j + e * e))


@triton.jit
def dphi_inline(j, e):
    a = j * j + e * e
    return tl.div_rn(0.5 * j * j + e * e, a * tl.sqrt_rn(tl.sqrt_rn(a)))


@triton.jit
def pair_sum_inline(a, b):
    s = libdevice.add_rn(a, b)
    v = libdevice.sub_rn(s, a)
    lo = libdevice.add_rn(
        libdevice.sub_rn(a, libdevice.sub_rn(s, v)),
        libdevice.sub_rn(b, v),
    )
    return s, lo


@triton.jit
def pair_diff_inline(a, b):
    return pair_sum_inline(a, -b)


@triton.jit
def pair_add_inline(ahi, alo, bhi, blo):
    shi, slo = pair_sum_inline(ahi, bhi)
    e = libdevice.add_rn(slo, libdevice.add_rn(alo, blo))
    return pair_sum_inline(shi, e)


@triton.jit
def head_pair_inline(ei, ed, ri, rd, xi, xd):
    d1h, d1l = pair_diff_inline(ei, ed)
    d2h, d2l = pair_diff_inline(ri, rd)
    ph, pl = pair_add_inline(d1h, d1l, d2h, d2l)
    d3h, d3l = pair_diff_inline(xi, xd)
    ph, pl = pair_add_inline(ph, pl, d3h, d3l)
    return libdevice.add_rn(ph, pl)


@triton.jit
def trial_pair_inline(ref, x, step, H):
    zero = tl.zeros_like(step)
    ph, pl = pair_add_inline(ref, x, step, zero)
    dry = libdevice.add_rn(libdevice.add_rn(ph, H), pl) < 0.0
    return tl.where(dry, -H, ph), tl.where(dry, zero, pl)


@triton.jit
def graph_product_inline(a, b):
    p = libdevice.mul_rn(a, b)
    return p, tl.fma(a, b, -p)


@triton.jit
def graph_scale_inline(ahi, alo, b):
    p1h, p1l = graph_product_inline(ahi, b)
    p2h, p2l = graph_product_inline(alo, b)
    return pair_add_inline(p1h, p1l, p2h, p2l)


@triton.jit
def continuation_beta_inline(phase):
    b = tl.full((), 1.0, tl.float32)
    b = tl.where(phase == 0, 1e-6, b)
    b = tl.where(phase == 1, 3e-6, b)
    b = tl.where(phase == 2, 1e-5, b)
    b = tl.where(phase == 3, 3e-5, b)
    b = tl.where(phase == 4, 1e-4, b)
    b = tl.where(phase == 5, 3e-4, b)
    b = tl.where(phase == 6, 1e-3, b)
    b = tl.where(phase == 7, 3e-3, b)
    b = tl.where(phase == 8, 1e-2, b)
    b = tl.where(phase == 9, 3e-2, b)
    b = tl.where(phase == 10, 1e-1, b)
    b = tl.where(phase == 11, 3e-1, b)
    return b


@triton.jit
def phase_offset_value_inline(idx):
    o = tl.full((), 0.5, tl.float32)
    o = tl.where(idx == 0, -0.01, o)
    o = tl.where(idx == 1, 0.01, o)
    o = tl.where(idx == 2, -0.05, o)
    o = tl.where(idx == 3, 0.05, o)
    o = tl.where(idx == 4, -0.2, o)
    o = tl.where(idx == 5, 0.2, o)
    o = tl.where(idx == 6, -0.5, o)
    return o


# --------------------------------------------------------------------------- #
# Channel & floodplain geometry
# --------------------------------------------------------------------------- #


@triton.jit
def channel_conveyance_inline(w, H, d):
    wet = d > 0.0
    a = w * d
    p = w + 2.0 * tl.minimum(d, H)
    p_safe = tl.where(wet, p, 1.0)
    ratio = tl.div_rn(a, p_safe)
    r = p23_inline(tl.where(wet, ratio, 0.0))
    c1 = tl.full((), 1.6666667, tl.float32)
    c2 = tl.full((), 1.3333334, tl.float32)
    term = tl.where(d < H, c2 * ratio, 0.0)
    k = tl.where(wet, a * r, 0.0)
    dk = tl.where(wet, r * (c1 * w - term), 0.0)
    return k, dk


@triton.jit
def flood_bands_inline(i, h, dw, tab_ptr, bed_ptr, mask, levels):
    k_acc = tl.zeros_like(h)
    dk_acc = tl.zeros_like(h)
    t0 = tl.zeros_like(h)
    b0 = tl.zeros_like(h)
    still = mask & (h > 0.0)
    base = i.to(tl.int64) * levels
    c1 = tl.full((), 1.6666667, tl.float32)
    c2 = tl.full((), 0.6666667, tl.float32)
    zero = tl.zeros_like(h)
    one = tl.full((), 1.0, tl.float32) + zero
    for b in range(levels):
        t = tl.load(tab_ptr + base + b, mask=still, other=0.0)
        b1 = tl.load(bed_ptr + base + b, mask=still, other=0.0)
        seg = b1 - b0
        still = still & (h > t0)
        full_band = ~(h < t)
        # Full band branch
        a_full = dw * ((h - t0) - 0.5 * (t - t0))
        is_last = b == (levels - 1)
        p_full = tl.where(is_last, seg + (h - t), seg)
        dp_full = tl.where(is_last, one, zero)
        da_full = dw
        # Partial band branch
        dt_b = tl.where(full_band | (~still), one, t - t0)
        x = h - t0
        run = x * tl.div_rn(dw, dt_b)
        a_part = 0.5 * run * x
        p_part = x * tl.div_rn(seg, dt_b)
        da_part = run
        dp_part = tl.div_rn(seg, dt_b)
        # Select
        a = tl.where(full_band, a_full, a_part)
        p = tl.where(full_band, p_full, p_part)
        da = tl.where(full_band, da_full, da_part)
        dp = tl.where(full_band, dp_full, dp_part)
        add_m = still & (a > 0.0)
        p_safe = tl.where(add_m, p, one)
        ratio = tl.div_rn(a, p_safe)
        r = p23_inline(tl.where(add_m, ratio, zero))
        k_acc = tl.where(add_m, k_acc + a * r, k_acc)
        dk_acc = tl.where(
            add_m, dk_acc + r * (c1 * da - tl.div_rn(c2 * a * dp, p_safe)), dk_acc
        )
        t0 = tl.where(still, t, t0)
        b0 = tl.where(still, b1, b0)
    return k_acc, dk_acc


@triton.jit
def stage_pair_inline(i, ref, x, length, w, hh, A, tab_ptr, bed_ptr, mask, levels):
    h = ref + x
    over = mask & (h > 0.0)
    d = tl.where(over, tl.maximum(h + hh, 0.0), tl.maximum((ref + hh) + x, 0.0))
    kc, dkc = channel_conveyance_inline(w, hh, d)
    S = length * w * d
    area = length * w
    kf = tl.zeros_like(h)
    dkf = tl.zeros_like(h)
    if tl.max(over.to(tl.int32), axis=0) != 0:
        dw = tl.div_rn(tl.div_rn(A, length), levels.to(tl.float32))
        ph = tl.zeros_like(h)
        pw = w
        acc = tl.zeros_like(h)
        nh_stop = tl.zeros_like(h)
        found = tl.zeros_like(over)
        stepping = over
        base = i.to(tl.int64) * levels
        for k in range(levels):
            nh = tl.load(tab_ptr + base + k, mask=stepping, other=0.0)
            nw = w + (k + 1).to(tl.float32) * dw
            pass_b = stepping & (h > nh)
            first_stop = stepping & (~pass_b)
            nh_stop = tl.where(first_stop, nh, nh_stop)
            found = found | first_stop
            acc = tl.where(pass_b, acc + 0.5 * (pw + nw) * (nh - ph), acc)
            ph = tl.where(pass_b, nh, ph)
            pw = tl.where(pass_b, nw, pw)
            stepping = pass_b
        grad = tl.where(found, tl.div_rn(nh_stop - ph, dw), 0.0)
        xb = h - ph
        grad_safe = tl.where(grad > 0.0, grad, 1.0)
        run = tl.where(grad > 0.0, tl.div_rn(xb, grad_safe), 0.0)
        S_over = length * (w * hh + acc + pw * xb + 0.5 * run * xb)
        area_over = length * (pw + run)
        kf_over, dkf_over = flood_bands_inline(i, h, dw, tab_ptr, bed_ptr, over, levels)
        S = tl.where(over, S_over, S)
        area = tl.where(over, area_over, area)
        kf = tl.where(over, kf_over, kf)
        dkf = tl.where(over, dkf_over, dkf)
    return S, area, kc, kf, dkc, dkf


@triton.jit
def invert_storage_inline(i, S_in, length, w, hh, A, tab_ptr, mask, levels):
    S = tl.maximum(S_in, 0.0)
    acc = length * w * hh
    chan_h = tl.div_rn(S, length * w) - hh
    over = mask & (S > acc)
    out = chan_h
    if tl.max(over.to(tl.int32), axis=0) != 0:
        dw = tl.div_rn(tl.div_rn(A, length), levels.to(tl.float32))
        ph = tl.zeros_like(S)
        pw = w
        nh_stop = tl.zeros_like(S)
        found = tl.zeros_like(over)
        stepping = over
        base = i.to(tl.int64) * levels
        for k in range(levels):
            nh = tl.load(tab_ptr + base + k, mask=stepping, other=0.0)
            nw = w + (k + 1).to(tl.float32) * dw
            nxt = acc + length * 0.5 * (pw + nw) * (nh - ph)
            pass_b = stepping & (S > nxt)
            first_stop = stepping & (~pass_b)
            nh_stop = tl.where(first_stop, nh, nh_stop)
            found = found | first_stop
            acc = tl.where(pass_b, nxt, acc)
            ph = tl.where(pass_b, nh, ph)
            pw = tl.where(pass_b, nw, pw)
            stepping = pass_b
        grad = tl.where(found, tl.div_rn(nh_stop - ph, dw), 0.0)
        extra = tl.maximum(S - acc, 0.0)
        grad_safe = tl.where(grad > 0.0, grad, 1.0)
        h_grad = ph + tl.div_rn(
            tl.div_rn(2.0 * extra, length),
            tl.sqrt_rn(pw * pw + tl.div_rn(2.0 * extra, grad_safe * length)) + pw,
        )
        h_flat = ph + tl.div_rn(extra, pw * length)
        out = tl.where(over, tl.where(grad > 0.0, h_grad, h_flat), chan_h)
    return out


# --------------------------------------------------------------------------- #
# Local inertia momentum flux
# --------------------------------------------------------------------------- #


@triton.jit
def momentum_flux_inline(q0, A, Ai, Ad, K, Ki, Kd, j, ji, jd, dt, eps):
    valid = (A > 1e-8) & (K > 1e-12)
    A_safe = tl.where(valid, A, 1.0)
    K_safe = tl.where(valid, K, 1.0)
    f = phi_inline(j, eps)
    fp = dphi_inline(j, eps)
    je = f * tl.abs(f)
    dje = 2.0 * tl.abs(f) * fp
    a = 9.8 * dt * A_safe
    b = tl.div_rn(a, K_safe * K_safe)
    u = q0 + a * je
    den = tl.sqrt_rn(1.0 + 4.0 * b * tl.abs(u))
    q = tl.div_rn(2.0 * u, 1.0 + den)
    fric = b * q * tl.abs(q)
    qi = tl.div_rn(
        (q - q0) * tl.div_rn(Ai, A_safe)
        + 2.0 * fric * tl.div_rn(Ki, K_safe)
        + a * dje * ji,
        den,
    )
    qd = tl.div_rn(
        (q - q0) * tl.div_rn(Ad, A_safe)
        + 2.0 * fric * tl.div_rn(Kd, K_safe)
        + a * dje * jd,
        den,
    )
    zero = tl.zeros_like(q)
    return (
        tl.where(valid, q, zero),
        tl.where(valid, qi, zero),
        tl.where(valid, qd, zero),
    )


@triton.jit
def eval_flux_vec_inline(
    i,
    dt,
    eps,
    mask,
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
):
    edge = tl.load(edge_ptr + i, mask=mask, other=False)
    down_raw = tl.load(down_ptr + i, mask=mask, other=0)
    d = tl.where(edge, down_raw, i)
    oi = tl.load(tref_ptr + i, mask=mask, other=0.0)
    xi = tl.load(ht_ptr + i, mask=mask, other=0.0)
    od = tl.load(tref_ptr + d, mask=mask, other=0.0)
    xd = tl.load(ht_ptr + d, mask=mask, other=0.0)
    ei = tl.load(elev_ptr + i, mask=mask, other=0.0)
    ed = tl.load(elev_ptr + d, mask=mask, other=0.0)
    dist = tl.load(dist_ptr + i, mask=mask, other=1.0)
    dist_safe = tl.where(edge & mask, dist, 1.0)
    head = tl.where(edge, head_pair_inline(ei, ed, oi, od, xi, xd), oi + xi)
    inv_10k = tl.div_rn(tl.full((), 1.0, tl.float32), tl.full((), 10000.0, tl.float32))
    j = tl.where(
        edge,
        tl.div_rn(head, dist_safe),
        tl.div_rn(oi + xi, tl.full((), 10000.0, tl.float32)),
    )

    length = tl.load(L_ptr + i, mask=mask, other=1.0)
    own_S = tl.load(S_ptr + i, mask=mask, other=0.0)
    own_area = tl.load(area_ptr + i, mask=mask, other=0.0)
    own_kc = tl.load(kc_ptr + i, mask=mask, other=0.0)
    own_kf = tl.load(kf_ptr + i, mask=mask, other=0.0)
    own_dkc = tl.load(dkc_ptr + i, mask=mask, other=0.0)
    own_dkf = tl.load(dkf_ptr + i, mask=mask, other=0.0)

    max_at_down = mask & edge & (head < 0.0)
    accel_S = own_S
    accel_area = own_area
    if tl.max(max_at_down.to(tl.int32), axis=0) != 0:
        w = tl.load(W_ptr + i, mask=max_at_down, other=1.0)
        hh = tl.load(H_ptr + i, mask=max_at_down, other=0.0)
        Ai_area = tl.load(A_ptr + i, mask=max_at_down, other=1.0)
        depthA = tl.maximum(
            ((oi + hh) + xi) - tl.minimum(head, 0.0) * tl.where(edge, 1.0, 0.0), 0.0
        )
        aS, aA, _, _, _, _ = stage_pair_inline(
            i,
            -hh,
            depthA,
            length,
            w,
            hh,
            Ai_area,
            tab_ptr,
            bed_ptr,
            max_at_down,
            levels,
        )
        accel_S = tl.where(max_at_down, aS, accel_S)
        accel_area = tl.where(max_at_down, aA, accel_area)

    Ae = tl.div_rn(accel_S, length)
    dAe = tl.div_rn(accel_area, length)
    f = phi_inline(j, eps)
    q0 = tl.load(Q0_ptr + i, mask=mask, other=0.0)
    reverse = mask & edge & ((q0 + 9.8 * dt * Ae * f * tl.abs(f)) < 0.0)

    nc_i = tl.load(nc_ptr + i, mask=mask, other=1.0)
    nf_i = tl.load(nf_ptr + i, mask=mask, other=1.0)
    inc = tl.div_rn(tl.full((), 1.0, tl.float32), nc_i)
    inf_ = tl.div_rn(tl.full((), 1.0, tl.float32), nf_i)
    K = own_kc * inc + own_kf * inf_
    dK = own_dkc * inc + own_dkf * inf_
    if tl.max(reverse.to(tl.int32), axis=0) != 0:
        w = tl.load(W_ptr + i, mask=reverse, other=1.0)
        hh = tl.load(H_ptr + i, mask=reverse, other=0.0)
        Ai_area = tl.load(A_ptr + i, mask=reverse, other=1.0)
        hd_val = tl.load(H_ptr + d, mask=reverse, other=0.0)
        beddiff = (ed - ei) + (hh - hd_val)
        depth = tl.maximum(((od + hd_val) + xd) + tl.minimum(0.0, beddiff), 0.0)
        _, _, ckc, ckf, cdkc, cdkf = stage_pair_inline(
            i, -hh, depth, length, w, hh, Ai_area, tab_ptr, bed_ptr, reverse, levels
        )
        K = tl.where(reverse, ckc * inc + ckf * inf_, K)
        dK = tl.where(reverse, cdkc * inc + cdkf * inf_, dK)

    inv_dist = tl.div_rn(tl.full((), 1.0, tl.float32), dist_safe)
    ji = tl.where(edge, inv_dist, inv_10k)
    jd = tl.where(edge, -inv_dist, 0.0)
    q, dqi, dqd = momentum_flux_inline(
        q0,
        Ae,
        tl.where(max_at_down, 0.0, dAe),
        tl.where(max_at_down, dAe, 0.0),
        K,
        tl.where(reverse, 0.0, dK),
        tl.where(reverse, dK, 0.0),
        j,
        ji,
        jd,
        dt,
        eps,
    )
    clamp_out = (~edge) & (q < 0.0)
    q = tl.where(clamp_out, 0.0, q)
    dqi = tl.where(clamp_out, 0.0, dqi)
    dqd = tl.where(clamp_out, 0.0, dqd)
    return q, dqi, dqd


# --------------------------------------------------------------------------- #
# Bank-relative volume coordinate & inverse (compensated FP32)
# --------------------------------------------------------------------------- #


@triton.jit
def terrain_coordinate_inline(i, h_hi, h_lo, length, w, A, tab_ptr, mask, levels):
    pos = mask & ((h_hi > 0.0) | ((h_hi == 0.0) & (h_lo > 0.0)))
    out_hi = h_hi
    out_lo = h_lo
    if tl.max(pos.to(tl.int32), axis=0) != 0:
        dw = tl.div_rn(tl.div_rn(A, length), levels.to(tl.float32))
        ph = tl.zeros_like(h_hi)
        pw = w
        acc_hi = tl.zeros_like(h_hi)
        acc_lo = tl.zeros_like(h_hi)
        nh_stop = tl.zeros_like(h_hi)
        found = tl.zeros_like(pos)
        stepping = pos
        base = i.to(tl.int64) * levels
        for k in range(levels):
            nh = tl.load(tab_ptr + base + k, mask=stepping, other=0.0)
            nw = w + (k + 1).to(tl.float32) * dw
            pass_b = stepping & (h_hi > nh)
            first_stop = stepping & (~pass_b)
            nh_stop = tl.where(first_stop, nh, nh_stop)
            found = found | first_stop
            width = 0.5 * (pw + nw)
            delta = nh - ph
            v = width * delta
            n_hi, n_lo = pair_add_inline(acc_hi, acc_lo, v, tl.fma(width, delta, -v))
            acc_hi = tl.where(pass_b, n_hi, acc_hi)
            acc_lo = tl.where(pass_b, n_lo, acc_lo)
            ph = tl.where(pass_b, nh, ph)
            pw = tl.where(pass_b, nw, pw)
            stepping = pass_b
        grad = tl.where(found, tl.div_rn(nh_stop - ph, dw), 0.0)
        xb = h_hi - ph
        grad_safe = tl.where(grad > 0.0, grad, 1.0)
        run = tl.where(grad > 0.0, tl.div_rn(xb, grad_safe), 0.0)
        width = pw + 0.5 * run
        v = width * xb
        acc_hi, acc_lo = pair_add_inline(acc_hi, acc_lo, v, tl.fma(width, xb, -v))
        acc_hi, acc_lo = pair_add_inline(
            acc_hi, acc_lo, (pw + run) * h_lo, tl.zeros_like(h_hi)
        )
        hi = tl.div_rn(acc_hi, w)
        r_hi, r_lo = pair_sum_inline(hi, tl.div_rn(tl.fma(-hi, w, acc_hi) + acc_lo, w))
        out_hi = tl.where(pos, r_hi, out_hi)
        out_lo = tl.where(pos, r_lo, out_lo)
    return out_hi, out_lo


@triton.jit
def terrain_inverse_inline(i, v_hi, v_lo, length, w, hh, A, tab_ptr, mask, levels):
    pos = mask & ((v_hi > 0.0) | ((v_hi == 0.0) & (v_lo > 0.0)))
    dry_neg = ((v_hi + hh) + v_lo) < 0.0
    out_hi = tl.where(dry_neg, -hh, v_hi)
    out_lo = tl.where(dry_neg, tl.zeros_like(v_lo), v_lo)
    if tl.max(pos.to(tl.int32), axis=0) != 0:
        dw = tl.div_rn(tl.div_rn(A, length), levels.to(tl.float32))
        ph = tl.zeros_like(v_hi)
        pw = w
        c = v_hi * w
        cross_hi, cross_lo = pair_sum_inline(c, tl.fma(v_hi, w, -c) + v_lo * w)
        acc_hi = tl.zeros_like(v_hi)
        acc_lo = tl.zeros_like(v_hi)
        nh_stop = tl.zeros_like(v_hi)
        found = tl.zeros_like(pos)
        stepping = pos
        base = i.to(tl.int64) * levels
        for k in range(levels):
            nh = tl.load(tab_ptr + base + k, mask=stepping, other=0.0)
            nw = w + (k + 1).to(tl.float32) * dw
            width = 0.5 * (pw + nw)
            delta = nh - ph
            a = width * delta
            nxt_hi, nxt_lo = pair_add_inline(
                acc_hi, acc_lo, a, tl.fma(width, delta, -a)
            )
            exc_hi, _ = pair_add_inline(cross_hi, cross_lo, -nxt_hi, -nxt_lo)
            pass_b = stepping & (exc_hi > 0.0)
            first_stop = stepping & (~pass_b)
            nh_stop = tl.where(first_stop, nh, nh_stop)
            found = found | first_stop
            acc_hi = tl.where(pass_b, nxt_hi, acc_hi)
            acc_lo = tl.where(pass_b, nxt_lo, acc_lo)
            ph = tl.where(pass_b, nh, ph)
            pw = tl.where(pass_b, nw, pw)
            stepping = pass_b
        exc_hi, exc_lo = pair_add_inline(cross_hi, cross_lo, -acc_hi, -acc_lo)
        extra = tl.maximum(exc_hi + exc_lo, 0.0)
        grad = tl.where(found, tl.div_rn(nh_stop - ph, dw), 0.0)
        grad_safe = tl.where(grad > 0.0, grad, 1.0)
        xb = tl.where(
            grad > 0.0,
            tl.div_rn(
                2.0 * extra,
                tl.sqrt_rn(pw * pw + tl.div_rn(2.0 * extra, grad_safe)) + pw,
            ),
            tl.div_rn(extra, pw),
        )
        h_hi, h_lo = pair_sum_inline(ph, xb)
        got_hi, got_lo = terrain_coordinate_inline(
            i, h_hi, h_lo, length, w, A, tab_ptr, pos, levels
        )
        err_hi, err_lo = pair_add_inline(v_hi, v_lo, -got_hi, -got_lo)
        slope = tl.div_rn(pw + tl.where(grad > 0.0, tl.div_rn(xb, grad_safe), 0.0), w)
        r_hi, r_lo = pair_add_inline(
            h_hi, h_lo, tl.div_rn(err_hi + err_lo, slope), tl.zeros_like(h_hi)
        )
        out_hi = tl.where(pos, r_hi, out_hi)
        out_lo = tl.where(pos, r_lo, out_lo)
    return out_hi, out_lo


# --------------------------------------------------------------------------- #
# 3-tangent Jet (W_i, H_i, H_down) + parameter partials + snapshot
# --------------------------------------------------------------------------- #


@triton.jit
def jet_mul_inline(av, a0, a1, a2, bv, b0, b1, b2):
    return (
        av * bv,
        a0 * bv + av * b0,
        a1 * bv + av * b1,
        a2 * bv + av * b2,
    )


@triton.jit
def jet_div_inline(av, a0, a1, a2, bv, b0, b1, b2):
    cv = tl.div_rn(av, bv)
    return (
        cv,
        tl.div_rn(a0 - cv * b0, bv),
        tl.div_rn(a1 - cv * b1, bv),
        tl.div_rn(a2 - cv * b2, bv),
    )


@triton.jit
def jet_pow23_inline(av, a0, a1, a2):
    pos = av > 0.0
    cv = cbrt(av * av)
    a_safe = tl.where(pos, av, 1.0)
    fac = tl.where(
        pos,
        tl.div_rn(
            tl.div_rn(tl.full((), 2.0, tl.float32), tl.full((), 3.0, tl.float32)) * cv,
            a_safe,
        ),
        0.0,
    )
    return cv, fac * a0, fac * a1, fac * a2


@triton.jit
def geometry_jet_inline(
    i,
    fv,
    f0,
    f1,
    f2,
    dv,
    d0,
    d1,
    d2,
    wv,
    Hv,
    length,
    A,
    tab_ptr,
    bed_ptr,
    mask,
    levels,
):
    # w has tangent (1, 0, 0); H has tangent (0, 1, 0)
    z = tl.zeros_like(fv)
    one = tl.full((), 1.0, tl.float32) + z
    dpos = dv > 0.0
    dv = tl.where(dpos, dv, z)
    d0 = tl.where(dpos, d0, z)
    d1 = tl.where(dpos, d1, z)
    d2 = tl.where(dpos, d2, z)

    # Channel conveyance jet
    acv, ac0, ac1, ac2 = jet_mul_inline(wv, one, z, z, dv, d0, d1, d2)
    d_lt_H = dv < Hv
    mv = tl.where(d_lt_H, dv, Hv)
    m0 = tl.where(d_lt_H, d0, z)
    m1 = tl.where(d_lt_H, d1, one)
    m2 = tl.where(d_lt_H, d2, z)
    pcv = wv + 2.0 * mv
    pc0 = one + 2.0 * m0
    pc1 = 2.0 * m1
    pc2 = 2.0 * m2
    pc_safe = tl.where(dpos, pcv, one)
    rv, r0, r1, r2 = jet_div_inline(acv, ac0, ac1, ac2, pc_safe, pc0, pc1, pc2)
    pv, p0, p1, p2 = jet_pow23_inline(tl.where(dpos, rv, z), r0, r1, r2)
    kcv, kc0, kc1, kc2 = jet_mul_inline(acv, ac0, ac1, ac2, pv, p0, p1, p2)
    kcv = tl.where(dpos, kcv, z)
    kc0 = tl.where(dpos, kc0, z)
    kc1 = tl.where(dpos, kc1, z)
    kc2 = tl.where(dpos, kc2, z)

    # Below bank storage
    Sv = length * acv
    S0 = length * ac0
    S1 = length * ac1
    S2 = length * ac2
    kfv, kf0, kf1, kf2 = z, z, z, z

    over = mask & (fv > 0.0)
    if tl.max(over.to(tl.int32), axis=0) != 0:
        dw = tl.div_rn(tl.div_rn(A, length), levels.to(tl.float32))
        ph = z
        pw = z
        acc = z
        nh_stop = z
        found = tl.zeros_like(over)
        stepping = over
        base = i.to(tl.int64) * levels
        for k in range(levels):
            nh = tl.load(tab_ptr + base + k, mask=stepping, other=0.0)
            nw = (k + 1).to(tl.float32) * dw
            pass_b = stepping & (fv > nh)
            first_stop = stepping & (~pass_b)
            nh_stop = tl.where(first_stop, nh, nh_stop)
            found = found | first_stop
            acc = tl.where(pass_b, acc + 0.5 * (pw + nw) * (nh - ph), acc)
            ph = tl.where(pass_b, nh, ph)
            pw = tl.where(pass_b, nw, pw)
            stepping = pass_b
        slope = tl.where(found, tl.div_rn(nh_stop - ph, dw), z)
        xv = fv - ph
        x0, x1, x2 = f0, f1, f2
        spos = slope > 0.0
        slope_safe = tl.where(spos, slope, one)
        runv = tl.where(spos, tl.div_rn(xv, slope_safe), z)
        run0 = tl.where(spos, tl.div_rn(x0, slope_safe), z)
        run1 = tl.where(spos, tl.div_rn(x1, slope_safe), z)
        run2 = tl.where(spos, tl.div_rn(x2, slope_safe), z)
        rxv, rx0, rx1, rx2 = jet_mul_inline(runv, run0, run1, run2, xv, x0, x1, x2)
        fpv = acc + pw * xv + 0.5 * rxv
        fp0 = pw * x0 + 0.5 * rx0
        fp1 = pw * x1 + 0.5 * rx1
        fp2 = pw * x2 + 0.5 * rx2
        fp_pos = fpv > 0.0
        fpv = tl.where(fp_pos, fpv, z)
        fp0 = tl.where(fp_pos, fp0, z)
        fp1 = tl.where(fp_pos, fp1, z)
        fp2 = tl.where(fp_pos, fp2, z)
        hfv = Hv + fv
        hf0, hf1, hf2 = f0, one + f1, f2
        whv, wh0, wh1, wh2 = jet_mul_inline(wv, one, z, z, hfv, hf0, hf1, hf2)
        Sv = tl.where(over, length * (whv + fpv), Sv)
        S0 = tl.where(over, length * (wh0 + fp0), S0)
        S1 = tl.where(over, length * (wh1 + fp1), S1)
        S2 = tl.where(over, length * (wh2 + fp2), S2)

        # Floodplain bands jet
        t0 = z
        b0 = z
        still = over
        for b in range(levels):
            t = tl.load(tab_ptr + base + b, mask=still, other=0.0)
            b1 = tl.load(bed_ptr + base + b, mask=still, other=0.0)
            seg = b1 - b0
            still = still & (fv > t0)
            full_b = ~(fv < t)
            # Full branch: a = ((fill - t0) - 0.5*(t - t0))*dw
            af_v = ((fv - t0) - 0.5 * (t - t0)) * dw
            af_0 = f0 * dw
            af_1 = f1 * dw
            af_2 = f2 * dw
            is_last = b == (levels - 1)
            pf_v = tl.where(is_last, seg + (fv - t), seg)
            pf_0 = tl.where(is_last, f0, z)
            pf_1 = tl.where(is_last, f1, z)
            pf_2 = tl.where(is_last, f2, z)
            # Partial branch
            dt_b = tl.where(full_b | (~still), one, t - t0)
            xb_v = fv - t0
            c_a = 0.5 * tl.div_rn(dw, dt_b)
            x2v, x20, x21, x22 = jet_mul_inline(xb_v, f0, f1, f2, xb_v, f0, f1, f2)
            ap_v, ap_0, ap_1, ap_2 = c_a * x2v, c_a * x20, c_a * x21, c_a * x22
            c_p = tl.div_rn(seg, dt_b)
            pp_v, pp_0, pp_1, pp_2 = c_p * xb_v, c_p * f0, c_p * f1, c_p * f2
            # Select
            av = tl.where(full_b, af_v, ap_v)
            a0 = tl.where(full_b, af_0, ap_0)
            a1 = tl.where(full_b, af_1, ap_1)
            a2 = tl.where(full_b, af_2, ap_2)
            pv_b = tl.where(full_b, pf_v, pp_v)
            p0_b = tl.where(full_b, pf_0, pp_0)
            p1_b = tl.where(full_b, pf_1, pp_1)
            p2_b = tl.where(full_b, pf_2, pp_2)
            add_m = still & (av > 0.0)
            p_safe = tl.where(add_m, pv_b, one)
            qv, q0_, q1_, q2_ = jet_div_inline(av, a0, a1, a2, p_safe, p0_b, p1_b, p2_b)
            uv, u0, u1, u2 = jet_pow23_inline(tl.where(add_m, qv, z), q0_, q1_, q2_)
            tv, t0_, t1_, t2_ = jet_mul_inline(av, a0, a1, a2, uv, u0, u1, u2)
            kfv = tl.where(add_m, kfv + tv, kfv)
            kf0 = tl.where(add_m, kf0 + t0_, kf0)
            kf1 = tl.where(add_m, kf1 + t1_, kf1)
            kf2 = tl.where(add_m, kf2 + t2_, kf2)
            t0 = tl.where(still, t, t0)
            b0 = tl.where(still, b1, b0)
    return Sv, S0, S1, S2, kcv, kc0, kc1, kc2, kfv, kf0, kf1, kf2


@triton.jit
def snapshot_unit_vec_inline(
    i,
    dt,
    eps,
    mask,
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
):
    edge = tl.load(edge_ptr + i, mask=mask, other=False)
    down_raw = tl.load(down_ptr + i, mask=mask, other=0)
    d = tl.where(edge, down_raw, i)
    ri = tl.load(ref_ptr + i, mask=mask, other=0.0)
    xi = tl.load(h_ptr + i, mask=mask, other=0.0)
    rd = tl.load(ref_ptr + d, mask=mask, other=0.0)
    xd = tl.load(h_ptr + d, mask=mask, other=0.0)
    ei = tl.load(elev_ptr + i, mask=mask, other=0.0)
    ed = tl.load(elev_ptr + d, mask=mask, other=0.0)
    dist = tl.load(dist_ptr + i, mask=mask, other=1.0)
    dist_safe = tl.where(edge & mask, dist, 1.0)
    head = tl.where(edge, head_pair_inline(ei, ed, ri, rd, xi, xd), ri + xi)
    j = tl.where(
        edge,
        tl.div_rn(head, dist_safe),
        tl.div_rn(ri + xi, tl.full((), 10000.0, tl.float32)),
    )

    length = tl.load(L_ptr + i, mask=mask, other=1.0)
    w = tl.load(W_ptr + i, mask=mask, other=1.0)
    hh = tl.load(H_ptr + i, mask=mask, other=0.0)
    hd_val = tl.load(H_ptr + d, mask=mask, other=0.0)
    A_i = tl.load(A_ptr + i, mask=mask, other=1.0)
    Q0_i = tl.load(Q0_ptr + i, mask=mask, other=0.0)
    nc_i = tl.load(nc_ptr + i, mask=mask, other=1.0)
    nf_i = tl.load(nf_ptr + i, mask=mask, other=1.0)
    Q_i = tl.load(Q_ptr + i, mask=mask, other=0.0)
    S_i = tl.load(S_ptr + i, mask=mask, other=0.0)
    area_i = tl.load(area_ptr + i, mask=mask, other=0.0)
    qi_i = tl.load(qi_ptr + i, mask=mask, other=0.0)
    qd_i = tl.load(qd_ptr + i, mask=mask, other=0.0)
    R_i = tl.load(R_ptr + i, mask=mask, other=0.0)

    z = tl.zeros_like(length)
    one = tl.full((), 1.0, tl.float32) + z

    ownfill_v = ri + xi
    owndepth_v = (hh + ri) + xi
    (
        own_Sv,
        own_S0,
        own_S1,
        own_S2,
        own_kcv,
        own_kc0,
        own_kc1,
        own_kc2,
        own_kfv,
        own_kf0,
        own_kf1,
        own_kf2,
    ) = geometry_jet_inline(
        i,
        ownfill_v,
        z,
        z,
        z,
        owndepth_v,
        z,
        one,
        z,
        w,
        hh,
        length,
        A_i,
        tab_ptr,
        bed_ptr,
        mask,
        levels,
    )
    ex_own_S, _, ex_own_kc, ex_own_kf, _, _ = stage_pair_inline(
        i, ri, xi, length, w, hh, A_i, tab_ptr, bed_ptr, mask, levels
    )

    # Acceleration area jet & exact value
    a_S0, a_S1, a_S2 = own_S0, own_S1, own_S2
    ex_acc_S = ex_own_S
    backwater = mask & edge & (head < 0.0)
    if tl.max(backwater.to(tl.int32), axis=0) != 0:
        da_raw = owndepth_v - head
        da_pos = da_raw > 0.0
        da_v = tl.where(da_pos, da_raw, z)
        da_1 = tl.where(da_pos, one, z)
        _, bS0, bS1, bS2, _, _, _, _, _, _, _, _ = geometry_jet_inline(
            i,
            da_v - hh,
            z,
            da_1 - one,
            z,
            da_v,
            z,
            da_1,
            z,
            w,
            hh,
            length,
            A_i,
            tab_ptr,
            bed_ptr,
            backwater,
            levels,
        )
        b_ex_S, _, _, _, _, _ = stage_pair_inline(
            i, -hh, da_v, length, w, hh, A_i, tab_ptr, bed_ptr, backwater, levels
        )
        a_S0 = tl.where(backwater, bS0, a_S0)
        a_S1 = tl.where(backwater, bS1, a_S1)
        a_S2 = tl.where(backwater, bS2, a_S2)
        ex_acc_S = tl.where(backwater, b_ex_S, ex_acc_S)

    Ae = tl.div_rn(ex_acc_S, length)
    Aj0, Aj1, Aj2 = (
        tl.div_rn(a_S0, length),
        tl.div_rn(a_S1, length),
        tl.div_rn(a_S2, length),
    )
    f = phi_inline(j, eps)
    reverse = mask & edge & ((Q0_i + 9.8 * dt * Ae * f * tl.abs(f)) < 0.0)

    # Conveyance jet & exact value
    c_kcv, c_kc0, c_kc1, c_kc2 = own_kcv, own_kc0, own_kc1, own_kc2
    c_kfv, c_kf0, c_kf1, c_kf2 = own_kfv, own_kf0, own_kf1, own_kf2
    ex_c_kc, ex_c_kf = ex_own_kc, ex_own_kf
    if tl.max(reverse.to(tl.int32), axis=0) != 0:
        beddiff_v = (ed - ei) + (hh - hd_val)
        bd_neg = beddiff_v < 0.0
        min_v = tl.where(bd_neg, beddiff_v, z)
        min_1 = tl.where(bd_neg, one, z)
        min_2 = tl.where(bd_neg, -one, z)
        dep_raw = ((hd_val + rd) + xd) + min_v
        dep_pos = dep_raw > 0.0
        dep_v = tl.where(dep_pos, dep_raw, z)
        dep_1 = tl.where(dep_pos, min_1, z)
        dep_2 = tl.where(dep_pos, one + min_2, z)
        _, _, _, _, rkcv, rkc0, rkc1, rkc2, rkfv, rkf0, rkf1, rkf2 = (
            geometry_jet_inline(
                i,
                dep_v - hh,
                z,
                dep_1 - one,
                dep_2,
                dep_v,
                z,
                dep_1,
                dep_2,
                w,
                hh,
                length,
                A_i,
                tab_ptr,
                bed_ptr,
                reverse,
                levels,
            )
        )
        _, _, ex_rkc, ex_rkf, _, _ = stage_pair_inline(
            i, -hh, dep_v, length, w, hh, A_i, tab_ptr, bed_ptr, reverse, levels
        )
        c_kcv = tl.where(reverse, rkcv, c_kcv)
        c_kc0 = tl.where(reverse, rkc0, c_kc0)
        c_kc1 = tl.where(reverse, rkc1, c_kc1)
        c_kc2 = tl.where(reverse, rkc2, c_kc2)
        c_kfv = tl.where(reverse, rkfv, c_kfv)
        c_kf0 = tl.where(reverse, rkf0, c_kf0)
        c_kf1 = tl.where(reverse, rkf1, c_kf1)
        c_kf2 = tl.where(reverse, rkf2, c_kf2)
        ex_c_kc = tl.where(reverse, ex_rkc, ex_c_kc)
        ex_c_kf = tl.where(reverse, ex_rkf, ex_c_kf)

    K = tl.div_rn(ex_c_kc, nc_i) + tl.div_rn(ex_c_kf, nf_i)
    Kj0 = tl.div_rn(c_kc0, nc_i) + tl.div_rn(c_kf0, nf_i)
    Kj1 = tl.div_rn(c_kc1, nc_i) + tl.div_rn(c_kf1, nf_i)
    Kj2 = tl.div_rn(c_kc2, nc_i) + tl.div_rn(c_kf2, nf_i)
    Kj3 = tl.div_rn(-tl.div_rn(c_kcv, nc_i), nc_i)
    Kj4 = tl.div_rn(-tl.div_rn(c_kfv, nf_i), nf_i)

    actualdepth = tl.maximum((ri + hh) + xi, 0.0)
    p5 = length * actualdepth
    p6 = length * w
    p7 = tl.where(reverse, one, z)

    active = (Ae > 1e-8) & (K > 1e-12) & (edge | (Q_i > 0.0))
    Ae_safe = tl.where(active, Ae, one)
    K_safe = tl.where(active, K, one)
    b = tl.div_rn(9.8 * dt * Ae_safe, K_safe * K_safe)
    friction = b * Q_i * tl.abs(Q_i)
    den = 1.0 + 2.0 * b * tl.abs(Q_i)
    dq = Q_i - Q0_i
    p0 = tl.where(
        active,
        tl.div_rn(
            tl.div_rn(dq * Aj0, Ae_safe) + tl.div_rn(2.0 * friction * Kj0, K_safe), den
        ),
        z,
    )
    p1 = tl.where(
        active,
        tl.div_rn(
            tl.div_rn(dq * Aj1, Ae_safe) + tl.div_rn(2.0 * friction * Kj1, K_safe), den
        ),
        z,
    )
    p2 = tl.where(
        active,
        tl.div_rn(
            tl.div_rn(dq * Aj2, Ae_safe) + tl.div_rn(2.0 * friction * Kj2, K_safe), den
        ),
        z,
    )
    p3 = tl.where(active, tl.div_rn(tl.div_rn(2.0 * friction * Kj3, K_safe), den), z)
    p4 = tl.where(active, tl.div_rn(tl.div_rn(2.0 * friction * Kj4, K_safe), den), z)

    dry_bound = (~(((ri + hh) + xi) > 0.0)) & (R_i > 0.0)
    j12 = tl.where(
        active,
        tl.div_rn(
            one,
            1.0 + 2.0 * tl.div_rn(9.8 * dt * Ae_safe, K_safe * K_safe) * tl.abs(Q_i),
        ),
        z,
    )

    base = i.to(tl.int64) * 17
    tl.store(saved_ptr + base + 0, Q_i, mask=mask)
    tl.store(saved_ptr + base + 1, S_i, mask=mask)
    tl.store(saved_ptr + base + 2, qi_i, mask=mask)
    tl.store(saved_ptr + base + 3, qd_i, mask=mask)
    tl.store(saved_ptr + base + 4, area_i, mask=mask)
    tl.store(saved_ptr + base + 5, tl.where(dry_bound, one, z), mask=mask)
    tl.store(saved_ptr + base + 6, ri + xi, mask=mask)
    tl.store(saved_ptr + base + 7, p5, mask=mask)
    tl.store(saved_ptr + base + 8, p6, mask=mask)
    tl.store(saved_ptr + base + 9, p0, mask=mask)
    tl.store(saved_ptr + base + 10, p1, mask=mask)
    tl.store(saved_ptr + base + 11, p2, mask=mask)
    tl.store(saved_ptr + base + 12, j12, mask=mask)
    tl.store(saved_ptr + base + 13, p3, mask=mask)
    tl.store(saved_ptr + base + 14, p4, mask=mask)
    tl.store(saved_ptr + base + 15, p7, mask=mask)
    tl.store(saved_ptr + base + 16, dt + z, mask=mask)


# --------------------------------------------------------------------------- #
# Newton phase / line-search / damped controller (works for scalar OR vector!)
# --------------------------------------------------------------------------- #


@triton.jit
def start_phase_state_inline(dt0, phase):
    i0 = tl.zeros_like(phase)
    mode = MODE_EVAL + i0
    st = i0
    steps = i0
    fevals = i0
    totalbt = i0
    reset = tl.where(phase <= 0, 1, tl.where(phase >= 13, 2, 0))
    use_unit = (phase < 0) | (phase >= 13)
    dt = dt0 * tl.where(use_unit, 1.0, continuation_beta_inline(phase))
    return mode, phase, st, steps, fevals, totalbt, reset, dt


@triton.jit
def newton_control_step_inline(
    # State integers (TI - 4 outputs: commit, save, reset, solve are freshly computed)
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
    # State floats (TF)
    dt,
    lam,
    merit,
    best,
    initial,
    sigma,
    # Inputs
    mt,
    line,
    dt0,
    tol,
    maxit,
    # Fixed-weights mode (False for forest; True bypasses phases 21/22)
    FIXED_WEIGHTS: tl.constexpr,
):
    """Advance one tree's Newton controller by one round (scalar or 1D vector).

    Returns updated state plus (commit, save, reset, solve, finished_now).
    """
    i0 = tl.zeros_like(mode)
    f0 = tl.zeros_like(dt)
    commit = i0
    save = i0
    reset = i0
    solve = i0
    finished_now = i0

    active = mode != MODE_DONE

    # Case 1: mode == FINAL (restored best surface of phase 20)
    is_final = active & (mode == MODE_FINAL)
    merit = tl.where(is_final, mt, merit)
    st = tl.where(is_final, tl.where(mt < tol, i0, best_st), st)
    mode = tl.where(is_final, MODE_DONE + i0, mode)
    finished_now = tl.where(is_final, 1 + i0, finished_now)

    # Case 2: damped Newton (when not FIXED_WEIGHTS, phase == DAMPED_PHASE and mode != RESTORE)
    if not FIXED_WEIGHTS:
        is_damped = (
            active & (~is_final) & (phase == DAMPED_PHASE) & (mode != MODE_RESTORE)
        )
    else:
        is_damped = active & False

    d_finish = is_damped & False
    if not FIXED_WEIGHTS:
        fevals = tl.where(is_damped, fevals + 1, fevals)
        d_trial = is_damped & (mode == MODE_TRIAL)
        d_fin_mt = isfinite_f32_inline(mt)
        d_bad = d_trial & (~d_fin_mt)
        d_good = d_trial & d_fin_mt
        # Non-finite trial: increase damping, retry EVAL (or RESTORE after 40 retries)
        sigma = tl.where(d_bad, tl.maximum(sigma * 10.0, 1e-3), sigma)
        new_tbt = totalbt + 1
        totalbt = tl.where(d_bad, new_tbt, totalbt)
        d_giveup = d_bad & (new_tbt >= 40)
        st = tl.where(d_giveup, 2 + i0, st)
        mode = tl.where(
            d_bad, tl.where(d_giveup, MODE_RESTORE + i0, MODE_EVAL + i0), mode
        )
        # Finite trial: always commit, update sigma from fixed-weight ratio
        old_d = merit
        commit = tl.where(d_good, 1 + i0, commit)
        merit = tl.where(d_good, mt, merit)
        steps = tl.where(d_good, steps + 1, steps)
        dsteps = tl.where(d_good, dsteps + 1, dsteps)
        old_safe = tl.where(d_good, old_d, 1.0)
        sig_new = sigma * tl.minimum(tl.div_rn(line, old_safe), DAMPED_GROWTH_C)
        sig_new = tl.where((sig_new == 0.0) & (line >= old_d), 1e-3, sig_new)
        sig_new = tl.where(sig_new < 1e-7, 0.0, sig_new)
        sigma = tl.where(d_good, sig_new, sigma)
        # Non-trial (EVAL) in damped phase
        d_eval = is_damped & (~d_trial)
        merit = tl.where(d_eval, mt, merit)
        # Continue decision for non-bad damped trees
        d_check = is_damped & (~d_bad)
        d_conv = d_check & (~(mt >= tol))
        d_maxed = d_check & (~d_conv) & (steps >= DAMPED_STEPS_C)
        d_finish = d_conv | d_maxed
        d_step = d_check & (~d_finish)
        st = tl.where(d_conv, i0, tl.where(d_maxed, 1 + i0, st))
        sigma = tl.where(d_finish, f0, sigma)
        solve = tl.where(d_step, 1 + i0, solve)
        mode = tl.where(d_step, MODE_TRIAL + i0, mode)
        lam = tl.where(d_step, 1.0 + f0, lam)

    # Case 3: standard Newton / line search / restore
    is_std = active & (~is_final) & (~is_damped)
    s_eval = is_std & (mode == MODE_EVAL)
    s_trial = is_std & (mode == MODE_TRIAL)
    s_restore = is_std & (~s_eval) & (~s_trial)

    # EVAL
    merit = tl.where(s_eval, mt, merit)
    fevals = tl.where(s_eval, fevals + 1, fevals)
    newton_done = s_eval & (~(mt >= tol))

    # TRIAL
    fevals = tl.where(s_trial, fevals + 1, fevals)
    totalbt = tl.where(s_trial, totalbt + 1, totalbt)
    old_s = merit
    if FIXED_WEIGHTS:
        check_val = line
    else:
        check_val = tl.where(phase == RESTART_PHASE, line, mt)
    accepted = (
        s_trial
        & isfinite_f32_inline(mt)
        & ((mt < tol) | (check_val <= (1.0 - 0.01 * lam) * old_s))
    )
    rejected = s_trial & (~accepted)

    commit = tl.where(accepted, 1 + i0, commit)
    merit = tl.where(accepted, mt, merit)
    steps = tl.where(accepted, steps + 1, steps)
    newton_done = newton_done | (accepted & (~((steps < maxit) & (mt >= tol))))

    # Rejected trial: halve lambda, retry or RESTORE
    lam = tl.where(rejected, lam * 0.5, lam)
    new_bt = bt + 1
    bt = tl.where(rejected, new_bt, bt)
    bt_fail = rejected & (~(new_bt < LINE_SEARCH))
    mode = tl.where(bt_fail, MODE_RESTORE + i0, mode)

    # RESTORE
    merit = tl.where(s_restore, mt, merit)
    fevals = tl.where(s_restore, fevals + 2, fevals)
    st = tl.where(s_restore, tl.where(mt < tol, i0, 2 + i0), st)

    # Need another linear solve (non-rejected, non-restore, not done)
    s_cont = (s_eval | accepted) & (~newton_done)
    solve = tl.where(s_cont, 1 + i0, solve)
    mode = tl.where(s_cont, MODE_TRIAL + i0, mode)
    lam = tl.where(s_cont, 1.0 + f0, lam)
    bt = tl.where(s_cont, i0, bt)

    # Standard phase done (from EVAL or accepted TRIAL)
    s_done = (s_eval | accepted) & newton_done
    fevals = tl.where(s_done, fevals + 1, fevals)
    st = tl.where(s_done, tl.where(merit < tol, i0, 1 + i0), st)

    # finish_phase for all trees finishing a phase (d_finish, s_restore, s_done)
    fp = d_finish | s_restore | s_done
    sum_g = tl.where(fp, sum_g + steps, sum_g)
    sum_fe = tl.where(fp, sum_fe + fevals, sum_fe)
    initial = tl.where(fp & (phase == -1), merit, initial)
    sigma = tl.where(fp & (phase == DAMPED_PHASE), f0, sigma)

    # Phase transitions inside finish_phase (mutually exclusive)
    if not FIXED_WEIGHTS:
        go_restart = fp & (st != 0) & (phase == -1)
        go_damped = fp & (~go_restart) & (st != 0) & (phase == RESTART_PHASE)
        go_cont0 = (
            fp & (~go_restart) & (~go_damped) & (st != 0) & (phase == DAMPED_PHASE)
        )
        go_done_forest = (
            fp
            & (~go_restart)
            & (~go_damped)
            & (~go_cont0)
            & ((phase == RESTART_PHASE) | (phase == DAMPED_PHASE) | (phase == -1))
        )
    else:
        go_restart = fp & False
        go_damped = fp & False
        go_cont0 = fp & False
        go_done_forest = fp & False

    rem = fp & (~go_restart) & (~go_damped) & (~go_cont0) & (~go_done_forest)
    go_next_cont = rem & (((phase == -1) & (st != 0)) | ((phase >= 0) & (phase < 12)))
    go_p13 = rem & (~go_next_cont) & (phase == 12) & (st != 0)
    in_p13p = rem & (~go_next_cont) & (~go_p13) & (phase >= 13)
    better = in_p13p & (merit < best)
    best = tl.where(better, merit, best)
    best_st = tl.where(better, st, best_st)
    save = tl.where(better, save | 2, save)
    go_p13_next = in_p13p & (st != 0) & (phase < 20)
    go_p20_final = in_p13p & (st != 0) & (~(phase < 20))
    go_done_other = rem & (~go_next_cont) & (~go_p13) & (~go_p13_next) & (~go_p20_final)

    # Apply start_phase transitions
    sp_any = go_restart | go_cont0 | go_next_cont | go_p13 | go_p13_next
    target_phase = tl.where(
        go_restart,
        -1 + i0,
        tl.where(go_cont0, i0, tl.where(go_p13, 13 + i0, phase + 1)),
    )
    sp_mode, sp_phase, sp_st, sp_steps, sp_fevals, sp_tbt, sp_reset, sp_dt = (
        start_phase_state_inline(dt0, target_phase)
    )
    # go_p13 saves trace & best before starting phase 13
    best = tl.where(go_p13, merit, best)
    best_st = tl.where(go_p13, st, best_st)
    save = tl.where(go_p13, save | 1, save)

    mode = tl.where(sp_any, sp_mode, mode)
    phase = tl.where(sp_any, tl.where(go_restart, RESTART_PHASE + i0, sp_phase), phase)
    st = tl.where(sp_any, sp_st, st)
    steps = tl.where(sp_any, sp_steps, steps)
    fevals = tl.where(sp_any, sp_fevals, fevals)
    totalbt = tl.where(sp_any, sp_tbt, totalbt)
    reset = tl.where(sp_any, sp_reset, reset)
    dt = tl.where(sp_any, sp_dt, dt)
    bt = tl.where(go_restart, i0, bt)

    # Apply start_damped transition
    if not FIXED_WEIGHTS:
        phase = tl.where(go_damped, DAMPED_PHASE + i0, phase)
        mode = tl.where(go_damped, MODE_EVAL + i0, mode)
        st = tl.where(go_damped, i0, st)
        steps = tl.where(go_damped, i0, steps)
        fevals = tl.where(go_damped, i0, fevals)
        totalbt = tl.where(go_damped, i0, totalbt)
        bt = tl.where(go_damped, i0, bt)
        sigma = tl.where(go_damped, DAMPED_SIGMA_C + f0, sigma)

    # Apply phase-20 restore best
    save = tl.where(go_p20_final, save | 4, save)
    mode = tl.where(go_p20_final, MODE_FINAL + i0, mode)

    # Apply finish_tree
    go_done = go_done_forest | go_done_other
    mode = tl.where(go_done, MODE_DONE + i0, mode)
    finished_now = tl.where(go_done, 1 + i0, finished_now)

    return (
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
    )
