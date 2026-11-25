// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// SPDX-License-Identifier: Apache-2.0
// Apple Metal (MSL) port of HydroNet implicit local-inertia forest solver,
// converged snapshot rebuild, and exact transposed-Jacobian adjoint.

// ------------------------------------------------------------------------
// Pure-MSL math & error-free transform helpers (no double, no CUDA-isms).

static inline bool hf_isfinite(float x) {
    return (as_type<uint>(x) & 0x7f800000u) != 0x7f800000u;
}

static inline float hf_inf() {
    return as_type<float>(0x7f800000u);
}

static inline int hf_iabs(int x) {
    return x < 0 ? -x : x;
}

static inline float hf_add_rn(float a, float b) {
#if HYDROFORGE_FAST_MATH
    return precise::fma(a, 1.0f, b);
#else
    return a + b;
#endif
}

static inline float hf_sub_rn(float a, float b) {
#if HYDROFORGE_FAST_MATH
    return precise::fma(a, 1.0f, -b);
#else
    return a - b;
#endif
}

static inline float hf_div(float a, float b) {
#if HYDROFORGE_FAST_MATH
    return fast::divide(a, b);
#else
    return a / b;
#endif
}

static inline float hf_sqrt(float x) {
#if HYDROFORGE_FAST_MATH
    return fast::sqrt(x);
#else
    return sqrt(x);
#endif
}

static inline float hf_rsqrt(float x) {
#if HYDROFORGE_FAST_MATH
    return fast::rsqrt(x);
#else
    return rsqrt(x);
#endif
}

static inline float hf_cbrt_nonneg(float x) {
    if (!(x > 0.0f)) return 0.0f;
#if HYDROFORGE_FAST_MATH
    return fast::powr(x, 1.0f / 3.0f);
#else
    float y = powr(x, 1.0f / 3.0f);
    // Overflow-safe Newton refinement to <= 1 ULP accuracy.
    return (2.0f * y + (x / y) / y) * (1.0f / 3.0f);
#endif
}

static inline float p23(float a) {
    float x = a * a;
    if (!(x > 0.0f)) return 0.0f;
#if HYDROFORGE_FAST_MATH
    return fast::powr(x, 1.0f / 3.0f);
#else
    return hf_cbrt_nonneg(x);
#endif
}

struct State {
    float S;
    float area;
    float kc;
    float kf;
    float dkc;
    float dkf;
};

static inline void channel_conv(
    float w, float H, float d, thread float& k, thread float& dk
) {
    if (!(d > 0.0f)) {
        k = 0.0f;
        dk = 0.0f;
        return;
    }
    float a = w * d;
    float p = w + 2.0f * min(d, H);
    float ap = hf_div(a, p);
    float r = p23(ap);
    k = a * r;
    dk = r * (1.6666667f * w - (d < H ? 1.3333334f * ap : 0.0f));
}

static inline void flood_bands(
    int i,
    float h,
    float dw,
    device const float* tab,
    device const float* bed,
    int levels,
    thread float& k,
    thread float& dk
) {
    k = 0.0f;
    dk = 0.0f;
    float t0 = 0.0f;
    float b0 = 0.0f;
    for (int b = 0; b < levels; ++b) {
        float t = tab[(long)i * levels + b];
        float b1 = bed[(long)i * levels + b];
        float seg = b1 - b0;
        if (!(h > t0)) break;
        float a;
        float p;
        float da;
        float dp;
        if (!(h < t)) {
            a = dw * ((h - t0) - 0.5f * (t - t0));
            p = seg;
            da = dw;
            dp = 0.0f;
            if (b == levels - 1) {
                p += h - t;
                dp = 1.0f;
            }
        } else {
            float x = h - t0;
            float inv_dt = hf_div(1.0f, t - t0);
            float run = x * (dw * inv_dt);
            a = 0.5f * run * x;
            p = x * (seg * inv_dt);
            da = run;
            dp = seg * inv_dt;
        }
        if (a > 0.0f) {
            float ap = hf_div(a, p);
            float r = p23(ap);
            k += a * r;
            dk += r * (1.6666667f * da - 0.6666667f * ap * dp);
        }
        t0 = t;
        b0 = b1;
    }
}

static inline State stage_eval(
    int i,
    float h,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* bed,
    int levels
) {
    State s;
    float l = L[i];
    float w = W[i];
    float hh = H[i];
    float d = max(h + hh, 0.0f);
    s.kf = 0.0f;
    s.dkf = 0.0f;
    channel_conv(w, hh, d, s.kc, s.dkc);
    if (!(h > 0.0f)) {
        s.S = l * w * d;
        s.area = l * w;
        return s;
    }
    float dw = A[i] / l / (float)levels;
    float ph = 0.0f;
    float pw = w;
    float acc = 0.0f;
    float nh = 0.0f;
    int lev = 0;
    for (int k = 0; k < levels; ++k) {
        nh = tab[(long)i * levels + k];
        float nw = w + (float)(k + 1) * dw;
        if (!(h > nh)) break;
        acc += 0.5f * (pw + nw) * (nh - ph);
        ph = nh;
        pw = nw;
        ++lev;
    }
    float grad = lev < levels ? (nh - ph) / dw : 0.0f;
    float x = h - ph;
    float run = grad > 0.0f ? x / grad : 0.0f;
    s.S = l * (w * hh + acc + pw * x + 0.5f * run * x);
    s.area = l * (pw + run);
    flood_bands(i, h, dw, tab, bed, levels, s.kf, s.dkf);
    return s;
}

static inline State stage_pair(
    int i,
    float ref,
    float x,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* bed,
    int levels
) {
    float h = ref + x;
    if (h > 0.0f) return stage_eval(i, h, L, W, H, A, tab, bed, levels);
    State s;
    float d = max((ref + H[i]) + x, 0.0f);
    s.S = L[i] * W[i] * d;
    s.area = L[i] * W[i];
    s.kf = 0.0f;
    s.dkf = 0.0f;
    channel_conv(W[i], H[i], d, s.kc, s.dkc);
    return s;
}

static inline float invert_storage(
    int i,
    float S,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    int levels
) {
    S = max(S, 0.0f);
    float l = L[i];
    float w = W[i];
    float hh = H[i];
    float acc = l * w * hh;
    if (S <= acc) return S / (l * w) - hh;
    float dw = A[i] / l / (float)levels;
    float ph = 0.0f;
    float pw = w;
    float nh = 0.0f;
    int lev = 0;
    for (int k = 0; k < levels; ++k) {
        nh = tab[(long)i * levels + k];
        float nw = w + (float)(k + 1) * dw;
        float next = acc + l * 0.5f * (pw + nw) * (nh - ph);
        if (!(S > next)) break;
        acc = next;
        ph = nh;
        pw = nw;
        ++lev;
    }
    float grad = lev < levels ? (nh - ph) / dw : 0.0f;
    float extra = max(S - acc, 0.0f);
    if (grad > 0.0f) {
        return ph + (2.0f * extra / l) / (hf_sqrt(pw * pw + 2.0f * extra / (grad * l)) + pw);
    }
    return ph + extra / (pw * l);
}

static inline float phi_reg(float j, float e) {
    return j * hf_rsqrt(hf_sqrt(j * j + e * e));
}

static inline float dphi_reg(float j, float e) {
    float a = j * j + e * e;
    return (0.5f * j * j + e * e) / (a * hf_sqrt(hf_sqrt(a)));
}

// Error-free FP32 transforms.
struct Pair {
    float hi;
    float lo;
};

static inline Pair pair_sum(float a, float b) {
    float s = hf_add_rn(a, b);
    float v = hf_sub_rn(s, a);
    return {s, hf_add_rn(hf_sub_rn(a, hf_sub_rn(s, v)), hf_sub_rn(b, v))};
}

static inline Pair pair_diff(float a, float b) {
    return pair_sum(a, -b);
}

static inline Pair pair_add(Pair a, Pair b) {
    Pair s = pair_sum(a.hi, b.hi);
    float e = hf_add_rn(s.lo, hf_add_rn(a.lo, b.lo));
    return pair_sum(s.hi, e);
}

static inline float head_pair(
    float ei, float ed, float ri, float rd, float xi, float xd
) {
    Pair p = pair_add(pair_diff(ei, ed), pair_diff(ri, rd));
    p = pair_add(p, pair_diff(xi, xd));
    return hf_add_rn(p.hi, p.lo);
}

static inline void trial_pair(
    float ref,
    float x,
    float step,
    float H,
    thread float& rnew,
    thread float& xnew
) {
    Pair p = pair_add(Pair{ref, x}, Pair{step, 0.0f});
    if (hf_add_rn(hf_add_rn(p.hi, H), p.lo) < 0.0f) {
        p.hi = -H;
        p.lo = 0.0f;
    }
    rnew = p.hi;
    xnew = p.lo;
}

constant float CONTINUATION_BETA[13] = {
    1e-6f, 3e-6f, 1e-5f, 3e-5f, 1e-4f, 3e-4f, 0.001f, 0.003f,
    0.01f, 0.03f, 0.1f, 0.3f, 1.0f
};
constant float RESTART_OFFSETS[8] = {
    -0.01f, 0.01f, -0.05f, 0.05f, -0.2f, 0.2f, -0.5f, 0.5f
};

static inline void momentum_flux(
    float q0,
    float A,
    float Ai,
    float Ad,
    float K,
    float Ki,
    float Kd,
    float j,
    float ji,
    float jd,
    float dt,
    float eps,
    thread float& q,
    thread float& qi,
    thread float& qd
) {
    q = 0.0f;
    qi = 0.0f;
    qd = 0.0f;
    if (!(A > 1e-8f) || !(K > 1e-12f)) return;
    float f = phi_reg(j, eps);
    float fp = dphi_reg(j, eps);
    float je = f * fabs(f);
    float dje = 2.0f * fabs(f) * fp;
    float a = 9.8f * dt * A;
    float b = a / (K * K);
    float u = q0 + a * je;
    float den = hf_sqrt(1.0f + 4.0f * b * fabs(u));
    q = 2.0f * u / (1.0f + den);
    float fric = b * q * fabs(q);
    qi = ((q - q0) * (Ai / A) + 2.0f * fric * (Ki / K) + a * dje * ji) / den;
    qd = ((q - q0) * (Ad / A) + 2.0f * fric * (Kd / K) + a * dje * jd) / den;
}

// Five local parameter tangents (W_i, H_i, H_down, n_channel, n_flood).
struct Jet {
    float v;
    float d[5];
};

static inline Jet jet_const(float a) {
    Jet r;
    r.v = a;
    for (int k = 0; k < 5; ++k) r.d[k] = 0.0f;
    return r;
}

static inline Jet jet_var(float v, int k) {
    Jet a = jet_const(v);
    a.d[k] = 1.0f;
    return a;
}

static inline Jet jet_add(Jet a, Jet b) {
    Jet c;
    c.v = a.v + b.v;
    for (int k = 0; k < 5; ++k) c.d[k] = a.d[k] + b.d[k];
    return c;
}

static inline Jet jet_add_f(Jet a, float b) {
    a.v += b;
    return a;
}

static inline Jet jet_sub(Jet a, Jet b) {
    Jet c;
    c.v = a.v - b.v;
    for (int k = 0; k < 5; ++k) c.d[k] = a.d[k] - b.d[k];
    return c;
}

static inline Jet jet_sub_f(Jet a, float b) {
    a.v -= b;
    return a;
}

static inline Jet jet_mul(Jet a, Jet b) {
    Jet c;
    c.v = a.v * b.v;
    for (int k = 0; k < 5; ++k) c.d[k] = a.d[k] * b.v + a.v * b.d[k];
    return c;
}

static inline Jet jet_mul_f(Jet a, float b) {
    Jet c;
    c.v = a.v * b;
    for (int k = 0; k < 5; ++k) c.d[k] = a.d[k] * b;
    return c;
}

static inline Jet jet_div(Jet a, Jet b) {
    Jet c;
    c.v = a.v / b.v;
    for (int k = 0; k < 5; ++k) c.d[k] = (a.d[k] - c.v * b.d[k]) / b.v;
    return c;
}

static inline Jet jet_div_f(Jet a, float b) {
    Jet c;
    c.v = a.v / b;
    for (int k = 0; k < 5; ++k) c.d[k] = a.d[k] / b;
    return c;
}

static inline Jet jet_pos(Jet a) {
    return a.v > 0.0f ? a : jet_const(0.0f);
}

static inline Jet jet_min(Jet a, Jet b) {
    return a.v < b.v ? a : b;
}

static inline Jet jet_pow23(Jet a) {
    Jet c = jet_const(p23(a.v));
    if (a.v > 0.0f) {
        float factor = (2.0f / 3.0f) * c.v / a.v;
        for (int k = 0; k < 5; ++k) c.d[k] = factor * a.d[k];
    }
    return c;
}

struct JState {
    Jet S;
    Jet kc;
    Jet kf;
};

static inline JState geometry_jet(
    int i,
    Jet fill,
    Jet depth,
    Jet w,
    Jet H,
    device const float* L,
    device const float* A,
    device const float* tab,
    device const float* bed,
    int levels
) {
    JState s;
    s.S = jet_const(0.0f);
    s.kc = jet_const(0.0f);
    s.kf = jet_const(0.0f);
    depth = jet_pos(depth);
    if (depth.v > 0.0f) {
        Jet ac = jet_mul(w, depth);
        Jet pc = jet_add(w, jet_mul_f(jet_min(depth, H), 2.0f));
        s.kc = jet_mul(ac, jet_pow23(jet_div(ac, pc)));
    }
    if (!(fill.v > 0.0f)) {
        s.S = jet_mul_f(jet_mul(w, depth), L[i]);
        return s;
    }
    float dw = A[i] / L[i] / (float)levels;
    float ph = 0.0f;
    float pw = 0.0f;
    float acc = 0.0f;
    float nh = 0.0f;
    int lev = 0;
    for (int k = 0; k < levels; ++k) {
        nh = tab[(long)i * levels + k];
        float nw = (float)(k + 1) * dw;
        if (!(fill.v > nh)) break;
        acc += 0.5f * (pw + nw) * (nh - ph);
        ph = nh;
        pw = nw;
        ++lev;
    }
    float slope = lev < levels ? (nh - ph) / dw : 0.0f;
    Jet x = jet_sub_f(fill, ph);
    Jet run = slope > 0.0f ? jet_div_f(x, slope) : jet_const(0.0f);
    Jet fp_area = jet_pos(
        jet_add_f(jet_add(jet_mul_f(x, pw), jet_mul_f(jet_mul(run, x), 0.5f)), acc)
    );
    s.S = jet_mul_f(jet_add(jet_mul(w, jet_add(H, fill)), fp_area), L[i]);
    float t0 = 0.0f;
    float b0 = 0.0f;
    for (int b = 0; b < levels; ++b) {
        float t = tab[(long)i * levels + b];
        float b1 = bed[(long)i * levels + b];
        float seg = b1 - b0;
        if (!(fill.v > t0)) break;
        Jet a;
        Jet p;
        if (!(fill.v < t)) {
            a = jet_mul_f(jet_sub_f(jet_sub_f(fill, t0), 0.5f * (t - t0)), dw);
            p = jet_const(seg);
            if (b == levels - 1) p = jet_add(p, jet_sub_f(fill, t));
        } else {
            Jet xb = jet_sub_f(fill, t0);
            a = jet_mul_f(jet_mul(xb, xb), 0.5f * (dw / (t - t0)));
            p = jet_mul_f(xb, seg / (t - t0));
        }
        if (a.v > 0.0f) s.kf = jet_add(s.kf, jet_mul(a, jet_pow23(jet_div(a, p))));
        t0 = t;
        b0 = b1;
    }
    return s;
}

static inline void parameter_partials(
    int i,
    device const float* ref,
    device const float* x,
    device const float* Q0,
    device const float* nc,
    device const float* nf,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* bed,
    device const float* elev,
    device const float* dist,
    device const int* down,
    device const uchar* edges,
    float eps,
    float dt,
    int levels,
    float q,
    thread float* out
) {
    bool edge = edges[i] != 0;
    int d = edge ? down[i] : i;
    float head = edge ? head_pair(elev[i], elev[d], ref[i], ref[d], x[i], x[d]) : ref[i] + x[i];
    float j = edge ? head / dist[i] : (ref[i] + x[i]) / 10000.0f;
    Jet w = jet_var(W[i], 0);
    Jet Hj = jet_var(H[i], 1);
    Jet Hd = jet_var(H[d], 2);
    Jet ownfill = jet_add_f(jet_const(ref[i]), x[i]);
    Jet owndepth = jet_add_f(jet_add_f(Hj, ref[i]), x[i]);
    JState accel = geometry_jet(i, ownfill, owndepth, w, Hj, L, A, tab, bed, levels);
    State exact_accel = stage_pair(i, ref[i], x[i], L, W, H, A, tab, bed, levels);
    if (edge && head < 0.0f) {
        Jet da = jet_pos(jet_sub_f(owndepth, head));
        accel = geometry_jet(i, jet_sub(da, Hj), da, w, Hj, L, A, tab, bed, levels);
        exact_accel = stage_pair(i, -H[i], da.v, L, W, H, A, tab, bed, levels);
    }
    float Ae = exact_accel.S / L[i];
    float f = phi_reg(j, eps);
    Jet Aj = jet_div_f(accel.S, L[i]);
    bool reverse = edge && (Q0[i] + 9.8f * dt * Ae * f * fabs(f) < 0.0f);
    Jet nj = jet_var(nc[i], 3);
    Jet fj = jet_var(nf[i], 4);
    JState c = geometry_jet(i, ownfill, owndepth, w, Hj, L, A, tab, bed, levels);
    State exact = stage_pair(i, ref[i], x[i], L, W, H, A, tab, bed, levels);
    if (reverse) {
        Jet beddiff = jet_add_f(jet_sub(Hj, Hd), elev[d] - elev[i]);
        Jet depth = jet_pos(
            jet_add(jet_add_f(jet_add_f(Hd, ref[d]), x[d]), jet_min(jet_const(0.0f), beddiff))
        );
        c = geometry_jet(i, jet_sub(depth, Hj), depth, w, Hj, L, A, tab, bed, levels);
        exact = stage_pair(i, -H[i], depth.v, L, W, H, A, tab, bed, levels);
    }
    Jet Kj = jet_add(jet_div(c.kc, nj), jet_div(c.kf, fj));
    float K = exact.kc / nc[i] + exact.kf / nf[i];
    for (int k = 0; k < 8; ++k) out[k] = 0.0f;
    float actualdepth = max((ref[i] + H[i]) + x[i], 0.0f);
    out[5] = L[i] * actualdepth;
    out[6] = L[i] * W[i];
    out[7] = reverse ? 1.0f : 0.0f;
    if (!(Ae > 1e-8f) || !(K > 1e-12f) || (!edge && !(q > 0.0f))) return;
    float b = 9.8f * dt * Ae / (K * K);
    float friction = b * q * fabs(q);
    float den = 1.0f + 2.0f * b * fabs(q);
    for (int k = 0; k < 5; ++k) {
        out[k] = ((q - Q0[i]) * Aj.d[k] / Ae + 2.0f * friction * Kj.d[k] / K) / den;
    }
}

// ------------------------------------------------------------------------
// Solver control enums, error reporting, and shared Net struct.

constant constexpr int LINE_SEARCH = 16;
constant constexpr int MAX_ROUNDS = (1 << 22);
constant constexpr int MODE_DONE = 0;
constant constexpr int MODE_EVAL = 1;
constant constexpr int MODE_TRIAL = 2;
constant constexpr int MODE_RESTORE = 3;
constant constexpr int MODE_FINAL = 4;

constant constexpr int TI_MODE = 0;
constant constexpr int TI_PHASE = 1;
constant constexpr int TI_STEPS = 2;
constant constexpr int TI_BT = 3;
constant constexpr int TI_ST = 4;
constant constexpr int TI_FEVALS = 5;
constant constexpr int TI_TOTALBT = 6;
constant constexpr int TI_SUM_G = 7;
constant constexpr int TI_SUM_FE = 8;
constant constexpr int TI_DSTEPS = 9;
constant constexpr int TI_BEST_ST = 10;
constant constexpr int TI_SOLVE = 11;
constant constexpr int TI_COMMIT = 12;
constant constexpr int TI_SAVE = 13;
constant constexpr int TI_RESET = 14;
constant constexpr int TI_STRIDE = TREE_INTEGERS;

constant constexpr int TF_DT = 0;
constant constexpr int TF_LAM = 1;
constant constexpr int TF_MERIT = 2;
constant constexpr int TF_BEST = 3;
constant constexpr int TF_INITIAL = 4;
constant constexpr int TF_SIGMA = 5;
constant constexpr int TF_STRIDE = TREE_SCALARS;

constant constexpr int RESTART_PHASE = 21;
constant constexpr int DAMPED_PHASE = 22;
constant constexpr int CTL_ERROR = 4;

static inline void report_err(device atomic_int* status, int code, int index) {
    if (atomic_fetch_add_explicit(&status[0], 1, memory_order_relaxed) == 0) {
        atomic_store_explicit(&status[1], code, memory_order_relaxed);
        atomic_store_explicit(&status[2], index, memory_order_relaxed);
    }
}

struct Net {
    device const float* S0;
    device const float* runoff;
    device const float* Q0;
    device const float* nc;
    device const float* nf;
    device const float* L;
    device const float* W;
    device const float* H;
    device const float* A;
    device const float* tab;
    device const float* bed;
    device const float* elev;
    device const float* dist;
    device const float* dt0;
    device const int* down;
    device const int* cp;
    device const int* kids;
    device const int* tree;
    device const int* pnodes;
    device const int* pptr;
    device const int* tunit;
    device const int* loff;
    device const int* tlev;
    device const uchar* edge;
    device float* h;
    device float* ht;
    device float* ref;
    device float* tref;
    device float* S;
    device float* area;
    device float* kc;
    device float* kf;
    device float* dkc;
    device float* dkf;
    device float* Q;
    device float* qi;
    device float* qd;
    device float* R;
    device float* wb;
    device float* wa;
    device float* ap;
    device float* rp;
    device float* dh;
    device float* b;
    device float* trace;
    device float* tf;
    device float* meta;
    device int* ti;
    device atomic_int* status;
    device atomic_uint* tm;
    device atomic_uint* tline;
    float eps;
    float tol;
    int n;
    int levels;
    int maxit;
};

static inline void eval_stage_unit(thread const Net& z, int i) {
    State s = stage_pair(
        i, z.tref[i], z.ht[i], z.L, z.W, z.H, z.A, z.tab, z.bed, z.levels
    );
    z.S[i] = s.S;
    z.area[i] = s.area;
    z.kc[i] = s.kc;
    z.kf[i] = s.kf;
    z.dkc[i] = s.dkc;
    z.dkf[i] = s.dkf;
}

static inline void eval_flux_fn(
    int i,
    float dt,
    float eps,
    int levels,
    device const float* Q0,
    device const float* nc,
    device const float* nf,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* bed,
    device const float* elev,
    device const float* dist,
    device const int* down,
    device const uchar* edges,
    device const float* tref,
    device const float* ht,
    device const float* S_arr,
    device const float* area_arr,
    device const float* kc_arr,
    device const float* kf_arr,
    device const float* dkc_arr,
    device const float* dkf_arr,
    device float* Q_out,
    device float* qi_out,
    device float* qd_out
) {
    bool edge = edges[i] != 0;
    int d = edge ? down[i] : i;
    float oi = tref[i];
    float xi = ht[i];
    float od = tref[d];
    float xd = ht[d];
    float head = edge ? head_pair(elev[i], elev[d], oi, od, xi, xd) : oi + xi;
    float j = edge ? head / dist[i] : (oi + xi) / 10000.0f;
    State own;
    own.S = S_arr[i];
    own.area = area_arr[i];
    own.kc = kc_arr[i];
    own.kf = kf_arr[i];
    own.dkc = dkc_arr[i];
    own.dkf = dkf_arr[i];
    float depthA = max(((oi + H[i]) + xi) - min(head, 0.0f) * (edge ? 1.0f : 0.0f), 0.0f);
    State accel = own;
    if (edge && head < 0.0f) {
        accel = stage_pair(i, -H[i], depthA, L, W, H, A, tab, bed, levels);
    }
    float Ae = accel.S / L[i];
    float dAe = accel.area / L[i];
    float f = phi_reg(j, eps);
    bool reverse = edge && (Q0[i] + 9.8f * dt * Ae * f * fabs(f) < 0.0f);
    float inc = 1.0f / nc[i];
    float inf = 1.0f / nf[i];
    float K = own.kc * inc + own.kf * inf;
    float dK = own.dkc * inc + own.dkf * inf;
    if (reverse) {
        float beddiff = (elev[d] - elev[i]) + (H[i] - H[d]);
        float depth = max(((od + H[d]) + xd) + min(0.0f, beddiff), 0.0f);
        State c = stage_pair(i, -H[i], depth, L, W, H, A, tab, bed, levels);
        K = c.kc * inc + c.kf * inf;
        dK = c.dkc * inc + c.dkf * inf;
    }
    bool max_at_down = edge && head < 0.0f;
    float q;
    float dqi;
    float dqd;
    momentum_flux(
        Q0[i],
        Ae,
        max_at_down ? 0.0f : dAe,
        max_at_down ? dAe : 0.0f,
        K,
        reverse ? 0.0f : dK,
        reverse ? dK : 0.0f,
        j,
        edge ? 1.0f / dist[i] : 1.0f / 10000.0f,
        edge ? -1.0f / dist[i] : 0.0f,
        dt,
        eps,
        q,
        dqi,
        dqd
    );
    if (!edge && q < 0.0f) {
        q = 0.0f;
        dqi = 0.0f;
        dqd = 0.0f;
    }
    Q_out[i] = q;
    qi_out[i] = dqi;
    qd_out[i] = dqd;
}

static inline void eval_flux_unit(thread const Net& z, int i, float dt) {
    eval_flux_fn(
        i, dt, z.eps, z.levels, z.Q0, z.nc, z.nf, z.L, z.W, z.H, z.A, z.tab, z.bed,
        z.elev, z.dist, z.down, z.edge, z.tref, z.ht, z.S, z.area, z.kc, z.kf, z.dkc, z.dkf,
        z.Q, z.qi, z.qd
    );
}

static inline float eval_residual_fn(
    int i,
    float dt,
    device const float* S0,
    device const float* runoff,
    device const float* S_arr,
    device const float* Q_arr,
    device const int* cp,
    device const int* kids,
    device float* R_out,
    device float* wb_out,
    device const float* wa_in,
    bool want_line,
    thread float& line_out
) {
    float in_q = 0.0f;
    float ain = 0.0f;
    for (int e = cp[i]; e < cp[i + 1]; ++e) {
        float q = Q_arr[kids[e]];
        in_q += q;
        ain += fabs(q);
    }
    float Qi = Q_arr[i];
    float r = (S_arr[i] - S0[i]) - dt * (runoff[i] + in_q - Qi);
    float scale = S0[i] + dt * (fabs(runoff[i]) + ain + fabs(Qi)) + 1000.0f;
    float score = hf_isfinite(r) && hf_isfinite(Qi) ? fabs(r) / scale : hf_inf();
    R_out[i] = r;
    wb_out[i] = 1.0f / scale;
    if (want_line) {
        line_out = hf_isfinite(score) ? fabs(r) * wa_in[i] : hf_inf();
    }
    return score;
}

static inline float eval_residual_unit(
    thread const Net& z, int i, float dt, bool want_line, thread float& line_out
) {
    return eval_residual_fn(
        i, dt, z.S0, z.runoff, z.S, z.Q, z.cp, z.kids, z.R, z.wb, z.wa, want_line, line_out
    );
}

static inline float damping_fn(
    int i,
    float dt,
    device const int* tree,
    device const float* tf,
    device const float* area_arr,
    device const float* qi_arr,
    device const float* qd_arr,
    device const int* cp,
    device const int* kids
) {
    float sigma = tf[(long)tree[i] * TF_STRIDE + TF_SIGMA];
    if (!(sigma > 0.0f)) return 0.0f;
    float d = area_arr[i] + dt * fabs(qi_arr[i]);
    for (int e = cp[i]; e < cp[i + 1]; ++e) {
        d += dt * fabs(qd_arr[kids[e]]);
    }
    return sigma * d;
}

static inline float damping_unit(thread const Net& z, int i, float dt) {
    return damping_fn(i, dt, z.tree, z.tf, z.area, z.qi, z.qd, z.cp, z.kids);
}

static inline void eliminate_unit(thread const Net& z, int i, float dt) {
    bool dry = !((z.ref[i] + z.H[i]) + z.h[i] > 0.0f) && z.R[i] > 0.0f;
    bool hd = z.edge[i] != 0;
    float a = z.area[i] + dt * z.qi[i] + damping_unit(z, i, dt);
    float r = -z.R[i];
    z.b[i] = (dry || !hd) ? 0.0f : dt * z.qd[i];
    for (int e = z.cp[i]; e < z.cp[i + 1]; ++e) {
        int j = z.kids[e];
        a -= dt * z.qd[j];
        float c = -dt * z.qi[j];
        a -= c * z.b[j] / z.ap[j];
        r -= c * z.rp[j] / z.ap[j];
    }
    z.ap[i] = dry ? 1.0f : a;
    z.rp[i] = dry ? 0.0f : r;
}

static inline void substitute_unit(thread const Net& z, int i) {
    bool hd = z.edge[i] != 0;
    z.dh[i] = (z.rp[i] - (hd ? z.b[i] * z.dh[z.down[i]] : 0.0f)) / z.ap[i];
}

static inline void eliminate_path_fn(
    int p,
    float dt,
    device const float* H,
    device const uchar* edge,
    device const int* cp,
    device const int* kids,
    device const int* tree,
    device const int* pnodes,
    device const int* pptr,
    device const float* ref,
    device const float* h,
    device const float* R,
    device const float* area,
    device const float* qi_arr,
    device const float* qd_arr,
    device const float* tf,
    device float* ap,
    device float* rp,
    device float* b_arr
) {
    int previous = -1;
    float pa = 0.0f;
    float pr = 0.0f;
    float pb = 0.0f;
    float pqi = 0.0f;
    float pqd = 0.0f;
    for (int k = pptr[p]; k < pptr[p + 1]; ++k) {
        int i = pnodes[k];
        bool dry = !((ref[i] + H[i]) + h[i] > 0.0f) && R[i] > 0.0f;
        float qi = qi_arr[i];
        float qd = qd_arr[i];
        float a = area[i] + dt * qi + damping_fn(i, dt, tree, tf, area, qi_arr, qd_arr, cp, kids);
        float r = -R[i];
        float b = (dry || edge[i] == 0) ? 0.0f : dt * qd;
        for (int e = cp[i]; e < cp[i + 1]; ++e) {
            int j = kids[e];
            bool carried = (j == previous);
            float qj = carried ? pqd : qd_arr[j];
            a -= dt * qj;
            float c = -dt * (carried ? pqi : qi_arr[j]);
            float aj = carried ? pa : ap[j];
            float bj = carried ? pb : b_arr[j];
            float rj = carried ? pr : rp[j];
            a -= c * bj / aj;
            r -= c * rj / aj;
        }
        pa = dry ? 1.0f : a;
        pr = dry ? 0.0f : r;
        pb = b;
        pqi = qi;
        pqd = qd;
        previous = i;
        ap[i] = pa;
        rp[i] = pr;
        b_arr[i] = b;
    }
}

static inline void substitute_path_fn(
    int p,
    device const int* down,
    device const uchar* edge,
    device const int* pnodes,
    device const int* pptr,
    device const float* ap,
    device const float* rp,
    device const float* b_arr,
    device float* dh
) {
    int out = pnodes[pptr[p + 1] - 1];
    float parent = edge[out] != 0 ? dh[down[out]] : 0.0f;
    for (int k = pptr[p + 1] - 1; k >= pptr[p]; --k) {
        int i = pnodes[k];
        float value = (rp[i] - (edge[i] != 0 ? b_arr[i] * parent : 0.0f)) / ap[i];
        dh[i] = value;
        parent = value;
    }
}

struct PathRow {
    float lower;
    float diag;
    float upper;
    float rhs;
    float boundary;
};

static inline PathRow path_row_fn(
    int p,
    int k,
    float dt,
    device const float* H,
    device const uchar* edge,
    device const int* cp,
    device const int* kids,
    device const int* tree,
    device const int* pnodes,
    device const int* pptr,
    device const float* ref,
    device const float* h,
    device const float* R,
    device const float* area,
    device const float* qi_arr,
    device const float* qd_arr,
    device const float* tf,
    device const float* ap,
    device const float* rp,
    device const float* b_arr
) {
    int begin = pptr[p];
    int length = pptr[p + 1] - begin;
    int i = pnodes[begin + k];
    int heavy = k > 0 ? pnodes[begin + k - 1] : -1;
    bool dry = !((ref[i] + H[i]) + h[i] > 0.0f) && R[i] > 0.0f;
    float diag = area[i] + dt * qi_arr[i] + damping_fn(i, dt, tree, tf, area, qi_arr, qd_arr, cp, kids);
    float rhs = -R[i];
    float lower = heavy >= 0 ? -dt * qi_arr[heavy] : 0.0f;
    float upper = (dry || edge[i] == 0) ? 0.0f : dt * qd_arr[i];
    for (int e = cp[i]; e < cp[i + 1]; ++e) {
        int j = kids[e];
        diag -= dt * qd_arr[j];
        if (j != heavy) {
            float c = -dt * qi_arr[j];
            diag -= c * b_arr[j] / ap[j];
            rhs -= c * rp[j] / ap[j];
        }
    }
    return {
        dry ? 0.0f : lower,
        dry ? 1.0f : diag,
        k + 1 < length ? upper : 0.0f,
        dry ? 0.0f : rhs,
        k + 1 == length ? upper : 0.0f
    };
}

static inline void start_phase(
    float dt0, device int* s, device float* f, int phase
) {
    s[TI_PHASE] = phase;
    s[TI_MODE] = MODE_EVAL;
    s[TI_ST] = 0;
    s[TI_STEPS] = 0;
    s[TI_FEVALS] = 0;
    s[TI_TOTALBT] = 0;
    s[TI_RESET] = phase <= 0 ? 1 : (phase >= 13 ? 2 : 0);
    f[TF_DT] = dt0 * ((phase < 0 || phase >= 13) ? 1.0f : CONTINUATION_BETA[phase]);
}

static inline void finish_tree(
    device float* meta,
    device atomic_int* status,
    int t,
    device int* s,
    device float* f
) {
    s[TI_MODE] = MODE_DONE;
    device float* m = meta + (long)t * 16;
    int phase = s[TI_PHASE];
    m[0] = f[TF_MERIT];
    m[1] = (float)s[TI_SUM_G];
    m[2] = (float)s[TI_ST];
    m[3] = (float)s[TI_TOTALBT];
    m[4] = (float)s[TI_SUM_G];
    m[5] = 0.0f;
    m[6] = (float)s[TI_SUM_FE];
    m[7] = 0.0f;
    m[8] = 0.0f;
    m[9] = 0.0f;
    m[10] = 0.0f;
    m[11] = 0.0f;
    m[12] = (float)s[TI_DSTEPS];
    m[13] = (float)(phase < 0 ? 1 : phase + 2);
    m[14] = phase >= 0 ? 1.0f : 0.0f;
    m[15] = f[TF_INITIAL];
    if (s[TI_ST] != 0) report_err(status, NEWTON_NOT_CONVERGED, t);
}

static inline void start_damped(device int* s, device float* f) {
    s[TI_PHASE] = DAMPED_PHASE;
    s[TI_MODE] = MODE_EVAL;
    s[TI_ST] = 0;
    s[TI_STEPS] = 0;
    s[TI_FEVALS] = 0;
    s[TI_TOTALBT] = 0;
    s[TI_BT] = 0;
    f[TF_SIGMA] = NEWTON_DAMPED_SIGMA;
}

static inline void finish_phase(
    float dt0,
    device float* meta,
    device atomic_int* status,
    int t,
    device int* s,
    device float* f
) {
    int st = s[TI_ST];
    int phase = s[TI_PHASE];
    float merit = f[TF_MERIT];
    s[TI_SUM_G] += s[TI_STEPS];
    s[TI_SUM_FE] += s[TI_FEVALS];
    if (phase == -1) f[TF_INITIAL] = merit;
    if (phase == DAMPED_PHASE) f[TF_SIGMA] = 0.0f;
    if (st != 0 && phase == -1) {
        start_phase(dt0, s, f, -1);
        s[TI_PHASE] = RESTART_PHASE;
        s[TI_BT] = 0;
        return;
    }
    if (st != 0 && phase == RESTART_PHASE) {
        start_damped(s, f);
        return;
    }
    if (st != 0 && phase == DAMPED_PHASE) {
        start_phase(dt0, s, f, 0);
        return;
    }
    if (phase == RESTART_PHASE || phase == DAMPED_PHASE || phase == -1) {
        finish_tree(meta, status, t, s, f);
        return;
    }
    if (phase >= 0 && phase < 12) {
        start_phase(dt0, s, f, phase + 1);
        return;
    }
    if (phase == 12 && st != 0) {
        f[TF_BEST] = merit;
        s[TI_BEST_ST] = st;
        s[TI_SAVE] |= 1;
        start_phase(dt0, s, f, 13);
        return;
    }
    if (phase >= 13) {
        if (merit < f[TF_BEST]) {
            f[TF_BEST] = merit;
            s[TI_BEST_ST] = st;
            s[TI_SAVE] |= 2;
        }
        if (st != 0 && phase < 20) {
            start_phase(dt0, s, f, phase + 1);
            return;
        }
        if (st != 0) {
            s[TI_SAVE] |= 4;
            s[TI_MODE] = MODE_FINAL;
            return;
        }
    }
    finish_tree(meta, status, t, s, f);
}

static inline void damped_control(
    float dt0,
    float tol,
    device float* meta,
    device atomic_int* status,
    int t,
    device int* s,
    device float* f,
    float mt,
    float line
) {
    ++s[TI_FEVALS];
    if (s[TI_MODE] == MODE_TRIAL) {
        float old = f[TF_MERIT];
        if (!hf_isfinite(mt)) {
            f[TF_SIGMA] = max(f[TF_SIGMA] * 10.0f, 1e-3f);
            s[TI_MODE] = MODE_EVAL;
            if (++s[TI_TOTALBT] >= 40) {
                s[TI_ST] = 2;
                s[TI_MODE] = MODE_RESTORE;
            }
            return;
        }
        s[TI_COMMIT] = 1;
        f[TF_MERIT] = mt;
        ++s[TI_STEPS];
        ++s[TI_DSTEPS];
        float sigma = f[TF_SIGMA] * min(line / old, NEWTON_DAMPED_GROWTH);
        if (sigma == 0.0f && line >= old) sigma = 1e-3f;
        f[TF_SIGMA] = sigma < 1e-7f ? 0.0f : sigma;
    } else {
        f[TF_MERIT] = mt;
    }
    if (!(mt >= tol)) {
        s[TI_ST] = 0;
        f[TF_SIGMA] = 0.0f;
        finish_phase(dt0, meta, status, t, s, f);
        return;
    }
    if (s[TI_STEPS] >= NEWTON_DAMPED_STEPS) {
        s[TI_ST] = 1;
        f[TF_SIGMA] = 0.0f;
        finish_phase(dt0, meta, status, t, s, f);
        return;
    }
    s[TI_SOLVE] = 1;
    s[TI_MODE] = MODE_TRIAL;
    f[TF_LAM] = 1.0f;
}

static inline void control_tree(
    float dt0,
    float tol,
    int maxit,
    device int* ti,
    device float* tf,
    device atomic_uint* tm,
    device float* meta,
    device atomic_int* status,
    int t,
    float line
) {
    device int* s = ti + (long)t * TI_STRIDE;
    device float* f = tf + (long)t * TF_STRIDE;
    s[TI_COMMIT] = 0;
    s[TI_SAVE] = 0;
    s[TI_RESET] = 0;
    s[TI_SOLVE] = 0;
    int mode = s[TI_MODE];
    if (mode == MODE_DONE) return;
    float mt = as_type<float>(atomic_load_explicit(&tm[t], memory_order_relaxed));
    atomic_store_explicit(&tm[t], 0u, memory_order_relaxed);
    if (mode == MODE_FINAL) {
        f[TF_MERIT] = mt;
        s[TI_ST] = mt < tol ? 0 : s[TI_BEST_ST];
        finish_tree(meta, status, t, s, f);
        return;
    }
    if (s[TI_PHASE] == DAMPED_PHASE && mode != MODE_RESTORE) {
        damped_control(dt0, tol, meta, status, t, s, f, mt, line);
        return;
    }
    bool newton_done = false;
    if (mode == MODE_EVAL) {
        f[TF_MERIT] = mt;
        ++s[TI_FEVALS];
        newton_done = !(mt >= tol);
    } else if (mode == MODE_TRIAL) {
        ++s[TI_FEVALS];
        ++s[TI_TOTALBT];
        float old = f[TF_MERIT];
        float lam = f[TF_LAM];
        bool fixed = (s[TI_PHASE] == RESTART_PHASE);
        if (hf_isfinite(mt) && (mt < tol || (fixed ? line : mt) <= (1.0f - 0.01f * lam) * old)) {
            s[TI_COMMIT] = 1;
            f[TF_MERIT] = mt;
            ++s[TI_STEPS];
            newton_done = !(s[TI_STEPS] < maxit && mt >= tol);
        } else {
            f[TF_LAM] = lam * 0.5f;
            if (++s[TI_BT] < LINE_SEARCH) return;
            s[TI_MODE] = MODE_RESTORE;
            return;
        }
    } else {
        f[TF_MERIT] = mt;
        s[TI_FEVALS] += 2;
        s[TI_ST] = mt < tol ? 0 : 2;
        finish_phase(dt0, meta, status, t, s, f);
        return;
    }
    if (!newton_done) {
        s[TI_SOLVE] = 1;
        s[TI_MODE] = MODE_TRIAL;
        f[TF_LAM] = 1.0f;
        s[TI_BT] = 0;
        return;
    }
    ++s[TI_FEVALS];
    s[TI_ST] = f[TF_MERIT] < tol ? 0 : 1;
    finish_phase(dt0, meta, status, t, s, f);
}

static inline void start_unit_fn(
    int i,
    device const float* S0,
    device const float* nc,
    device const float* nf,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    int levels,
    device float* ref,
    device float* h,
    device float* tref,
    device float* ht,
    device atomic_int* status
) {
    float r0 = invert_storage(i, S0[i], L, W, H, A, tab, levels);
    ref[i] = r0;
    h[i] = 0.0f;
    tref[i] = r0;
    ht[i] = 0.0f;
    if (!(nc[i] > 0.0f && hf_isfinite(nc[i]) && nf[i] > 0.0f && hf_isfinite(nf[i]))) {
        report_err(status, INVALID_ROUGHNESS, i);
    }
}

static inline void start_tree_fn(
    int t,
    float dt0,
    device int* ti,
    device float* tf,
    device atomic_uint* tm,
    device atomic_uint* tline
) {
    device int* s = ti + (long)t * TI_STRIDE;
    device float* f = tf + (long)t * TF_STRIDE;
    for (int k = 0; k < TI_STRIDE; ++k) s[k] = 0;
    for (int k = 0; k < TF_STRIDE; ++k) f[k] = 0.0f;
    start_phase(dt0, s, f, -1);
    s[TI_RESET] = 0;
    s[TI_BEST_ST] = 2;
    f[TF_BEST] = 1e30f;
    atomic_store_explicit(&tm[t], 0u, memory_order_relaxed);
    atomic_store_explicit(&tline[t], 0u, memory_order_relaxed);
}

static inline void apply_unit_fn(
    int i,
    int t,
    int n,
    int levels,
    device const int* ti,
    device const float* S0,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device float* ref,
    device float* h,
    device const float* tref,
    device const float* ht,
    device const float* wb,
    device float* wa,
    device float* trace
) {
    device const int* s = ti + (long)t * TI_STRIDE;
    if (s[TI_COMMIT] != 0) {
        ref[i] = tref[i];
        h[i] = ht[i];
    }
    if (s[TI_SOLVE] != 0) {
        int phase = s[TI_PHASE];
        if (phase == RESTART_PHASE || phase == DAMPED_PHASE) {
            wa[i] = wb[i];
        }
    }
    int save = s[TI_SAVE];
    int reset = s[TI_RESET];
    if ((save & 1) != 0) {
        trace[i] = ref[i] + h[i];
        trace[(long)n + i] = ref[i];
        trace[2L * (long)n + i] = h[i];
    }
    if ((save & 2) != 0) {
        trace[(long)n + i] = ref[i];
        trace[2L * (long)n + i] = h[i];
    }
    if ((save & 4) != 0) {
        ref[i] = trace[(long)n + i];
        h[i] = trace[2L * (long)n + i];
    }
    if (reset == 1) {
        ref[i] = invert_storage(i, S0[i], L, W, H, A, tab, levels);
        h[i] = 0.0f;
    }
    if (reset == 2) {
        ref[i] = max(-H[i], trace[i] + RESTART_OFFSETS[s[TI_PHASE] - 13]);
        h[i] = 0.0f;
    }
}

static inline Pair terrain_coordinate(
    int i,
    Pair h,
    device const float* L,
    device const float* W,
    device const float* A,
    device const float* tab,
    int levels
) {
    if (!(h.hi > 0.0f || (h.hi == 0.0f && h.lo > 0.0f))) return h;
    float w = W[i];
    float dw = A[i] / L[i] / (float)levels;
    float ph = 0.0f;
    float pw = w;
    float nh = 0.0f;
    Pair acc = {0.0f, 0.0f};
    int lev = 0;
    for (int k = 0; k < levels; ++k) {
        nh = tab[(long)i * levels + k];
        float nw = w + (float)(k + 1) * dw;
        if (!(h.hi > nh)) break;
        float width = 0.5f * (pw + nw);
        float delta = nh - ph;
        float v = width * delta;
        acc = pair_add(acc, Pair{v, precise::fma(width, delta, -v)});
        ph = nh;
        pw = nw;
        ++lev;
    }
    float grad = lev < levels ? (nh - ph) / dw : 0.0f;
    float x = h.hi - ph;
    float run = grad > 0.0f ? x / grad : 0.0f;
    float width = pw + 0.5f * run;
    float v = width * x;
    acc = pair_add(acc, Pair{v, precise::fma(width, x, -v)});
    acc = pair_add(acc, Pair{(pw + run) * h.lo, 0.0f});
    float hi = acc.hi / w;
    return pair_sum(hi, (precise::fma(-hi, w, acc.hi) + acc.lo) / w);
}

static inline Pair terrain_inverse(
    int i,
    Pair v,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    int levels
) {
    if (!(v.hi > 0.0f || (v.hi == 0.0f && v.lo > 0.0f))) {
        if ((v.hi + H[i]) + v.lo < 0.0f) return {-H[i], 0.0f};
        return v;
    }
    float w = W[i];
    float dw = A[i] / L[i] / (float)levels;
    float ph = 0.0f;
    float pw = w;
    float nh = 0.0f;
    float c = v.hi * w;
    Pair cross = pair_sum(c, precise::fma(v.hi, w, -c) + v.lo * w);
    Pair acc = {0.0f, 0.0f};
    int lev = 0;
    for (int k = 0; k < levels; ++k) {
        nh = tab[(long)i * levels + k];
        float nw = w + (float)(k + 1) * dw;
        float width = 0.5f * (pw + nw);
        float delta = nh - ph;
        float a = width * delta;
        Pair next = pair_add(acc, Pair{a, precise::fma(width, delta, -a)});
        Pair excess = pair_add(cross, Pair{-next.hi, -next.lo});
        if (!(excess.hi > 0.0f)) break;
        acc = next;
        ph = nh;
        pw = nw;
        ++lev;
    }
    Pair excess = pair_add(cross, Pair{-acc.hi, -acc.lo});
    float extra = max(excess.hi + excess.lo, 0.0f);
    float grad = lev < levels ? (nh - ph) / dw : 0.0f;
    float x = grad > 0.0f
        ? 2.0f * extra / (hf_sqrt(pw * pw + 2.0f * extra / grad) + pw)
        : extra / pw;
    Pair h = pair_sum(ph, x);
    Pair got = terrain_coordinate(i, h, L, W, A, tab, levels);
    Pair error = pair_add(v, Pair{-got.hi, -got.lo});
    float slope = (pw + (grad > 0.0f ? x / grad : 0.0f)) / w;
    return pair_add(h, Pair{(error.hi + error.lo) / slope, 0.0f});
}

static inline void trial_unit_fn(
    int i,
    int t,
    int levels,
    device const int* ti,
    device const float* tf,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* ref_arr,
    device const float* h_arr,
    device float* tref_arr,
    device float* ht_arr,
    device const float* dh_arr,
    device const float* area_arr,
    device float* ap_arr,
    device float* rp_arr,
    device float* b_arr
) {
    device const int* s = ti + (long)t * TI_STRIDE;
    int mode = s[TI_MODE];
    if (mode == MODE_DONE) return;
    if (mode == MODE_TRIAL) {
        float ref = ref_arr[i];
        float off = h_arr[i];
        float rn = ref;
        float xn = off;
        float lambda = tf[(long)t * TF_STRIDE + TF_LAM];
        float direction = dh_arr[i];
        trial_pair(ref, off, lambda * direction, H[i], rn, xn);
        if (((ref + off) > 0.0f) != ((rn + xn) > 0.0f)) {
            Pair v;
            float ratio;
            if (s[TI_SOLVE] != 0) {
                v = terrain_coordinate(i, Pair{ref, off}, L, W, A, tab, levels);
                ratio = area_arr[i] / (L[i] * W[i]);
                ap_arr[i] = v.hi;
                rp_arr[i] = v.lo;
                b_arr[i] = ratio;
            } else {
                v = Pair{ap_arr[i], rp_arr[i]};
                ratio = b_arr[i];
            }
            if (s[TI_BT] >= 2 || (s[TI_PHASE] == DAMPED_PHASE && direction > 0.0f)) {
                Pair h = terrain_inverse(
                    i,
                    pair_add(v, Pair{lambda * (ratio * direction), 0.0f}),
                    L, W, H, A, tab, levels
                );
                rn = h.hi;
                xn = h.lo;
            }
        }
        tref_arr[i] = rn;
        ht_arr[i] = xn;
    } else {
        tref_arr[i] = ref_arr[i];
        ht_arr[i] = h_arr[i];
    }
}

static inline void snapshot_unit(
    int i,
    float full_dt,
    device const float* Q0,
    device const float* nc,
    device const float* nf,
    device const float* L,
    device const float* W,
    device const float* H,
    device const float* A,
    device const float* tab,
    device const float* bed,
    device const float* elev,
    device const float* dist,
    device const int* down,
    device const uchar* edges,
    device const float* ref,
    device const float* h,
    device const float* S,
    device const float* area,
    device const float* Q,
    device const float* qi,
    device const float* qd,
    device const float* R,
    device float* saved,
    float eps,
    int levels
) {
    float p[8];
    parameter_partials(
        i, ref, h, Q0, nc, nf, L, W, H, A, tab, bed, elev, dist, down, edges,
        eps, full_dt, levels, Q[i], p
    );
    long base = (long)i * 17;
    saved[base + 0] = Q[i];
    saved[base + 1] = S[i];
    saved[base + 2] = qi[i];
    saved[base + 3] = qd[i];
    saved[base + 4] = area[i];
    saved[base + 5] = (!((ref[i] + H[i]) + h[i] > 0.0f) && R[i] > 0.0f) ? 1.0f : 0.0f;
    saved[base + 6] = ref[i] + h[i];
    saved[base + 7] = p[5];
    saved[base + 8] = p[6];
    saved[base + 9] = p[0];
    saved[base + 10] = p[1];
    saved[base + 11] = p[2];
    saved[base + 13] = p[3];
    saved[base + 14] = p[4];
    saved[base + 15] = p[7];
    saved[base + 16] = full_dt;

    bool edge = edges[i] != 0;
    int d = edge ? down[i] : i;
    float head = edge ? head_pair(elev[i], elev[d], ref[i], ref[d], h[i], h[d]) : ref[i] + h[i];
    State accel = stage_pair(i, ref[i], h[i], L, W, H, A, tab, bed, levels);
    if (edge && head < 0.0f) {
        accel = stage_pair(
            i, -H[i], max(((ref[i] + H[i]) + h[i]) - head, 0.0f),
            L, W, H, A, tab, bed, levels
        );
    }
    State c = stage_pair(i, ref[i], h[i], L, W, H, A, tab, bed, levels);
    if (p[7] > 0.5f) {
        float depth = max(
            ((ref[d] + H[d]) + h[d]) + min(0.0f, (elev[d] - elev[i]) + (H[i] - H[d])),
            0.0f
        );
        c = stage_pair(i, -H[i], depth, L, W, H, A, tab, bed, levels);
    }
    float Ae = accel.S / L[i];
    float K = c.kc / nc[i] + c.kf / nf[i];
    saved[base + 12] = (Ae > 1e-8f && K > 1e-12f && (edge || Q[i] > 0.0f))
        ? 1.0f / (1.0f + 2.0f * 9.8f * full_dt * Ae / (K * K) * fabs(Q[i]))
        : 0.0f;
}

// ========================================================================
// Sub-kernel bodies
// ========================================================================

// HYDROFORGE METAL KERNEL BODY: inertia_solve_trees
{
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_block_trees) return;
    int t = args.block_tree_ptr[b_idx];
    int tid = (int)lid;
    int B = (int)tpg;

    Net z;
    z.S0 = args.storage_ptr;
    z.runoff = args.runoff_ptr;
    z.Q0 = args.outflow_ptr;
    z.nc = args.river_manning_ptr;
    z.nf = args.flood_manning_ptr;
    z.L = args.river_length_ptr;
    z.W = args.river_width_ptr;
    z.H = args.river_height_ptr;
    z.A = args.catchment_area_ptr;
    z.tab = args.flood_depth_table_ptr;
    z.bed = args.flood_bed_length_ptr;
    z.elev = args.catchment_elevation_ptr;
    z.dist = args.downstream_distance_ptr;
    z.dt0 = args.time_step_ptr;
    z.down = args.downstream_idx_ptr;
    z.cp = args.upstream_offset_ptr;
    z.kids = args.upstream_index_ptr;
    z.tree = args.tree_index_ptr;
    z.pnodes = nullptr;
    z.pptr = nullptr;
    z.tunit = args.tree_unit_ptr;
    z.loff = args.level_offset_ptr;
    z.tlev = args.tree_level_ptr;
    z.edge = args.downstream_edge_ptr;
    z.h = args.newton_offset_ptr;
    z.ht = args.trial_offset_ptr;
    z.ref = args.newton_reference_ptr;
    z.tref = args.trial_reference_ptr;
    z.S = args.trial_storage_ptr;
    z.area = args.trial_area_ptr;
    z.kc = args.channel_conveyance_ptr;
    z.kf = args.flood_conveyance_ptr;
    z.dkc = args.channel_conveyance_slope_ptr;
    z.dkf = args.flood_conveyance_slope_ptr;
    z.Q = args.trial_outflow_ptr;
    z.qi = args.outflow_own_slope_ptr;
    z.qd = args.outflow_down_slope_ptr;
    z.R = args.residual_ptr;
    z.wb = args.balance_weight_ptr;
    z.wa = args.accepted_weight_ptr;
    z.ap = args.pivot_ptr;
    z.rp = args.pivot_rhs_ptr;
    z.dh = args.newton_step_ptr;
    z.b = args.coupling_ptr;
    z.trace = args.phase_trace_ptr;
    z.tf = args.tree_scalars_ptr;
    z.meta = args.newton_record_ptr;
    z.ti = args.tree_control_ptr;
    z.status = args.newton_status_ptr;
    z.tm = args.tree_merit_ptr;
    z.tline = args.tree_line_ptr;
    z.eps = *args.slope_regularization;
    z.tol = *args.newton_tolerance;
    z.n = *args.num_catchments;
    z.levels = *args.num_flood_levels;
    z.maxit = *args.newton_max_iterations;

    threadgroup int s_active;
    threadgroup int s_solving;
    threadgroup uint s_merit[32];
    threadgroup uint s_line[32];

    float dt0 = *args.time_step_ptr;
    int first = z.loff[z.tlev[t]];
    int last = z.loff[z.tlev[t + 1]];
    for (int k = first + tid; k < last; k += B) {
        start_unit_fn(
            z.tunit[k], z.S0, z.nc, z.nf, z.L, z.W, z.H, z.A, z.tab, z.levels,
            z.ref, z.h, z.tref, z.ht, z.status
        );
    }
    if (tid == 0) {
        start_tree_fn(t, dt0, z.ti, z.tf, z.tm, z.tline);
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

    uint num_sg = (tpg + simd_width - 1u) / simd_width;
    for (int round = 0; round < MAX_ROUNDS; ++round) {
        if (z.ti[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE) {
            for (int k = first + tid; k < last; k += B) {
                eval_stage_unit(z, z.tunit[k]);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (z.ti[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE) {
            float dt = z.tf[(long)t * TF_STRIDE + TF_DT];
            for (int k = first + tid; k < last; k += B) {
                eval_flux_unit(z, z.tunit[k], dt);
            }
        }
        threadgroup_barrier(mem_flags::mem_device);

        uint loc_m = 0u;
        uint loc_l = 0u;
        if (z.ti[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE) {
            float dt = z.tf[(long)t * TF_STRIDE + TF_DT];
            int phase = z.ti[(long)t * TI_STRIDE + TI_PHASE];
            bool weighted = (phase == RESTART_PHASE || phase == DAMPED_PHASE);
            for (int k = first + tid; k < last; k += B) {
                float lk = 0.0f;
                float sc = eval_residual_unit(z, z.tunit[k], dt, weighted, lk);
                loc_m = max(loc_m, as_type<uint>(sc));
                if (weighted) loc_l = max(loc_l, as_type<uint>(lk));
            }
        }
        uint sg_m = simd_max(loc_m);
        uint sg_l = simd_max(loc_l);
        if (simd_lane == 0u) {
            s_merit[simd_group] = sg_m;
            s_line[simd_group] = sg_l;
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        if (tid == 0) {
            uint tot_m = 0u;
            uint tot_l = 0u;
            for (uint sg = 0u; sg < num_sg; ++sg) {
                tot_m = max(tot_m, s_merit[sg]);
                tot_l = max(tot_l, s_line[sg]);
            }
            uint prev_m = atomic_load_explicit(&z.tm[t], memory_order_relaxed);
            atomic_store_explicit(&z.tm[t], max(prev_m, tot_m), memory_order_relaxed);
            control_tree(
                dt0, z.tol, z.maxit, z.ti, z.tf, z.tm, z.meta, z.status, t,
                as_type<float>(tot_l)
            );
            s_active = (z.ti[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE) ? 1 : 0;
            s_solving = z.ti[(long)t * TI_STRIDE + TI_SOLVE];
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);

        for (int k = first + tid; k < last; k += B) {
            apply_unit_fn(
                z.tunit[k], t, z.n, z.levels, z.ti, z.S0, z.L, z.W, z.H, z.A, z.tab,
                z.ref, z.h, z.tref, z.ht, z.wb, z.wa, z.trace
            );
        }
        if (s_active == 0) return;
        threadgroup_barrier(mem_flags::mem_device);

        if (s_solving != 0) {
            float dt = z.tf[(long)t * TF_STRIDE + TF_DT];
            for (int l = z.tlev[t]; l < z.tlev[t + 1]; ++l) {
                for (int k = z.loff[l] + tid; k < z.loff[l + 1]; k += B) {
                    eliminate_unit(z, z.tunit[k], dt);
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
            for (int l = z.tlev[t + 1] - 1; l >= z.tlev[t]; --l) {
                for (int k = z.loff[l] + tid; k < z.loff[l + 1]; k += B) {
                    substitute_unit(z, z.tunit[k]);
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
        }
        for (int k = first + tid; k < last; k += B) {
            trial_unit_fn(
                z.tunit[k], t, z.levels, z.ti, z.tf, z.L, z.W, z.H, z.A, z.tab,
                z.ref, z.h, z.tref, z.ht, z.dh, z.area, z.ap, z.rp, z.b
            );
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (tid == 0) report_err(z.status, NEWTON_ROUND_LIMIT, t);
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_start_units
{
    int k = (int)i;
    if (k >= *args.num_grid_units) return;
    int u = args.grid_unit_ptr[k];
    start_unit_fn(
        u,
        args.storage_ptr,
        args.river_manning_ptr,
        args.flood_manning_ptr,
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        *args.num_flood_levels,
        args.newton_reference_ptr,
        args.newton_offset_ptr,
        args.trial_reference_ptr,
        args.trial_offset_ptr,
        args.newton_status_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_start_trees
{
    int k = (int)i;
    if (k >= *args.num_grid_trees) return;
    int t = args.grid_tree_ptr[k];
    start_tree_fn(
        t,
        *args.time_step_ptr,
        args.tree_control_ptr,
        args.tree_scalars_ptr,
        args.tree_merit_ptr,
        args.tree_line_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_stage
{
    int k = (int)i;
    if (k >= *args.num_grid_units) return;
    int u = args.grid_unit_ptr[k];
    int t = args.tree_index_ptr[u];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_MODE] == MODE_DONE) return;
    State s = stage_pair(
        u,
        args.trial_reference_ptr[u],
        args.trial_offset_ptr[u],
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        args.flood_bed_length_ptr,
        *args.num_flood_levels
    );
    args.trial_storage_ptr[u] = s.S;
    args.trial_area_ptr[u] = s.area;
    args.channel_conveyance_ptr[u] = s.kc;
    args.flood_conveyance_ptr[u] = s.kf;
    args.channel_conveyance_slope_ptr[u] = s.dkc;
    args.flood_conveyance_slope_ptr[u] = s.dkf;
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_flux
{
    int k = (int)i;
    if (k >= *args.num_grid_units) return;
    int u = args.grid_unit_ptr[k];
    int t = args.tree_index_ptr[u];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_MODE] == MODE_DONE) return;
    float dt = args.tree_scalars_ptr[(long)t * TF_STRIDE + TF_DT];
    eval_flux_fn(
        u, dt, *args.slope_regularization, *args.num_flood_levels,
        args.outflow_ptr, args.river_manning_ptr, args.flood_manning_ptr,
        args.river_length_ptr, args.river_width_ptr, args.river_height_ptr,
        args.catchment_area_ptr, args.flood_depth_table_ptr, args.flood_bed_length_ptr,
        args.catchment_elevation_ptr, args.downstream_distance_ptr, args.downstream_idx_ptr,
        args.downstream_edge_ptr, args.trial_reference_ptr, args.trial_offset_ptr,
        args.trial_storage_ptr, args.trial_area_ptr, args.channel_conveyance_ptr,
        args.flood_conveyance_ptr, args.channel_conveyance_slope_ptr,
        args.flood_conveyance_slope_ptr, args.trial_outflow_ptr,
        args.outflow_own_slope_ptr, args.outflow_down_slope_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_residual
{
    threadgroup uint s_m[32];
    threadgroup uint s_l[32];
    int k = (int)i;
    int ngu = *args.num_grid_units;
    int single = *args.single_tree;

    if (single != 0) {
        int t = args.grid_tree_ptr[0];
        bool active = (args.tree_control_ptr[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE);
        int phase = args.tree_control_ptr[(long)t * TI_STRIDE + TI_PHASE];
        bool weighted = (phase == RESTART_PHASE || phase == DAMPED_PHASE);
        uint loc_m = 0u;
        uint loc_l = 0u;
        if (active && k < ngu) {
            float dt = args.tree_scalars_ptr[(long)t * TF_STRIDE + TF_DT];
            float lk = 0.0f;
            float sc = eval_residual_fn(
                args.grid_unit_ptr[k], dt, args.storage_ptr, args.runoff_ptr,
                args.trial_storage_ptr, args.trial_outflow_ptr, args.upstream_offset_ptr,
                args.upstream_index_ptr, args.residual_ptr, args.balance_weight_ptr,
                args.accepted_weight_ptr, weighted, lk
            );
            loc_m = as_type<uint>(sc);
            if (weighted) loc_l = as_type<uint>(lk);
        }
        uint sg_m = simd_max(loc_m);
        uint sg_l = simd_max(loc_l);
        if (simd_lane == 0u) {
            s_m[simd_group] = sg_m;
            s_l[simd_group] = sg_l;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0u && active) {
            long total = (long)(*args.total_threads);
            uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
            uint num_sg = (group_threads + simd_width - 1u) / simd_width;
            uint bm = 0u;
            uint bl = 0u;
            for (uint sg = 0u; sg < num_sg; ++sg) {
                bm = max(bm, s_m[sg]);
                bl = max(bl, s_l[sg]);
            }
            if (bm != 0u) atomic_fetch_max_explicit(&args.tree_merit_ptr[t], bm, memory_order_relaxed);
            if (weighted && bl != 0u) atomic_fetch_max_explicit(&args.tree_line_ptr[t], bl, memory_order_relaxed);
        }
        return;
    }

    if (k >= ngu) return;
    int u = args.grid_unit_ptr[k];
    int t = args.tree_index_ptr[u];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_MODE] == MODE_DONE) return;
    int phase = args.tree_control_ptr[(long)t * TI_STRIDE + TI_PHASE];
    bool weighted = (phase == RESTART_PHASE || phase == DAMPED_PHASE);
    float dt = args.tree_scalars_ptr[(long)t * TF_STRIDE + TF_DT];
    float lk = 0.0f;
    float sc = eval_residual_fn(
        u, dt, args.storage_ptr, args.runoff_ptr, args.trial_storage_ptr,
        args.trial_outflow_ptr, args.upstream_offset_ptr, args.upstream_index_ptr,
        args.residual_ptr, args.balance_weight_ptr, args.accepted_weight_ptr,
        weighted, lk
    );
    atomic_fetch_max_explicit(&args.tree_merit_ptr[t], as_type<uint>(sc), memory_order_relaxed);
    if (weighted) {
        atomic_fetch_max_explicit(&args.tree_line_ptr[t], as_type<uint>(lk), memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_control
{
    int k = (int)i;
    if (k >= *args.num_grid_trees) return;
    int t = args.grid_tree_ptr[k];
    float line_val = as_type<float>(atomic_load_explicit(&args.tree_line_ptr[t], memory_order_relaxed));
    atomic_store_explicit(&args.tree_line_ptr[t], 0u, memory_order_relaxed);
    control_tree(
        *args.time_step_ptr,
        *args.newton_tolerance,
        *args.newton_max_iterations,
        args.tree_control_ptr,
        args.tree_scalars_ptr,
        args.tree_merit_ptr,
        args.newton_record_ptr,
        args.newton_status_ptr,
        t,
        line_val
    );
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_MODE] != MODE_DONE) {
        atomic_fetch_add_explicit(&args.grid_control_ptr[0], 1, memory_order_relaxed);
    }
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_SOLVE] != 0) {
        atomic_fetch_add_explicit(&args.grid_control_ptr[2], 1, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_apply
{
    int k = (int)i;
    if (k >= *args.num_grid_units) return;
    int u = args.grid_unit_ptr[k];
    int t = args.tree_index_ptr[u];
    apply_unit_fn(
        u,
        t,
        *args.num_catchments,
        *args.num_flood_levels,
        args.tree_control_ptr,
        args.storage_ptr,
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        args.newton_reference_ptr,
        args.newton_offset_ptr,
        args.trial_reference_ptr,
        args.trial_offset_ptr,
        args.balance_weight_ptr,
        args.accepted_weight_ptr,
        args.phase_trace_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_eliminate_serial
{
    int k = (int)i;
    if (k >= *args.num_paths) return;
    int p = args.paths_ptr[k];
    int t = args.tree_index_ptr[args.path_unit_ptr[args.path_offset_ptr[p]]];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_SOLVE] == 0) return;
    float dt = args.tree_scalars_ptr[(long)t * TF_STRIDE + TF_DT];
    eliminate_path_fn(
        p, dt, args.river_height_ptr, args.downstream_edge_ptr,
        args.upstream_offset_ptr, args.upstream_index_ptr, args.tree_index_ptr,
        args.path_unit_ptr, args.path_offset_ptr, args.newton_reference_ptr,
        args.newton_offset_ptr, args.residual_ptr, args.trial_area_ptr,
        args.outflow_own_slope_ptr, args.outflow_down_slope_ptr, args.tree_scalars_ptr,
        args.pivot_ptr, args.pivot_rhs_ptr, args.coupling_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_substitute_serial
{
    int k = (int)i;
    if (k >= *args.num_paths) return;
    int p = args.paths_ptr[k];
    int t = args.tree_index_ptr[args.path_unit_ptr[args.path_offset_ptr[p]]];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_SOLVE] == 0) return;
    substitute_path_fn(
        p, args.downstream_idx_ptr, args.downstream_edge_ptr, args.path_unit_ptr,
        args.path_offset_ptr, args.pivot_ptr, args.pivot_rhs_ptr, args.coupling_ptr,
        args.newton_step_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_eliminate_pcr
{
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_pcr_paths) return;
    int p = args.pcr_paths_ptr[b_idx];
    int begin = args.path_offset_ptr[p];
    int length = args.path_offset_ptr[p + 1] - begin;
    int t = args.tree_index_ptr[args.path_unit_ptr[begin]];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_SOLVE] == 0) return;

    float dt = args.tree_scalars_ptr[(long)t * TF_STRIDE + TF_DT];
    int slot = args.pcr_slots_ptr[b_idx];
    long stride = (long)(*args.pcr_stride);
    device float* scratch = args.pcr_scratch_ptr;

    long cur = 0L;
    long nxt = 5L * stride;
    for (int k = (int)lid; k < length; k += (int)tpg) {
        PathRow row = path_row_fn(
            p, k, dt, args.river_height_ptr, args.downstream_edge_ptr,
            args.upstream_offset_ptr, args.upstream_index_ptr, args.tree_index_ptr,
            args.path_unit_ptr, args.path_offset_ptr, args.newton_reference_ptr,
            args.newton_offset_ptr, args.residual_ptr, args.trial_area_ptr,
            args.outflow_own_slope_ptr, args.outflow_down_slope_ptr, args.tree_scalars_ptr,
            args.pivot_ptr, args.pivot_rhs_ptr, args.coupling_ptr
        );
        scratch[cur + 0L * stride + slot + k] = row.lower;
        scratch[cur + 1L * stride + slot + k] = row.diag;
        scratch[cur + 2L * stride + slot + k] = row.upper;
        scratch[cur + 3L * stride + slot + k] = row.rhs;
        scratch[cur + 4L * stride + slot + k] = row.boundary;
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (int dist = 1; dist < length; dist *= 2) {
        for (int k = (int)lid; k < length; k += (int)tpg) {
            float a = scratch[cur + 0L * stride + slot + k];
            float d = scratch[cur + 1L * stride + slot + k];
            float c = scratch[cur + 2L * stride + slot + k];
            float r = scratch[cur + 3L * stride + slot + k];
            float bv = scratch[cur + 4L * stride + slot + k];
            float na = 0.0f;
            float nc = 0.0f;
            if (k >= dist) {
                int j = k - dist;
                float f = a / scratch[cur + 1L * stride + slot + j];
                d -= f * scratch[cur + 2L * stride + slot + j];
                r -= f * scratch[cur + 3L * stride + slot + j];
                bv -= f * scratch[cur + 4L * stride + slot + j];
                na = -f * scratch[cur + 0L * stride + slot + j];
            }
            if (k + dist < length) {
                int j = k + dist;
                float f = c / scratch[cur + 1L * stride + slot + j];
                d -= f * scratch[cur + 0L * stride + slot + j];
                r -= f * scratch[cur + 3L * stride + slot + j];
                bv -= f * scratch[cur + 4L * stride + slot + j];
                nc = -f * scratch[cur + 2L * stride + slot + j];
            }
            scratch[nxt + 0L * stride + slot + k] = na;
            scratch[nxt + 1L * stride + slot + k] = d;
            scratch[nxt + 2L * stride + slot + k] = nc;
            scratch[nxt + 3L * stride + slot + k] = r;
            scratch[nxt + 4L * stride + slot + k] = bv;
        }
        threadgroup_barrier(mem_flags::mem_device);
        long tmp = cur;
        cur = nxt;
        nxt = tmp;
    }
    for (int k = (int)lid; k < length; k += (int)tpg) {
        int unit = args.path_unit_ptr[begin + k];
        float d = scratch[cur + 1L * stride + slot + k];
        args.pivot_ptr[unit] = 1.0f;
        args.pivot_rhs_ptr[unit] = scratch[cur + 3L * stride + slot + k] / d;
        args.coupling_ptr[unit] = scratch[cur + 4L * stride + slot + k] / d;
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_substitute_pcr
{
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_pcr_paths) return;
    int p = args.pcr_paths_ptr[b_idx];
    int begin = args.path_offset_ptr[p];
    int end = args.path_offset_ptr[p + 1];
    int t = args.tree_index_ptr[args.path_unit_ptr[begin]];
    if (args.tree_control_ptr[(long)t * TI_STRIDE + TI_SOLVE] == 0) return;
    int out = args.path_unit_ptr[end - 1];
    float parent = args.downstream_edge_ptr[out] != 0
        ? args.newton_step_ptr[args.downstream_idx_ptr[out]]
        : 0.0f;
    for (int k = begin + (int)lid; k < end; k += (int)tpg) {
        int u = args.path_unit_ptr[k];
        args.newton_step_ptr[u] = args.pivot_rhs_ptr[u] - args.coupling_ptr[u] * parent;
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_grid_trial
{
    int k = (int)i;
    if (k >= *args.num_grid_units) return;
    int u = args.grid_unit_ptr[k];
    int t = args.tree_index_ptr[u];
    trial_unit_fn(
        u,
        t,
        *args.num_flood_levels,
        args.tree_control_ptr,
        args.tree_scalars_ptr,
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        args.newton_reference_ptr,
        args.newton_offset_ptr,
        args.trial_reference_ptr,
        args.trial_offset_ptr,
        args.newton_step_ptr,
        args.trial_area_ptr,
        args.pivot_ptr,
        args.pivot_rhs_ptr,
        args.coupling_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_commit
{
    int idx = (int)i;
    if (idx == 0) {
        int err = args.grid_control_ptr[CTL_ERROR];
        if (err != 0) report_err(args.newton_status_ptr, err, 0);
    }
    if (idx >= *args.num_catchments) return;
    float r = args.newton_reference_ptr[idx];
    float h = args.newton_offset_ptr[idx];
    if (*args.save_surface != 0 && args.converged_surface_ptr) {
        args.converged_surface_ptr[2L * (long)idx] = r;
        args.converged_surface_ptr[2L * (long)idx + 1L] = h;
    }
    float q = args.trial_outflow_ptr[idx];
    float s = args.trial_storage_ptr[idx];
    float in_q = args.downstream_edge_ptr[idx] != 0 ? max(-q, 0.0f) : 0.0f;
    for (int e = args.upstream_offset_ptr[idx]; e < args.upstream_offset_ptr[idx + 1]; ++e) {
        in_q += max(args.trial_outflow_ptr[args.upstream_index_ptr[e]], 0.0f);
    }
    if (!(hf_isfinite(q) && s >= 0.0f && hf_isfinite(s))) {
        report_err(args.newton_status_ptr, INVALID_STATE, idx);
    }
    args.storage_ptr[idx] = s;
    args.outflow_ptr[idx] = q;
    args.water_surface_ptr[idx] = r + h;
    args.inflow_ptr[idx] = in_q;
}

// HYDROFORGE METAL KERNEL BODY: inertia_rebuild_stage
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    float r = args.converged_surface_ptr[2L * (long)idx];
    float x = args.converged_surface_ptr[2L * (long)idx + 1L];
    args.newton_reference_ptr[idx] = r;
    args.newton_offset_ptr[idx] = x;
    args.trial_reference_ptr[idx] = r;
    args.trial_offset_ptr[idx] = x;
    State s = stage_pair(
        idx,
        r,
        x,
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        args.flood_bed_length_ptr,
        *args.num_flood_levels
    );
    args.trial_storage_ptr[idx] = s.S;
    args.trial_area_ptr[idx] = s.area;
    args.channel_conveyance_ptr[idx] = s.kc;
    args.flood_conveyance_ptr[idx] = s.kf;
    args.channel_conveyance_slope_ptr[idx] = s.dkc;
    args.flood_conveyance_slope_ptr[idx] = s.dkf;
}

// HYDROFORGE METAL KERNEL BODY: inertia_rebuild_flux
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    eval_flux_fn(
        idx, *args.time_step_ptr, *args.slope_regularization, *args.num_flood_levels,
        args.outflow_ptr, args.river_manning_ptr, args.flood_manning_ptr,
        args.river_length_ptr, args.river_width_ptr, args.river_height_ptr,
        args.catchment_area_ptr, args.flood_depth_table_ptr, args.flood_bed_length_ptr,
        args.catchment_elevation_ptr, args.downstream_distance_ptr, args.downstream_idx_ptr,
        args.downstream_edge_ptr, args.trial_reference_ptr, args.trial_offset_ptr,
        args.trial_storage_ptr, args.trial_area_ptr, args.channel_conveyance_ptr,
        args.flood_conveyance_ptr, args.channel_conveyance_slope_ptr,
        args.flood_conveyance_slope_ptr, args.trial_outflow_ptr,
        args.outflow_own_slope_ptr, args.outflow_down_slope_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_rebuild_residual
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    float dummy_line = 0.0f;
    eval_residual_fn(
        idx, *args.time_step_ptr, args.storage_ptr, args.runoff_ptr,
        args.trial_storage_ptr, args.trial_outflow_ptr, args.upstream_offset_ptr,
        args.upstream_index_ptr, args.residual_ptr, args.balance_weight_ptr,
        nullptr, false, dummy_line
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_rebuild_snapshot
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    snapshot_unit(
        idx,
        *args.time_step_ptr,
        args.outflow_ptr,
        args.river_manning_ptr,
        args.flood_manning_ptr,
        args.river_length_ptr,
        args.river_width_ptr,
        args.river_height_ptr,
        args.catchment_area_ptr,
        args.flood_depth_table_ptr,
        args.flood_bed_length_ptr,
        args.catchment_elevation_ptr,
        args.downstream_distance_ptr,
        args.downstream_idx_ptr,
        args.downstream_edge_ptr,
        args.newton_reference_ptr,
        args.newton_offset_ptr,
        args.trial_storage_ptr,
        args.trial_area_ptr,
        args.trial_outflow_ptr,
        args.outflow_own_slope_ptr,
        args.outflow_down_slope_ptr,
        args.residual_ptr,
        args.jacobian_snapshot_ptr,
        *args.slope_regularization,
        *args.num_flood_levels
    );
}

// HYDROFORGE METAL KERNEL BODY: inertia_adjoint_up
{
    int k_idx = (int)i;
    if (k_idx >= *args.count) return;
    int p = *args.r_begin + k_idx;
    device const float* saved = args.jacobian_snapshot_ptr;
    device const float* gQ = args.grad_outflow_ptr;
    device const float* gS = args.grad_storage_ptr;
    device const float* gh = args.grad_water_surface_ptr;
    device const int* cp = args.upstream_offset_ptr;
    device const int* kids = args.upstream_index_ptr;
    device const int* pnodes = args.path_unit_ptr;
    device const int* pptr = args.path_offset_ptr;
    device float* ap = args.pivot_ptr;
    device float* rp = args.pivot_rhs_ptr;

    float dt = saved[(long)pnodes[pptr[p]] * 17L + 16L];
    for (int k = pptr[p]; k < pptr[p + 1]; ++k) {
        int u = pnodes[k];
        long bu = (long)u * 17L;
        bool bi = saved[bu + 5L] > 0.5f;
        float a = bi ? 1.0f : saved[bu + 4L] + dt * saved[bu + 2L];
        float r = saved[bu + 4L] * gS[u] + saved[bu + 2L] * gQ[u] + gh[u];
        for (int e = cp[u]; e < cp[u + 1]; ++e) {
            int j = kids[e];
            long bj = (long)j * 17L;
            r += saved[bj + 3L] * gQ[j];
            if (!bi) a -= dt * saved[bj + 3L];
            float upper = saved[bj + 5L] > 0.5f ? 0.0f : dt * saved[bj + 3L];
            float lower = bi ? 0.0f : -dt * saved[bj + 2L];
            a -= upper * lower / ap[j];
            r -= upper * rp[j] / ap[j];
        }
        ap[u] = a;
        rp[u] = r;
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_adjoint_down
{
    int k_idx = (int)i;
    if (k_idx >= *args.count) return;
    int p = *args.r_begin + k_idx;
    device const float* saved = args.jacobian_snapshot_ptr;
    device float* gQ = args.grad_outflow_ptr;
    device const int* down = args.downstream_idx_ptr;
    device const uchar* edges = args.downstream_edge_ptr;
    device const int* pnodes = args.path_unit_ptr;
    device const int* pptr = args.path_offset_ptr;
    device const float* ap = args.pivot_ptr;
    device const float* rp = args.pivot_rhs_ptr;
    device float* lam = args.adjoint_continuity_ptr;
    device float* ce = args.adjoint_flux_ptr;
    device float* g_runoff = args.grad_runoff_ptr;

    float dt = saved[(long)pnodes[pptr[p]] * 17L + 16L];
    for (int k = pptr[p + 1] - 1; k >= pptr[p]; --k) {
        int u = pnodes[k];
        long bu = (long)u * 17L;
        bool edge = edges[u] != 0;
        int d = edge ? down[u] : -1;
        bool bound_d = edge && (saved[(long)d * 17L + 5L] > 0.5f);
        float lower = (edge && !bound_d) ? -dt * saved[bu + 2L] : 0.0f;
        float v = (rp[u] - (edge ? lower * lam[d] : 0.0f)) / ap[u];
        lam[u] = v;
        float li = saved[bu + 5L] > 0.5f ? 0.0f : v;
        float ld_ = (edge && !bound_d) ? lam[d] : 0.0f;
        ce[u] = gQ[u] - dt * (li - ld_);
        g_runoff[u] = dt * li;
        gQ[u] = ce[u] * saved[bu + 12L];
    }
}

// HYDROFORGE METAL KERNEL BODY: inertia_adjoint_finish
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    device const float* saved = args.jacobian_snapshot_ptr;
    device float* gS = args.grad_storage_ptr;
    device const int* cp = args.upstream_offset_ptr;
    device const int* kids = args.upstream_index_ptr;
    device const float* lam = args.adjoint_continuity_ptr;
    device const float* ce = args.adjoint_flux_ptr;

    long b = (long)idx * 17L;
    bool bound = saved[b + 5L] > 0.5f;
    float li = bound ? 0.0f : lam[idx];
    args.grad_river_width_ptr[idx] += (gS[idx] - li) * saved[b + 7L] + ce[idx] * saved[b + 9L];
    float v = (gS[idx] - li) * saved[b + 8L] + ce[idx] * saved[b + 10L];
    gS[idx] = li;
    if (bound) v -= lam[idx];
    float cn = ce[idx] * saved[b + 13L];
    float fn = ce[idx] * saved[b + 14L];
    for (int e = cp[idx]; e < cp[idx + 1]; ++e) {
        int j = kids[e];
        v += ce[j] * saved[(long)j * 17L + 11L];
    }
    args.grad_river_height_ptr[idx] += v;
    args.grad_river_manning_ptr[idx] += cn;
    args.grad_flood_manning_ptr[idx] += fn;
}
