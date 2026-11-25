// Copyright (c) 2025 Shengyu Kang (Wuhan University)
// SPDX-License-Identifier: Apache-2.0
// Apple Metal (MSL) port of HydroNet coupled bifurcation graph solver,
// sparse-LU + BiCGStab linear solver, snapshot rebuild, and transpose adjoint.
// Note: implicit_inertia.metal helpers and graph constants are prepended via prelude.

// ------------------------------------------------------------------------
// Graph math & coupled physics helpers.

static inline Pair graph_product(float a, float b) {
    float p = a * b;
    return {p, precise::fma(a, b, -p)};
}

static inline Pair graph_scale(Pair a, float b) {
    return pair_add(graph_product(a.hi, b), graph_product(a.lo, b));
}

static inline float comp_coupling_weight(int phase) {
    if (phase == 0) return 0.0f;
    if (phase > 0 && phase < GRAPH_COUPLING_FULL_PHASE) {
        return ldexp(1.0f, phase - GRAPH_COUPLING_FULL_PHASE);
    }
    return 1.0f;
}

static inline void graph_flux_cw(
    int p,
    float dt,
    float cw,
    int bl,
    device const int* src,
    device const int* dst,
    device const float* elev,
    device const float* H,
    device const float* tref,
    device const float* ht,
    device const float* width,
    device const float* length,
    device const float* elevation,
    device const float* roughness,
    device const float* previous,
    device float* path_values,
    device float* path_sum,
    device atomic_int* status
) {
    int u = src[p];
    int d = dst[p];
    float ou = tref[u];
    float xu = ht[u];
    float od = tref[d];
    float xd = ht[d];
    float head = head_pair(elev[u], elev[d], ou, od, xu, xd);
    float raw = head / length[p];
    int high = head >= 0.0f ? u : d;
    float slope = clamp(raw, -GRAPH_SLOPE_LIMIT, GRAPH_SLOPE_LIMIT);
    float ds = fabs(raw) < GRAPH_SLOPE_LIMIT ? 1.0f / length[p] : 0.0f;
    float net = 0.0f;
    float di = 0.0f;
    float ddown = 0.0f;
    for (int l = 0; l < bl; ++l) {
        int k = p * bl + l;
        float q = 0.0f;
        float qi = 0.0f;
        float qd = 0.0f;
        float qp = 0.0f;
        float qn = 0.0f;
        float hu = 0.0f;
        float hd = 0.0f;
        Pair height = pair_add(
            pair_diff(elev[high], elevation[k]),
            Pair{tref[high], ht[high]}
        );
        float depth = max(height.hi + height.lo, 0.0f);
        float n = roughness[k];
        float w = width[k];
        if (w > 0.0f && elevation[k] < GRAPH_CLOSED_ELEVATION && !(n > 0.0f && hf_isfinite(n))) {
            report_err(status, INVALID_ROUGHNESS, k);
        }
        if (depth > GRAPH_DRY_DEPTH && w > 0.0f && n > 0.0f && hf_isfinite(n)) {
            float a = 9.8f * dt * w * depth;
            float v = previous[k] + a * slope;
            int donor = v >= 0.0f ? u : d;
            float bank_bed = elev[donor] - H[donor];
            float donor_depth;
            if (bank_bed > elevation[k]) {
                donor_depth = max((tref[donor] + H[donor]) + ht[donor], 0.0f);
            } else {
                Pair filled = pair_add(
                    pair_diff(elev[donor], elevation[k]),
                    Pair{tref[donor], ht[donor]}
                );
                donor_depth = max(filled.hi + filled.lo, 0.0f);
            }
            if (donor_depth > GRAPH_DRY_DEPTH) {
                float K = hf_div(
                    hf_div(w * donor_depth * donor_depth, hf_cbrt_nonneg(donor_depth)),
                    n
                );
                float b = 9.8f * dt * w * depth / (K * K);
                float den = hf_sqrt(1.0f + 4.0f * b * fabs(v));
                q = 2.0f * v / (1.0f + den);
                float friction = b * q * fabs(q);
                float high_part = (a * slope - friction) / depth;
                float donor_part = (10.0f / 3.0f) * friction / donor_depth;
                qi = ((high == u ? high_part : 0.0f) + (donor == u ? donor_part : 0.0f) + a * ds) / den;
                qd = ((high == d ? high_part : 0.0f) + (donor == d ? donor_part : 0.0f) - a * ds) / den;
                qp = 1.0f / den;
                qn = -2.0f * friction / (n * den);
                if (bank_bed > elevation[k]) {
                    float h_val = donor_part / den;
                    if (donor == u) hu = h_val;
                    else hd = h_val;
                }
            }
        }
        q *= cw;
        qi *= cw;
        qd *= cw;
        qp *= cw;
        qn *= cw;
        hu *= cw;
        hd *= cw;
        long base = (long)k * 7L;
        path_values[base + 0L] = q;
        path_values[base + 1L] = qi;
        path_values[base + 2L] = qd;
        path_values[base + 3L] = qp;
        path_values[base + 4L] = qn;
        path_values[base + 5L] = hu;
        path_values[base + 6L] = hd;
        net += q;
        di += qi;
        ddown += qd;
    }
    long sbase = (long)p * 3L;
    path_sum[sbase + 0L] = net;
    path_sum[sbase + 1L] = di;
    path_sum[sbase + 2L] = ddown;
}

static inline float graph_residual_fn(
    int i,
    float dt,
    device const float* S0,
    device const float* runoff,
    device const float* S,
    device const float* Q,
    device const int* cp,
    device const int* kids,
    device const int* boffset,
    device const int* blink,
    device const float* path_sum,
    device float* R,
    device float* wb,
    device float* rhs
) {
    float in_q = 0.0f;
    float ain = 0.0f;
    for (int e = cp[i]; e < cp[i + 1]; ++e) {
        float q = Q[kids[e]];
        in_q += q;
        ain += fabs(q);
    }
    float out_q = 0.0f;
    float absout = 0.0f;
    for (int e = boffset[i]; e < boffset[i + 1]; ++e) {
        int link = blink[e];
        int p = hf_iabs(link) - 1;
        float q = path_sum[(long)p * 3L];
        out_q += (link > 0 ? 1.0f : -1.0f) * q;
        absout += fabs(q);
    }
    float q = Q[i];
    float r = (S[i] - S0[i]) - dt * (runoff[i] + in_q - q - out_q);
    float scale = S0[i] + dt * (fabs(runoff[i]) + ain + fabs(q) + absout) + 1000.0f;
    float score = hf_isfinite(r) && hf_isfinite(q) ? fabs(r) / scale : hf_inf();
    float inv_s = 1.0f / scale;
    rhs[i] = inv_s;
    R[i] = r;
    wb[i] = inv_s;
    return score;
}

static inline float graph_damping_fn(
    int i,
    float dt,
    device const int* u_comp,
    device const float* tf,
    device const float* area,
    device const float* qi,
    device const float* qd,
    device const int* cp,
    device const int* kids,
    device const int* boffset,
    device const int* blink,
    device const float* path_sum
) {
    int c = u_comp[i];
    if (c < 0) return 0.0f;
    float sigma = tf[(long)c * TF_STRIDE + TF_SIGMA];
    if (!(sigma > 0.0f)) return 0.0f;
    float d = area[i] + dt * fabs(qi[i]);
    for (int e = cp[i]; e < cp[i + 1]; ++e) {
        d += dt * fabs(qd[kids[e]]);
    }
    for (int e = boffset[i]; e < boffset[i + 1]; ++e) {
        int link = blink[e];
        int p = hf_iabs(link) - 1;
        d += dt * fabs(link > 0 ? path_sum[(long)p * 3L + 1L] : path_sum[(long)p * 3L + 2L]);
    }
    return sigma * d;
}

static inline void finish_graph_comp(
    float dt0,
    int c,
    device const int* c_rep,
    device float* meta,
    device atomic_int* status,
    device int* s,
    device float* f
) {
    int st = s[TI_ST];
    int phase = s[TI_PHASE];
    float merit = f[TF_MERIT];
    float init0 = f[TF_INITIAL];
    s[TI_SUM_G] += s[TI_STEPS];
    s[TI_SUM_FE] += s[TI_FEVALS];
    if (phase == -1) f[TF_INITIAL] = merit;
    if (phase == DAMPED_PHASE) f[TF_SIGMA] = 0.0f;
    if (st != 0 && phase == -1) {
        bool keep_surf = hf_isfinite(merit) && merit < init0;
        start_phase(dt0, s, f, -1);
        s[TI_PHASE] = RESTART_PHASE;
        s[TI_BT] = 0;
        s[TI_RESET] = keep_surf ? 0 : 1;
        return;
    }
    if (st != 0 && phase == RESTART_PHASE) {
        start_damped(s, f);
        return;
    }
    if (phase == DAMPED_PHASE) {
        if (st != 0) {
            start_phase(dt0, s, f, 0);
            return;
        }
        finish_tree(meta, status, c_rep[c], s, f);
        return;
    }
    if (phase == RESTART_PHASE || phase == -1) {
        finish_tree(meta, status, c_rep[c], s, f);
        return;
    }
    if (phase >= 0 && phase < GRAPH_COUPLING_FULL_PHASE) {
        start_phase(dt0, s, f, phase + 1);
        return;
    }
    if (phase == GRAPH_COUPLING_FULL_PHASE && st != 0) {
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
    finish_tree(meta, status, c_rep[c], s, f);
}

static inline void control_graph_comp(
    float dt0,
    float tol,
    int maxit,
    int c,
    device const int* c_rep,
    device int* ti,
    device float* tf,
    device float* meta,
    device atomic_int* status,
    float mt,
    float line
) {
    device int* s = ti + (long)c * TI_STRIDE;
    device float* f = tf + (long)c * TF_STRIDE;
    s[TI_COMMIT] = 0;
    s[TI_SAVE] = 0;
    s[TI_RESET] = 0;
    s[TI_SOLVE] = 0;
    int mode = s[TI_MODE];
    if (mode == MODE_DONE) return;
    if (mode == MODE_FINAL) {
        f[TF_MERIT] = mt;
        s[TI_ST] = mt < tol ? 0 : s[TI_BEST_ST];
        finish_tree(meta, status, c_rep[c], s, f);
        return;
    }
    if (s[TI_PHASE] == DAMPED_PHASE && mode != MODE_RESTORE) {
        ++s[TI_FEVALS];
        if (mode == MODE_TRIAL) {
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
            finish_graph_comp(dt0, c, c_rep, meta, status, s, f);
            return;
        }
        if (s[TI_STEPS] >= NEWTON_DAMPED_STEPS) {
            s[TI_ST] = 1;
            f[TF_SIGMA] = 0.0f;
            finish_graph_comp(dt0, c, c_rep, meta, status, s, f);
            return;
        }
        s[TI_SOLVE] = 1;
        s[TI_MODE] = MODE_TRIAL;
        f[TF_LAM] = 1.0f;
        s[TI_BT] = 0;
        return;
    }
    bool newton_done = false;
    bool fixed_w = (s[TI_PHASE] != -1);
    int step_cap = (!fixed_w && maxit > 36) ? 36 : maxit;
    if (mode == MODE_EVAL) {
        if (s[TI_STEPS] == 0 && s[TI_PHASE] == -1) f[TF_INITIAL] = mt;
        f[TF_MERIT] = mt;
        ++s[TI_FEVALS];
        newton_done = !(mt >= tol);
    } else if (mode == MODE_TRIAL) {
        ++s[TI_FEVALS];
        ++s[TI_TOTALBT];
        float old = f[TF_MERIT];
        float lam = f[TF_LAM];
        float cmp_v = fixed_w ? line : mt;
        if (hf_isfinite(mt) && (mt < tol || cmp_v <= (1.0f - 0.01f * lam) * old)) {
            s[TI_COMMIT] = 1;
            f[TF_MERIT] = mt;
            ++s[TI_STEPS];
            newton_done = !(s[TI_STEPS] < step_cap && mt >= tol);
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
        finish_graph_comp(dt0, c, c_rep, meta, status, s, f);
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
    finish_graph_comp(dt0, c, c_rep, meta, status, s, f);
}

static inline float factor_entry_serial(
    device const int* plan, device const float* factor, int e
) {
    device const int* start = plan + plan[FACTOR_PRODUCT_PTR];
    device const int* lower = plan + plan[FACTOR_PRODUCT_LOWER];
    device const int* upper = plan + plan[FACTOR_PRODUCT_UPPER];
    float value = factor[e];
    for (int k = start[e]; k < start[e + 1]; ++k) {
        value -= factor[lower[k]] * factor[upper[k]];
    }
    return value;
}

static inline float factor_entry_simd(
    device const int* plan,
    device const float* factor,
    int e,
    uint simd_lane,
    uint simd_width
) {
    device const int* start = plan + plan[FACTOR_PRODUCT_PTR];
    device const int* lower = plan + plan[FACTOR_PRODUCT_LOWER];
    device const int* upper = plan + plan[FACTOR_PRODUCT_UPPER];
    float sum = 0.0f;
    for (int k = start[e] + (int)simd_lane; k < start[e + 1]; k += (int)simd_width) {
        sum += factor[lower[k]] * factor[upper[k]];
    }
    sum = simd_sum(sum);
    return factor[e] - sum;
}

static inline float factor_dot_tg(
    device const int* entry,
    device const int* index,
    int begin,
    int end,
    int skip,
    device const float* factor,
    device const float* x,
    uint lid,
    uint tpg,
    uint simd_lane,
    uint simd_group,
    uint simd_width,
    threadgroup float* s_buf
) {
    float sum = 0.0f;
    for (int q = begin + (int)lid; q < end; q += (int)tpg) {
        if (index[q] != skip) sum += factor[entry[q]] * x[index[q]];
    }
    float sg_sum = simd_sum(sum);
    if (tpg <= simd_width) return sg_sum;
    if (simd_lane == 0u) s_buf[simd_group] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    if (lid == 0u) {
        uint num_sg = (tpg + simd_width - 1u) / simd_width;
        for (uint sg = 0u; sg < num_sg; ++sg) total += s_buf[sg];
    }
    return total;
}

static inline Pair graph_lambda_fn(
    device const float* saved,
    device const float* dh,
    device const float* sol_low,
    int i
) {
    return saved[(long)i * 17L + 5L] > 0.5f
        ? Pair{0.0f, 0.0f}
        : Pair{dh[i], sol_low[i]};
}

static inline float graph_flux_adjoint_fn(
    float dt, Pair u, Pair d, float objective
) {
    Pair value = pair_add(
        Pair{objective, 0.0f},
        graph_scale(pair_add(u, Pair{-d.hi, -d.lo}), -dt)
    );
    return value.hi + value.lo;
}

// ========================================================================
// Sub-kernel bodies
// ========================================================================

// HYDROFORGE METAL KERNEL BODY: graph_coupled_start_units
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
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
    args.graph_bound_ptr[u] = 0.0f;
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_start_comps
{
    int c = (int)i;
    if (c >= *args.num_graph_comps) return;
    start_tree_fn(
        c,
        *args.time_step_ptr,
        args.tree_control_ptr,
        args.tree_scalars_ptr,
        args.tree_merit_ptr,
        args.tree_line_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_stage
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
    int c = args.unit_comp_ptr[u];
    if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_MODE] == MODE_DONE) return;
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

// HYDROFORGE METAL KERNEL BODY: graph_coupled_main_flux
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
    int c = args.unit_comp_ptr[u];
    if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_MODE] == MODE_DONE) return;
    eval_flux_fn(
        u, *args.time_step_ptr, *args.slope_regularization, *args.num_flood_levels,
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

// HYDROFORGE METAL KERNEL BODY: graph_coupled_bif_flux
{
    int p = (int)i;
    if (p >= *args.num_bifurcation_paths) return;
    int c = args.path_comp_ptr[p];
    if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_MODE] == MODE_DONE) return;
    float cw = comp_coupling_weight(args.tree_control_ptr[(long)c * TI_STRIDE + TI_PHASE]);
    graph_flux_cw(
        p,
        *args.time_step_ptr,
        cw,
        *args.num_bifurcation_levels,
        args.bifurcation_catchment_idx_ptr,
        args.bifurcation_downstream_idx_ptr,
        args.catchment_elevation_ptr,
        args.river_height_ptr,
        args.trial_reference_ptr,
        args.trial_offset_ptr,
        args.bifurcation_width_ptr,
        args.bifurcation_length_ptr,
        args.bifurcation_elevation_ptr,
        args.bifurcation_manning_ptr,
        args.bifurcation_outflow_ptr,
        args.graph_path_values_ptr,
        args.graph_path_sum_ptr,
        args.newton_status_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_residual
{
    threadgroup uint s_m[32];
    threadgroup uint s_l[32];
    int k = (int)i;
    int ngu = *args.num_graph_units;
    float dt = *args.time_step_ptr;
    if (*args.single_comp != 0) {
        int mode0 = args.tree_control_ptr[TI_MODE];
        bool active = (mode0 != MODE_DONE);
        bool need_line = (mode0 == MODE_TRIAL) && (args.tree_control_ptr[TI_PHASE] != -1);
        uint loc_m = 0u;
        uint loc_l = 0u;
        if (active && k < ngu) {
            int u = args.graph_unit_ptr[k];
            float sc = graph_residual_fn(
                u, dt, args.storage_ptr, args.runoff_ptr, args.trial_storage_ptr,
                args.trial_outflow_ptr, args.upstream_offset_ptr, args.upstream_index_ptr,
                args.incident_offset_ptr, args.incident_path_ptr, args.graph_path_sum_ptr,
                args.residual_ptr, args.balance_weight_ptr, args.graph_rhs_ptr
            );
            loc_m = as_type<uint>(sc);
            if (need_line) {
                float lv = hf_isfinite(sc)
                    ? fabs(args.residual_ptr[u]) * args.graph_balance_scale_ptr[u]
                    : hf_inf();
                loc_l = as_type<uint>(lv);
            }
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
            if (bm != 0u) atomic_fetch_max_explicit(&args.tree_merit_ptr[0], bm, memory_order_relaxed);
            if (need_line && bl != 0u) atomic_fetch_max_explicit(&args.tree_line_ptr[0], bl, memory_order_relaxed);
        }
        return;
    }

    if (k >= ngu) return;
    int u = args.graph_unit_ptr[k];
    int c = args.unit_comp_ptr[u];
    int mode_c = args.tree_control_ptr[(long)c * TI_STRIDE + TI_MODE];
    if (mode_c == MODE_DONE) return;
    float sc = graph_residual_fn(
        u, dt, args.storage_ptr, args.runoff_ptr, args.trial_storage_ptr,
        args.trial_outflow_ptr, args.upstream_offset_ptr, args.upstream_index_ptr,
        args.incident_offset_ptr, args.incident_path_ptr, args.graph_path_sum_ptr,
        args.residual_ptr, args.balance_weight_ptr, args.graph_rhs_ptr
    );
    atomic_fetch_max_explicit(&args.tree_merit_ptr[c], as_type<uint>(sc), memory_order_relaxed);
    if (mode_c == MODE_TRIAL && args.tree_control_ptr[(long)c * TI_STRIDE + TI_PHASE] != -1) {
        float lv = hf_isfinite(sc)
            ? fabs(args.residual_ptr[u]) * args.graph_balance_scale_ptr[u]
            : hf_inf();
        atomic_fetch_max_explicit(&args.tree_line_ptr[c], as_type<uint>(lv), memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_control
{
    int c = (int)i;
    if (c >= *args.num_graph_comps) return;
    float merit = as_type<float>(atomic_load_explicit(&args.tree_merit_ptr[c], memory_order_relaxed));
    float line = as_type<float>(atomic_load_explicit(&args.tree_line_ptr[c], memory_order_relaxed));
    atomic_store_explicit(&args.tree_merit_ptr[c], 0u, memory_order_relaxed);
    atomic_store_explicit(&args.tree_line_ptr[c], 0u, memory_order_relaxed);
    control_graph_comp(
        *args.time_step_ptr,
        *args.newton_tolerance,
        *args.newton_max_iterations,
        c,
        args.comp_rep_tree_ptr,
        args.tree_control_ptr,
        args.tree_scalars_ptr,
        args.newton_record_ptr,
        args.newton_status_ptr,
        merit,
        line
    );
    if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_MODE] != MODE_DONE) {
        atomic_fetch_add_explicit(&args.grid_control_ptr[5], 1, memory_order_relaxed);
    }
    if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] != 0) {
        atomic_fetch_add_explicit(&args.grid_control_ptr[6], 1, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_apply
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
    int c = args.unit_comp_ptr[u];
    apply_unit_fn(
        u,
        c,
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

// HYDROFORGE METAL KERNEL BODY: graph_coupled_prep_rhs
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
    if (*args.all_active == 0) {
        int c = args.unit_comp_ptr[u];
        if (args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    args.graph_balance_scale_ptr[u] = args.graph_rhs_ptr[u];
    bool bound = !((args.newton_reference_ptr[u] + args.river_height_ptr[u]) + args.newton_offset_ptr[u] > 0.0f)
        && args.residual_ptr[u] > 0.0f;
    args.graph_rhs_ptr[u] = bound ? 0.0f : -args.residual_ptr[u];
    args.graph_bound_ptr[u] = bound ? 1.0f : 0.0f;
}

// HYDROFORGE METAL KERNEL BODY: graph_coupled_trial
{
    int k = (int)i;
    if (k >= *args.num_graph_units) return;
    int u = args.graph_unit_ptr[k];
    int c = args.unit_comp_ptr[u];
    trial_unit_fn(
        u,
        c,
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

// HYDROFORGE METAL KERNEL BODY: graph_commit_paths
{
    int k = (int)i;
    if (k >= *args.num_bifurcation_values) return;
    args.bifurcation_outflow_ptr[k] = args.graph_path_values_ptr[(long)k * 7L];
}

// HYDROFORGE METAL KERNEL BODY: graph_commit_inflow
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    int bl = *args.num_bifurcation_levels;
    float received = 0.0f;
    for (int e = args.incident_offset_ptr[idx]; e < args.incident_offset_ptr[idx + 1]; ++e) {
        int link = args.incident_path_ptr[e];
        int p = hf_iabs(link) - 1;
        for (int k = 0; k < bl; ++k) {
            float q = args.graph_path_values_ptr[((long)p * bl + k) * 7L];
            received += max(link > 0 ? -q : q, 0.0f);
        }
    }
    args.inflow_ptr[idx] += received;
}

// ------------------------------------------------------------------------
// Unified sparse-LU + BiCGStab linear solver sub-kernels.

// HYDROFORGE METAL KERNEL BODY: graph_matrix_and_init
{
    threadgroup uint s_max[32];
    int idx = (int)i;
    int nu = *args.num_units;
    int full_dom = *args.full_domain;
    int all_act = *args.all_active;
    int use_damp = *args.use_damping;
    float dt = *args.dt_ptr;
    uint loc_m = 0u;
    if (idx < nu) {
        int u = full_dom != 0 ? idx : args.graph_unit_ptr[idx];
        bool active = (all_act != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            bool dry = args.graph_bound_ptr[u] > 0.5f;
            float cdiag = args.trial_area_ptr[u] + dt * args.outflow_own_slope_ptr[u];
            if (use_damp != 0) {
                cdiag += graph_damping_fn(
                    u, dt, args.unit_comp_ptr, args.tree_scalars_ptr,
                    args.trial_area_ptr, args.outflow_own_slope_ptr, args.outflow_down_slope_ptr,
                    args.upstream_offset_ptr, args.upstream_index_ptr,
                    args.incident_offset_ptr, args.incident_path_ptr, args.graph_path_sum_ptr
                );
            }
            for (int e = args.upstream_offset_ptr[u]; e < args.upstream_offset_ptr[u + 1]; ++e) {
                cdiag -= dt * args.outflow_down_slope_ptr[args.upstream_index_ptr[e]];
            }
            for (int e = args.incident_offset_ptr[u]; e < args.incident_offset_ptr[u + 1]; ++e) {
                int link = args.incident_path_ptr[e];
                int p = hf_iabs(link) - 1;
                cdiag += dt * (
                    link > 0
                    ? args.graph_path_sum_ptr[(long)p * 3L + 1L]
                    : -args.graph_path_sum_ptr[(long)p * 3L + 2L]
                );
            }
            float sc = dry ? 1.0f : 1.0f / cdiag;
            args.graph_scaling_ptr[u] = sc;
            float r_val = args.graph_rhs_ptr[u] * sc;
            args.graph_rhs_ptr[u] = r_val;
            args.newton_step_ptr[u] = 0.0f;
            args.graph_solution_low_ptr[u] = 0.0f;
            float av = fabs(r_val);
            loc_m = as_type<uint>(hf_isfinite(av) ? av : hf_inf());
        }
    }
    uint sg_m = simd_max(loc_m);
    if (simd_lane == 0u) s_max[simd_group] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        uint bm = 0u;
        for (uint sg = 0u; sg < num_sg; ++sg) bm = max(bm, s_max[sg]);
        if (bm != 0u) atomic_fetch_max_explicit(&args.norm_buf_ptr[0], bm, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_assemble
{
    int e = (int)i;
    if (e >= *args.num_entries) return;
    if (*args.all_active == 0) {
        int c = args.entry_comp_ptr[e];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    device const int* plan = args.plan_ptr;
    device const int* start = plan + plan[FACTOR_SOURCE_PTR];
    device const int* code = plan + plan[FACTOR_SOURCE_CODE];
    float dt = *args.dt_ptr;
    float value = 0.0f;
    for (int q = start[e]; q < start[e + 1]; ++q) {
        int kind = code[q] & 7;
        int k = code[q] >> 3;
        if (kind == 0) {
            value += 1.0f;
        } else if (kind == 1) {
            if (args.downstream_edge_ptr[k] != 0 && !(args.graph_bound_ptr[k] > 0.5f)) {
                value += dt * args.outflow_down_slope_ptr[k] * args.graph_scaling_ptr[k];
            }
        } else if (kind == 2) {
            int d = args.downstream_idx_ptr[k];
            if (!(args.graph_bound_ptr[d] > 0.5f)) {
                value -= dt * args.outflow_own_slope_ptr[k] * args.graph_scaling_ptr[d];
            }
        } else if (kind == 3) {
            int u = args.bifurcation_catchment_idx_ptr[k];
            if (!(args.graph_bound_ptr[u] > 0.5f)) {
                value += dt * args.graph_path_sum_ptr[(long)k * 3L + 2L] * args.graph_scaling_ptr[u];
            }
        } else {
            int d = args.bifurcation_downstream_idx_ptr[k];
            if (!(args.graph_bound_ptr[d] > 0.5f)) {
                value -= dt * args.graph_path_sum_ptr[(long)k * 3L + 1L] * args.graph_scaling_ptr[d];
            }
        }
    }
    args.graph_factor_ptr[e] = value;
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_numeric_heavy
{
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_heavy) return;
    int p = *args.r_begin + b_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    device const int* urow = plan + plan[FACTOR_UROW_PTR];
    device const int* uentry = plan + plan[FACTOR_UROW_ENTRY];
    device const int* lcol = plan + plan[FACTOR_LCOL_PTR];
    device const int* lentry = plan + plan[FACTOR_LCOL_ENTRY];
    device float* factor = args.graph_factor_ptr;

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    uint num_sg = (tpg + simd_width - 1u) / simd_width;
    for (int t = pb; t < pe; ++t) {
        int k = unit[t];
        for (int q = urow[k] + (int)simd_group; q < urow[k + 1]; q += (int)num_sg) {
            float v = factor_entry_simd(plan, factor, uentry[q], simd_lane, simd_width);
            if (simd_lane == 0u) factor[uentry[q]] = v;
        }
        threadgroup_barrier(mem_flags::mem_device);
        float pivot = factor[diag[k]];
        for (int q = lcol[k] + (int)simd_group; q < lcol[k + 1]; q += (int)num_sg) {
            float v = factor_entry_simd(plan, factor, lentry[q], simd_lane, simd_width);
            if (simd_lane == 0u) factor[lentry[q]] = v / pivot;
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_numeric_light
{
    int k_idx = (int)i;
    if (k_idx >= *args.count) return;
    int p = *args.split + k_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    device const int* urow = plan + plan[FACTOR_UROW_PTR];
    device const int* uentry = plan + plan[FACTOR_UROW_ENTRY];
    device const int* lcol = plan + plan[FACTOR_LCOL_PTR];
    device const int* lentry = plan + plan[FACTOR_LCOL_ENTRY];
    device float* factor = args.graph_factor_ptr;

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    for (int t = pb; t < pe; ++t) {
        int k = unit[t];
        for (int q = urow[k]; q < urow[k + 1]; ++q) {
            factor[uentry[q]] = factor_entry_serial(plan, factor, uentry[q]);
        }
        float pivot = factor[diag[k]];
        for (int q = lcol[k]; q < lcol[k + 1]; ++q) {
            factor[lentry[q]] = factor_entry_serial(plan, factor, lentry[q]) / pivot;
        }
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_solve_fwd_heavy
{
    threadgroup float s_buf[8];
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_heavy) return;
    int p = *args.r_begin + b_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    bool trans = (*args.trans != 0);
    device const int* fptr = plan + plan[trans ? FACTOR_UCOL_PTR : FACTOR_LROW_PTR];
    device const int* fentry = plan + plan[trans ? FACTOR_UCOL_ENTRY : FACTOR_LROW_ENTRY];
    device const int* findex = plan + plan[trans ? FACTOR_UCOL_INDEX : FACTOR_LROW_INDEX];

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    for (int t = pb; t < pe; ++t) {
        int k = unit[t];
        float dot = factor_dot_tg(
            fentry, findex, fptr[k], fptr[k + 1], -1,
            args.graph_factor_ptr, args.lu_forward_ptr,
            lid, tpg, simd_lane, simd_group, simd_width, s_buf
        );
        if (lid == 0u) {
            float value = args.c_vec_ptr[order[k]] - dot;
            args.lu_forward_ptr[k] = trans ? value / args.graph_factor_ptr[diag[k]] : value;
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_solve_fwd_light
{
    int k_idx = (int)i;
    if (k_idx >= *args.count) return;
    int p = *args.split + k_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    bool trans = (*args.trans != 0);
    device const int* fptr = plan + plan[trans ? FACTOR_UCOL_PTR : FACTOR_LROW_PTR];
    device const int* fentry = plan + plan[trans ? FACTOR_UCOL_ENTRY : FACTOR_LROW_ENTRY];
    device const int* findex = plan + plan[trans ? FACTOR_UCOL_INDEX : FACTOR_LROW_INDEX];

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    for (int t = pb; t < pe; ++t) {
        int k = unit[t];
        float value = args.c_vec_ptr[order[k]];
        for (int q = fptr[k]; q < fptr[k + 1]; ++q) {
            value -= args.graph_factor_ptr[fentry[q]] * args.lu_forward_ptr[findex[q]];
        }
        args.lu_forward_ptr[k] = trans ? value / args.graph_factor_ptr[diag[k]] : value;
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_solve_bwd_heavy
{
    threadgroup float s_buf[8];
    int b_idx = (int)(i / tpg);
    if (b_idx >= *args.num_heavy) return;
    int p = *args.r_begin + b_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    bool trans = (*args.trans != 0);
    device const int* bptr = plan + plan[trans ? FACTOR_LCOL_PTR : FACTOR_UROW_PTR];
    device const int* bentry = plan + plan[trans ? FACTOR_LCOL_ENTRY : FACTOR_UROW_ENTRY];
    device const int* bindex = plan + plan[trans ? FACTOR_LCOL_INDEX : FACTOR_UROW_INDEX];

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    for (int t = pe - 1; t >= pb; --t) {
        int k = unit[t];
        float dot = factor_dot_tg(
            bentry, bindex, bptr[k], bptr[k + 1], k,
            args.graph_factor_ptr, args.lu_backward_ptr,
            lid, tpg, simd_lane, simd_group, simd_width, s_buf
        );
        if (lid == 0u) {
            float value = args.lu_forward_ptr[k] - dot;
            if (!trans) value /= args.graph_factor_ptr[diag[k]];
            args.lu_backward_ptr[k] = value;
            int ord = order[k];
            args.out_vec_ptr[ord] = trans ? value * args.graph_scaling_ptr[ord] : value;
        }
        threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_factor_solve_bwd_light
{
    int k_idx = (int)i;
    if (k_idx >= *args.count) return;
    int p = *args.split + k_idx;
    device const int* plan = args.plan_ptr;
    device const int* unit = plan + plan[FACTOR_PATH_UNIT];
    device const int* path = plan + plan[FACTOR_PATH_OFFSET];
    device const int* order = plan + plan[FACTOR_ORDER];
    device const int* diag = plan + plan[FACTOR_DIAG];
    bool trans = (*args.trans != 0);
    device const int* bptr = plan + plan[trans ? FACTOR_LCOL_PTR : FACTOR_UROW_PTR];
    device const int* bentry = plan + plan[trans ? FACTOR_LCOL_ENTRY : FACTOR_UROW_ENTRY];
    device const int* bindex = plan + plan[trans ? FACTOR_LCOL_INDEX : FACTOR_UROW_INDEX];

    int pb = path[p];
    int pe = path[p + 1];
    if (*args.all_active == 0 && pb < pe) {
        int c = args.unit_comp_ptr[order[unit[pb]]];
        if (c < 0 || args.tree_control_ptr[(long)c * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    for (int t = pe - 1; t >= pb; --t) {
        int k = unit[t];
        float value = args.lu_forward_ptr[k];
        for (int q = bptr[k]; q < bptr[k + 1]; ++q) {
            if (bindex[q] != k) value -= args.graph_factor_ptr[bentry[q]] * args.lu_backward_ptr[bindex[q]];
        }
        if (!trans) value /= args.graph_factor_ptr[diag[k]];
        args.lu_backward_ptr[k] = value;
        int ord = order[k];
        args.out_vec_ptr[ord] = trans ? value * args.graph_scaling_ptr[ord] : value;
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_matvec
{
    int idx = (int)i;
    if (idx >= *args.num_units) return;
    int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
    if (*args.all_active == 0) {
        if (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    float dt = *args.dt_ptr;
    bool comp = (*args.compensated != 0);
    bool trans = (*args.trans != 0);
    bool bound = args.graph_bound_ptr[u] > 0.5f;
    Pair xi = {args.x_ptr[u], comp ? args.graph_solution_low_ptr[u] : 0.0f};
    Pair mi = bound ? Pair{0.0f, 0.0f} : xi;
    Pair value = bound ? xi : graph_scale(xi, args.trial_area_ptr[u]);

    if (trans) {
        Pair md = Pair{0.0f, 0.0f};
        if (args.downstream_edge_ptr[u] != 0) {
            int d = args.downstream_idx_ptr[u];
            if (args.graph_bound_ptr[d] < 0.5f) {
                md = Pair{args.x_ptr[d], comp ? args.graph_solution_low_ptr[d] : 0.0f};
            }
        }
        value = pair_add(
            value,
            graph_scale(graph_scale(pair_add(mi, Pair{-md.hi, -md.lo}), args.outflow_own_slope_ptr[u]), dt)
        );
        for (int e = args.upstream_offset_ptr[u]; e < args.upstream_offset_ptr[u + 1]; ++e) {
            int j = args.upstream_index_ptr[e];
            Pair mj = args.graph_bound_ptr[j] > 0.5f
                ? Pair{0.0f, 0.0f}
                : Pair{args.x_ptr[j], comp ? args.graph_solution_low_ptr[j] : 0.0f};
            value = pair_add(
                value,
                graph_scale(graph_scale(pair_add(mj, Pair{-mi.hi, -mi.lo}), args.outflow_down_slope_ptr[j]), dt)
            );
        }
        for (int e = args.incident_offset_ptr[u]; e < args.incident_offset_ptr[u + 1]; ++e) {
            int link = args.incident_path_ptr[e];
            int p = hf_iabs(link) - 1;
            int pu = args.bifurcation_catchment_idx_ptr[p];
            int pd = args.bifurcation_downstream_idx_ptr[p];
            Pair mu = args.graph_bound_ptr[pu] > 0.5f
                ? Pair{0.0f, 0.0f}
                : Pair{args.x_ptr[pu], comp ? args.graph_solution_low_ptr[pu] : 0.0f};
            Pair mpd = args.graph_bound_ptr[pd] > 0.5f
                ? Pair{0.0f, 0.0f}
                : Pair{args.x_ptr[pd], comp ? args.graph_solution_low_ptr[pd] : 0.0f};
            float slope_p = args.graph_path_sum_ptr[(long)p * 3L + (link > 0 ? 1L : 2L)];
            value = pair_add(
                value,
                graph_scale(graph_scale(pair_add(mu, Pair{-mpd.hi, -mpd.lo}), slope_p), dt)
            );
        }
    } else if (!bound) {
        if (*args.use_damping != 0) {
            float damp = graph_damping_fn(
                u, dt, args.unit_comp_ptr, args.tree_scalars_ptr,
                args.trial_area_ptr, args.outflow_own_slope_ptr, args.outflow_down_slope_ptr,
                args.upstream_offset_ptr, args.upstream_index_ptr,
                args.incident_offset_ptr, args.incident_path_ptr, args.graph_path_sum_ptr
            );
            if (damp > 0.0f) value = pair_add(value, graph_scale(xi, damp));
        }
        Pair flux = graph_scale(xi, args.outflow_own_slope_ptr[u]);
        if (args.downstream_edge_ptr[u] != 0) {
            int d = args.downstream_idx_ptr[u];
            Pair xd = {args.x_ptr[d], comp ? args.graph_solution_low_ptr[d] : 0.0f};
            flux = pair_add(flux, graph_scale(xd, args.outflow_down_slope_ptr[u]));
        }
        value = pair_add(value, graph_scale(flux, dt));
        for (int e = args.upstream_offset_ptr[u]; e < args.upstream_offset_ptr[u + 1]; ++e) {
            int j = args.upstream_index_ptr[e];
            Pair xj = {args.x_ptr[j], comp ? args.graph_solution_low_ptr[j] : 0.0f};
            Pair flux_j = pair_add(
                graph_scale(xj, args.outflow_own_slope_ptr[j]),
                graph_scale(xi, args.outflow_down_slope_ptr[j])
            );
            value = pair_add(value, graph_scale(flux_j, -dt));
        }
        for (int e = args.incident_offset_ptr[u]; e < args.incident_offset_ptr[u + 1]; ++e) {
            int link = args.incident_path_ptr[e];
            int p = hf_iabs(link) - 1;
            int pu = args.bifurcation_catchment_idx_ptr[p];
            int pd = args.bifurcation_downstream_idx_ptr[p];
            Pair xu = {args.x_ptr[pu], comp ? args.graph_solution_low_ptr[pu] : 0.0f};
            Pair xd = {args.x_ptr[pd], comp ? args.graph_solution_low_ptr[pd] : 0.0f};
            Pair flux_p = pair_add(
                graph_scale(xu, args.graph_path_sum_ptr[(long)p * 3L + 1L]),
                graph_scale(xd, args.graph_path_sum_ptr[(long)p * 3L + 2L])
            );
            value = pair_add(value, graph_scale(flux_p, link > 0 ? dt : -dt));
        }
    }
    args.out_ptr[u] = (value.hi + value.lo) * args.graph_scaling_ptr[u];
}

// HYDROFORGE METAL KERNEL BODY: graph_vec_copy_dh
{
    int idx = (int)i;
    if (idx >= *args.num_units) return;
    int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
    if (*args.all_active == 0) {
        if (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    args.newton_step_ptr[u] = args.graph_delta_ptr[u];
    args.graph_solution_low_ptr[u] = 0.0f;
}

// HYDROFORGE METAL KERNEL BODY: graph_vec_norm
{
    threadgroup uint s_max[32];
    int idx = (int)i;
    uint loc_m = 0u;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            float v = fabs(args.x_ptr[u]);
            loc_m = as_type<uint>(hf_isfinite(v) ? v : hf_inf());
        }
    }
    uint sg_m = simd_max(loc_m);
    if (simd_lane == 0u) s_max[simd_group] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        uint bm = 0u;
        for (uint sg = 0u; sg < num_sg; ++sg) bm = max(bm, s_max[sg]);
        if (bm != 0u) atomic_fetch_max_explicit(&args.norm_buf_ptr[0], bm, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_vec_dot
{
    threadgroup float s_sum[32];
    int idx = (int)i;
    float val = 0.0f;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) val = args.a_ptr[u] * args.b_ptr[u];
    }
    float sg_s = simd_sum(val);
    if (simd_lane == 0u) s_sum[simd_group] = sg_s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        float tot = 0.0f;
        for (uint sg = 0u; sg < num_sg; ++sg) tot += s_sum[sg];
        uint bid = i / tpg;
        args.red_ptr[bid] = tot;
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_vec_dot_pair
{
    threadgroup float s_tt[32];
    threadgroup float s_ts[32];
    int idx = (int)i;
    float v_tt = 0.0f;
    float v_ts = 0.0f;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            float tv = args.t_ptr[u];
            v_tt = tv * tv;
            v_ts = tv * args.s_ptr[u];
        }
    }
    float sg_tt = simd_sum(v_tt);
    float sg_ts = simd_sum(v_ts);
    if (simd_lane == 0u) {
        s_tt[simd_group] = sg_tt;
        s_ts[simd_group] = sg_ts;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        float tot_tt = 0.0f;
        float tot_ts = 0.0f;
        for (uint sg = 0u; sg < num_sg; ++sg) {
            tot_tt += s_tt[sg];
            tot_ts += s_ts[sg];
        }
        uint bid = i / tpg;
        args.red_ptr[2u * bid] = tot_tt;
        args.red_ptr[2u * bid + 1u] = tot_ts;
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_init
{
    int idx = (int)i;
    if (idx >= *args.num_units) return;
    int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
    if (*args.all_active == 0) {
        if (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    float r = args.tmp_ptr[u] / (*args.initial);
    args.r_ptr[u] = r;
    args.rhat_ptr[u] = r;
    args.p_ptr[u] = 0.0f;
    args.v_ptr[u] = 0.0f;
    args.delta_ptr[u] = 0.0f;
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_upd_p
{
    int idx = (int)i;
    if (idx >= *args.num_units) return;
    int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
    if (*args.all_active == 0) {
        if (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    args.p_ptr[u] = args.r_ptr[u] + (*args.beta) * (args.p_ptr[u] - (*args.omega) * args.v_ptr[u]);
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_upd_s
{
    threadgroup uint s_max[32];
    int idx = (int)i;
    uint loc_m = 0u;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            float alpha = *args.alpha;
            float sv = args.r_ptr[u] - alpha * args.v_ptr[u];
            args.s_ptr[u] = sv;
            args.delta_ptr[u] += alpha * args.tmp_ptr[u];
            float av = fabs(sv);
            loc_m = as_type<uint>(hf_isfinite(av) ? av : hf_inf());
        }
    }
    uint sg_m = simd_max(loc_m);
    if (simd_lane == 0u) s_max[simd_group] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        uint bm = 0u;
        for (uint sg = 0u; sg < num_sg; ++sg) bm = max(bm, s_max[sg]);
        if (bm != 0u) atomic_fetch_max_explicit(&args.norm_buf_ptr[0], bm, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_upd_r
{
    threadgroup uint s_max[32];
    int idx = (int)i;
    uint loc_m = 0u;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            float omega = *args.omega;
            float rv = args.s_ptr[u] - omega * args.t_ptr[u];
            args.r_ptr[u] = rv;
            args.delta_ptr[u] += omega * args.tmp_ptr[u];
            float av = fabs(rv);
            loc_m = as_type<uint>(hf_isfinite(av) ? av : hf_inf());
        }
    }
    uint sg_m = simd_max(loc_m);
    if (simd_lane == 0u) s_max[simd_group] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        uint bm = 0u;
        for (uint sg = 0u; sg < num_sg; ++sg) bm = max(bm, s_max[sg]);
        if (bm != 0u) atomic_fetch_max_explicit(&args.norm_buf_ptr[0], bm, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_accum
{
    int idx = (int)i;
    if (idx >= *args.num_units) return;
    int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
    if (*args.all_active == 0) {
        if (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] == 0) return;
    }
    Pair x = pair_add(
        Pair{args.newton_step_ptr[u], args.graph_solution_low_ptr[u]},
        graph_product(*args.initial, args.delta_ptr[u])
    );
    args.newton_step_ptr[u] = x.hi;
    args.graph_solution_low_ptr[u] = x.lo;
}

// HYDROFORGE METAL KERNEL BODY: graph_bicg_res_diff
{
    threadgroup uint s_max[32];
    int idx = (int)i;
    uint loc_m = 0u;
    if (idx < *args.num_units) {
        int u = (*args.full_domain != 0) ? idx : args.graph_unit_ptr[idx];
        bool active = (*args.all_active != 0)
            || (args.tree_control_ptr[(long)args.unit_comp_ptr[u] * TI_STRIDE + TI_SOLVE] != 0);
        if (active) {
            float diff = args.rhs_ptr[u] - args.tmp_ptr[u];
            args.tmp_ptr[u] = diff;
            float av = fabs(diff);
            loc_m = as_type<uint>(hf_isfinite(av) ? av : hf_inf());
        }
    }
    uint sg_m = simd_max(loc_m);
    if (simd_lane == 0u) s_max[simd_group] = sg_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        long total = (long)(*args.total_threads);
        uint group_threads = (uint)min((long)tpg, total - ((long)i - (long)lid));
        uint num_sg = (group_threads + simd_width - 1u) / simd_width;
        uint bm = 0u;
        for (uint sg = 0u; sg < num_sg; ++sg) bm = max(bm, s_max[sg]);
        if (bm != 0u) atomic_fetch_max_explicit(&args.norm_buf_ptr[0], bm, memory_order_relaxed);
    }
}

// HYDROFORGE METAL KERNEL BODY: graph_scale_div
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    args.out_ptr[idx] = args.in_ptr[idx] / args.scaling_ptr[idx];
}

// HYDROFORGE METAL KERNEL BODY: graph_report_error
{
    if (i == 0u) {
        report_err(args.newton_status_ptr, *args.code, *args.info);
    }
}

// ------------------------------------------------------------------------
// Graph snapshot rebuild sub-kernels.

// HYDROFORGE METAL KERNEL BODY: graph_rebuild_stage
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

// HYDROFORGE METAL KERNEL BODY: graph_rebuild_main_flux
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

// HYDROFORGE METAL KERNEL BODY: graph_rebuild_bif_flux
{
    int p = (int)i;
    if (p >= *args.num_bifurcation_paths) return;
    graph_flux_cw(
        p,
        *args.time_step_ptr,
        1.0f,
        *args.num_bifurcation_levels,
        args.bifurcation_catchment_idx_ptr,
        args.bifurcation_downstream_idx_ptr,
        args.catchment_elevation_ptr,
        args.river_height_ptr,
        args.trial_reference_ptr,
        args.trial_offset_ptr,
        args.bifurcation_width_ptr,
        args.bifurcation_length_ptr,
        args.bifurcation_elevation_ptr,
        args.bifurcation_manning_ptr,
        args.bifurcation_outflow_ptr,
        args.graph_path_values_ptr,
        args.graph_path_sum_ptr,
        args.newton_status_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: graph_rebuild_residual
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    graph_residual_fn(
        idx,
        *args.time_step_ptr,
        args.storage_ptr,
        args.runoff_ptr,
        args.trial_storage_ptr,
        args.trial_outflow_ptr,
        args.upstream_offset_ptr,
        args.upstream_index_ptr,
        args.incident_offset_ptr,
        args.incident_path_ptr,
        args.graph_path_sum_ptr,
        args.residual_ptr,
        args.balance_weight_ptr,
        args.graph_rhs_ptr
    );
}

// HYDROFORGE METAL KERNEL BODY: graph_rebuild_snapshot
{
    int idx = (int)i;
    if (idx < *args.num_catchments) {
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
    if (idx < *args.num_bifurcation_values) {
        long base = (long)idx * 7L;
        for (int k = 0; k < 7; ++k) {
            args.bifurcation_jacobian_ptr[base + k] = args.graph_path_values_ptr[base + k];
        }
    }
}

// ------------------------------------------------------------------------
// Graph transpose adjoint sub-kernels.

// HYDROFORGE METAL KERNEL BODY: graph_adjoint_prep_cells
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    device const float* saved = args.jacobian_snapshot_ptr;
    device const float* bsaved = args.bifurcation_jacobian_ptr;
    long b = (long)idx * 17L;
    float qi = saved[b + 2L];
    float qd = saved[b + 3L];
    float area = saved[b + 4L];
    float bound = saved[b + 5L];
    args.outflow_own_slope_ptr[idx] = qi;
    args.outflow_down_slope_ptr[idx] = qd;
    args.trial_area_ptr[idx] = area;
    args.graph_bound_ptr[idx] = bound;

    Pair value = pair_add(
        pair_add(
            graph_product(area, args.grad_storage_ptr[idx]),
            graph_product(qi, args.grad_outflow_ptr[idx])
        ),
        Pair{args.grad_water_surface_ptr[idx], 0.0f}
    );
    for (int e = args.upstream_offset_ptr[idx]; e < args.upstream_offset_ptr[idx + 1]; ++e) {
        int j = args.upstream_index_ptr[e];
        value = pair_add(value, graph_product(saved[(long)j * 17L + 3L], args.grad_outflow_ptr[j]));
    }
    int bl = *args.num_bifurcation_levels;
    for (int e = args.incident_offset_ptr[idx]; e < args.incident_offset_ptr[idx + 1]; ++e) {
        int link = args.incident_path_ptr[e];
        int p = hf_iabs(link) - 1;
        long col = link > 0 ? 1L : 2L;
        for (int l = 0; l < bl; ++l) {
            int k = p * bl + l;
            value = pair_add(
                value,
                graph_product(bsaved[(long)k * 7L + col], args.grad_bifurcation_outflow_ptr[k])
            );
        }
    }
    args.graph_rhs_ptr[idx] = value.hi + value.lo;
}

// HYDROFORGE METAL KERNEL BODY: graph_adjoint_prep_paths
{
    int p = (int)i;
    if (p >= *args.num_bifurcation_paths) return;
    int bl = *args.num_bifurcation_levels;
    float a = 0.0f;
    float b = 0.0f;
    for (int l = 0; l < bl; ++l) {
        int k = p * bl + l;
        a += args.bifurcation_jacobian_ptr[(long)k * 7L + 1L];
        b += args.bifurcation_jacobian_ptr[(long)k * 7L + 2L];
    }
    args.graph_path_sum_ptr[(long)p * 3L + 1L] = a;
    args.graph_path_sum_ptr[(long)p * 3L + 2L] = b;
}

// HYDROFORGE METAL KERNEL BODY: graph_adjoint_step1
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    device const float* saved = args.jacobian_snapshot_ptr;
    float dt = saved[16L];
    Pair lambda = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, idx);
    Pair d_lam = args.downstream_edge_ptr[idx] != 0
        ? graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, args.downstream_idx_ptr[idx])
        : Pair{0.0f, 0.0f};
    float ce = graph_flux_adjoint_fn(dt, lambda, d_lam, args.grad_outflow_ptr[idx]);
    args.graph_temporary_ptr[idx] = ce;
    Pair rg = graph_scale(lambda, dt);
    long b = (long)idx * 17L;
    args.grad_runoff_ptr[idx] = rg.hi + rg.lo;
    args.grad_outflow_ptr[idx] = ce * saved[b + 12L];
    Pair diff = pair_add(Pair{args.grad_storage_ptr[idx], 0.0f}, Pair{-lambda.hi, -lambda.lo});
    args.grad_river_width_ptr[idx] += (diff.hi + diff.lo) * saved[b + 7L] + ce * saved[b + 9L];
    args.grad_river_manning_ptr[idx] += ce * saved[b + 13L];
    args.grad_flood_manning_ptr[idx] += ce * saved[b + 14L];
}

// HYDROFORGE METAL KERNEL BODY: graph_adjoint_step2
{
    int idx = (int)i;
    if (idx >= *args.num_catchments) return;
    device const float* saved = args.jacobian_snapshot_ptr;
    device const float* bsaved = args.bifurcation_jacobian_ptr;
    float dt = saved[16L];
    long b = (long)idx * 17L;
    Pair lambda = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, idx);
    Pair diff = pair_add(Pair{args.grad_storage_ptr[idx], 0.0f}, Pair{-lambda.hi, -lambda.lo});
    float height = (diff.hi + diff.lo) * saved[b + 8L] + args.graph_temporary_ptr[idx] * saved[b + 10L];
    if (saved[b + 5L] > 0.5f) {
        height -= (args.newton_step_ptr[idx] + args.graph_solution_low_ptr[idx]);
    }
    for (int e = args.upstream_offset_ptr[idx]; e < args.upstream_offset_ptr[idx + 1]; ++e) {
        int j = args.upstream_index_ptr[e];
        height += args.graph_temporary_ptr[j] * saved[(long)j * 17L + 11L];
    }
    int bl = *args.num_bifurcation_levels;
    for (int e = args.incident_offset_ptr[idx]; e < args.incident_offset_ptr[idx + 1]; ++e) {
        int link = args.incident_path_ptr[e];
        int p = hf_iabs(link) - 1;
        int pu = args.bifurcation_catchment_idx_ptr[p];
        int pd = args.bifurcation_downstream_idx_ptr[p];
        Pair lu = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, pu);
        Pair ld_ = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, pd);
        long col = link > 0 ? 5L : 6L;
        for (int l = 0; l < bl; ++l) {
            int k = p * bl + l;
            float ce = graph_flux_adjoint_fn(dt, lu, ld_, args.grad_bifurcation_outflow_ptr[k]);
            height += ce * bsaved[(long)k * 7L + col];
        }
    }
    args.grad_river_height_ptr[idx] += height;
    args.grad_storage_ptr[idx] = lambda.hi + lambda.lo;
}

// HYDROFORGE METAL KERNEL BODY: graph_adjoint_step3
{
    int p = (int)i;
    if (p >= *args.num_bifurcation_paths) return;
    device const float* saved = args.jacobian_snapshot_ptr;
    device const float* bsaved = args.bifurcation_jacobian_ptr;
    float dt = saved[16L];
    int pu = args.bifurcation_catchment_idx_ptr[p];
    int pd = args.bifurcation_downstream_idx_ptr[p];
    Pair lu = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, pu);
    Pair ld_ = graph_lambda_fn(saved, args.newton_step_ptr, args.graph_solution_low_ptr, pd);
    int bl = *args.num_bifurcation_levels;
    for (int l = 0; l < bl; ++l) {
        int k = p * bl + l;
        float ce = graph_flux_adjoint_fn(dt, lu, ld_, args.grad_bifurcation_outflow_ptr[k]);
        args.grad_bifurcation_outflow_ptr[k] = ce * bsaved[(long)k * 7L + 3L];
        args.grad_bifurcation_manning_ptr[k] += ce * bsaved[(long)k * 7L + 4L];
    }
}
