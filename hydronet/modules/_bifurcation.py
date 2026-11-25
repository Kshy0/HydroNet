"""Online zero-scipy static sparse LU planner and component condensation for bifurcations.

The planner builds the exact symbolic LU pattern directly on GPU in pure PyTorch
in three stages:

1. Complete degree-1 peeling (zero fill-in) via vectorized GPU wavefronts.
2. Lazy degree-2 chain discovery on the 2-core so bridge chains between bifurcation
   loops peel at degree 1 with zero fill once an endpoint collapses, while true
   cycles contract cleanly at degree 2.
3. Exact integer-bucket Minimum-Fill ordering on the compact junction 3-core,
   followed by vectorized GPU assembly of all 23 packed arrays in ``FACTOR_LAYOUT``.

Because pure-forest trees have no bifurcation edges and zero fill in topological
order, :func:`condensed_graph_plan_tensors` runs the 2-core ordering only once on
the coupled subgraph and composes both ``core_factor_layout`` (forward solver)
and ``factor_layout`` (full-domain adjoint solver) in a single pass.
"""

from __future__ import annotations

from itertools import accumulate

import torch

from hydronet.config import FACTOR_HEAVY_WORK, FACTOR_LAYOUT
from hydronet.modules._forest import decompose_tensor

# Assembly sources of a factor entry: code = kind + 8 * index.
DIAGONAL, DOWN, UP, SOURCE_ROW, TARGET_ROW = range(5)

CORE_FACTOR_HEAVY_WORK = 512
_MAX_FILL_DEG = 16


def _empty_factor_plan(device: torch.device) -> torch.Tensor:
    lengths = [1 if k.endswith(("_ptr", "_offset")) else 0 for k in FACTOR_LAYOUT]
    offsets = list(accumulate([len(FACTOR_LAYOUT), *lengths[:-1]]))
    body = [0] * sum(lengths)
    return torch.tensor(offsets + body, dtype=torch.int32, device=device)


def _connected_components(
    num_nodes: int, u: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected connected components on GPU via pointer-jumping hook-and-compress.

    Returns ``(unique_roots, labels)`` where ``unique_roots[c]`` is the smallest
    node index in component ``c`` (ordered ascending) and ``labels[i] == c``.
    """
    device = u.device
    parent = torch.arange(num_nodes, device=device, dtype=torch.int64)
    if u.numel() > 0:
        ul = u.long()
        vl = v.long()
        while True:
            pu = parent[ul]
            pv = parent[vl]
            diff = pu != pv
            if not diff.any():
                break
            hi = torch.maximum(pu[diff], pv[diff])
            lo = torch.minimum(pu[diff], pv[diff])
            parent.scatter_reduce_(0, hi, lo, reduce="amin", include_self=True)
            while True:
                gp = parent[parent]
                if torch.equal(gp, parent):
                    break
                parent = gp
    uniq_roots, labels = torch.unique(parent, sorted=True, return_inverse=True)
    return uniq_roots, labels.to(torch.int32)


def _deg1_peel(
    eu: torch.Tensor, ev: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Complete degree-1 peeling of a simple undirected graph in pure PyTorch GPU."""
    device = eu.device
    m = eu.numel()
    deg = torch.zeros(n, device=device, dtype=torch.int32)
    sum_nbr = torch.zeros(n, device=device, dtype=torch.int64)
    if m > 0:
        eu_l, ev_l = eu.long(), ev.long()
        ones = torch.ones(m, device=device, dtype=torch.int32)
        deg.scatter_add_(0, eu_l, ones)
        deg.scatter_add_(0, ev_l, ones)
        sum_nbr.scatter_add_(0, eu_l, ev_l)
        sum_nbr.scatter_add_(0, ev_l, eu_l)

    status = torch.zeros(n, device=device, dtype=torch.int32)
    status[deg == 0] = 2
    in_front = torch.zeros(n, device=device, dtype=torch.bool)
    neg_ones = torch.full((n,), -1, device=device, dtype=torch.int32)

    frontier = torch.nonzero(deg == 1, as_tuple=False).squeeze(1)
    ord_chunks: list[torch.Tensor] = []
    par_chunks: list[torch.Tensor] = []

    while frontier.numel() > 0:
        in_front[frontier] = True
        v = sum_nbr[frontier]
        head_on = in_front[v]
        in_front[frontier] = False

        if head_on.any():
            u_ho = frontier[head_on]
            v_ho = v[head_on]
            smaller = u_ho < v_ho
            u_peel_ho = u_ho[smaller]
            v_peel_ho = v_ho[smaller]
            u_root_ho = u_ho[~smaller]
            if u_peel_ho.numel() > 0:
                status[u_peel_ho] = 1
                ord_chunks.append(u_peel_ho.to(torch.int32))
                par_chunks.append(v_peel_ho.to(torch.int32))
            if u_root_ho.numel() > 0:
                status[u_root_ho] = 2
            deg[u_ho] = 0

        normal = ~head_on
        if not normal.any():
            break
        u_n = frontier[normal]
        v_n = v[normal]
        status[u_n] = 1
        deg[u_n] = 0
        ord_chunks.append(u_n.to(torch.int32))
        par_chunks.append(v_n.to(torch.int32))

        deg.scatter_add_(0, v_n, neg_ones[: v_n.numel()])
        sum_nbr.scatter_add_(0, v_n, -u_n)
        touched = torch.unique(v_n)
        d_t = deg[touched]
        roots_now = touched[d_t == 0]
        if roots_now.numel() > 0:
            status[roots_now] = 2
        frontier = touched[d_t == 1]

    if ord_chunks:
        ord_p = torch.cat(ord_chunks)
        p_h1 = torch.cat(par_chunks)
    else:
        ord_p = torch.empty(0, device=device, dtype=torch.int32)
        p_h1 = torch.empty(0, device=device, dtype=torch.int32)
    return ord_p, p_h1, status


def _discover_chains(eu: torch.Tensor, ev: torch.Tensor, n_core: int):
    """Discover maximal degree-2 chains and junctions of a 2-core graph in pure PyTorch GPU."""
    device = eu.device
    m = eu.numel()
    deg = torch.zeros(n_core, device=device, dtype=torch.int32)
    sum_nbr = torch.zeros(n_core, device=device, dtype=torch.int64)
    if m > 0:
        eu_l, ev_l = eu.long(), ev.long()
        ones = torch.ones(m, device=device, dtype=torch.int32)
        deg.scatter_add_(0, eu_l, ones)
        deg.scatter_add_(0, ev_l, ones)
        sum_nbr.scatter_add_(0, eu_l, ev_l)
        sum_nbr.scatter_add_(0, ev_l, eu_l)

    du = deg[eu.long()]
    dv = deg[ev.long()]
    dir0 = (du >= 3) & (dv == 2)
    dir1 = (dv >= 3) & (du == 2)
    active_edges = dir0 | dir1

    if active_edges.any():
        e_idx = torch.nonzero(active_edges, as_tuple=False).squeeze(1)
        is_d0 = dir0[e_idx]
        start = torch.where(is_d0, eu[e_idx], ev[e_idx]).long()
        first = torch.where(is_d0, ev[e_idx], eu[e_idx]).long()
        n_cand = start.numel()

        prev = start.clone()
        cur = first.clone()
        cand_ids = torch.arange(n_cand, device=device, dtype=torch.int64)
        active = cand_ids
        step_cids: list[torch.Tensor] = [cand_ids]
        step_nodes: list[torch.Tensor] = [cur]
        end = torch.empty(n_cand, device=device, dtype=torch.int64)
        last_node = cur.clone()

        while active.numel() > 0:
            nxt = sum_nbr[cur] - prev
            cont = deg[nxt] == 2
            done = ~cont
            if done.any():
                done_ids = active[done]
                end[done_ids] = nxt[done]
                last_node[done_ids] = cur[done]
            if not cont.any():
                break
            active = active[cont]
            prev = cur[cont]
            cur = nxt[cont]
            step_cids.append(active)
            step_nodes.append(cur)

        win = (start < end) | ((start == end) & (first <= last_node))
        win_ids = torch.nonzero(win, as_tuple=False).squeeze(1)
        n_chains = win_ids.numel()
    else:
        n_chains = 0

    if n_chains > 0:
        ws = start[win_ids].to(torch.int32)
        we = end[win_ids].to(torch.int32)
        cid_map = torch.full((n_cand,), -1, device=device, dtype=torch.int64)
        cid_map[win_ids] = torch.arange(n_chains, device=device, dtype=torch.int64)
        all_cids = cid_map[torch.cat(step_cids)]
        all_nodes = torch.cat(step_nodes)
        keep = all_cids >= 0
        w_cids = all_cids[keep]
        w_nodes = all_nodes[keep]
        perm = torch.argsort(w_cids, stable=True)
        cnodes = w_nodes[perm].to(torch.int32)
        counts = torch.bincount(w_cids, minlength=n_chains).to(torch.int32)
        woff = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int32),
                counts.cumsum(0, dtype=torch.int32),
            ]
        )
        in_chain = torch.zeros(n_core, device=device, dtype=torch.bool)
        in_chain[cnodes.long()] = True
    else:
        ws = we = cnodes = torch.empty(0, device=device, dtype=torch.int32)
        woff = torch.zeros(1, device=device, dtype=torch.int32)
        in_chain = torch.zeros(n_core, device=device, dtype=torch.bool)

    jj_mask = (du >= 3) & (dv >= 3)
    jj_u = eu[jj_mask]
    jj_v = ev[jj_mask]
    iso_mask = (deg == 2) & (~in_chain)
    iso_edges = iso_mask[eu.long()] & iso_mask[ev.long()]
    iso_nodes = torch.nonzero(iso_mask, as_tuple=False).squeeze(1)
    iso_u = eu[iso_edges]
    iso_v = ev[iso_edges]
    junc_nodes = torch.nonzero(deg >= 3, as_tuple=False).squeeze(1)
    return (
        junc_nodes,
        jj_u,
        jj_v,
        ws,
        we,
        woff,
        cnodes,
        iso_nodes,
        iso_u,
        iso_v,
    )


def _solve_2core_from_chains(
    junc_list: list[int],
    jj_u: list[int],
    jj_v: list[int],
    ws: list[int],
    we: list[int],
    woff: list[int],
    cnodes: list[int],
    iso_nodes: list[int],
    iso_u: list[int],
    iso_v: list[int],
) -> tuple[list[int], list[int], list[int]]:
    """Order a 2-core with lazy degree-2 chain peeling and integer-bucket MinFill."""
    order: list[int] = []
    fill_u: list[int] = []
    fill_v: list[int] = []

    # 1. Pure isolated cycles (degree 2 everywhere)
    if iso_nodes:
        iso_adj: dict[int, list[int]] = {u: [] for u in iso_nodes}
        for u, v in zip(iso_u, iso_v):
            iso_adj[u].append(v)
            iso_adj[v].append(u)
        seen: set[int] = set()
        for start in iso_nodes:
            if start in seen:
                continue
            cyc = [start]
            seen.add(start)
            prev, cur = start, iso_adj[start][0]
            while cur != start:
                cyc.append(cur)
                seen.add(cur)
                a, b = iso_adj[cur]
                nxt = b if a == prev else a
                prev, cur = cur, nxt
            length = len(cyc)
            for k in range(1, length - 1):
                u = cyc[k]
                order.append(u)
                fill_u.append(u)
                fill_v.append(cyc[k + 1])
                fill_u.append(u)
                fill_v.append(start)
            if length >= 2:
                u = cyc[length - 1]
                order.append(u)
                fill_u.append(u)
                fill_v.append(start)
            order.append(start)

    nj = len(junc_list)
    if nj == 0:
        return order, fill_u, fill_v

    jmap = {u: i for i, u in enumerate(junc_list)}
    adj: list[set[int]] = [set() for _ in range(nj)]
    lazy_chain: dict[tuple[int, int], tuple[int, int, int]] = {}

    def contract_chain_deg2(sj_orig: int, ej_orig: int, cid: int) -> None:
        nodes = cnodes[woff[cid] : woff[cid + 1]]
        k = len(nodes)
        for i in range(k - 1):
            u = nodes[i]
            order.append(u)
            fill_u.append(u)
            fill_v.append(nodes[i + 1])
            fill_u.append(u)
            fill_v.append(sj_orig)
        u = nodes[k - 1]
        order.append(u)
        fill_u.append(u)
        fill_v.append(ej_orig)
        if sj_orig != ej_orig:
            fill_u.append(u)
            fill_v.append(sj_orig)

    def peel_chain_deg1(
        from_j_orig: int, to_j_orig: int, sj_orig: int, cid: int
    ) -> None:
        nodes = cnodes[woff[cid] : woff[cid + 1]]
        if from_j_orig != sj_orig:
            nodes = nodes[::-1]
        k = len(nodes)
        for i in range(k - 1):
            u = nodes[i]
            order.append(u)
            fill_u.append(u)
            fill_v.append(nodes[i + 1])
        u = nodes[k - 1]
        order.append(u)
        fill_u.append(u)
        fill_v.append(to_j_orig)

    for u_orig, v_orig in zip(jj_u, jj_v):
        u, v = jmap[u_orig], jmap[v_orig]
        adj[u].add(v)
        adj[v].add(u)

    for cid, (ws_orig, we_orig) in enumerate(zip(ws, we)):
        a, b = jmap[ws_orig], jmap[we_orig]
        if a == b:
            contract_chain_deg2(junc_list[a], junc_list[a], cid)
            continue
        key = (a, b) if a < b else (b, a)
        if b in adj[a]:
            contract_chain_deg2(junc_list[a], junc_list[b], cid)
        else:
            adj[a].add(b)
            adj[b].add(a)
            lazy_chain[key] = (a, b, cid)

    def materialize_edge(a: int, b: int) -> None:
        key = (a, b) if a < b else (b, a)
        ch = lazy_chain.pop(key, None)
        if ch is not None:
            sa, eb, cid = ch
            contract_chain_deg2(junc_list[sa], junc_list[eb], cid)

    alive = [True] * nj
    rem = nj

    def peel_deg01(u: int) -> int | None:
        nonlocal rem
        if not alive[u] or len(adj[u]) > 1:
            return None
        alive[u] = False
        rem -= 1
        au = adj[u]
        if not au:
            order.append(junc_list[u])
            return None
        v = next(iter(au))
        au.clear()
        adj[v].discard(u)
        key = (u, v) if u < v else (v, u)
        ch = lazy_chain.pop(key, None)
        if ch is not None:
            sa, eb, cid = ch
            b0, b1 = woff[cid], woff[cid + 1]
            first_node = cnodes[b0] if u == sa else cnodes[b1 - 1]
            order.append(junc_list[u])
            fill_u.append(junc_list[u])
            fill_v.append(first_node)
            peel_chain_deg1(junc_list[u], junc_list[v], junc_list[sa], cid)
        else:
            order.append(junc_list[u])
            fill_u.append(junc_list[u])
            fill_v.append(junc_list[v])
        return v

    # Initial degree-0/1 cascade (exposes bridge chains with zero fill-in)
    q01 = [u for u in range(nj) if len(adj[u]) <= 1]
    while q01:
        u = q01.pop()
        v = peel_deg01(u)
        if v is not None and alive[v] and len(adj[v]) <= 1:
            q01.append(v)

    rem_nodes = [u for u in range(nj) if alive[u]]
    if not rem_nodes:
        return order, fill_u, fill_v

    def fast_score(u: int) -> int:
        au = adj[u]
        d = len(au)
        if d <= 1:
            return 0
        if d == 2:
            a, b = au
            return 2 if b in adj[a] else (1 << 6) + 2
        if d == 3:
            a, b, c = au
            aa = adj[a]
            f = (b not in aa) + (c not in aa) + (c not in adj[b])
            return (f << 6) + 3
        if d <= _MAX_FILL_DEG:
            tot_pairs = (d * (d - 1)) >> 1
            existing = 0
            for v in au:
                existing += len(au & adj[v])
            f = tot_pairs - (existing >> 1)
            return (f << 6) + d
        return (((d * (d - 1)) >> 1) << 6) + min(d, 63)

    buckets: dict[int, list[int]] = {}
    cur_score = [-1] * nj
    min_s = 10**9
    for u in rem_nodes:
        s = fast_score(u)
        cur_score[u] = s
        b = buckets.get(s)
        if b is None:
            buckets[s] = [u]
        else:
            b.append(u)
        if s < min_s:
            min_s = s

    while rem > 0:
        while q01:
            u = q01.pop()
            v = peel_deg01(u)
            if v is not None and alive[v]:
                if len(adj[v]) <= 1:
                    q01.append(v)
                else:
                    sv = fast_score(v)
                    cur_score[v] = sv
                    b = buckets.get(sv)
                    if b is None:
                        buckets[sv] = [v]
                    else:
                        b.append(v)
                    if sv < min_s:
                        min_s = sv
        if rem == 0:
            break

        b = buckets.get(min_s)
        while not b:
            if min_s in buckets:
                del buckets[min_s]
            min_s = min(buckets)
            b = buckets[min_s]

        u = b.pop()
        if not alive[u] or cur_score[u] != min_s:
            continue
        d = len(adj[u])
        if d <= 1:
            q01.append(u)
            continue

        alive[u] = False
        rem -= 1
        nbrs = adj[u]
        adj[u] = set()
        for v in nbrs:
            materialize_edge(u, v)
        uo = junc_list[u]
        order.append(uo)
        for v in nbrs:
            fill_u.append(uo)
            fill_v.append(junc_list[v])

        nl = list(nbrs)
        if len(nl) == 2:
            a, b_n = nl[0], nl[1]
            adj[a].discard(u)
            adj[b_n].discard(u)
            if b_n not in adj[a]:
                adj[a].add(b_n)
                adj[b_n].add(a)
            else:
                materialize_edge(a, b_n)
            for w in (a, b_n):
                if len(adj[w]) <= 1:
                    q01.append(w)
                else:
                    sw = fast_score(w)
                    if sw != cur_score[w]:
                        cur_score[w] = sw
                        bk = buckets.get(sw)
                        if bk is None:
                            buckets[sw] = [w]
                        else:
                            bk.append(w)
                        if sw < min_s:
                            min_s = sw
        else:
            has_lazy = bool(lazy_chain)
            for i in range(len(nl)):
                vi = nl[i]
                avi = adj[vi]
                avi.discard(u)
                if has_lazy:
                    for j in range(i + 1, len(nl)):
                        vj = nl[j]
                        if vj not in avi:
                            avi.add(vj)
                            adj[vj].add(vi)
                        else:
                            materialize_edge(vi, vj)
                else:
                    avi.update(nbrs)
                    avi.discard(vi)
            for w in nl:
                if len(adj[w]) <= 1:
                    q01.append(w)
                else:
                    sw = fast_score(w)
                    if sw != cur_score[w]:
                        cur_score[w] = sw
                        bk = buckets.get(sw)
                        if bk is None:
                            buckets[sw] = [w]
                        else:
                            bk.append(w)
                        if sw < min_s:
                            min_s = sw

    return order, fill_u, fill_v


def _order_and_fill(
    n: int,
    eu_t: torch.Tensor,
    ev_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute elimination order ``order_t`` and sorted unique off-diagonal fill pairs ``(e_lo_t, e_hi_t)``."""
    device = eu_t.device
    if device.type not in ("cuda", "mps"):
        raise ValueError(
            f"HydroNet bifurcation planner requires a CUDA device, got {device}"
        )
    if n == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        return empty, empty, empty

    ord_p, p_h1, status = _deg1_peel(eu_t, ev_t, n)
    roots_loc = torch.nonzero(status == 2, as_tuple=False).squeeze(1).to(torch.int32)
    core_mask = status == 0
    core_loc = torch.nonzero(core_mask, as_tuple=False).squeeze(1).to(torch.int32)
    nc = core_loc.numel()
    if nc > 0:
        cmap = torch.full((n,), -1, device=device, dtype=torch.int32)
        cmap[core_loc.long()] = torch.arange(nc, device=device, dtype=torch.int32)
        ke = core_mask[eu_t.long()] & core_mask[ev_t.long()]
        eu_c = cmap[eu_t[ke].long()]
        ev_c = cmap[ev_t[ke].long()]
        chain_tensors = _discover_chains(eu_c, ev_c, nc)
        c_ord_loc, c_fu_loc, c_fv_loc = _solve_2core_from_chains(
            *(t.tolist() for t in chain_tensors)
        )
        c_ord_glob = core_loc[torch.tensor(c_ord_loc, device=device, dtype=torch.long)]
        c_fu_glob = core_loc[torch.tensor(c_fu_loc, device=device, dtype=torch.long)]
        c_fv_glob = core_loc[torch.tensor(c_fv_loc, device=device, dtype=torch.long)]
    else:
        c_ord_glob = c_fu_glob = c_fv_glob = torch.empty(
            0, device=device, dtype=torch.int32
        )

    order_t = torch.cat([ord_p, c_ord_glob, roots_loc])
    pos_t = torch.empty(n, device=device, dtype=torch.int32)
    pos_t[order_t.long()] = torch.arange(n, device=device, dtype=torch.int32)

    pu = pos_t[torch.cat([ord_p, c_fu_glob]).long()]
    pv = pos_t[torch.cat([p_h1, c_fv_glob]).long()]
    if pu.numel() > 0:
        flo = torch.minimum(pu, pv)
        fhi = torch.maximum(pu, pv)
        fkeys = torch.unique(flo.to(torch.int64) * n + fhi.to(torch.int64))
        e_lo_t = (fkeys // n).to(torch.int32)
        e_hi_t = (fkeys % n).to(torch.int32)
    else:
        e_lo_t = e_hi_t = torch.empty(0, device=device, dtype=torch.int32)
    return order_t, e_lo_t, e_hi_t


def _assemble_factor_plan_torch(
    n: int,
    gnode_t: torch.Tensor,
    order_t: torch.Tensor,
    e_lo_t: torch.Tensor,
    e_hi_t: torch.Tensor,
    link_u_t: torch.Tensor,
    link_d_t: torch.Tensor,
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
    *,
    heavy_work: int = FACTOR_HEAVY_WORK,
) -> tuple[torch.Tensor, int, int]:
    """Assemble the packed int32 ``FACTOR_LAYOUT`` tensor on ``order_t.device``.

    Returns ``(packed, n_entries, n_rounds)``.
    """
    device = order_t.device
    if n == 0:
        return _empty_factor_plan(device), 0, 0

    m_off = e_lo_t.numel()
    n_entries = n + 2 * m_off

    pos_t = torch.empty(n, device=device, dtype=torch.int32)
    pos_t[order_t.long()] = torch.arange(n, device=device, dtype=torch.int32)
    order_glob = gnode_t[order_t.long()]
    diag_t = torch.arange(n, device=device, dtype=torch.int32)

    # Entry numbering:
    #   [0, n)              : diagonal (k, k)
    #   [n, n + m_off)      : upper off-diagonal U(lo[e], hi[e])
    #   [n + m_off, n+2*m)  : lower off-diagonal L(hi[e], lo[e])
    lcol_cnt = torch.bincount(e_lo_t.long(), minlength=n).to(torch.int32)
    lcol_ptr = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            lcol_cnt.cumsum(0, dtype=torch.int32),
        ]
    )
    lcol_idx = e_hi_t
    lcol_ent = torch.arange(n + m_off, n + 2 * m_off, device=device, dtype=torch.int32)

    urow_cnt = lcol_cnt + 1
    urow_ptr = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            urow_cnt.cumsum(0, dtype=torch.int32),
        ]
    )
    urow_tot = n + m_off
    urow_idx = torch.empty(urow_tot, device=device, dtype=torch.int32)
    urow_ent = torch.empty(urow_tot, device=device, dtype=torch.int32)
    diag_pos = urow_ptr[:-1].long()
    urow_idx[diag_pos] = diag_t
    urow_ent[diag_pos] = diag_t
    if m_off > 0:
        off_urow_pos = (
            torch.arange(m_off, device=device, dtype=torch.int64) + e_lo_t.long() + 1
        )
        urow_idx[off_urow_pos] = e_hi_t
        urow_ent[off_urow_pos] = torch.arange(
            n, n + m_off, device=device, dtype=torch.int32
        )

    perm_hi = torch.argsort(e_hi_t, stable=True)
    lrow_cnt = torch.bincount(e_hi_t.long(), minlength=n).to(torch.int32)
    lrow_ptr = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            lrow_cnt.cumsum(0, dtype=torch.int32),
        ]
    )
    lrow_idx = e_lo_t[perm_hi]
    lrow_ent = (n + m_off) + perm_hi.to(torch.int32)
    ucol_ptr = lrow_ptr
    ucol_idx = lrow_idx
    ucol_ent = n + perm_hi.to(torch.int32)

    # Products: diagonal updates need no search; off-diagonal pairs use one symmetric search.
    p_diag_tgt = e_hi_t
    p_diag_el = lcol_ent
    p_diag_eu = torch.arange(n, n + m_off, device=device, dtype=torch.int32)
    p_diag_j = e_lo_t

    keys_sorted = e_lo_t.to(torch.int64) * n + e_hi_t.to(torch.int64)
    col2 = torch.nonzero(lcol_cnt >= 2, as_tuple=False).squeeze(1)
    if col2.numel() > 0:
        d2 = lcol_cnt[col2].long()
        pairs_per_col = (d2 * (d2 - 1)) >> 1
        total_pairs = int(pairs_per_col.sum().item())
        col_rep = torch.repeat_interleave(col2, pairs_per_col).to(torch.int32)
        pair_off = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.int64),
                pairs_per_col.cumsum(0)[:-1],
            ]
        )
        local_idx = torch.arange(
            total_pairs, device=device, dtype=torch.int64
        ) - torch.repeat_interleave(pair_off, pairs_per_col)
        b_tri = ((1.0 + torch.sqrt(1.0 + 8.0 * local_idx.to(torch.float64))) * 0.5).to(
            torch.int64
        )
        b_tri = torch.where(((b_tri * (b_tri - 1)) >> 1) > local_idx, b_tri - 1, b_tri)
        b_tri = torch.where(((b_tri + 1) * b_tri) >> 1 <= local_idx, b_tri + 1, b_tri)
        a_tri = local_idx - ((b_tri * (b_tri - 1)) >> 1)

        base_e = lcol_ptr[col_rep.long()].long()
        ea = base_e + a_tri
        eb = base_e + b_tri
        ra = e_hi_t[ea]
        rb = e_hi_t[eb]

        q_keys = ra.to(torch.int64) * n + rb.to(torch.int64)
        hit = torch.searchsorted(keys_sorted, q_keys).to(torch.int32)
        p_tgt = torch.cat([p_diag_tgt, n + hit, (n + m_off) + hit])
        p_el = torch.cat(
            [
                p_diag_el,
                (n + m_off) + ea.to(torch.int32),
                (n + m_off) + eb.to(torch.int32),
            ]
        )
        p_eu = torch.cat([p_diag_eu, n + eb.to(torch.int32), n + ea.to(torch.int32)])
        p_j = torch.cat([p_diag_j, col_rep, col_rep])
    else:
        p_tgt, p_el, p_eu, p_j = p_diag_tgt, p_diag_el, p_diag_eu, p_diag_j

    if p_tgt.numel() > 0:
        p_key = p_tgt.to(torch.int64) * n + p_j.to(torch.int64)
        p_perm = torch.argsort(p_key)
        p_cnt = torch.bincount(p_tgt.long(), minlength=n_entries).to(torch.int32)
        product_lower = p_el[p_perm]
        product_upper = p_eu[p_perm]
    else:
        p_cnt = torch.zeros(n_entries, device=device, dtype=torch.int32)
        product_lower = torch.empty(0, device=device, dtype=torch.int32)
        product_upper = torch.empty(0, device=device, dtype=torch.int32)
    product_ptr = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            p_cnt.cumsum(0, dtype=torch.int32),
        ]
    )

    # Assembly sources (DIAGONAL, main links, bifurcation paths)
    s_ent_list = [diag_t]
    s_code_list = [DIAGONAL + 8 * order_glob]
    if link_u_t.numel() > 0:
        pu = pos_t[link_u_t.long()]
        pd = pos_t[link_d_t.long()]
        ug = gnode_t[link_u_t.long()]
        lo_m = torch.minimum(pu, pd)
        hi_m = torch.maximum(pu, pd)
        hit_m = torch.searchsorted(
            keys_sorted, lo_m.to(torch.int64) * n + hi_m.to(torch.int64)
        ).to(torch.int32)
        e_ud = torch.where(pu < pd, n + hit_m, (n + m_off) + hit_m)
        e_du = torch.where(pd < pu, n + hit_m, (n + m_off) + hit_m)
        s_ent_list.extend([e_ud, e_du])
        s_code_list.extend([DOWN + 8 * ug, UP + 8 * ug])
    if src_t.numel() > 0:
        ps = pos_t[src_t.long()]
        pt = pos_t[dst_t.long()]
        p_id = torch.arange(src_t.numel(), device=device, dtype=torch.int32)
        lo_b = torch.minimum(ps, pt)
        hi_b = torch.maximum(ps, pt)
        hit_b = torch.searchsorted(
            keys_sorted, lo_b.to(torch.int64) * n + hi_b.to(torch.int64)
        ).to(torch.int32)
        e_st = torch.where(ps < pt, n + hit_b, (n + m_off) + hit_b)
        e_ts = torch.where(pt < ps, n + hit_b, (n + m_off) + hit_b)
        s_ent_list.extend([e_st, e_ts])
        s_code_list.extend([SOURCE_ROW + 8 * p_id, TARGET_ROW + 8 * p_id])

    s_ent = torch.cat(s_ent_list)
    s_code = torch.cat(s_code_list)
    s_perm = torch.argsort(s_ent, stable=True)
    s_cnt = torch.bincount(s_ent.long(), minlength=n_entries).to(torch.int32)
    source_ptr = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            s_cnt.cumsum(0, dtype=torch.int32),
        ]
    )
    source_code = s_code[s_perm]

    # Elimination tree heavy-path layout
    has_h = lcol_cnt > 0
    parent_t = torch.full((n,), -1, device=device, dtype=torch.int32)
    if has_h.any():
        h_cols = torch.nonzero(has_h, as_tuple=False).squeeze(1)
        parent_t[h_cols] = e_hi_t[lcol_ptr[h_cols].long()]
    etree = decompose_tensor(parent_t)
    hd_t, rnd_t = etree["head"], etree["round"]

    p_ord = torch.argsort(hd_t, stable=True)
    p_ord = p_ord[torch.argsort(rnd_t[p_ord], stable=True)].to(torch.int32)
    hd_s = hd_t[p_ord.long()]
    starts = torch.ones(n, device=device, dtype=torch.bool)
    if n > 1:
        starts[1:] = hd_s[1:] != hd_s[:-1]
    path_offset = torch.cat(
        [
            torch.nonzero(starts, as_tuple=False).squeeze(1).to(torch.int32),
            torch.tensor([n], device=device, dtype=torch.int32),
        ]
    )
    n_paths = path_offset.numel() - 1
    first_u = p_ord[path_offset[:-1].long()].long()
    p_rnd = rnd_t[first_u]
    r_cnt = torch.bincount(p_rnd.long()).to(torch.int32)
    round_offset = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            r_cnt.cumsum(0, dtype=torch.int32),
        ]
    )
    n_rounds = round_offset.numel() - 1

    # Heavy paths (exceeding heavy_work products + entries) lead each round
    work = (1 + p_cnt[:n]).clone()
    if m_off > 0:
        off_w = 2 + p_cnt[n : n + m_off] + p_cnt[n + m_off : n + 2 * m_off]
        work.scatter_add_(0, e_lo_t.long(), off_w)

    w_cs = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int64),
            work[p_ord.long()].cumsum(0, dtype=torch.int64),
        ]
    )
    path_work = w_cs[path_offset[1:].long()] - w_cs[path_offset[:-1].long()]
    is_heavy = (path_work > heavy_work).to(torch.int32)

    p_key_sort = p_rnd.to(torch.int64) * 2 + (1 - is_heavy).to(torch.int64)
    path_perm = torch.argsort(p_key_sort, stable=True)
    round_heavy = torch.zeros(n_rounds, device=device, dtype=torch.int32)
    round_heavy.scatter_add_(0, p_rnd.long(), is_heavy)

    p_len = path_offset[1:] - path_offset[:-1]
    new_len = p_len[path_perm]
    new_path_offset = torch.cat(
        [
            torch.zeros(1, device=device, dtype=torch.int32),
            new_len.cumsum(0, dtype=torch.int32),
        ]
    )
    path_id_per_elem = torch.repeat_interleave(
        torch.arange(n_paths, device=device, dtype=torch.int32), new_len.long()
    )
    local_off = (
        torch.arange(n, device=device, dtype=torch.int32)
        - new_path_offset[path_id_per_elem.long()]
    )
    old_start = path_offset[path_perm[path_id_per_elem.long()]]
    new_path_unit = p_ord[(old_start + local_off).long()]

    arrays = {
        "order": order_glob,
        "urow_ptr": urow_ptr,
        "urow_entry": urow_ent,
        "urow_index": urow_idx,
        "lcol_ptr": lcol_ptr,
        "lcol_entry": lcol_ent,
        "lcol_index": lcol_idx,
        "lrow_ptr": lrow_ptr,
        "lrow_entry": lrow_ent,
        "lrow_index": lrow_idx,
        "ucol_ptr": ucol_ptr,
        "ucol_entry": ucol_ent,
        "ucol_index": ucol_idx,
        "diag": diag_t,
        "product_ptr": product_ptr,
        "product_lower": product_lower,
        "product_upper": product_upper,
        "source_ptr": source_ptr,
        "source_code": source_code,
        "path_unit": new_path_unit,
        "path_offset": new_path_offset,
        "round_offset": round_offset,
        "round_heavy": round_heavy,
    }
    assert tuple(arrays) == FACTOR_LAYOUT
    lengths = [len(FACTOR_LAYOUT)] + [arrays[k].numel() for k in FACTOR_LAYOUT]
    offsets_list = list(accumulate(lengths))
    if offsets_list[-1] >= 2**31 or (
        8 * max(int(gnode_t.max().item()), src_t.numel()) + 4 >= 2**31
    ):
        raise ValueError("the graph factor exceeds int32 indexing")
    offsets = torch.tensor(offsets_list[:-1], device=device, dtype=torch.int32)
    packed = torch.cat([offsets] + [arrays[k].to(torch.int32) for k in FACTOR_LAYOUT])
    return packed, n_entries, n_rounds


def _build_unique_undirected_edges(
    n: int,
    link_u_t: torch.Tensor,
    link_d_t: torch.Tensor,
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = link_u_t.device
    all_u = torch.cat([link_u_t, src_t])
    all_v = torch.cat([link_d_t, dst_t])
    if all_u.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        return empty, empty
    lo = torch.minimum(all_u, all_v)
    hi = torch.maximum(all_u, all_v)
    nz = lo != hi
    lo, hi = lo[nz], hi[nz]
    if lo.numel() == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        return empty, empty
    keys = torch.unique(lo.to(torch.int64) * n + hi.to(torch.int64))
    return (keys // n).to(torch.int32), (keys % n).to(torch.int32)


def condensed_graph_plan_tensors(
    downstream_idx: torch.Tensor,
    downstream_edge: torch.Tensor,
    tree_index: torch.Tensor,
    tree_size: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    """Build component condensation, ``core_factor_layout`` and full ``factor_layout`` in one pass.

    Runs the 2-core ordering only once on the coupled subgraph and composes the
    full-domain plan by appending the pure-forest trees (which are already in
    zero-fill topological order).
    """
    device = downstream_idx.device
    if device.type not in ("cuda", "mps"):
        raise ValueError(
            f"HydroNet bifurcation planner requires a CUDA device, got {device}"
        )
    n = downstream_idx.numel()
    m = src_idx.numel()
    nt = tree_size.numel()

    down_t = torch.where(
        downstream_edge.bool(),
        downstream_idx.to(torch.int32),
        torch.full((n,), -1, device=device, dtype=torch.int32),
    )
    tree_idx_t = tree_index.to(torch.int32)
    tsize_t = tree_size.to(torch.int32)
    src_t = src_idx.to(torch.int32)
    dst_t = dst_idx.to(torch.int32)

    # 1. Separate pure-forest trees from coupled trees and find coupled components.
    tree_coupled = torch.zeros(nt, device=device, dtype=torch.bool)
    if m > 0:
        tree_coupled[tree_idx_t[src_t.long()].long()] = True
        tree_coupled[tree_idx_t[dst_t.long()].long()] = True

    pure_raw = torch.nonzero(~tree_coupled, as_tuple=False).squeeze(1)
    if pure_raw.numel() > 1:
        pure_raw = pure_raw[torch.argsort(-tsize_t[pure_raw], stable=True)]
    pure_block_tree = pure_raw.to(torch.int32)
    coupled_trees = (
        torch.nonzero(tree_coupled, as_tuple=False).squeeze(1).to(torch.int32)
    )

    n_ct = coupled_trees.numel()
    if n_ct > 0:
        ct_map = torch.full((nt,), -1, device=device, dtype=torch.int32)
        ct_map[coupled_trees.long()] = torch.arange(
            n_ct, device=device, dtype=torch.int32
        )
        tsrc = ct_map[tree_idx_t[src_t.long()].long()]
        tdst = ct_map[tree_idx_t[dst_t.long()].long()]
        keep_e = tsrc != tdst
        if keep_e.any():
            lo_t = torch.minimum(tsrc[keep_e], tdst[keep_e])
            hi_t = torch.maximum(tsrc[keep_e], tdst[keep_e])
            ek = torch.unique(lo_t.to(torch.int64) * n_ct + hi_t.to(torch.int64))
            eu_ct = ek // n_ct
            ev_ct = ek % n_ct
        else:
            eu_ct = ev_ct = torch.empty(0, device=device, dtype=torch.int64)

        uniq_roots, ct_comp = _connected_components(n_ct, eu_ct, ev_ct)
        n_comps = int(uniq_roots.numel())
        comp_rep_tree = coupled_trees[uniq_roots]
        tree_to_comp = torch.full((nt,), -1, device=device, dtype=torch.int32)
        tree_to_comp[coupled_trees.long()] = ct_comp
        unit_comp = tree_to_comp[tree_idx_t.long()]
        path_comp = unit_comp[src_t.long()].to(torch.int32)
        graph_unit = (
            torch.nonzero(unit_comp >= 0, as_tuple=False).squeeze(1).to(torch.int32)
        )
    else:
        n_comps = 0
        ct_comp = torch.empty(0, device=device, dtype=torch.int32)
        comp_rep_tree = torch.empty(0, device=device, dtype=torch.int32)
        unit_comp = torch.full((n,), -1, device=device, dtype=torch.int32)
        path_comp = torch.empty(m, device=device, dtype=torch.int32)
        graph_unit = torch.empty(0, device=device, dtype=torch.int32)

    # 2. Order & fill the coupled subgraph ONCE -> core_factor_layout + core_entry_comp.
    nc = graph_unit.numel()
    idx_all = torch.arange(n, device=device, dtype=torch.int32)
    all_link_mask = (down_t >= 0) & (down_t != idx_all)
    all_link_u = torch.nonzero(all_link_mask, as_tuple=False).squeeze(1).to(torch.int32)
    all_link_d = down_t[all_link_u.long()]

    if nc > 0:
        cmap = torch.full((n,), -1, device=device, dtype=torch.int32)
        cmap[graph_unit.long()] = torch.arange(nc, device=device, dtype=torch.int32)
        cdown = down_t[graph_unit.long()]
        cdown = torch.where(cdown >= 0, cmap[cdown.clamp(min=0).long()], cdown)
        cidx = torch.arange(nc, device=device, dtype=torch.int32)
        clink_mask = (cdown >= 0) & (cdown != cidx)
        clink_u = torch.nonzero(clink_mask, as_tuple=False).squeeze(1).to(torch.int32)
        clink_d = cdown[clink_u.long()]
        csrc = cmap[src_t.long()]
        cdst = cmap[dst_t.long()]

        ceu, cev = _build_unique_undirected_edges(nc, clink_u, clink_d, csrc, cdst)
        c_order_loc, c_elo, c_ehi = _order_and_fill(nc, ceu, cev)
        core_plan, n_core_entries, n_core_rounds = _assemble_factor_plan_torch(
            nc,
            graph_unit,
            c_order_loc,
            c_elo,
            c_ehi,
            clink_u,
            clink_d,
            csrc,
            cdst,
            heavy_work=CORE_FACTOR_HEAVY_WORK,
        )
        # Entry component ID directly from our [diag, U, L] entry layout (zero loops!)
        col_comp = unit_comp[graph_unit[c_order_loc.long()].long()]
        core_entry_comp = torch.cat(
            [col_comp, col_comp[c_elo.long()], col_comp[c_elo.long()]]
        ).to(torch.int32)
    else:
        core_plan = _empty_factor_plan(device)
        n_core_entries = 0
        n_core_rounds = 0
        core_entry_comp = torch.empty(0, device=device, dtype=torch.int32)
        c_order_loc = c_elo = c_ehi = torch.empty(0, device=device, dtype=torch.int32)

    # 3. Compose full-domain factor_layout by appending pure-forest nodes (zero fill!).
    #    Pure-forest nodes are already ordered upstream-before-downstream, so every pure
    #    link u -> down[u] has pos[u] < pos[down[u]] and produces zero fill-in.
    pure_units = torch.nonzero(unit_comp < 0, as_tuple=False).squeeze(1).to(torch.int32)
    if nc == n:
        full_order = graph_unit[c_order_loc.long()]
        full_elo = c_elo
        full_ehi = c_ehi
    elif nc == 0:
        full_order = idx_all
        full_elo = all_link_u
        full_ehi = all_link_d
    else:
        full_order = torch.cat([graph_unit[c_order_loc.long()], pure_units])
        full_pos = torch.empty(n, device=device, dtype=torch.int32)
        full_pos[full_order.long()] = idx_all
        pure_link_mask = (unit_comp < 0) & all_link_mask
        pure_lu = torch.nonzero(pure_link_mask, as_tuple=False).squeeze(1)
        pure_ld = down_t[pure_lu].long()
        # pure_lu is strictly increasing -> full_pos[pure_lu] is strictly increasing >= nc!
        p_elo = full_pos[pure_lu]
        p_ehi = full_pos[pure_ld]
        full_elo = torch.cat([c_elo, p_elo])
        full_ehi = torch.cat([c_ehi, p_ehi])

    full_plan, n_full_entries, n_full_rounds = _assemble_factor_plan_torch(
        n,
        idx_all,
        full_order,
        full_elo,
        full_ehi,
        all_link_u,
        all_link_d,
        src_t,
        dst_t,
        heavy_work=FACTOR_HEAVY_WORK,
    )

    return dict(
        pure_block_tree=pure_block_tree,
        graph_unit=graph_unit,
        unit_comp=unit_comp,
        path_comp=path_comp,
        coupled_trees=coupled_trees,
        coupled_tree_comp=ct_comp,
        comp_rep_tree=comp_rep_tree,
        num_graph_comps=n_comps,
        core_factor_layout=core_plan,
        core_entry_comp=core_entry_comp,
        num_core_factor_entries=n_core_entries,
        num_core_factor_rounds=n_core_rounds,
        factor_layout=full_plan,
        num_factor_entries=n_full_entries,
        num_factor_rounds=n_full_rounds,
    )
