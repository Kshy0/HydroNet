"""Heavy-path decomposition of the river forest.

Units are ordered upstream before downstream (``downstream[i] > i``; outlets
point to themselves or outside the map). Every unit continues the path of its
upstream unit with the largest subtree; paths run from their most upstream
unit to their outlet and fall into rounds by the number of light links between
them and the root. Eliminating rounds from the deepest to the root, and
substituting in the reverse order, solves a tree system in at most
log2(n) + 1 independent path rounds.

:func:`decompose_tensor` computes the exact decomposition directly on GPU via
$O(\\log N)$ Miller-Reif rake-and-compress tree contraction and pointer jumping
in pure PyTorch, backing the online ``computed_tensor_field`` layout of
:mod:`hydronet.modules.base` and the graph elimination tree of
:mod:`hydronet.modules._bifurcation`.
"""

from __future__ import annotations

import torch


def decompose_tensor(downstream: torch.Tensor) -> dict[str, torch.Tensor]:
    """Outlet, path head, path round and height of ``downstream`` as int32 GPU tensors."""
    device = downstream.device
    if device.type not in ("cuda", "mps"):
        raise ValueError(
            f"HydroNet forest decomposition requires a CUDA device, got {device}"
        )
    n = downstream.numel()
    if n == 0:
        empty = torch.empty(0, device=device, dtype=torch.int32)
        return dict(down=empty, outlet=empty, head=empty, round=empty, height=empty)

    idx = torch.arange(n, device=device, dtype=torch.int64)
    down64 = downstream.to(torch.int64)
    down64 = torch.where(down64 == idx, torch.full_like(down64, -1), down64)
    if (
        torch.any(down64 >= n).item()
        or torch.any(down64 < -1).item()
        or torch.any((down64 >= 0) & (down64 <= idx)).item()
    ):
        raise ValueError("catchments must be ordered upstream before downstream")

    down32 = down64.to(torch.int32)
    valid = down64 >= 0
    p = down64.clone()
    indeg = torch.bincount(p[valid], minlength=n).to(torch.int32)
    rem = indeg.clone()
    alive = torch.ones(n, device=device, dtype=torch.bool)

    size = torch.ones(n, device=device, dtype=torch.int32)
    height = torch.zeros(n, device=device, dtype=torch.int32)
    best = torch.zeros(n, device=device, dtype=torch.int64)

    # Edge summary along compressed edge u -> p[u]:
    #   entering_size(u -> p[u])   = size[u] + w_sz[u]
    #   entering_height(u -> p[u]) = max(height[u] + d_len[u], w_ht[u])
    #   entering_child(u -> p[u])  = ec[u] (direct original child of p[u])
    w_sz = torch.zeros(n, device=device, dtype=torch.int32)
    d_len = torch.zeros(n, device=device, dtype=torch.int32)
    w_ht = torch.full((n,), -1, device=device, dtype=torch.int32)
    ec = idx.clone()

    stack: list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ] = []
    active_nodes = idx
    bit = 0
    neg_ones = torch.full((n,), -1, device=device, dtype=torch.int32)

    # O(log N) Miller-Reif rake-and-compress tree contraction.
    while True:
        rem_a = rem[active_nodes]
        p_a = p[active_nodes]
        is_leaf = (rem_a == 0) & (p_a >= 0)
        if is_leaf.any():
            u = active_nodes[is_leaf]
            pu = p_a[is_leaf]
            sz_in = size[u] + w_sz[u]
            ht_in = torch.maximum(height[u] + d_len[u], w_ht[u])
            size.scatter_add_(0, pu, sz_in)
            height.scatter_reduce_(0, pu, ht_in + 1, reduce="amax", include_self=True)
            key = (sz_in.to(torch.int64) << 32) | (0x7FFFFFFF - ec[u])
            best.scatter_reduce_(0, pu, key, reduce="amax", include_self=True)
            rem.scatter_add_(0, pu, neg_ones[: pu.numel()])
            alive[u] = False
            active_nodes = active_nodes[~is_leaf]
            if active_nodes.numel() == 0:
                break

        p_a = p[active_nodes]
        has_p = p_a >= 0
        if not has_p.any():
            break

        u_cand = active_nodes[has_p]
        v_cand = p_a[has_p]
        v_ok = (rem[v_cand] == 1) & (p[v_cand] >= 0)
        if v_ok.any():
            u_c = u_cand[v_ok]
            v_c = v_cand[v_ok]
            shift = bit % 16
            h_v = (((v_c * 2654435761) >> shift) & 1) == 1
            u_is_comp = (rem[u_c] == 1) & (p[u_c] >= 0)
            h_u = (((u_c * 2654435761) >> shift) & 1) == 1
            pick = h_v & (~(u_is_comp & h_u))
            if not pick.any():
                pick = ~u_is_comp
                if not pick.any():
                    pick = torch.zeros_like(h_v)
                    pick[0] = True
            u_s = u_c[pick]
            v_s = v_c[pick]
            w_s = p[v_s]

            old_ec = ec[u_s].clone()
            old_wsz = w_sz[u_s].clone()
            old_dlen = d_len[u_s].clone()
            old_wht = w_ht[u_s].clone()
            stack.append((u_s, v_s, old_ec, old_wsz, old_dlen, old_wht))

            dv = d_len[v_s]
            w_sz[u_s] = old_wsz + size[v_s] + w_sz[v_s]
            d_len[u_s] = old_dlen + 1 + dv
            ht_from_u_edge = torch.where(old_wht >= 0, old_wht + 1 + dv, old_wht)
            ht_from_v = height[v_s] + dv
            w_ht[u_s] = torch.maximum(
                torch.maximum(ht_from_u_edge, ht_from_v), w_ht[v_s]
            )
            ec[u_s] = ec[v_s]
            p[u_s] = w_s

            alive[v_s] = False
            active_nodes = active_nodes[alive[active_nodes]]
            bit += 1

    # Unwind compressed nodes in reverse order to finalize exact size, height and heavy child.
    for u_s, v_s, old_ec, old_wsz, old_dlen, old_wht in reversed(stack):
        sz_in = size[u_s] + old_wsz
        ht_in = torch.maximum(height[u_s] + old_dlen, old_wht)
        size[v_s] = size[v_s] + sz_in
        height[v_s] = torch.maximum(height[v_s], ht_in + 1)
        key = (sz_in.to(torch.int64) << 32) | (0x7FFFFFFF - old_ec)
        best[v_s] = torch.maximum(best[v_s], key)

    # Pointer jumping for outlet, heavy-path head, and light-edge round: O(log max_h).
    idx32 = idx.to(torch.int32)
    outlet = torch.where(valid, down32, idx32)
    heavy_of = (0x7FFFFFFF - (best & 0x7FFFFFFF)).to(torch.int32)
    is_heavy = valid & (heavy_of[outlet.long()] == idx32)
    head = torch.where(is_heavy, outlet, idx32)
    rounds = (valid & (~is_heavy)).to(torch.int32)

    max_h = int(height.max().item())
    steps = max_h.bit_length() if max_h > 0 else 0
    for _ in range(steps):
        o_long = outlet.long()
        rounds = rounds + rounds[o_long]
        outlet = outlet[o_long]
        head = head[head.long()]

    return dict(down=down32, outlet=outlet, head=head, round=rounds, height=height)
