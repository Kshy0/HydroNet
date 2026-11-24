"""The training domain: the trees that can reach a gauge.

The loss reads only gauged catchments. A tree without a gauge, and without a
chain of bifurcation paths into a gauged tree, has no route to the loss, so its
runoff, roughness and routing contribute nothing to any gradient. The driver
keeps the other trees: whole trees, in the map's order, with all their
bifurcation paths.
"""

from __future__ import annotations

import netCDF4 as nc
import numpy as np

CATCHMENT, PATH, SAVED = "catchment", "bifurcation_path", "saved_points"


def _roots(index: np.ndarray, down: np.ndarray) -> np.ndarray:
    root = np.where(down >= 0, down, index)
    while True:
        jumped = root[root]
        if np.array_equal(jumped, root):
            return root
        root = jumped


def river_groups(params_nc, *, bifurcation: bool) -> tuple[np.ndarray, np.ndarray]:
    """The map's ``catchment_id`` and the group of each: the outlet ID of its tree, or with
    ``bifurcation`` the smallest outlet ID of the trees its bifurcation paths join to it.

    A group routes on its own (no path leaves it), so it can take its own
    calendar window and cross-validation fold.
    """
    with nc.Dataset(params_nc) as d:
        cid = np.asarray(d["catchment_id"][:], dtype=np.int64)
        downstream = np.asarray(d["downstream_id"][:], dtype=np.int64)
        paths = ((np.asarray(d["bifurcation_catchment_id"][:], dtype=np.int64),
                  np.asarray(d["bifurcation_downstream_id"][:], dtype=np.int64))
                 if bifurcation and "bifurcation_catchment_id" in d.variables else None)
    order = np.argsort(cid)
    position = lambda ids: order[np.searchsorted(cid, ids, sorter=order)]  # noqa: E731
    index = np.arange(len(cid))
    outlet = (downstream < 0) | (downstream == cid)
    down = np.where(outlet, -1, position(np.where(outlet, cid, downstream)))
    group = _roots(index, down)
    if paths is not None:  # union-find over the tree roots; the smaller outlet ID represents
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for u, v in zip(group[position(paths[0])].tolist(), group[position(paths[1])].tolist()):
            u, v = find(u), find(v)
            if u != v:
                low, high = (u, v) if cid[u] < cid[v] else (v, u)
                parent[high] = low
        roots = np.unique(group)
        joined = np.array([find(r) for r in roots.tolist()], np.int64)
        group = joined[np.searchsorted(roots, group)]
    return cid, cid[group]


def gauged_catchments(params_nc, gauge_cids, *, bifurcation: bool) -> np.ndarray:
    """Mask of the catchments in trees linked to a gauge (through bifurcations if open)."""
    cid, group = river_groups(params_nc, bifurcation=bifurcation)
    gauges = np.asarray(gauge_cids, dtype=np.int64)
    if not np.isin(gauges, cid).all():
        raise ValueError("gauges lie outside the map")
    return np.isin(group, group[np.isin(cid, gauges)])


def restrict_proxy(proxy, params_nc, keep: np.ndarray, *, bifurcation: bool):
    """``proxy`` restricted to the kept catchments, their saved points and their paths.

    With ``bifurcation`` every path touching a kept tree must stay inside the
    kept trees; otherwise paths leaving them are dropped (the module is closed).
    The forest heavy-path decomposition and bifurcation LU plan are computed
    online by the model modules for the restricted proxy.
    """
    with nc.Dataset(params_nc) as d:
        leading = {name: v.dimensions[0] for name, v in d.variables.items() if v.dimensions}
        dims = set(d.dimensions)
    kept_ids = np.asarray(proxy["catchment_id"])[keep]
    rows = {CATCHMENT: keep}
    if PATH in dims:
        inside = [np.isin(np.asarray(proxy[name]), kept_ids)
                  for name in ("bifurcation_catchment_id", "bifurcation_downstream_id")]
        if bifurcation and (inside[0] != inside[1]).any():
            raise ValueError("a bifurcation path leaves the kept trees")
        rows[PATH] = inside[0] & inside[1]
    if SAVED in dims:
        rows[SAVED] = np.isin(np.asarray(proxy["output_catchment_id"]), kept_ids)
    values = {name: np.asarray(proxy[name])[rows[dim]] for name, dim in leading.items() if dim in rows}
    dimensions = {dim: int(mask.sum()) for dim, mask in rows.items()}
    return proxy.updated(values=values, dimensions=dimensions)
