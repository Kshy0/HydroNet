"""Step 2: assign stations in a gauge file to k cross-validation folds by whole river groups.

    python -m scripts.s02_assign_folds GAUGES.nc PARAMS.nc [--folds 5] [--seed 42] [--bifurcation]

A river group is a tree, or with ``--bifurcation`` the trees its bifurcation
paths join (``scripts.training.domain.river_groups``): all its stations share
one fold, so a held-out fold is a set of rivers the model never saw. Groups are
dealt region by region (continents by the outlet's location), largest first,
each to the fold with the fewest stations of that region, so every fold has
about the same stations and mix of regions. Writes ``fold`` into GAUGES.nc.
"""

from __future__ import annotations

import argparse

import netCDF4 as nc
import numpy as np

from scripts.training.data import read_complete
from scripts.training.domain import river_groups


def region(lon, lat):
    """A coarse continent of a location."""
    if lat < -60:
        return "AN"
    if lon < -30:
        return "NA" if lat > 12 else "SA"
    if lat > 35 and lon < 60:
        return "EU"
    if lat <= 35 and lon < 55:
        return "AF"
    if lon > 110 and lat < -10:
        return "OC"
    return "AS"


def assign(groups, regions, folds, seed):
    """The fold of every station, from its group and its group's region."""
    rng = np.random.default_rng(seed)
    names, sizes = np.unique(groups, return_counts=True)
    fold_of = {}
    totals = np.zeros(folds, np.int64)
    for r in sorted(set(regions.values())):
        mine = [i for i, g in enumerate(names) if regions[int(g)] == r]
        mine = [mine[i] for i in rng.permutation(len(mine))]
        counts = np.zeros(folds, np.int64)
        for i in sorted(mine, key=lambda i: -sizes[i]):  # stable: ties keep the shuffled order
            f = int(np.lexsort((np.arange(folds), totals, counts))[0])
            fold_of[int(names[i])] = f
            counts[f] += sizes[i]
            totals[f] += sizes[i]
    return np.array([fold_of[int(g)] for g in groups], np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gauges")
    parser.add_argument("params")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bifurcation", action="store_true")
    a = parser.parse_args()
    cid, group = river_groups(a.params, bifurcation=a.bifurcation)
    with nc.Dataset(a.params) as d:
        located = "longitude" in d.variables and "latitude" in d.variables
        lon, lat = ((read_complete(d["longitude"]).astype(float), read_complete(d["latitude"]).astype(float))
                    if located else (np.zeros(len(cid)), np.zeros(len(cid))))
    index = {int(c): i for i, c in enumerate(cid)}
    with nc.Dataset(a.gauges, "a") as d:
        stations = np.asarray(d["catchment_id"][:], np.int64)
        groups = group[[index[int(c)] for c in stations]]
        regions = {int(g): region(lon[index[int(g)]], lat[index[int(g)]]) for g in np.unique(groups)}
        fold = assign(groups, regions, a.folds, a.seed)
        v = d["fold"] if "fold" in d.variables else d.createVariable("fold", "i4", ("station",))
        v[:] = fold
        v.folds, v.seed, v.bifurcation = a.folds, a.seed, int(a.bifurcation)
    for f in range(a.folds):
        mine = fold == f
        print(f"fold {f}: {int(mine.sum())} stations, {len(np.unique(groups[mine]))} groups, "
              + " ".join(f"{r}={sum(regions[int(g)] == r for g in groups[mine])}"
                         for r in sorted(set(regions.values()))))


if __name__ == "__main__":
    main()
