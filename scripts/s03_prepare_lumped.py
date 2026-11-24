"""Step 3: build lumped inputs of every station (forcing and attributes over upstream catchments).

    python -m scripts.s03_prepare_lumped PARAMS GAUGES FORCING_DIR ATTRIBUTES OUT_DIR
        [--first 1979] [--last 2019] [--outlet NAME ...] [--workers 8]

For the catchment of every station in GAUGES, its upstream catchments (itself
and all that drain to it, from ``downstream_id`` of PARAMS) weighted by
``catchment_area``. Writes, in the layout of the catchment forcing (hydroforge
exported data: ``(time, saved_points)`` with the ``catchment_id`` coordinate,
here the stations' catchments):

- ``OUT_DIR/{variable}_{year}.nc``: the area-weighted upstream mean of every
  variable of ``FORCING_DIR/{variable}_{year}.nc``;
- ``OUT_DIR/attributes.nc``: every attribute of ATTRIBUTES as the area-weighted
  upstream mean, except the ``--outlet`` ones (the channel at the station:
  geometry by default), which keep the station's own value; and
  ``drainage_area`` (m2), the summed upstream catchment area.

Every upstream catchment of every station must be a point of each input file
(once, with finite values), or nothing is written for that file.

``scripts.s06_lumped_baseline`` reads them like the catchment inputs.
"""

from __future__ import annotations

import argparse
from multiprocessing import get_context
from pathlib import Path

import netCDF4 as nc
import numpy as np
from hydroforge.io.rank_output.schema import (
    POINT_DIM,
    TIME_DIM,
    create_time_axis,
    write_point_coordinate,
)

from scripts.training.data import read_complete

OUTLET = ("river_length", "river_width", "river_height", "catchment_area", "upstream_area",
          "catchment_elevation", "downstream_distance", "reference_bed_slope")


class CSRWeights:
    """Lightweight pure-NumPy CSR matrix supporting ``weights @ x`` for 1D and 2D arrays."""

    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
                 shape: tuple[int, int]):
        self.data = np.ascontiguousarray(data, dtype=np.float64)
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def __matmul__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x)
        if arr.shape[0] != self.shape[1]:
            raise ValueError(f"shape mismatch: {self.shape} @ {arr.shape}")
        n_rows = self.shape[0]
        if arr.ndim == 1:
            out = np.empty(n_rows, dtype=np.float64)
            arr64 = np.asarray(arr, dtype=np.float64)
            for s in range(n_rows):
                a, b = int(self.indptr[s]), int(self.indptr[s + 1])
                out[s] = float(np.dot(self.data[a:b], arr64[self.indices[a:b]]))
            return out
        if arr.ndim == 2:
            # arr: (points, time) -> out: (stations, time)
            out = np.empty((n_rows, arr.shape[1]), dtype=np.float64)
            for s in range(n_rows):
                a, b = int(self.indptr[s]), int(self.indptr[s + 1])
                out[s] = self.data[a:b] @ arr[self.indices[a:b]]
            return out
        raise ValueError(f"unsupported array ndim {arr.ndim}")


def upstream_weights(catchment_id, downstream_id, area, stations):
    """Sparse (station, catchment) area weights of every station's upstream catchments
    (rows sum to 1) and each station's drainage area."""
    n = len(catchment_id)
    index = {int(c): i for i, c in enumerate(catchment_id)}
    down = np.array([index.get(int(d), -1) for d in downstream_id], dtype=np.int64)
    child = np.flatnonzero((down >= 0) & (down != np.arange(n)))
    # Downstream -> upstream CSR adjacency for BFS from each station.
    counts = np.bincount(down[child], minlength=n)
    up_ptr = np.r_[0, np.cumsum(counts)]
    up_idx = child[np.argsort(down[child], kind="stable")]

    indptr = [0]
    columns, values, drainage = [], [], []
    for cid in stations:
        root = index[int(cid)]
        basin = [root]
        head = 0
        while head < len(basin):
            u = basin[head]
            head += 1
            a, b = int(up_ptr[u]), int(up_ptr[u + 1])
            if a < b:
                basin.extend(up_idx[a:b].tolist())
        b_arr = np.asarray(basin, dtype=np.int64)
        weight = area[b_arr].astype(np.float64)
        total = float(weight.sum())
        drainage.append(total)
        columns.append(b_arr)
        values.append(weight / total)
        indptr.append(indptr[-1] + len(b_arr))

    all_cols = np.concatenate(columns) if columns else np.zeros(0, dtype=np.int64)
    all_vals = np.concatenate(values) if values else np.zeros(0, dtype=np.float64)
    matrix = CSRWeights(all_vals, all_cols, np.asarray(indptr, dtype=np.int64),
                        shape=(len(stations), n))
    return matrix, np.array(drainage, dtype=np.float64)


def on_columns(weights, map_ids, ids, what):
    """``weights`` (station, map catchment) on the columns of a file whose points are ``ids``.

    Every upstream catchment of every station must be a point of the file, once."""
    ids = np.asarray(ids, np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{what}: duplicate catchment IDs")
    lookup = {int(c): j for j, c in enumerate(ids)}
    needed = np.unique(weights.indices)
    missing = [int(map_ids[k]) for k in needed if int(map_ids[k]) not in lookup]
    if missing:
        raise ValueError(f"{what} lacks {len(missing)} upstream catchments of the stations, "
                         f"e.g. {missing[:5]}")
    column = np.full(len(map_ids), -1, np.int64)
    column[needed] = [lookup[int(map_ids[k])] for k in needed]
    return CSRWeights(weights.data, column[weights.indices], weights.indptr,
                      shape=(weights.shape[0], len(ids)))


def _finite(values, what):
    if not np.isfinite(values).all():
        raise ValueError(f"{what}: nonfinite upstream means")
    return values


_WORK = {}


def _aggregate(job):
    """One ``{variable}_{year}.nc`` file: the upstream means of the stations."""
    name, source, target = job
    stations = _WORK["stations"]
    with nc.Dataset(source) as d:
        ids = np.asarray(d["catchment_id"][:], np.int64)
        values = read_complete(d[name]).astype(np.float32)  # (time, points)
        time = d["time"]
        times, units, calendar = time[:], time.units, getattr(time, "calendar", "standard")
        attributes = {k: d[name].getncattr(k) for k in d[name].ncattrs() if k != "_FillValue"}
    if not np.array_equal(_WORK.get("ids"), ids):  # reindex once per point order
        _WORK.update(ids=ids, local=on_columns(_WORK["weights"], _WORK["map_ids"], ids, str(source)))
    lumped = _finite(np.asarray(_WORK["local"] @ values.T, np.float32).T, str(source))
    temporary = Path(str(target) + ".tmp")
    with nc.Dataset(temporary, "w", format="NETCDF4") as d:
        d.createDimension(TIME_DIM, len(times))
        d.createDimension(POINT_DIM, len(stations))
        t = create_time_axis(d, calendar=calendar)
        if t.units != units:
            raise ValueError(f"{source}: time units {units!r}, expected {t.units!r}")
        t[:] = times
        write_point_coordinate(d, "catchment_id", stations)
        v = d.createVariable(name, "f4", (TIME_DIM, POINT_DIM), contiguous=True)
        v.setncatts(attributes)
        v[:] = lumped
    temporary.replace(target)
    return target.name


def build(params, gauges, forcing, attributes, out, *, first, last, outlet=OUTLET, workers=8):
    with nc.Dataset(params) as d:
        cid = np.asarray(d["catchment_id"][:], np.int64)
        downstream = np.asarray(d["downstream_id"][:], np.int64)
        area = read_complete(d["catchment_area"]).astype(np.float64)
    with nc.Dataset(gauges) as d:
        stations = np.asarray(d["catchment_id"][:], np.int64)
    weights, drainage = upstream_weights(cid, downstream, area, stations)
    print(f"{len(stations)} stations, {weights.nnz} station-catchment weights", flush=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Attributes: upstream means, outlet values at the station.
    temporary = out / "attributes.nc.tmp"
    with nc.Dataset(attributes) as src, nc.Dataset(temporary, "w", format="NETCDF4") as dst:
        ids = np.asarray(src["catchment_id"][:], np.int64)
        local = on_columns(weights, cid, ids, str(attributes))
        at_station = {int(c): j for j, c in enumerate(ids)}
        at_station = np.array([at_station[int(c)] for c in stations])
        dst.createDimension("catchment", len(stations))
        dst.createVariable("catchment_id", "i8", ("catchment",))[:] = stations
        for name, var in src.variables.items():
            if name == "catchment_id":
                continue
            value = read_complete(var).astype(np.float64)
            lumped = _finite(value[at_station] if name in outlet else local @ value, f"{attributes}:{name}")
            out_var = dst.createVariable(name, "f4", ("catchment",))
            out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs() if k != "_FillValue"})
            out_var[:] = lumped.astype(np.float32)
        v = dst.createVariable("drainage_area", "f8", ("catchment",))
        v.units = "m2"
        v[:] = drainage
    temporary.replace(out / "attributes.nc")

    jobs = [(path.name.split("_")[0], path, out / path.name)
            for year in range(first, last + 1) for path in sorted(Path(forcing).glob(f"*_{year}.nc"))]
    _WORK.update(weights=weights, map_ids=cid, stations=stations)
    with get_context("fork").Pool(workers) as pool:
        for k, name in enumerate(pool.imap_unordered(_aggregate, jobs), 1):
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("params", "gauges", "forcing", "attributes", "out"):
        parser.add_argument(name)
    parser.add_argument("--first", type=int, default=1979)
    parser.add_argument("--last", type=int, default=2019)
    parser.add_argument("--outlet", nargs="*", default=list(OUTLET))
    parser.add_argument("--workers", type=int, default=8)
    a = parser.parse_args()
    build(a.params, a.gauges, a.forcing, a.attributes, a.out, first=a.first, last=a.last,
          outlet=set(a.outlet), workers=a.workers)


if __name__ == "__main__":
    main()
