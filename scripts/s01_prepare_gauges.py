"""Step 1: allocate GRDC daily discharge to a CaMa-Flood map into the standard gauge file.

    python -m scripts.s01_prepare_gauges GRDC_DIR MAP_DIR PARAMS OUT.nc
        [--start 1979-01-01] [--end 2019-12-31] [--max-error 0.2] [--min-years 3]

``GRDC_DIR`` holds the GRDC ``{country}.zip`` archives, ``MAP_DIR`` the
CaMa-Flood map with its high-resolution tiles (cmfgpu ``HiResMap``) and
``PARAMS`` the HydroNet parameter file whose catchments the stations must lie
in. A station is kept when it is allocated with a relative upstream-area
error of at most ``max-error`` and has at least ``min-years`` x 365 valid days
in ``[start, end]``, wherever they lie; of several in one catchment the one with
the most valid days. Negative discharge is missing; nothing is interpolated.

Outputs ``catchment_id``, ``station_id``, metadata, and ``discharge(time, station)``
(m3/s, NaN missing) on a CF daily ``time`` axis. Next step: ``scripts.s02_assign_folds``.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import netCDF4 as nc
import numpy as np

EPOCH = np.datetime64("1900-01-01")
YEAR = 365


def write_gauges(path, dates, catchment_id, discharge, **stations):
    """Write a gauge NetCDF file: ``discharge`` (time, station) on ``dates`` and per-station fields."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with nc.Dataset(temporary, "w", format="NETCDF4") as d:
        d.createDimension("time", len(dates))
        d.createDimension("station", len(catchment_id))
        t = d.createVariable("time", "f8", ("time",))
        t.units, t.calendar = "days since 1900-01-01 00:00:00", "standard"
        t[:] = (np.asarray(dates, "datetime64[D]") - EPOCH).astype(np.float64)
        d.createVariable("catchment_id", "i8", ("station",))[:] = catchment_id
        for name, values in stations.items():
            values = np.asarray(values)
            kind = str if values.dtype.kind in "US" else values.dtype
            d.createVariable(name, kind, ("station",))[:] = values.astype(str) if kind is str else values
        q = d.createVariable("discharge", "f4", ("time", "station"), zlib=True, complevel=4,
                             chunksizes=(len(dates), 1), fill_value=np.float32(np.nan))
        q.units = "m3/s"
        q[:] = discharge
    temporary.replace(path)


def one_per_catchment(catchment_id, valid_days, keep):
    """Among the kept stations, pick the one with the most valid days in each catchment."""
    order = np.lexsort((-valid_days, catchment_id))
    order = order[keep[order]]
    first = np.r_[True, catchment_id[order][1:] != catchment_id[order][:-1]]
    chosen = np.zeros(len(catchment_id), bool)
    chosen[order[first]] = True
    return chosen


def allocate(map_dir, station_id, lat, lon, area):
    """cmfgpu's catchment and upstream area (km2) of every station on the map."""
    from cmfgpu.params.allocation.hires_map import HiResMap

    with tempfile.TemporaryDirectory(prefix="hydronet-gauges-") as temporary:
        listing = Path(temporary) / "gauges.csv"
        with listing.open("w") as stream:
            stream.write("ID,Lat,Lon,Area_km2\n")
            for row in zip(station_id, lat, lon, area):
                stream.write("{},{:.6f},{:.6f},{:.3f}\n".format(*row))
        hires = HiResMap(map_dir=Path(map_dir), gauge_list=str(listing), out_dir=Path(temporary))
        hires.load_lowres()
        hires.load_hires()
        hires.build_upstream_table()
        hires.calc_outlet_pixels()
        hires.load_gauge_list()
        hires.allocate()
        result = hires.results_as_structured_array()
    np.testing.assert_array_equal(result["id"], station_id)
    return result["catchment_id1"].astype(np.int64), result["area1"].astype(np.float64)


def build(grdc_dir, map_dir, params, out, *, start, end, max_error, min_years):
    from cmfgpu.analysis.gauge_reader import extract_grdc_from_zips, scan_grdc_zips

    series = extract_grdc_from_zips(grdc_dir, scan_grdc_zips(grdc_dir), start, end)
    station_id = np.array(sorted(series), np.int64)
    meta = [series[int(s)][2] for s in station_id]
    lat, lon, area = (np.array([m.get(k, np.nan) for m in meta], np.float64)
                      for k in ("lat", "lon", "area_km2"))
    located = np.isfinite(lat) & np.isfinite(lon) & (area > 0)
    dates = np.arange(np.datetime64(start), np.datetime64(end) + 1)
    discharge = np.full((len(dates), located.sum()), np.nan, np.float32)
    for j, sid in enumerate(station_id[located]):
        when, values, _ = series[int(sid)]
        offset = (np.asarray(when, "datetime64[D]") - dates[0]).astype(np.int64)
        inside = (offset >= 0) & (offset < len(dates))
        discharge[offset[inside], j] = values[inside]
    discharge[discharge < 0] = np.nan
    station_id, lat, lon, area = (v[located] for v in (station_id, lat, lon, area))
    catchment, allocated = allocate(map_dir, station_id, lat, lon, area)
    error = (allocated - area) / area
    with nc.Dataset(params) as d:
        mapped = np.isin(catchment, np.asarray(d["catchment_id"][:], np.int64))
    valid_days = np.isfinite(discharge).sum(0)
    keep = (catchment >= 0) & mapped & (np.abs(error) <= max_error) & (valid_days >= min_years * YEAR)
    chosen = one_per_catchment(catchment, valid_days, keep)
    print(f"GRDC stations {len(series)} | located {len(station_id)} | allocated {int((catchment >= 0).sum())}"
          f" | in the map {int(((catchment >= 0) & mapped).sum())}"
          f" | |error| <= {max_error} {int(((catchment >= 0) & mapped & (np.abs(error) <= max_error)).sum())}"
          f" | >= {min_years} years {int(keep.sum())} | one per catchment {int(chosen.sum())}")
    write_gauges(out, dates, catchment[chosen], discharge[:, chosen], station_id=station_id[chosen],
                 lat=lat[chosen], lon=lon[chosen], reported_area_km2=area[chosen],
                 alloc_error=error[chosen].astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("grdc_dir", "map_dir", "params", "out"):
        parser.add_argument(name)
    parser.add_argument("--start", default="1979-01-01")
    parser.add_argument("--end", default="2019-12-31")
    parser.add_argument("--max-error", type=float, default=0.2)
    parser.add_argument("--min-years", type=float, default=3)
    a = parser.parse_args()
    build(a.grdc_dir, a.map_dir, a.params, a.out, start=a.start, end=a.end, max_error=a.max_error,
          min_years=a.min_years)


if __name__ == "__main__":
    main()
