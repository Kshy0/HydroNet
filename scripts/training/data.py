"""Training inputs, all NetCDF: forcing, attributes, gauges and the CaMa-Flood restart.

- Forcing: one hydroforge ``ExportedDataset`` file set per variable,
  ``{forcing}/{variable}_{year}.nc`` with ``(time, saved_points)`` values and
  the ``catchment_id`` coordinate (the layout of exported runoff forcing).
- Attributes: ``catchment_id`` and one variable per static LSTM attribute
  (already transformed and standardized), plus the runoff prior
  ``runoff_mean``/``runoff_std`` (mm/day), on the ``catchment`` dimension.
- Gauges: ``catchment_id`` and ``discharge(time, station)`` (m3/s, NaN missing)
  with a CF ``time`` axis in days, optionally the cross-validation ``fold`` of
  every station (``scripts.s01_prepare_gauges``, ``scripts.s02_assign_folds``).
"""

from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
from hydroforge.data import InputProxy
from hydroforge.data.datasets import ExportedDataset, yearly_time_to_key

DAY = timedelta(days=1)


def align_ids(source, target):
    """Rows of ``source`` holding the ``target`` IDs in order."""
    lookup = {int(cid): i for i, cid in enumerate(source)}
    rows = np.array([lookup[int(cid)] for cid in target], np.int64)
    np.testing.assert_array_equal(np.asarray(source)[rows], target)
    return rows


def read_complete(variable, rows=None):
    """The values of a NetCDF variable (``rows`` of its first axis), all present and finite.

    NetCDF reads missing values as masked; dropping the mask would leave their
    fill values as ordinary numbers, so a masked value is an error."""
    values = variable[:] if rows is None else variable[:][rows]
    if np.ma.is_masked(values):
        raise ValueError(f"{variable.name}: {int(np.ma.getmaskarray(values).sum())} missing values")
    values = np.asarray(np.ma.getdata(values))
    if values.dtype.kind == "f" and not np.isfinite(values).all():
        raise ValueError(f"{variable.name}: nonfinite values")
    return values


def days(first_year, last_year):
    return np.arange(np.datetime64(f"{first_year}-01-01"), np.datetime64(f"{last_year + 1}-01-01"))


class Forcing:
    """Daily forcing of the model's units over whole years, held in host RAM.

    Only these units are read (``ExportedDataset.selected`` and
    ``load_to_memory``). :meth:`read` returns days ``[a, b)`` on the device:
    the variables of ``spec`` (``frontend.forcing``) standardized by its fixed
    statistics, the ``log1p`` ones transformed first, then the season as the
    sine and cosine of the day of the year. :meth:`shifted` gives a view in
    which every unit reads its own calendar window.
    """

    def __init__(self, directory, spec, ids, first_year, last_year, *, device):
        self.dates = days(first_year, last_year)
        self.device, self.ids, self.shift = device, np.asarray(ids, np.int64), None
        start, end = datetime(first_year, 1, 1), datetime(last_year, 12, 31)
        self.variables = []
        for name in spec.variables:
            source = ExportedDataset(
                base_dir=Path(directory), var_name=name, prefix=f"{name}_", suffix=".nc",
                time_to_key=yearly_time_to_key, start_date=start, end_date=end,
                time_interval=DAY, model_step=DAY)
            if source.num_main_source_steps != len(self.dates):
                raise ValueError(f"{name} does not hold every day of {first_year}-{last_year}")
            local = source.selected(self.ids)
            local.load_to_memory()
            local.close()
            self.variables.append(local)
        self.log1p = torch.tensor([v in spec.log1p for v in spec.variables], device=device)
        self.mean = torch.tensor(spec.mean, device=device, dtype=torch.float32)
        self.std = torch.tensor(spec.std, device=device, dtype=torch.float32)
        year_day = (self.dates - self.dates.astype("datetime64[Y]")).astype(np.float32)
        self.angle = torch.tensor(2 * np.pi * year_day / 365.2425, device=device)

    def shifted(self, shift, length):
        """A view of ``length`` routed days in which unit ``u`` reads day ``shift[u] + t`` at
        routed day ``t`` (the resident values are shared). Its ``dates`` are the routed day
        numbers."""
        shift = np.asarray(shift, np.int64)
        if shift.shape != self.ids.shape or shift.min() < 0 or shift.max() + length > len(self.dates):
            raise ValueError("shifted forcing windows must lie inside the reader period")
        view = copy.copy(self)
        view.variables = [v.selected(self.ids, time_shift_steps=shift) for v in self.variables]
        view.shift = torch.tensor(shift, device=self.device)
        view.dates = np.arange(length)
        return view

    def read(self, a, b):
        if not 0 <= a < b <= len(self.dates):
            raise ValueError("Forcing slice lies outside this reader period")
        raw = np.stack([v.read_window(a, b - a) for v in self.variables], -1)
        forcing = torch.from_numpy(raw).pin_memory().to(self.device, non_blocking=True)
        forcing = torch.where(self.log1p, torch.log1p(forcing.clamp_min(0.0)), forcing)
        forcing = (forcing - self.mean) / self.std
        steps = torch.arange(a, b, device=self.device)
        angle = (self.angle[steps][:, None].expand(-1, forcing.shape[1]) if self.shift is None
                 else self.angle[steps[:, None] + self.shift[None]])
        seasonal = torch.stack([torch.sin(angle), torch.cos(angle)], -1)
        return torch.cat([forcing, seasonal], -1), self.dates[a:b]


def load_attributes(path, names, ids):
    """Static attributes ``names`` (N, F) and the runoff prior mean and std of ``ids``, all
    finite (``InputProxy`` already refuses missing values)."""
    data = InputProxy.from_nc(path, visible_vars=["catchment_id", "runoff_mean", "runoff_std", *names])
    rows = align_ids(np.asarray(data["catchment_id"]), ids)
    values = {name: np.asarray(data[name])[rows] for name in (*names, "runoff_mean", "runoff_std")}
    bad = [name for name, v in values.items() if not np.isfinite(v).all()]
    if bad:
        raise ValueError(f"{path}: nonfinite {bad}")
    static = (np.stack([values[name].astype(np.float32) for name in names], 1) if names
              else np.zeros((len(rows), 0), np.float32))
    return static, values["runoff_mean"], values["runoff_std"]


ROLES = ("train", "validation", "test")


class Gauges:
    """Daily discharge of a gauge file's stations on ``dates`` (``cid``, ``fold``, ``q``).

    Without ``exclude_fold`` every station is read; with it the stations of that
    fold are not read at all. The valid days are split in time order into the
    roles of :data:`ROLES` by the ``roles`` fractions, per river group: with
    ``rivers = (catchment_id, group)`` of the map (``domain.river_groups``) all
    stations of a group share the calendar bounds that split their pooled valid
    days, so no station is tested on days its river is trained on; without it
    every station is split on its own. ``bounds[:, r]`` and ``bounds[:, r + 1]``
    delimit role ``r`` (day indices).
    """

    def __init__(self, path, dates, roles, *, rivers=None, exclude_fold=None):
        with nc.Dataset(path) as d:
            cid = np.asarray(d["catchment_id"][:], np.int64)
            fold = (np.asarray(d["fold"][:], np.int64) if "fold" in d.variables
                    else np.full(len(cid), -1, np.int64))
            if exclude_fold is not None and "fold" not in d.variables:
                raise ValueError(f"{path} has no cross-validation folds (scripts.s02_assign_folds)")
            columns = np.flatnonzero(fold != exclude_fold) if exclude_fold is not None else slice(None)
            time = d["time"]
            own = np.array(nc.num2date(time[:], time.units, only_use_cftime_datetimes=False,
                                       only_use_python_datetimes=True), "datetime64[D]")
            q = np.ma.filled(d["discharge"][:, columns].astype(np.float32), np.nan)
        self.cid, self.fold, self.dates = cid[columns], fold[columns], np.asarray(dates, "datetime64[D]")
        if len(np.unique(self.cid)) != len(self.cid):
            raise ValueError(f"{path}: gauges must lie in distinct catchments")
        self.q = np.full((len(self.dates), len(self.cid)), np.nan, np.float32)
        _, rows, mine = np.intersect1d(self.dates, own, return_indices=True)
        self.q[rows] = q[mine]
        if rivers is None:
            self.group = np.arange(len(self.cid))
        else:
            self.group = rivers[1][align_ids(rivers[0], self.cid)]
        names, member = np.unique(self.group, return_inverse=True)
        valid = np.zeros((len(self.dates), len(names)), np.int64)
        np.add.at(valid.T, member, np.isfinite(self.q).T)  # valid station-days of each group
        rank = np.cumsum(valid, 0)  # rank of the last pooled valid day, day by day
        total = rank[-1]
        bounds = [(rank < np.rint(f * total)[None] + 1).sum(0) for f in np.cumsum(np.r_[0.0, roles[:-1]])]
        # The last role ends after the last valid day.
        bounds.append(np.where(total > 0, len(self.dates) - (valid[::-1] > 0).argmax(0), len(self.dates)))
        self.bounds = np.stack(bounds, 1)[member]

    def identity(self):
        """The stations, their folds and role dates as a digest, the number of folds and of
        stations: ``scripts.s09_crossval`` combines only runs that share them."""
        digest = hashlib.sha256()
        for array in (self.cid, self.fold, self.bounds, self.dates[[0, -1]].astype(np.int64)):
            digest.update(np.ascontiguousarray(array).tobytes())
        folds = int(self.fold.max()) + 1 if (self.fold >= 0).any() else 0
        return dict(split=digest.hexdigest()[:16], folds=folds, stations=len(self.cid))

    def on(self, role, cids, shift=None, length=None):
        """Discharge of stations ``cids`` in ``role`` (NaN on other days): on ``dates``, or with
        ``shift`` (per station) on ``length`` routed days, day ``shift + t`` at day ``t``."""
        columns = align_ids(self.cid, cids)
        r = ROLES.index(role)
        lo, hi = self.bounds[columns, r], self.bounds[columns, r + 1]
        steps = np.arange(len(self.dates))[:, None] if shift is None else (
            np.arange(length)[:, None] + np.asarray(shift, np.int64)[None])
        inside = (steps >= lo) & (steps < hi)
        values = self.q[np.clip(steps, 0, len(self.dates) - 1), columns[None]]
        return np.where(inside, values, np.nan).astype(np.float32)


def source_cids(params_nc, model_cids):
    """Model ``catchment_id`` translated to ``catchment_source_id`` where the map has one."""
    model_cids = np.asarray(model_cids, dtype=np.int64)
    with nc.Dataset(params_nc) as dataset:
        if "catchment_source_id" not in dataset.variables:
            return model_cids
        ids = np.asarray(dataset["catchment_id"][:], dtype=np.int64)
        sources = np.asarray(dataset["catchment_source_id"][:], dtype=np.int64)
    lookup = {int(c): int(s) for c, s in zip(ids, sources)}
    return np.array([lookup[int(c)] for c in model_cids], dtype=np.int64)


def load_restart_state(restart_nc, params_nc, model_cids, device, dtype=torch.float32):
    """Total storage (river + flood + protected) and total outflow in model order."""
    with nc.Dataset(restart_nc) as d:
        ids = np.asarray(d["catchment_id"][:], dtype=np.int64)
        index = {int(c): i for i, c in enumerate(ids)}
        rows = np.asarray([index[int(c)] for c in source_cids(params_nc, model_cids)], np.int64)
        missing = [k for k in ("river_outflow", "flood_outflow") if k not in d.variables]
        if missing:
            raise ValueError(f"the restart lacks {missing}; HydroNet needs CaMa S and Q")
        get = lambda k: read_complete(d[k], rows)  # noqa: E731
        storage = (get("river_storage").astype(np.float32) + get("flood_storage").astype(np.float32)
                   + (get("protected_storage").astype(np.float32) if "protected_storage" in d.variables
                      else np.float32(0)))
        outflow = get("river_outflow").astype(np.float32) + get("flood_outflow").astype(np.float32)
    return (torch.from_numpy(np.ascontiguousarray(storage)).to(device=device, dtype=dtype).clamp_min_(0.0),
            torch.from_numpy(np.ascontiguousarray(outflow)).to(device=device, dtype=dtype))


def load_bifurcation_restart(restart_nc, params_nc, model_path_ids, device, dtype=torch.float32):
    """Restore pathway momentum by identifier, translating subset source IDs."""
    selected = np.asarray(model_path_ids, dtype=np.int64)
    with nc.Dataset(params_nc) as params:
        if "bifurcation_source_id" in params.variables:
            ids = np.asarray(params["bifurcation_path_id"][:], dtype=np.int64)
            original = np.asarray(params["bifurcation_source_id"][:], dtype=np.int64)
            lookup = dict(zip(ids.tolist(), original.tolist()))
            selected = np.asarray([lookup[int(i)] for i in selected], dtype=np.int64)
    with nc.Dataset(restart_nc) as restart:
        required = ("bifurcation_path_id", "bifurcation_outflow")
        if any(name not in restart.variables for name in required):
            raise ValueError("Bifurcation routing requires a restart with pathway discharge; "
                             "spin up CaMa with --bifurcation")
        ids = np.asarray(restart["bifurcation_path_id"][:], dtype=np.int64)
        lookup = {int(v): i for i, v in enumerate(ids)}
        try:
            rows = [lookup[int(v)] for v in selected]
        except KeyError as exc:
            raise ValueError(f"Restart lacks bifurcation path {exc.args[0]}") from exc
        values = read_complete(restart["bifurcation_outflow"], rows).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(values)).to(device=device, dtype=dtype)
