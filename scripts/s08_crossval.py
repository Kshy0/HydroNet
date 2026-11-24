"""Step 8: combine k-fold evaluations into space-only, time-only, and space-time metrics.

Single-seed runs (one run directory per fold):

    python -m scripts.s08_crossval OUT/fold0 OUT/fold1 ... [--out crossval.json] [--plot cdf.png] [--partial]

Multi-seed daily-discharge ensemble (multiple seed run directories per fold; averages
daily simulated discharge across seeds before scoring):

    python -m scripts.s08_crossval --ensemble --gauges GAUGES.nc --params PARAMS.nc \\
        OUT/fold0-s42 OUT/fold0-s43 OUT/fold0-s44 OUT/fold1-s42 ... [--out crossval.json] [--plot cdf.png]

Each run directory holds the ``results.json`` (and, for ``--ensemble``, ``daily_rank*.npz``)
of ``scripts.s05_evaluate``, ``scripts.s06_lumped_baseline``, or ``scripts.s07_gnn_baseline``
for one ``gauges.heldout_fold``. Together the folds hold every river group out once:

- **space only**: every station scored by the run that held its river group out,
  on its train days (unseen basins, training period);
- **time only**: every station scored on its test days by the runs that trained
  on it, the median of each score over those k - 1 runs (seen basins, test period);
- **space-time**: every station scored by the run that held its river group out,
  on its test days (unseen basins, test period).

Runs must belong to one model and one split (``model``, ``split``, and ``folds`` of
``results.json``); ``--partial`` accepts a subset of folds and reports the missing ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.training.audit import gauge_metrics
from scripts.training.data import Gauges, align_ids
from scripts.training.domain import river_groups

SCORES = ("nse", "kge", "pbias_percent")


def _summary(rows):
    return dict(n_gauges=len(rows), **{f"{k}_median": float(np.median([r[k] for r in rows]))
                                       for k in SCORES} if rows else {})


def _validate_results(results, *, partial=False, allow_duplicate_folds=False):
    for key in ("model", "split", "folds"):
        values = {r.get(key) for r in results}
        if len(values) != 1 or None in values:
            raise ValueError(f"the runs differ in {key}: {sorted(map(str, values))}")
    folds = [r["heldout_fold"] for r in results]
    if None in folds or (not allow_duplicate_folds and len(set(folds)) != len(folds)):
        raise ValueError(f"every run must hold out its own fold, got {folds}")
    unique_folds = sorted(set(folds))
    missing = sorted(set(range(results[0]["folds"])) - set(unique_folds))
    if missing and not partial:
        raise ValueError(f"folds {missing} have no run (use --partial for a subset)")
    return unique_folds, missing


def combine(runs, *, partial=False):
    """Combine single-seed k-fold ``results.json`` files."""
    results = [json.loads((Path(run) / "results.json").read_text()) for run in runs]
    folds, missing = _validate_results(results, partial=partial, allow_duplicate_folds=False)
    space = [row for r in results for row in r["metrics"]["heldout"]["train"]["per_gauge"]]
    if len({row["cid"] for row in space}) != len(space):
        raise ValueError("a station is held out by more than one run")
    space_time = [row for r in results
                  for row in r["metrics"]["heldout"].get("test", {}).get("per_gauge", [])]
    by_station = {}
    for r in results:
        for row in r["metrics"]["trained"]["test"]["per_gauge"]:
            by_station.setdefault(row["cid"], []).append(row)
    time = [dict(cid=cid, runs=len(rows), **{k: float(np.median([x[k] for x in rows])) for k in SCORES})
            for cid, rows in sorted(by_station.items())]
    return dict(model=results[0]["model"], split=results[0]["split"], folds=folds,
                missing_folds=missing, space_only=dict(**_summary(space), per_gauge=space),
                time_only=dict(**_summary(time), per_gauge=time),
                space_time=dict(**_summary(space_time), per_gauge=space_time))


def _load_daily(run_dir: Path):
    """Load and concatenate all ranks' ``daily_rank*.npz`` sorted by catchment ID."""
    files = sorted(run_dir.glob("daily_rank*.npz"))
    if not files:
        raise ValueError(f"{run_dir} has no daily_rank*.npz files for ensemble scoring")
    cids, qs, dates = [], [], None
    for path in files:
        with np.load(path) as data:
            if dates is None:
                dates = np.asarray(data["dates"], "datetime64[D]")
            elif not np.array_equal(dates, np.asarray(data["dates"], "datetime64[D]")):
                raise ValueError(f"{path} dates differ from other ranks in {run_dir}")
            cids.append(np.asarray(data["cid"], np.int64))
            qs.append(np.asarray(data["q"], np.float32))
    cid = np.concatenate(cids)
    q = np.concatenate(qs, axis=1)
    order = np.argsort(cid)
    return cid[order], dates, q[:, order]


def combine_ensemble(runs, gauges_nc, params_nc, *, roles=(0.6, 0.2, 0.2), warmup_days=365,
                     bifurcation=False, partial=False):
    """Combine multi-seed k-fold runs by averaging daily discharge per fold before scoring."""
    run_paths = [Path(run) for run in runs]
    results = [json.loads((path / "results.json").read_text()) for path in run_paths]
    folds, missing = _validate_results(results, partial=partial, allow_duplicate_folds=True)
    by_fold: dict[int, list[Path]] = {}
    for path, result in zip(run_paths, results):
        by_fold.setdefault(int(result["heldout_fold"]), []).append(path)
    seed_counts = {len(paths) for paths in by_fold.values()}
    if len(seed_counts) != 1:
        raise ValueError(f"every fold must have the same number of seeds, got {sorted(seed_counts)}")

    rivers = river_groups(params_nc, bifurcation=bifurcation)
    gauges = None
    space, space_time = [], []
    by_station = {}
    for fold in folds:
        members = [_load_daily(path) for path in by_fold[fold]]
        cid, dates = members[0][0], members[0][1]
        for other_cid, other_dates, _ in members[1:]:
            if not np.array_equal(cid, other_cid) or not np.array_equal(dates, other_dates):
                raise ValueError(f"fold {fold} seed runs differ in station IDs or dates")
        if gauges is None:
            gauges = Gauges(gauges_nc, dates, roles, rivers=rivers)
            identity = gauges.identity()
            if identity["split"] != results[0]["split"]:
                raise ValueError(f"gauges split {identity['split']} differs from runs {results[0]['split']}")
        q_mean = np.mean([q.astype(np.float64) for _, _, q in members], axis=0).astype(np.float32)
        station_fold = gauges.fold[align_ids(gauges.cid, cid)]
        heldout = station_fold == fold
        for target, mask, role in ((space, heldout, "train"), (space_time, heldout, "test")):
            cids = cid[mask]
            observed = gauges.on(role, cids)
            observed[:warmup_days] = np.nan
            target.extend(gauge_metrics(q_mean[:, mask], observed, cids))
        trained_cids = cid[~heldout]
        observed_test = gauges.on("test", trained_cids)
        observed_test[:warmup_days] = np.nan
        for row in gauge_metrics(q_mean[:, ~heldout], observed_test, trained_cids):
            by_station.setdefault(row["cid"], []).append(row)

    time = [dict(cid=cid, runs=len(rows), **{k: float(np.median([x[k] for x in rows])) for k in SCORES})
            for cid, rows in sorted(by_station.items())]
    return dict(model=results[0]["model"], split=results[0]["split"], folds=folds,
                missing_folds=missing, seeds_per_fold=next(iter(seed_counts)),
                space_only=dict(**_summary(space), per_gauge=space),
                time_only=dict(**_summary(time), per_gauge=time),
                space_time=dict(**_summary(space_time), per_gauge=space_time))


def plot(combined, path, score="nse"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 3.5), dpi=200)
    curves = (("space_only", "Space only (unseen basins)", "#2a78d6"),
              ("time_only", "Time only (test period)", "#eb6834"),
              ("space_time", "Space-time (unseen + test)", "#2f9e44"))
    for key, label, color in curves:
        rows = combined.get(key, {}).get("per_gauge", [])
        if not rows:
            continue
        values = np.sort([row[score] for row in rows])
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, lw=2,
                label=f"{label}, n={len(values)}")
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(score.upper())
    ax.set_ylabel("Fraction of stations")
    ax.grid(color="#e4e3de", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="crossval.json")
    parser.add_argument("--plot")
    parser.add_argument("--partial", action="store_true", help="accept a subset of the folds")
    parser.add_argument("--ensemble", action="store_true",
                        help="average daily discharge across multiple seed runs per fold before scoring")
    parser.add_argument("--gauges", help="gauge NetCDF file (required with --ensemble)")
    parser.add_argument("--params", help="HydroNet parameter NetCDF file (required with --ensemble)")
    parser.add_argument("--roles", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--warmup-days", type=int, default=365)
    parser.add_argument("--bifurcation", action="store_true")
    args = parser.parse_args()
    if args.ensemble:
        if not args.gauges or not args.params:
            parser.error("--ensemble requires --gauges and --params")
        combined = combine_ensemble(
            args.runs, args.gauges, args.params, roles=tuple(args.roles),
            warmup_days=args.warmup_days, bifurcation=args.bifurcation, partial=args.partial)
    else:
        combined = combine(args.runs, partial=args.partial)
    Path(args.out).write_text(json.dumps(combined, indent=2) + "\n")
    if args.plot:
        plot(combined, args.plot)
    for key in ("space_only", "time_only", "space_time"):
        if combined.get(key, {}).get("n_gauges", 0):
            print(key, {k: v for k, v in combined[key].items() if k != "per_gauge"})


if __name__ == "__main__":
    main()
