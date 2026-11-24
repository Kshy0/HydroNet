"""Step 6: replay a trained HydroNet over the whole period and score every station.

    python -m scripts.s06_evaluate configs/tiny.yaml [--checkpoint out/tiny/best.pt] [--set key=value]

Reads every station, including the held-out fold. The model is cropped to
the trees of all stations and routed once, continuously from the restart, over
``period`` with the checkpoint's weights (``best.pt`` by default); the first
``warmup_days`` are not scored. Each station is scored on its train,
validation and test days (``gauges.roles``), in the group ``trained`` or
``heldout`` (``gauges.heldout_fold``). Writes ``results.json`` (per group and
role: medians and every station's scores) and each rank's daily outflow at its
stations, ``daily_rank{R}.npz``. ``scripts.s09_crossval`` combines the folds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from hydroforge.parallel import setup_distributed

from scripts.training.audit import gauge_metrics
from scripts.training.config import (
    Config,
    add_config_override_args,
    load_config,
    training_differences,
)
from scripts.training.data import ROLES, Forcing, Gauges, align_ids, days
from scripts.training.domain import river_groups
from scripts.training.loop import build, gather, load_weights, route, save_json, summary


def run(cfg: Config, checkpoint=None):
    dd = setup_distributed(allowed_devices=("cuda",))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg.paths.checkpoints_dir)
    saved = torch.load(checkpoint or out / "best.pt", map_location="cpu", weights_only=False)
    changed = training_differences(saved["config"], cfg)
    if changed:
        raise ValueError(f"Configuration differs from the checkpoint's: {changed}")
    span = days(*cfg.period)
    gauges = Gauges(cfg.paths.gauges, span, cfg.gauges.roles,
                    rivers=river_groups(cfg.paths.params, bifurcation=cfg.model.bifurcation))
    native, proxy, network, initial, ids, _ = build(cfg, gauges.cid, dd.device)
    try:
        load_weights(network, saved["weights"])
        network.eval()
        local = gauges.cid[np.isin(gauges.cid, ids)]
        index = {int(c): i for i, c in enumerate(ids)}
        outputs = torch.tensor([index[int(c)] for c in local], device=dd.device, dtype=torch.long)
        forcing = Forcing(cfg.paths.forcing, cfg.frontend.forcing, ids, *cfg.period, device=dd.device)
        with torch.no_grad():
            routed, _, audit = route(network, forcing, initial(), outputs,
                                     chunk_days=cfg.training.chunk_days, phase="evaluate",
                                     output_dir=out)
        del forcing
        fold = gauges.fold[align_ids(gauges.cid, local)]
        heldout = (np.zeros(len(local), bool) if cfg.gauges.heldout_fold is None
                   else fold == cfg.gauges.heldout_fold)
        results = {}
        for group, members in (("trained", ~heldout), ("heldout", heldout)):
            cids = local[members]
            results[group] = {}
            for role in ROLES:
                observed = gauges.on(role, cids)
                observed[:cfg.training.warmup_days] = np.nan
                rows = [row for part in gather(gauge_metrics(routed["q"][:, members], observed, cids))
                        for row in part]
                results[group][role] = dict(**summary(rows), per_gauge=rows)
        np.savez_compressed(out / f"daily_rank{dd.rank}.npz", cid=local, dates=routed["dates"],
                            q=routed["q"])
        audits = gather(audit)
        if dd.rank == 0:
            save_json(out / "results.json", dict(model="hydronet", **gauges.identity(), epoch=saved["epoch"],
                                                 heldout_fold=cfg.gauges.heldout_fold,
                                                 metrics=results, audits=audits))
    finally:
        native.close()
        proxy.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--checkpoint", help="Weights to evaluate (default: best.pt of the run)")
    add_config_override_args(parser)
    args = parser.parse_args()
    run(load_config(args.config, args.config_overrides), checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
