"""Step 5: train an ERA5 runoff LSTM through HydroNet's implicit local-inertia steps.

    python -m scripts.s05_train configs/tiny.yaml [--resume out/tiny/last.pt] [--set key=value]

Every river's pooled station record is split in time into train, validation
and test days (``gauges.roles``, shared by the river's stations); the stations of ``gauges.heldout_fold`` (whole river
groups) are not read. Every epoch routes each river group over its own
calendar window (``scripts.training.windows``): ``warmup_days`` without loss,
then ``window_days`` holding as many of its stations' training days as fit,
drawn anew every epoch. Truncated BPTT with continuous physical and recurrent
states, a gauge loss of standardized MSE and correlation and Adam: every
``chunk_days`` window is one update, the routing backpropagating through that
window only and the LSTM through at least ``lstm_memory_days`` (see
``loop.route``); roughness keeps the model's defaults. Every epoch is
scored on the stations' validation days (fixed windows of
``validation_days``); the best epoch's weights are kept in ``best.pt`` for
``scripts.s06_evaluate``.
"""

from __future__ import annotations

import argparse
import math
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
from scripts.training.windows import group_windows, unit_shift


def _restore(cfg, resume, out):
    """The checkpoint to resume, after checking its directory and configuration."""
    if Path(resume).resolve().parent != out.resolve():
        raise ValueError("Resume in the checkpoint's output directory so best.pt and history stay aligned")
    saved = torch.load(resume, map_location="cpu", weights_only=False)
    changed = training_differences(saved["config"], cfg)
    if changed:
        raise ValueError(f"Resume configuration differs from the checkpoint's: {changed}")
    return saved


def run(cfg: Config, *, resume=None):
    dd = setup_distributed(allowed_devices=("cuda",))
    protocol = cfg.training
    torch.set_num_threads(2)
    torch.manual_seed(protocol.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(cfg.paths.checkpoints_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume is None and (out / "last.pt").exists():
        raise ValueError("Refuse to overwrite existing training; use --resume or a new output directory")
    saved = None if resume is None else _restore(cfg, resume, out)
    span = days(*cfg.period)
    rivers = river_groups(cfg.paths.params, bifurcation=cfg.model.bifurcation)
    gauges = Gauges(cfg.paths.gauges, span, cfg.gauges.roles, rivers=rivers,
                    exclude_fold=cfg.gauges.heldout_fold)
    native, proxy, network, initial, ids, groups = build(cfg, gauges.cid, dd.device)
    try:
        # This rank's stations, their units and river groups.
        local = gauges.cid[np.isin(gauges.cid, ids)]
        index = {int(c): i for i, c in enumerate(ids)}
        rows = np.array([index[int(c)] for c in local], np.int64)
        units = torch.tensor(rows, device=dd.device, dtype=torch.long)
        columns = align_ids(gauges.cid, local)
        forcing = Forcing(cfg.paths.forcing, cfg.frontend.forcing, ids, *cfg.period, device=dd.device)
        # Each station's standard deviation over its training days, at least 1 m3/s.
        train_days = gauges.on("train", local)
        count = np.isfinite(train_days).sum(0)
        std = np.ones(len(local))
        std[count > 1] = np.fmax(np.nanstd(train_days[:, count > 1].astype(np.float64), 0), 1.0)
        std = torch.tensor(std, device=dd.device, dtype=torch.float32)
        del train_days

        def phase(role, length, seed=None):
            """The shifted forcing and the stations' ``role`` discharge of every group's window."""
            r = ROLES.index(role)
            lo, hi = gauges.bounds[columns, r], gauges.bounds[columns, r + 1]
            valid = np.isfinite(gauges.on(role, local))
            starts = group_windows(groups[rows], lo, hi, valid, length, protocol.warmup_days,
                                   len(span), seed=seed)
            shift = unit_shift(starts, groups)
            routed = protocol.warmup_days + length
            observed = gauges.on(role, local, shift[rows], routed)
            observed[:protocol.warmup_days] = np.nan  # the warm-up is never scored
            return forcing.shifted(shift, routed), observed

        optimizer = torch.optim.Adam(network.parameters(), lr=protocol.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol.epochs,
                                                               eta_min=protocol.lr_min)
        history, best, first_epoch = [], -math.inf, 1
        if saved is not None:
            load_weights(network, saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            history, best, first_epoch = saved["history"], saved["best_nse"], saved["epoch"] + 1
        validation, checked = phase("validation", protocol.validation_days)
        common = dict(chunk_days=protocol.chunk_days, output_dir=out)
        for epoch in range(first_epoch, protocol.epochs + 1):
            # Every epoch draws each group's training window anew (seeded by the epoch).
            training, observed = phase("train", protocol.window_days, seed=(protocol.seed, epoch))
            network.train()
            _, _, train_info = route(
                network, training, initial(), units, targets=(units, observed, std),
                optimizer=optimizer,
                warmup_days=protocol.warmup_days, memory_days=protocol.lstm_memory_days,
                grad_clip=protocol.grad_clip, correlation_weight=protocol.correlation_weight,
                epoch=epoch, phase="train", **common)
            del training, observed
            network.eval()
            with torch.no_grad():
                val, _, val_info = route(network, validation, initial(), units, epoch=epoch,
                                         phase="validation", **common)
            scores = summary([row for part in gather(gauge_metrics(val["q"], checked, local))
                              for row in part])
            chosen = scores["n_gauges"] > 0 and scores["nse_median"] > best
            best = max(best, scores.get("nse_median", -math.inf))
            history.append(dict(epoch=epoch, train=gather(train_info), validation=gather(val_info),
                                scores=scores, learning_rate=optimizer.param_groups[0]["lr"]))
            scheduler.step()
            if dd.rank == 0:
                state = dict(weights={k: p.detach().cpu().clone()
                                      for k, p in network.named_parameters()},
                             optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                             epoch=epoch, best_nse=best,
                             config=cfg.model_dump(mode="json"),
                             history=history)
                for name in ("last.pt",) + (("best.pt",) if chosen else ()):
                    torch.save(state, out / f"{name}.tmp")
                    (out / f"{name}.tmp").replace(out / name)
                save_json(out / "history.json", history)
            if dist.is_initialized():
                dist.barrier()
    finally:
        native.close()
        proxy.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", help="Resume at the next epoch from a complete last.pt checkpoint")
    parser.add_argument("--check-config", action="store_true",
                        help="Validate and print the configuration without starting a GPU job")
    add_config_override_args(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.config_overrides)
    if args.check_config:
        print(cfg.model_dump_json(indent=2))
    else:
        run(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
