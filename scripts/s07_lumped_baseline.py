"""Step 7: lumped LSTM baseline — every station on its own upstream inputs, without routing.

    python -m scripts.s07_lumped_baseline train CONFIG [--resume OUT/last.pt] [--set key=value]
    python -m scripts.s07_lumped_baseline evaluate CONFIG [--checkpoint OUT/best.pt] [--set key=value]

The inputs are each station's upstream forcing and attributes
(``scripts.s04_prepare_lumped``). The LSTM of HydroNet (``training.lstm.RunoffLSTM``,
without the hydraulic inputs) predicts a station's specific discharge (mm/day),
times its ``drainage_area`` the discharge. Everything else follows
``scripts.s05_train`` and ``scripts.s06_evaluate``: the river-group time roles and folds,
each group's calendar windows, the loss, truncated BPTT with ``chunk_days``
updates and ``lstm_memory_days`` of LSTM memory, ``best.pt`` by the median
validation NSE, and the same ``results.json``, so ``scripts.s09_crossval`` combines
either model's folds.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch

from scripts.training.audit import gauge_metrics
from scripts.training.config import (
    LumpedConfig,
    add_config_override_args,
    load_config,
    training_differences,
)
from scripts.training.data import (
    ROLES,
    Forcing,
    Gauges,
    align_ids,
    days,
    load_attributes,
    read_complete,
)
from scripts.training.domain import river_groups
from scripts.training.loop import load_weights, loss, save_json, summary
from scripts.training.lstm import RunoffLSTM
from scripts.training.windows import group_windows, unit_shift

SECONDS_PER_DAY = 86400.0


class Lumped:
    """The LSTM of the stations ``cids`` and their discharge per mm/day of runoff."""

    def __init__(self, cfg, cids, device):
        f = cfg.frontend
        static, prior_mean, prior_std = load_attributes(cfg.paths.attributes, f.attributes, cids)
        with nc.Dataset(cfg.paths.attributes) as d:
            rows = align_ids(np.asarray(d["catchment_id"][:], np.int64), cids)
            drainage = read_complete(d["drainage_area"], rows).astype(np.float64)
        self.static = torch.tensor(static, device=device)
        self.scale = torch.tensor(drainage * 1e-3 / SECONDS_PER_DAY, device=device, dtype=torch.float32)
        self.network = RunoffLSTM(static.shape[1], prior_std, prior_mean, hidden=f.hidden,
                                  spatial_checkpoint_cells=f.spatial_checkpoint_cells,
                                  temporal_checkpoint_days=f.temporal_checkpoint_days,
                                  inputs=len(f.forcing.variables) + 2).to(device)

    def __call__(self, forcing, state):
        rate, h, c = self.network(forcing, self.static, *state)
        return rate * self.scale, (h, c)

    def load(self, weights):
        load_weights(self.network, weights)

    def initial_state(self):
        n, hidden = len(self.scale), self.network.hidden
        zeros = self.scale.new_zeros((1, n, hidden))
        return zeros, zeros.clone()


def run_lstm(model, forcing, *, warmup_days, chunk_days, memory_days=0, observed=None, std=None,
             optimizer=None, grad_clip=None, correlation_weight=0.0, keep=True):
    """Run ``forcing`` from the zero state in ``chunk_days`` windows (as ``loop.route``).

    With an optimizer every window after the warm-up is one update of the loss
    on ``observed``: the LSTM recomputes with gradients from its detached state
    at the latest window boundary at least ``memory_days`` before the window's
    end. Returns the daily discharge (if ``keep``) and a summary.
    """
    total = len(forcing.dates)
    bounds = sorted({0, total, *range(0, min(warmup_days, total), chunk_days),
                     *range(warmup_days, total, chunk_days)})
    states, outputs, losses, tick = {0: model.initial_state()}, [], [], time.monotonic()
    observation = None if observed is None else torch.tensor(observed, device=model.scale.device)
    windows = list(zip(bounds[:-1], bounds[1:]))
    for wi, (a, b) in enumerate(windows):
        training = optimizer is not None and a >= warmup_days
        first = max(day for day in states if day <= max(b - memory_days, 0)) if training else a
        inputs, _ = forcing.read(first, b)
        with torch.set_grad_enabled(training):
            q, (h, c) = model(inputs, states[first])
            q = q[a - first:]
            if training:
                value, logged = loss(q, observation[a:b], std, correlation_weight)
                if value is not None:
                    if not bool(torch.isfinite(value)):
                        raise RuntimeError("Nonfinite gauge loss; training stopped")
                    optimizer.zero_grad(set_to_none=True)
                    value.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.network.parameters(), grad_clip)
                    if not bool(torch.isfinite(norm)):
                        raise RuntimeError("Nonfinite gradient; training stopped")
                    optimizer.step()
                    losses.append(logged)
        states[b] = (h.detach(), c.detach())
        reach = max(windows[wi + 1][1] - memory_days, 0) if wi + 1 < len(windows) else b
        oldest = max(day for day in states if day <= reach)
        states = {d: v for d, v in states.items() if d >= oldest}
        if keep:
            outputs.append(q.detach().cpu().numpy())
    info = dict(seconds=time.monotonic() - tick, updated_windows=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None)
    return (np.concatenate(outputs) if keep else None), info


def _setup(cfg, *, exclude_fold):
    torch.manual_seed(cfg.training.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    span = days(*cfg.period)
    rivers = river_groups(cfg.paths.params, bifurcation=False)
    gauges = Gauges(cfg.paths.gauges, span, cfg.gauges.roles, rivers=rivers, exclude_fold=exclude_fold)
    model = Lumped(cfg, gauges.cid, device)
    forcing = Forcing(cfg.paths.forcing, cfg.frontend.forcing, gauges.cid, *cfg.period, device=device)
    return span, gauges, model, forcing, device


def train(cfg: LumpedConfig, *, resume=None):
    protocol = cfg.training
    out = Path(cfg.paths.checkpoints_dir)
    out.mkdir(parents=True, exist_ok=True)
    if resume is None and (out / "last.pt").exists():
        raise ValueError("Refuse to overwrite existing training; use --resume or a new output directory")
    saved = None
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=False)
        changed = training_differences(saved["config"], cfg)
        if changed:
            raise ValueError(f"Resume configuration differs from the checkpoint's: {changed}")
    span, gauges, model, forcing, device = _setup(cfg, exclude_fold=cfg.gauges.heldout_fold)
    cids = gauges.cid
    train_days = gauges.on("train", cids)
    count = np.isfinite(train_days).sum(0)
    std = np.ones(len(cids))
    std[count > 1] = np.fmax(np.nanstd(train_days[:, count > 1].astype(np.float64), 0), 1.0)
    std = torch.tensor(std, device=device, dtype=torch.float32)
    del train_days

    def phase(role, length, seed=None):
        r = ROLES.index(role)
        valid = np.isfinite(gauges.on(role, cids))
        starts = group_windows(gauges.group, gauges.bounds[:, r], gauges.bounds[:, r + 1], valid, length,
                               protocol.warmup_days, len(span), seed=seed)
        shift = unit_shift(starts, gauges.group)
        routed = protocol.warmup_days + length
        observed = gauges.on(role, cids, shift, routed)
        observed[:protocol.warmup_days] = np.nan
        return forcing.shifted(shift, routed), observed

    optimizer = torch.optim.Adam(model.network.parameters(), lr=protocol.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol.epochs, eta_min=protocol.lr_min)
    history, best, first_epoch = [], -math.inf, 1
    if saved is not None:
        model.load(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best, first_epoch = saved["history"], saved["best_nse"], saved["epoch"] + 1
    validation, checked = phase("validation", protocol.validation_days)
    common = dict(warmup_days=protocol.warmup_days, chunk_days=protocol.chunk_days)
    for epoch in range(first_epoch, protocol.epochs + 1):
        training, observed = phase("train", protocol.window_days, seed=(protocol.seed, epoch))
        model.network.train()
        _, train_info = run_lstm(model, training, memory_days=protocol.lstm_memory_days,
                                 observed=observed, std=std, optimizer=optimizer,
                                 grad_clip=protocol.grad_clip,
                                 correlation_weight=protocol.correlation_weight, keep=False, **common)
        del training, observed
        model.network.eval()
        with torch.no_grad():
            q, val_info = run_lstm(model, validation, **common)
        scores = summary(gauge_metrics(q, checked, cids))
        chosen = scores["n_gauges"] > 0 and scores["nse_median"] > best
        best = max(best, scores.get("nse_median", -math.inf))
        history.append(dict(epoch=epoch, train=[train_info], validation=[val_info], scores=scores,
                            learning_rate=optimizer.param_groups[0]["lr"]))
        print("EPOCH", epoch, train_info, scores, flush=True)
        scheduler.step()
        # Parameters only: the per-station buffers (runoff prior) come from the attributes.
        state = dict(weights={k: v.detach().cpu().clone() for k, v in model.network.named_parameters()},
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                     best_nse=best, config=cfg.model_dump(mode="json"), history=history)
        for name in ("last.pt",) + (("best.pt",) if chosen else ()):
            torch.save(state, out / f"{name}.tmp")
            (out / f"{name}.tmp").replace(out / name)
        save_json(out / "history.json", history)


def evaluate(cfg: LumpedConfig, checkpoint=None):
    out = Path(cfg.paths.checkpoints_dir)
    saved = torch.load(checkpoint or out / "best.pt", map_location="cpu", weights_only=False)
    changed = training_differences(saved["config"], cfg)
    if changed:
        raise ValueError(f"Configuration differs from the checkpoint's: {changed}")
    span, gauges, model, forcing, _ = _setup(cfg, exclude_fold=None)
    model.load(saved["weights"])
    model.network.eval()
    with torch.no_grad():
        q, info = run_lstm(model, forcing, warmup_days=0, chunk_days=cfg.training.chunk_days)
    heldout = (np.zeros(len(gauges.cid), bool) if cfg.gauges.heldout_fold is None
               else gauges.fold == cfg.gauges.heldout_fold)
    results = {}
    for group, members in (("trained", ~heldout), ("heldout", heldout)):
        cids = gauges.cid[members]
        results[group] = {}
        for role in ROLES:
            observed = gauges.on(role, cids)
            observed[:cfg.training.warmup_days] = np.nan
            rows = gauge_metrics(q[:, members], observed, cids)
            results[group][role] = dict(**summary(rows), per_gauge=rows)
    np.savez_compressed(out / "daily_rank0.npz", cid=gauges.cid, dates=span, q=q)
    save_json(out / "results.json", dict(model="lumped", **gauges.identity(), epoch=saved["epoch"],
                                         heldout_fold=cfg.gauges.heldout_fold,
                                         metrics=results, audits=[info]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("train", "evaluate"))
    parser.add_argument("config")
    parser.add_argument("--resume", help="train: continue from this last.pt")
    parser.add_argument("--checkpoint", help="evaluate: these weights (default: best.pt of the run)")
    add_config_override_args(parser)
    a = parser.parse_args()
    cfg = load_config(a.config, a.config_overrides, schema=LumpedConfig)
    if a.step == "train":
        train(cfg, resume=a.resume)
    else:
        evaluate(cfg, checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
