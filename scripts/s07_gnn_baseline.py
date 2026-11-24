"""Step 7: graph neural network baseline — learned routing on HydroNet's river network, no water balance.

    python -m scripts.s07_gnn_baseline train CONFIG [--resume OUT/last.pt] [--set key=value]
    python -m scripts.s07_gnn_baseline evaluate CONFIG [--checkpoint OUT/best.pt] [--set key=value]

The nodes are HydroNet's catchments (the trees of the stations) and the edges
their ``downstream_id`` links. Every day, for every catchment ``i``:

- ``z_i``: HydroNet's LSTM (``training.lstm.RunoffLSTM`` with a feature head)
  on the catchment forcing and static attributes, without hydraulic inputs;
- ``m_i^k``: the mean of ``m^(k-1)`` over the catchments draining into ``i``,
  from ``m^0 = h^(t-1)``, for ``k = 1..graph.hops`` (information moves at most
  that many catchments downstream a day; the state carries it on);
- ``h_i^t = GRU([z_i, m_i^1, ..., m_i^K], h_i^(t-1))``;
- ``Q_i = A_i sigma softplus(w h_i + b)``: the upstream area ``A_i`` times
  the runoff scale ``sigma``, with ``b`` placing a zero head at the mean
  runoff (the global runoff prior, ``frontend.runoff_prior``).

Nothing moves water: a catchment's discharge need not be its inflow plus its
runoff. The model is in the family of the diffusion-convolution recurrent
networks of GraphWaveNet (Sun et al., 2022), directed along the drainage and
recurrent in time so that large rivers are reached. The process model's
discharge is not an input. Training and evaluation follow ``scripts.s04_train``
and ``scripts.s05_evaluate``: the river-group roles and folds, each group's
calendar windows, the loss, truncated BPTT with ``chunk_days`` updates (the
graph state continuous, the LSTM recomputed over ``lstm_memory_days``),
``best.pt`` by the median validation NSE and ``results.json`` (model
``gnn``) for ``scripts.s08_crossval``. Ranks hold whole river groups, so no edge
crosses ranks; gradients are averaged over the ranks.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import torch
import torch.distributed as dist
from hydroforge.parallel import setup_distributed
from torch import nn

from scripts.training.audit import gauge_metrics
from scripts.training.config import (
    GNNConfig,
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
from scripts.training.domain import gauged_catchments, river_groups
from scripts.training.loop import (
    gather,
    gauge_scales,
    load_weights,
    loss,
    save_json,
    summary,
)
from scripts.training.lstm import RunoffLSTM
from scripts.training.windows import group_windows, unit_shift

SECONDS_PER_DAY = 86400.0


class RiverGNN(nn.Module):
    """The encoder and the recurrent graph network of one rank's catchments (see module docs)."""

    def __init__(self, static, prior, upstream_area, children, *, inputs, hidden,
                 spatial_checkpoint_cells, temporal_checkpoint_days, graph_hidden, hops):
        super().__init__()
        n = len(upstream_area)
        self.hops, self.graph_hidden = hops, graph_hidden
        self.encoder = RunoffLSTM(static.shape[1], 1.0, 1.0, hidden=hidden,
                                  spatial_checkpoint_cells=spatial_checkpoint_cells,
                                  temporal_checkpoint_days=temporal_checkpoint_days, inputs=inputs,
                                  features=graph_hidden)
        self.cell = nn.GRUCell(graph_hidden * (hops + 1), graph_hidden)
        self.head = nn.Linear(graph_hidden, 1)
        nn.init.normal_(self.head.weight, std=0.01)
        bias = math.log(math.expm1(2.0))
        nn.init.constant_(self.head.bias, bias)
        self.register_buffer("static", torch.as_tensor(static, dtype=torch.float32))
        # softplus(offset + bias) * std = mean for a zero head.
        self.offset = math.log(math.expm1(prior.mean / prior.std)) - bias
        self.register_buffer("scale", prior.std * torch.as_tensor(upstream_area, dtype=torch.float32)
                             * (1e-3 / SECONDS_PER_DAY))
        # Mean over the catchments draining into each one: row i holds 1/n_i at its n_i children.
        parent, child = children
        count = np.bincount(parent, minlength=n).astype(np.float32)
        weights = torch.as_tensor(1.0 / count[parent], dtype=torch.float32)
        self.register_buffer("upstream", torch.sparse_coo_tensor(
            torch.as_tensor(np.stack([parent, child]), dtype=torch.long), weights, (n, n)).coalesce())

    def initial_state(self):
        n, hidden = len(self.scale), self.encoder.hidden
        zeros = self.scale.new_zeros((1, n, hidden))
        return (zeros, zeros.clone()), self.scale.new_zeros((n, self.graph_hidden))

    def encode(self, forcing, state):
        z, h, c = self.encoder(forcing, self.static, *state)
        return z, (h, c)

    def route(self, z, h):
        """Discharge (T, N) of the encodings ``z`` (T, N, F) from the graph state ``h``."""
        out = []
        for day in z:
            m, messages = h, []
            for _ in range(self.hops):
                m = torch.sparse.mm(self.upstream, m)
                messages.append(m)
            h = self.cell(torch.cat([day, *messages], -1), h)
            out.append(torch.nn.functional.softplus(self.head(h).squeeze(-1) + self.offset) * self.scale)
        return torch.stack(out), h


def _upstream_areas(params):
    """The map's catchment IDs, downstream positions and upstream areas (the summed areas of
    every catchment's upstream catchments)."""
    with nc.Dataset(params) as d:
        cid = np.asarray(d["catchment_id"][:], np.int64)
        downstream = np.asarray(d["downstream_id"][:], np.int64)
        area = read_complete(d["catchment_area"]).astype(np.float64)
    index = {int(c): i for i, c in enumerate(cid)}
    down = np.array([index.get(int(x), -1) for x in downstream])
    down[down == np.arange(len(cid))] = -1
    depth = np.zeros(len(cid), np.int64)
    current = down.copy()
    while (current >= 0).any():
        alive = current >= 0
        depth[alive] += 1
        current = np.where(alive, down[np.maximum(current, 0)], -1)
    sums = area.copy()
    for level in range(int(depth.max()), 0, -1):  # headwaters first
        nodes = np.flatnonzero(depth == level)
        np.add.at(sums, down[nodes], sums[nodes])
    return cid, down, sums


def _partition(groups, world):
    """The rank of every river group: largest first, to the least loaded rank."""
    names, sizes = np.unique(groups, return_counts=True)
    load, owner = np.zeros(world), {}
    for k in np.lexsort((names, -sizes)):
        rank = int(np.argmin(load))
        owner[int(names[k])] = rank
        load[rank] += sizes[k]
    return np.array([owner[int(g)] for g in groups])


class Setup:
    """This rank's stations, catchments, inputs and network."""

    def __init__(self, cfg, *, exclude_fold, dd):
        span = days(*cfg.period)
        rivers = river_groups(cfg.paths.params, bifurcation=False)
        gauges = Gauges(cfg.paths.gauges, span, cfg.gauges.roles, rivers=rivers, exclude_fold=exclude_fold)
        cid, down, area = _upstream_areas(cfg.paths.params)
        np.testing.assert_array_equal(cid, rivers[0])
        domain = np.flatnonzero(gauged_catchments(cfg.paths.params, gauges.cid, bifurcation=False))
        world = dist.get_world_size() if dist.is_initialized() else 1
        mine = domain[_partition(rivers[1][domain], world) == dd.rank]
        position = np.full(len(cid), -1)
        position[mine] = np.arange(len(mine))
        child = np.flatnonzero((down >= 0) & (position >= 0))
        parent = position[down[child]]
        if (parent < 0).any():
            raise ValueError("a river group is split between ranks")
        self.ids, self.groups, self.span, self.gauges = cid[mine], rivers[1][mine], span, gauges
        self.local = gauges.cid[np.isin(gauges.cid, self.ids)]
        self.rows = align_ids(self.ids, self.local)
        self.columns = align_ids(gauges.cid, self.local)
        f = cfg.frontend
        static = load_attributes(cfg.paths.attributes, f.attributes, self.ids)
        self.area = area[mine][self.rows]  # of the stations
        self.network = RiverGNN(
            static, f.runoff_prior, area[mine], (parent, position[child]),
            inputs=len(f.forcing.variables) + 2, hidden=f.hidden,
            spatial_checkpoint_cells=f.spatial_checkpoint_cells,
            temporal_checkpoint_days=f.temporal_checkpoint_days,
            graph_hidden=cfg.graph.hidden, hops=cfg.graph.hops).to(dd.device)
        if dist.is_initialized():
            for parameter in self.network.parameters():
                dist.broadcast(parameter.data, src=0)
        self.forcing = Forcing(cfg.paths.forcing, f.forcing, self.ids, *cfg.period, device=dd.device,
                               on_device=cfg.training.forcing_on_device)


def run(net, forcing, rows, *, warmup_days, chunk_days, memory_days=0, observed=None, scales=None,
        optimizer=None, grad_clip=None, correlation_weight=0.0, keep=True):
    """Route ``forcing`` from the initial state in ``chunk_days`` windows (as ``loop.route``).

    With an optimizer every window after the warm-up is one update of the loss
    on ``observed`` at the units ``rows`` (``scales = (depth, std)`` of ``loop.loss``): the graph state continues from its
    detached value at the window's start, the LSTM recomputes with gradients
    from its detached state at the latest boundary at least ``memory_days``
    before the window's end. Returns the daily discharge at ``rows`` (if
    ``keep``) and a summary.
    """
    total = len(forcing.dates)
    bounds = sorted({0, total, *range(0, min(warmup_days, total), chunk_days),
                     *range(warmup_days, total, chunk_days)})
    windows = list(zip(bounds[:-1], bounds[1:]))
    lstm, graph = net.initial_state()
    states, outputs, losses, tick = {0: lstm}, [], [], time.monotonic()
    device = net.scale.device
    observation = None if observed is None else torch.tensor(observed, device=device)
    index = torch.as_tensor(rows, device=device, dtype=torch.long)
    for wi, (a, b) in enumerate(windows):
        training = optimizer is not None and a >= warmup_days
        first = max(day for day in states if day <= max(b - memory_days, 0)) if training else a
        inputs, _ = forcing.read(first, b)
        with torch.set_grad_enabled(training):
            z, (h, c) = net.encode(inputs, states[first])
            q, graph_end = net.route(z[a - first:], graph)
            q = q[:, index]
            if training:
                value, logged = loss(q, observation[a:b], *scales, correlation_weight)
                if value is not None:
                    _update(net, optimizer, value, grad_clip)
                    losses.append(logged)
        states[b], graph = (h.detach(), c.detach()), graph_end.detach()
        reach = max(windows[wi + 1][1] - memory_days, 0) if wi + 1 < len(windows) else b
        oldest = max(day for day in states if day <= reach)
        states = {d: v for d, v in states.items() if d >= oldest}
        if keep:
            outputs.append(q.detach().cpu().numpy())
    info = dict(units=len(net.scale), seconds=time.monotonic() - tick, updated_windows=len(losses),
                mean_loss=float(np.mean(losses)) if losses else None)
    return (np.concatenate(outputs) if keep else None), info


def _update(net, optimizer, value, grad_clip):
    if not bool(torch.isfinite(value)):
        raise RuntimeError("Nonfinite gauge loss; training stopped")
    optimizer.zero_grad(set_to_none=True)
    value.backward()
    gradients = [p.grad if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]
    flat = torch.cat([g.flatten() for g in gradients])
    if dist.is_initialized():  # the loss carries the global denominators: average the ranks
        dist.all_reduce(flat)
        flat /= dist.get_world_size()
    offset = 0
    for p, g in zip(net.parameters(), gradients):
        p.grad = flat[offset:offset + g.numel()].view_as(g).clone()
        offset += g.numel()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite gradient; training stopped")
    optimizer.step()


def _distributed(cfg):
    dd = setup_distributed(allowed_devices=("cuda",))
    torch.backends.cuda.matmul.allow_tf32 = cfg.training.tf32
    torch.backends.cudnn.allow_tf32 = cfg.training.tf32
    return dd


def train(cfg: GNNConfig, *, resume=None):
    dd = _distributed(cfg)
    protocol = cfg.training
    torch.manual_seed(protocol.seed)
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
    s = Setup(cfg, exclude_fold=cfg.gauges.heldout_fold, dd=dd)
    net, gauges = s.network, s.gauges
    scales = tuple(torch.tensor(v, device=dd.device) for v in gauge_scales(
        gauges.on("train", s.local), s.area, protocol.depth_epsilon))

    def phase(role, length, seed=None):
        r = ROLES.index(role)
        valid = np.isfinite(gauges.on(role, s.local))
        starts = group_windows(s.groups[s.rows], gauges.bounds[s.columns, r], gauges.bounds[s.columns, r + 1],
                               valid, length, protocol.warmup_days, len(s.span), seed=seed)
        shift = unit_shift(starts, s.groups)
        routed = protocol.warmup_days + length
        observed = gauges.on(role, s.local, shift[s.rows], routed)
        observed[:protocol.warmup_days] = np.nan
        return s.forcing.shifted(shift, routed), observed

    optimizer = torch.optim.Adam(net.parameters(), lr=protocol.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, protocol.epochs, eta_min=protocol.lr_min)
    history, best, first_epoch = [], -math.inf, 1
    if saved is not None:
        load_weights(net, saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        history, best, first_epoch = saved["history"], saved["best_nse"], saved["epoch"] + 1
    validation, checked = phase("validation", protocol.validation_days)
    common = dict(warmup_days=protocol.warmup_days, chunk_days=protocol.chunk_days)
    for epoch in range(first_epoch, protocol.epochs + 1):
        training, observed = phase("train", protocol.window_days, seed=(protocol.seed, epoch))
        net.train()
        _, train_info = run(net, training, s.rows, memory_days=protocol.lstm_memory_days, observed=observed,
                            scales=scales, optimizer=optimizer, grad_clip=protocol.grad_clip,
                            correlation_weight=protocol.correlation_weight, keep=False, **common)
        del training, observed
        net.eval()
        with torch.no_grad():
            q, val_info = run(net, validation, s.rows, **common)
        scores = summary([row for part in gather(gauge_metrics(q, checked, s.local)) for row in part])
        chosen = scores["n_gauges"] > 0 and scores["nse_median"] > best
        best = max(best, scores.get("nse_median", -math.inf))
        history.append(dict(epoch=epoch, train=gather(train_info), validation=gather(val_info),
                            scores=scores, learning_rate=optimizer.param_groups[0]["lr"]))
        scheduler.step()
        if dd.rank == 0:
            print("EPOCH", epoch, history[-1]["train"][0], scores, flush=True)
            # Parameters only: the per-catchment buffers come from the inputs.
            state = dict(weights={k: v.detach().cpu().clone() for k, v in net.named_parameters()},
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), epoch=epoch,
                         best_nse=best, config=cfg.model_dump(mode="json"), history=history)
            for name in ("last.pt",) + (("best.pt",) if chosen else ()):
                torch.save(state, out / f"{name}.tmp")
                (out / f"{name}.tmp").replace(out / name)
            save_json(out / "history.json", history)
        if dist.is_initialized():
            dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate(cfg: GNNConfig, checkpoint=None):
    dd = _distributed(cfg)
    out = Path(cfg.paths.checkpoints_dir)
    saved = torch.load(checkpoint or out / "best.pt", map_location="cpu", weights_only=False)
    changed = training_differences(saved["config"], cfg)
    if changed:
        raise ValueError(f"Configuration differs from the checkpoint's: {changed}")
    s = Setup(cfg, exclude_fold=None, dd=dd)
    load_weights(s.network, saved["weights"])
    s.network.eval()
    with torch.no_grad():
        q, info = run(s.network, s.forcing, s.rows, warmup_days=0, chunk_days=cfg.training.chunk_days)
    gauges, fold = s.gauges, s.gauges.fold[s.columns]
    heldout = (np.zeros(len(s.local), bool) if cfg.gauges.heldout_fold is None
               else fold == cfg.gauges.heldout_fold)
    results = {}
    for group, members in (("trained", ~heldout), ("heldout", heldout)):
        cids = s.local[members]
        results[group] = {}
        for role in ROLES:
            observed = gauges.on(role, cids)
            observed[:cfg.training.warmup_days] = np.nan
            rows = [row for part in gather(gauge_metrics(q[:, members], observed, cids)) for row in part]
            results[group][role] = dict(**summary(rows), per_gauge=rows)
    np.savez_compressed(out / f"daily_rank{dd.rank}.npz", cid=s.local, dates=s.span, q=q)
    audits = gather(info)
    if dd.rank == 0:
        save_json(out / "results.json", dict(model="gnn", **gauges.identity(), epoch=saved["epoch"],
                                             heldout_fold=cfg.gauges.heldout_fold, metrics=results,
                                             audits=audits))
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("train", "evaluate"))
    parser.add_argument("config")
    parser.add_argument("--resume", help="train: continue from this last.pt")
    parser.add_argument("--checkpoint", help="evaluate: these weights (default: best.pt of the run)")
    add_config_override_args(parser)
    a = parser.parse_args()
    cfg = load_config(a.config, a.config_overrides, schema=GNNConfig)
    if a.step == "train":
        train(cfg, resume=a.resume)
    else:
        evaluate(cfg, checkpoint=a.checkpoint)


if __name__ == "__main__":
    main()
