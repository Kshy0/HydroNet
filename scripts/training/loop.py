"""What training and evaluation share: building the network, routing a period, the loss."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from scripts.training.audit import mass_balance
from scripts.training.coupling import HydroNet
from scripts.training.data import (
    align_ids,
    load_attributes,
    load_bifurcation_restart,
    load_restart_state,
)
from scripts.training.domain import gauged_catchments, river_groups
from scripts.training.model import build_model, check_newton_status, routing_edges


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def gather(value):
    """``value`` of every rank, in rank order."""
    if not dist.is_initialized():
        return [value]
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def _sum(value):
    value = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(value)
    return value


def load_weights(module, weights):
    """Trained parameters into ``module``: exactly its parameters, by name (its buffers are
    rebuilt from the inputs)."""
    names = {name for name, _ in module.named_parameters()}
    if set(weights) != names:
        raise ValueError(f"checkpoint parameters differ from the network's: missing "
                         f"{sorted(names - set(weights))}, unexpected {sorted(set(weights) - names)}")
    module.load_state_dict({**module.state_dict(), **weights})


def summary(rows):
    """Gauge count and median NSE, KGE and percent bias of ``gauge_metrics`` rows."""
    if not rows:
        return dict(n_gauges=0)
    return dict(n_gauges=len(rows), **{name + "_median": float(np.median([r[name] for r in rows]))
                                       for name in ("nse", "kge", "pbias_percent")})


def build(cfg, gauge_cids, device):
    """The model cropped to the trees of ``gauge_cids``, its input proxy, the network
    (LSTM), a factory of the initial state, the unit IDs (in model order) of this rank and
    their river groups. Roughness keeps the model's defaults."""
    options = {**cfg.model.model_dump(), "differentiable": True}
    params = cfg.paths.params
    keep = gauged_catchments(params, gauge_cids, bifurcation=cfg.model.bifurcation)
    native, proxy = build_model(params, device, keep=keep, options=options,
                                output_cids=gauge_cids,
                                initial_time=datetime(cfg.period[0], 1, 1))
    ids = native.base.catchment_id.cpu().numpy()
    map_ids, map_groups = river_groups(params, bifurcation=cfg.model.bifurcation)
    groups = map_groups[align_ids(map_ids, ids)]
    f = cfg.frontend
    static, prior_mean, prior_std = load_attributes(cfg.paths.attributes, f.attributes, ids)
    upstream = np.asarray(proxy["upstream_area"], np.float64)[
        align_ids(np.asarray(proxy["catchment_id"]), ids)]
    network = HydroNet(
        native, torch.tensor(static, device=device), prior_mean, prior_std,
        torch.tensor(upstream, device=device, dtype=torch.float32),
        hydraulic_mean=f.hydraulic.mean, hydraulic_std=f.hydraulic.std,
        forcing_inputs=len(f.forcing.variables) + 2,  # and the season's sine and cosine
        lstm_hidden=f.hidden, spatial_checkpoint_cells=f.spatial_checkpoint_cells,
        temporal_checkpoint_days=f.temporal_checkpoint_days)
    if dist.is_initialized():
        for parameter in network.parameters():
            dist.broadcast(parameter.data, src=0)
    s0, q0 = load_restart_state(cfg.paths.cama_restart, params, ids, device, torch.float32)
    bif = native.bifurcation
    qb0 = (load_bifurcation_restart(cfg.paths.cama_restart, params,
                                    bif.bifurcation_path_id.cpu().numpy(), device)
           if bif is not None and bif.num_bifurcation_paths else None)
    return native, proxy, network, lambda: network.initial_state(s0, q0, qb0), ids, groups


def loss(prediction, observation, std, correlation_weight):
    """Gauge loss of a window and its logged value: ``(1 - w) * zMSE + w * (1 - r)``.

    zMSE is each gauge's mean squared error scaled by its training standard
    deviation ``std``, averaged over gauges with at least 3 observations. ``r``
    is the earlier HydroNet's variability-scaled correlation, the window
    covariance over the simulated window and the training standard deviations,
    ``r_window * std_obs_window / std``, averaged over gauges with at least 30
    observations in the window; windows without any contribute no ``1 - r``.
    Missing and nonfinite observations are zeroed before any arithmetic.
    """
    valid = torch.isfinite(observation)
    mask = valid.to(prediction.dtype)
    y = torch.where(valid, observation, 0.0)
    count = valid.sum(0)
    safe = count.clamp_min(1).to(prediction.dtype)
    eligible = count >= 3
    mse = (((prediction - y) / std).square() * mask).sum(0) / safe
    numerator = mse[eligible].sum()
    denominator = _sum(eligible.sum()).to(prediction.dtype)
    if int(denominator) == 0:
        return None, None
    world = dist.get_world_size() if dist.is_initialized() else 1
    weight = float(correlation_weight)
    value = (1.0 - weight) * numerator / denominator
    logged = (1.0 - weight) * float(_sum(numerator) / denominator)
    if weight:
        correlated = count >= 30
        mean_p = (prediction * mask).sum(0) / safe
        mean_y = (y * mask).sum(0) / safe
        dp, dy = (prediction - mean_p) * mask, (y - mean_y) * mask
        r = ((dp * dy).sum(0) / safe) / (
            ((dp.square().sum(0) / safe).clamp_min(1e-6).sqrt()) * std)
        terms = (1.0 - r)[correlated].sum()
        pairs = _sum(correlated.sum()).to(prediction.dtype)
        if int(pairs):
            value = value + weight * terms / pairs
            logged += weight * float(_sum(terms) / pairs)
    # Gradient averaging over ranks cancels world, giving the exact global loss.
    return world * value, logged


def _update(network, optimizer, value, grad_clip):
    """Backpropagate ``value`` and step once, refusing nonfinite or uncertified gradients.

    The LSTM's gradients come through the runoff (autograd)."""
    if not bool(torch.isfinite(value)):
        raise RuntimeError("Nonfinite gauge loss; training stopped")
    value.backward()
    check_newton_status(network.physics)
    gradients = [p.grad for p in network.parameters() if p.grad is not None]
    flat = torch.cat([g.flatten() for g in gradients])
    if dist.is_initialized():
        dist.all_reduce(flat)
        flat /= dist.get_world_size()
    offset = 0
    for gradient in gradients:
        gradient.copy_(flat[offset : offset + gradient.numel()].view_as(gradient))
        offset += gradient.numel()
    norm = torch.nn.utils.clip_grad_norm_(network.parameters(), grad_clip)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("Nonfinite gradient; training stopped")
    optimizer.step()


def progress(output_dir, **fields):
    rank = dist.get_rank() if dist.is_initialized() else 0
    fields = dict(rank=rank, **fields,
                  gpu_peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
                  gpu_reserved_peak_gib=torch.cuda.max_memory_reserved() / 1024**3)
    save_json(Path(output_dir) / f"progress_rank{rank}.json", fields)
    if rank == 0:
        print("PROGRESS", json.dumps(fields), flush=True)


def route(network, forcing, state, outputs, *, chunk_days, phase, output_dir,
          epoch=0, targets=None, optimizer=None, warmup_days=0, memory_days=0,
          grad_clip=None, correlation_weight=0.0):
    """Route ``forcing``'s period continuously from ``state``.

    Returns the daily outflow of units ``outputs`` (with the dates), the final
    state and a summary. Windows of ``chunk_days`` tile the warm-up and the
    days after it; states are detached at every window boundary. ``targets =
    (units, observations on forcing.dates, std)`` adds the loss of every window
    after the warm-up. With an optimizer every such window is one update: its
    routing starts from the detached state at the window's start, and its LSTM
    from the detached LSTM state at the latest boundary at least
    ``memory_days`` before the window's end (recomputed with gradients from
    there, reading the hydraulic inputs recorded when those days were routed).
    """
    state = state.detach()
    physical = network.physics
    edges = routing_edges(physical)
    bif = physical.bifurcation
    # The audit's residual is the Newton merit recomputed in FP64 from the stored FP32 state.
    limit = 1.05 * physical.base.newton_tolerance
    total = len(forcing.dates)
    warmup_days = min(warmup_days, total)
    bounds = sorted({0, total, *range(0, warmup_days, chunk_days),
                     *range(warmup_days, total, chunk_days)})
    windows = list(zip(bounds[:-1], bounds[1:]))
    # Detached LSTM states and recorded hydraulic inputs at window boundaries within reach.
    memory = optimizer is not None
    lstm_states, recorded = {0: (state.hidden, state.cell)}, {}
    predictions, dates, losses = [], [], []
    tick, maximum, updates = time.monotonic(), 0.0, 0
    for wi, (a, b) in enumerate(windows):
        training = memory and a >= warmup_days
        lead = 0
        if training:
            first = max(day for day in lstm_states if day <= max(b - memory_days, 0))
            lead = a - first
        inputs, days = forcing.read(a - lead, b)
        start, hydraulic = state, None
        if lead:
            hidden, cell = lstm_states[a - lead]
            start = state._replace(hidden=hidden, cell=cell)
            hydraulic = torch.cat([recorded[d] for d in sorted(recorded) if a - lead <= d < a])
        previous, previous_q = state.storage, state.previous_q
        if training:
            optimizer.zero_grad(set_to_none=True)
            # The reverse steps add up the roughness adjoints (not trained): keep them bounded.
            physical.zero_parameter_grads()
        with torch.set_grad_enabled(training):
            q, s, r, terminal, qb = network(inputs, start, route_from=lead, hydraulic=hydraulic)
            check_newton_status(physical)
            maximum = max(maximum, mass_balance(
                *edges, previous, q, s, r, limit=limit,
                bifurcation=None if qb is None else (bif.bifurcation_catchment_idx,
                                                     bif.bifurcation_downstream_idx, qb)))
            if targets is not None and a >= warmup_days:
                units, observed, std = targets
                observation = torch.tensor(observed[a:b], device=q.device)
                value, logged = loss(q[:, units], observation, std, correlation_weight)
                if training and value is not None:
                    _update(network, optimizer, value, grad_clip)
                    updates += 1
                if logged is not None:
                    losses.append(logged)
        predictions.append(q[:, outputs].detach().cpu().numpy())
        dates.extend(days[lead:])
        state = terminal.detach()
        if memory:
            with torch.no_grad():
                recorded[a] = network.hydraulic_features(
                    torch.cat((previous[None], s[:-1].detach())),
                    torch.cat((previous_q[None], q[:-1].detach())))
            lstm_states[b] = (state.hidden, state.cell)
            if wi + 1 < len(windows):
                reach = max(windows[wi + 1][1] - memory_days, 0)
                oldest = max(day for day in lstm_states if day <= reach)
                lstm_states = {d: v for d, v in lstm_states.items() if d >= oldest}
                recorded = {d: v for d, v in recorded.items() if d >= oldest}
        if wi % 10 == 0 or wi == len(windows) - 1:
            progress(output_dir, phase=phase, epoch=epoch, chunk=wi + 1, chunks=len(windows),
                     last_date=str(days[-1]), seconds=time.monotonic() - tick,
                     mean_loss=float(np.mean(losses)) if losses else None,
                     updated_windows=updates, max_mass_residual=maximum)
        del inputs
    info = dict(units=len(state.storage), seconds=time.monotonic() - tick, updated_windows=updates,
                mean_loss=float(np.mean(losses)) if losses else None, max_mass_residual=maximum)
    return dict(dates=np.array(dates), q=np.concatenate(predictions)), state, info
