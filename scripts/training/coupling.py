"""Forcing LSTM -> local runoff -> HydroNetModel steps.

Each physical day is one autograd node (:class:`PhysicsStep`). Its forward
assigns the day's start state, stages the runoff, advances one step and
returns the results; it keeps the start state and runoff it was given, the
converged surface and the clock at the end of the day. Its backward puts the
model back at the end of that day, stages the same start state, runoff and
surface with the result adjoints, and runs the model's reverse step
(``step_adjoint``), which moves the clock back to the day's start; it returns
the state and runoff adjoints on the inputs' devices. The model itself keeps
no history: everything a backward needs is held by its node.

Only the forcing (runoff) and the state pass through autograd here. The
model's parameters (roughness) stay fixed over a window; the reverse steps add
their adjoints to the model's ``grad_*`` parameter fields.

The clock travels with the state: :meth:`HydroNet.initial_state` starts a new
timeline at the model's initial time, every routed day assigns the clock it
starts from and the final state carries the clock it ended at.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.autograd.function import once_differentiable

from scripts.training.lstm import RunoffLSTM

DAY = timedelta(days=1)


class RoutingState(NamedTuple):
    storage: torch.Tensor
    previous_q: torch.Tensor
    hidden: torch.Tensor
    cell: torch.Tensor
    bifurcation_outflow: torch.Tensor | None = None
    # The model clock the state belongs to (hydroforge ModelClock).
    clock: Any = None

    def detach(self):
        return RoutingState(*(v.detach() if isinstance(v, torch.Tensor) else v for v in self))


# Adjoint fields of PhysicsStep's tensor inputs, in their order.
_GRADS = ("base.grad_storage", "base.grad_outflow", "base.grad_runoff")
_PATHWAY_GRAD = "bifurcation.grad_bifurcation_outflow"


class PhysicsStep(torch.autograd.Function):
    """One day of ``HydroNetModel`` as an autograd node (see module docs).

    ``PhysicsStep.apply(physics, time_step, storage, outflow, runoff,
    pathway=None)``: start storage and outflow, the day's runoff and, with
    bifurcations, the start pathway discharge. Returns end outflow, storage and
    pathway discharge (``None`` without bifurcations).
    """

    @staticmethod
    def forward(ctx, physics, time_step, storage, outflow, runoff, pathway=None):
        graph = pathway is not None
        start = {"base.storage": storage.detach(), "base.outflow": outflow.detach()}
        if graph:
            start["bifurcation.bifurcation_outflow"] = pathway.detach()
        physics.assign_state(start)
        physics.set_inputs(runoff=runoff.detach())
        physics.step_advance(time_step=time_step)
        keys = ("base.outflow", "base.storage") + (("bifurcation.bifurcation_outflow",) if graph else ())
        results = physics.capture_state(keys)
        ctx.set_materialize_grads(False)
        if any(ctx.needs_input_grad):
            ctx.physics, ctx.time_step, ctx.graph = physics, time_step, graph
            ctx.end = physics.capture_clock()
            surface = physics.capture_state(("base.converged_surface",))["base.converged_surface"]
            ctx.save_for_backward(storage, outflow, runoff, pathway, surface)
        return tuple(results[k] for k in keys) + ((None,) if not graph else ())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_outflow, grad_storage, grad_pathway):
        physics, graph = ctx.physics, ctx.graph
        storage, outflow, runoff, pathway, surface = inputs = ctx.saved_tensors

        def zeros(key):
            module, _, name = key.partition(".")
            return torch.zeros_like(getattr(getattr(physics, module), name))

        staged = {
            "base.storage": storage.detach(), "base.outflow": outflow.detach(),
            "base.converged_surface": surface,
            "base.grad_outflow": zeros("base.grad_outflow") if grad_outflow is None else grad_outflow,
            "base.grad_storage": zeros("base.grad_storage") if grad_storage is None else grad_storage,
            "base.grad_water_surface": zeros("base.grad_water_surface"),
        }
        if graph:
            staged["bifurcation.bifurcation_outflow"] = pathway.detach()
            staged[_PATHWAY_GRAD] = zeros(_PATHWAY_GRAD) if grad_pathway is None else grad_pathway
        # Back at the end of the day; the reverse step moves the clock to its start.
        physics.assign_clock(ctx.end)
        physics.set_inputs(runoff=runoff.detach())
        physics.assign_state(staged)
        physics.step_adjoint(time_step=ctx.time_step)
        keys = _GRADS + ((_PATHWAY_GRAD,) if graph else (None,))
        grads = physics.capture_state(k for k in keys if k is not None)
        # Adjoints are on the model's device; autograd wants each on its input's.
        result = (None, None, *(
            grads[k].to(x.device) if needed and x is not None else None
            for k, x, needed in zip(keys, inputs, ctx.needs_input_grad[2:])
        ))
        return result[:len(ctx.needs_input_grad)]


class HydroNet(nn.Module):
    """The trainable model: an LSTM two-way coupled to a differentiable ``HydroNetModel``.

    Day by day the LSTM reads the forcing, the static attributes and the
    hydraulic state at the day's start (:meth:`hydraulic_features`) and
    predicts every unit's runoff. The model routes the day with its configured
    parameters, and gradients pass both ways
    across all days of a window. ``time_step`` is the physical step of one
    forcing record (runoff stays in mm/day).
    """

    def __init__(self, model, static, prior_mean, prior_std, upstream_area, *, hydraulic_mean,
                 hydraulic_std, forcing_inputs,
                 lstm_hidden=256, spatial_checkpoint_cells=4096, temporal_checkpoint_days=0,
                 time_step=DAY):
        super().__init__()
        if not model.options.differentiable:
            raise ValueError("HydroNet training needs a model built with options.differentiable")
        self.physics = model
        bif = model.bifurcation
        self.has_bifurcation = bif is not None and bif.num_bifurcation_paths > 0
        self.register_buffer("initial_bifurcation_outflow",
            bif.bifurcation_outflow.clone() if self.has_bifurcation else None)
        self.time_step = time_step
        self.register_buffer("static", static)
        self.register_buffer("local_area", model.base.catchment_area)
        self.register_buffer("upstream_area", torch.as_tensor(upstream_area, dtype=static.dtype,
                                                              device=static.device))
        self.register_buffer("hydraulic_mean", static.new_tensor(hydraulic_mean))
        self.register_buffer("hydraulic_std", static.new_tensor(hydraulic_std))
        self.frontend = RunoffLSTM(
            static.shape[1], prior_std, prior_mean, hidden=lstm_hidden,
            spatial_checkpoint_cells=spatial_checkpoint_cells,
            temporal_checkpoint_days=temporal_checkpoint_days, inputs=forcing_inputs + 2,
        ).to(static.device)

    def initial_state(self, storage, previous_q, bifurcation_outflow=None):
        """The state at the model's initial time, on a new clock timeline."""
        n, hidden = len(storage), self.frontend.hidden
        qb = self.initial_bifurcation_outflow if bifurcation_outflow is None else bifurcation_outflow
        self.physics.reset_clock()
        return RoutingState(storage.clone(), previous_q.clone(),
                            storage.new_zeros((1, n, hidden)), storage.new_zeros((1, n, hidden)),
                            None if qb is None else qb.clone(), self.physics.capture_clock())

    def hydraulic_features(self, storage, outflow):
        """The hydraulic LSTM inputs of the state (storage, outflow), (..., N, 2).

        Storage is log1p of its depth over the upstream area (mm) and signed
        outflow asinh of the specific discharge over the upstream area
        (mm/day): per upstream area small and large rivers are comparable, so
        the inputs carry their change in time rather than river size (the
        static attributes give that), and asinh keeps reverse flow. Both are
        standardized by the fixed hydraulic mean/std and clipped to +-5 (lakes
        and flat floodplains store much relative to their upstream area).
        """
        depth = storage.clamp_min(0.0) * (1000.0 / self.upstream_area)
        specific = outflow * (86400.0 * 1000.0 / self.upstream_area)
        raw = torch.stack((torch.log1p(depth), torch.asinh(specific)), -1)
        return ((raw - self.hydraulic_mean) / self.hydraulic_std).clamp(-5.0, 5.0)

    def forward(self, forcing, state, *, route_from=0, hydraulic=None):
        """Route ``forcing`` (T, N, F) from ``state``.

        The LSTM runs all T days from ``state.hidden``/``state.cell``; the
        routing starts from ``state``'s storage at day ``route_from``. The
        ``route_from`` days before it read ``hydraulic`` (route_from, N, 2),
        recorded when they were routed. Returns the routed days' signed outflow
        (N, m3/s), storage (N, m3), runoff (N, m3/s), the final state and the
        pathway discharge (``None`` without bifurcations).
        """
        h, c = state.hidden, state.cell
        if route_from:
            if hydraulic is None or len(hydraulic) != route_from:
                raise ValueError("the days before route_from need their recorded hydraulic inputs")
            _, h, c = self.frontend(torch.cat((forcing[:route_from], hydraulic), -1),
                                    self.static, h, c)
        s, q, qb = state.storage, state.previous_q, state.bifurcation_outflow
        if self.has_bifurcation and qb is None:
            raise ValueError("RoutingState must include bifurcation discharge")
        if state.clock is None:
            raise ValueError("RoutingState must carry the model clock (see initial_state)")
        physics, clock = self.physics, state.clock
        qs, ss, rs, qbs = [], [], [], []
        for t in range(route_from, len(forcing)):
            inputs = torch.cat((forcing[t], self.hydraulic_features(s, q)), -1)[None]
            predicted, h, c = self.frontend(inputs, self.static, h, c)
            runoff = predicted[0] * (self.local_area * (0.001 / 86400.0))
            physics.assign_clock(clock)
            q, s, qb = PhysicsStep.apply(physics, self.time_step, s, q, runoff,
                                         qb if self.has_bifurcation else None)
            clock = physics.capture_clock()
            # Each day's results join the window's series right away, so autograd sums
            # the adjoints of a day's state in a fixed order (series, then later days).
            qs.append(q[None])
            ss.append(s[None])
            rs.append(runoff[None])
            if qb is not None:
                qbs.append(qb[None])
        return (torch.cat(qs), torch.cat(ss), torch.cat(rs), RoutingState(s, q, h, c, qb, clock),
                torch.cat(qbs) if qbs else None)
