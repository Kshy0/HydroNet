# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""HydroNet: implicit local-inertia routing of a river network."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
from hydroforge.contracts import BackendRequirement
from hydroforge.execution import ManagedStep, adjoint_step, between_steps, managed_step
from hydroforge.model import (
    AbstractModel,
    copy_tensor_inputs,
    module_ref,
    optional_module_ref,
)
from pydantic import Field

from hydronet.config import HydroNetOptions
from hydronet.modules import BaseModule, BifurcationModule
from hydronet.phys import (
    solve_inertia_adjoint,
    solve_inertia_graph_adjoint,
    solve_inertia_graph_rebuild,
    solve_inertia_graph_step,
    solve_inertia_rebuild,
    solve_inertia_step,
)


class HydroNetModel(AbstractModel):
    """Fully implicit continuity and local inertia, with optional bifurcations.

    Every step solves the end-of-step water surfaces of each tree together by
    Newton's method, with backwater and reverse flow, for the staged runoff
    (:meth:`set_inputs`, every step) and the model's parameters, among them
    the channel/floodplain Manning roughness (:meth:`set_parameters`, constant
    over the steps they serve). With the ``bifurcation`` module the
    pathways join the same implicit solve. With ``options.differentiable`` a
    step also saves its converged surface, and :meth:`step_adjoint` reverses
    the step that ended at the current clock (``hydroforge.execution.step``).
    The driver stages what the reverse step reads: the step's start state and
    converged surface (``assign_state``), its runoff (:meth:`set_inputs`) and
    the result adjoints (``assign_state``). The state and runoff adjoints are
    then in their ``grad_*`` fields; the parameter adjoints (roughness, channel
    width and height) are added to theirs, so after the reverse steps of a
    window they hold its total (:meth:`zero_parameter_grads` starts one).
    ``scripts/training/coupling`` wraps this in autograd.
    """

    base = module_ref(BaseModule)
    bifurcation = optional_module_ref(BifurcationModule)
    partition_key: ClassVar[str] = "catchment_id"
    partition_group: ClassVar[str] = "catchment_basin_id"
    backend_requirements: ClassVar[Mapping[str, BackendRequirement]] = {
        "cuda": BackendRequirement(
            precision=frozenset({"float32"}),
            mixed_precision=False,
            default_mixed_precision=False,
            ensemble=False,
        ),
        "triton": BackendRequirement(
            precision=frozenset({"float32"}),
            mixed_precision=False,
            default_mixed_precision=False,
            ensemble=False,
        ),
        "metal": BackendRequirement(
            precision=frozenset({"float32"}),
            mixed_precision=False,
            default_mixed_precision=False,
            ensemble=False,
        ),
    }
    options: HydroNetOptions = Field(default_factory=HydroNetOptions)

    @property
    def graph(self) -> bool:
        """Whether steps solve the coupled bifurcation graph."""
        return (
            self.bifurcation is not None and self.bifurcation.num_bifurcation_paths > 0
        )

    @between_steps
    @torch.inference_mode()
    def set_inputs(self, *, runoff: torch.Tensor) -> None:
        """Stage one step of runoff."""
        copy_tensor_inputs({"runoff": (runoff, self.base.runoff)})

    @between_steps
    @torch.inference_mode()
    def set_parameters(
        self,
        *,
        river_manning: torch.Tensor | None = None,
        flood_manning: torch.Tensor | None = None,
        bifurcation_manning: torch.Tensor | None = None,
    ) -> None:
        """Replace the given roughness parameters for the following steps."""
        staged = {}
        if river_manning is not None:
            staged["river_manning"] = (river_manning, self.base.river_manning)
        if flood_manning is not None:
            staged["flood_manning"] = (flood_manning, self.base.flood_manning)
        if bifurcation_manning is not None:
            if self.bifurcation is None:
                raise ValueError("bifurcation_manning requires the bifurcation module")
            staged["bifurcation_manning"] = (
                bifurcation_manning,
                self.bifurcation.bifurcation_manning,
            )
        copy_tensor_inputs(staged)

    @torch.inference_mode()
    def zero_parameter_grads(self) -> None:
        """Zero the parameter adjoints that reverse steps add to."""
        if not self.options.differentiable:
            raise ValueError("parameter adjoints require options.differentiable")
        base = self.base
        for grad in (
            base.grad_river_manning,
            base.grad_flood_manning,
            base.grad_river_width,
            base.grad_river_height,
        ):
            grad.zero_()
        if self.bifurcation is not None:
            self.bifurcation.grad_bifurcation_manning.zero_()

    @managed_step
    @torch.inference_mode()
    def step_advance(self, step: ManagedStep) -> None:
        """Advance one step with the staged forcing: one implicit solve."""
        if step.requested_sub_steps != 1:
            raise ValueError(
                "HydroNet solves each step implicitly; it takes no sub-steps"
            )
        if self.graph:
            if self.plan.spatial_world_size != 1:
                raise ValueError(
                    "Coupled bifurcation routing currently requires one spatial rank"
                )
            solve_inertia_graph_step()
        else:
            solve_inertia_step()

    @adjoint_step
    @torch.inference_mode()
    def step_adjoint(self, step: ManagedStep) -> None:
        """Adjoint of the step that ended at the current clock and converged to
        ``base.converged_surface``, whose start state and forcing the driver
        staged into their fields, from the result adjoints
        ``base.grad_outflow``, ``base.grad_storage``, ``base.grad_water_surface``
        (and ``bifurcation.grad_bifurcation_outflow``). It first evaluates the
        converged Jacobians into ``base.jacobian_snapshot`` (and
        ``bifurcation.bifurcation_jacobian``) again, bit for bit as Newton ended.
        The outflow, storage and pathway discharge adjoints are replaced in place
        by those of the step's start and the runoff adjoint is written; the
        parameter adjoints (roughness, width, height) are added to theirs."""
        if not self.options.differentiable:
            raise ValueError("step_adjoint requires options.differentiable")
        if step.requested_sub_steps != 1:
            raise ValueError(
                "HydroNet solves each step implicitly; it takes no sub-steps"
            )
        if self.graph:
            solve_inertia_graph_rebuild()
            solve_inertia_graph_adjoint()
        else:
            solve_inertia_rebuild()
            solve_inertia_adjoint()
