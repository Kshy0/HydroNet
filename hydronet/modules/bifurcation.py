"""Multi-level Manning pathways, their recurrent discharge and implicit adjoint."""

from typing import ClassVar

import torch
from hydroforge.model import (
    AbstractModule,
    CoordinateField,
    ReferenceField,
    ReferenceIndexField,
    TensorField,
    computed_cache,
    computed_field,
    computed_tensor_field,
    kernel_field,
    module_ref,
)
from pydantic import model_validator

from hydronet.config import GRAPH_CLOSED_ELEVATION
from hydronet.modules._bifurcation import condensed_graph_plan_tensors
from hydronet.modules.base import ADJOINT, BaseModule, computed_layout_field


class BifurcationModule(AbstractModule):
    module_name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = (
        "Implicit pathways connecting catchments, including cycles"
    )
    base = module_ref(BaseModule)

    @computed_cache
    def _cond(self) -> dict:
        b = self.base
        return condensed_graph_plan_tensors(
            b.downstream_idx,
            b.downstream_edge,
            b.tree_index,
            b.tree_size,
            self.bifurcation_catchment_idx,
            self.bifurcation_downstream_idx,
        )

    @kernel_field
    def num_factor_rounds(self) -> int:
        return int(self._cond["num_factor_rounds"])

    @kernel_field
    def num_core_factor_rounds(self) -> int:
        return int(self._cond["num_core_factor_rounds"])

    @kernel_field
    def num_graph_comps(self) -> int:
        return int(self._cond["num_graph_comps"])

    bifurcation_path_id: torch.Tensor = CoordinateField(
        description="Identifier of a bifurcation path",
        dtype="idx",
        shape=("num_bifurcation_paths",),
        partition_by="bifurcation_catchment_id",
    )
    bifurcation_catchment_id: torch.Tensor = ReferenceField(
        description="Source catchment identifier",
        dtype="idx",
        shape=("num_bifurcation_paths",),
        dim_coords="bifurcation_path_id",
        references="base.catchment_id",
    )
    bifurcation_downstream_id: torch.Tensor = ReferenceField(
        description="Receiving catchment identifier",
        dtype="idx",
        shape=("num_bifurcation_paths",),
        dim_coords="bifurcation_path_id",
        references="base.catchment_id",
    )
    bifurcation_catchment_idx = ReferenceIndexField("bifurcation_catchment_id")
    bifurcation_downstream_idx = ReferenceIndexField("bifurcation_downstream_id")

    bifurcation_width: torch.Tensor = TensorField(
        description="Width of each rectangular pathway level",
        units="m",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="param",
    )

    bifurcation_elevation: torch.Tensor = TensorField(
        description="Bed elevation of each pathway level above mean sea level",
        units="m",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="param",
    )

    bifurcation_manning: torch.Tensor = TensorField(
        description="Manning roughness of each pathway level",
        units="s m-1/3",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="param",
        default=0.03,
    )

    bifurcation_outflow: torch.Tensor = TensorField(
        description="Signed discharge of each pathway level",
        units="m3 s-1",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="init_state",
        default=0,
    )

    grad_bifurcation_outflow: torch.Tensor | None = TensorField(
        description="Adjoint of the end discharge; the adjoint step replaces it by that of the "
        "start discharge",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="state",
        default=0,
        depends_on=ADJOINT,
        output="disabled",
    )

    grad_bifurcation_manning: torch.Tensor | None = TensorField(
        description="Adjoint of the pathway roughness",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels"),
        dim_coords="bifurcation_path_id",
        category="state",
        default=0,
        depends_on=ADJOINT,
        output="disabled",
    )

    bifurcation_length: torch.Tensor = TensorField(
        description="Length of a bifurcation pathway",
        units="m",
        dtype="float",
        shape=("num_bifurcation_paths",),
        dim_coords="bifurcation_path_id",
        gt=0,
    )
    bifurcation_jacobian: torch.Tensor | None = TensorField(
        description="Converged Q, dQ/deta_source, dQ/deta_target, dQ/dQ_start, dQ/dn, dQ/dH_source, "
        "dQ/dH_target, evaluated by step_adjoint at base.converged_surface",
        dtype="float",
        shape=("num_bifurcation_paths", "num_bifurcation_levels", 7),
        dim_coords="bifurcation_path_id",
        category="state",
        default=0,
        depends_on=ADJOINT,
        output="disabled",
    )
    linear_record: torch.Tensor = TensorField(
        description="Last graph linear solve: status, iterations, true equilibrated relative residual, total forward iterations",
        dtype="float",
        shape=(4,),
        dim_coords=None,
        category="state",
        default=0,
        output="disabled",
    )

    @computed_tensor_field(
        description="Start of incident signed bifurcation paths",
        dtype="idx",
        category="topology",
        shape=("base.num_upstream_offsets",),
        output="disabled",
    )
    def incident_offset(self) -> torch.Tensor:
        nodes = torch.cat(
            [self.bifurcation_catchment_idx, self.bifurcation_downstream_idx]
        ).long()
        counts = torch.bincount(nodes, minlength=self.base.num_catchments)
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    @computed_tensor_field(
        description="Signed path indices: positive source, negative receiver",
        dtype="idx",
        category="topology",
        shape=("num_incidents",),
        output="disabled",
    )
    def incident_path(self) -> torch.Tensor:
        # Grouped by catchment; path p is +(p + 1) at its source and -(p + 1) at its target.
        nodes = torch.cat(
            [self.bifurcation_catchment_idx, self.bifurcation_downstream_idx]
        ).long()
        path = torch.arange(1, self.num_bifurcation_paths + 1, device=nodes.device)
        return torch.cat([path, -path])[torch.argsort(nodes, stable=True)].to(
            torch.int32
        )

    @computed_layout_field(
        "Pure-forest main trees with zero bifurcation coupling", ("num_pure_trees",)
    )
    def pure_block_tree(self) -> torch.Tensor:
        return self._cond["pure_block_tree"]

    @computed_layout_field(
        "Catchments belonging to coupled bifurcation trees", ("num_graph_units",)
    )
    def graph_unit(self) -> torch.Tensor:
        return self._cond["graph_unit"]

    @computed_layout_field(
        "Coupled component id of each catchment (-1 for pure forest)",
        ("base.num_catchments",),
    )
    def unit_comp(self) -> torch.Tensor:
        return self._cond["unit_comp"]

    @computed_layout_field(
        "Coupled component id of each bifurcation path", ("num_bifurcation_paths",)
    )
    def path_comp(self) -> torch.Tensor:
        return self._cond["path_comp"]

    @computed_layout_field(
        "Main-channel trees belonging to coupled components", ("num_coupled_trees",)
    )
    def coupled_trees(self) -> torch.Tensor:
        return self._cond["coupled_trees"]

    @computed_layout_field(
        "Component id of each coupled main-channel tree", ("num_coupled_trees",)
    )
    def coupled_tree_comp(self) -> torch.Tensor:
        return self._cond["coupled_tree_comp"]

    @computed_layout_field(
        "Representative coupled tree index for each coupled component",
        ("num_graph_comps",),
    )
    def comp_rep_tree(self) -> torch.Tensor:
        return self._cond["comp_rep_tree"]

    @computed_layout_field(
        "Packed static LU plan of the full graph Jacobian", ("num_factor_plan",)
    )
    def factor_layout(self) -> torch.Tensor:
        return self._cond["factor_layout"]

    @computed_layout_field(
        "Packed static LU plan over coupled catchments", ("num_core_factor_plan",)
    )
    def core_factor_layout(self) -> torch.Tensor:
        return self._cond["core_factor_layout"]

    @computed_layout_field(
        "Component id of each coupled LU entry", ("num_core_factor_entries",)
    )
    def core_entry_comp(self) -> torch.Tensor:
        return self._cond["core_entry_comp"]

    @computed_field(description="Number of pure-forest trees")
    def num_pure_trees(self) -> int:
        return int(self._cond["pure_block_tree"].numel())

    @computed_field(description="Number of catchments in coupled bifurcation trees")
    def num_graph_units(self) -> int:
        return int(self._cond["graph_unit"].numel())

    @computed_field(description="Number of coupled main-channel trees")
    def num_coupled_trees(self) -> int:
        return int(self._cond["coupled_trees"].numel())

    @computed_field(description="Length of packed coupled LU plan")
    def num_core_factor_plan(self) -> int:
        return int(self._cond["core_factor_layout"].numel())

    @computed_field(description="Number of entries of coupled LU factors")
    def num_core_factor_entries(self) -> int:
        return int(self._cond["num_core_factor_entries"])

    @computed_field(description="Workspace capacity for graph LU factors")
    def max_factor_entries(self) -> int:
        return max(self.num_factor_entries, self.num_core_factor_entries, 1)

    @computed_field(description="Length of the packed factor plan")
    def num_factor_plan(self) -> int:
        return int(self._cond["factor_layout"].numel())

    @computed_field(description="Number of entries of the graph LU factors")
    def num_factor_entries(self) -> int:
        return int(self._cond["num_factor_entries"])

    @computed_field(description="Number of bifurcation pathways")
    def num_bifurcation_paths(self) -> int:
        return self.bifurcation_path_id.numel()

    @computed_field(description="Number of rectangular levels per pathway")
    def num_bifurcation_levels(self) -> int:
        return self.bifurcation_width.shape[-1]

    @computed_field(description="Number of pathway-level discharges")
    def num_bifurcation_values(self) -> int:
        return self.num_bifurcation_paths * self.num_bifurcation_levels

    @computed_field(description="Two endpoint incidences per bifurcation path")
    def num_incidents(self) -> int:
        return 2 * self.num_bifurcation_paths

    @model_validator(mode="after")
    def validate_pathways(self):
        source, target = self.bifurcation_catchment_idx, self.bifurcation_downstream_idx
        if torch.any((source < 0) | (target < 0) | (source == target)).item():
            raise ValueError("bifurcation endpoints must be distinct local catchments")
        if self.num_bifurcation_paths and self.num_bifurcation_levels < 1:
            raise ValueError("nonempty bifurcation paths require at least one level")
        if not all(
            torch.isfinite(v).all().item()
            for v in (
                self.bifurcation_width,
                self.bifurcation_length,
                self.bifurcation_elevation,
            )
        ):
            raise ValueError("bifurcation geometry must be finite")
        if torch.any(self.bifurcation_width < 0).item():
            raise ValueError("bifurcation widths must be nonnegative")
        active = self.bifurcation_elevation < GRAPH_CLOSED_ELEVATION
        if torch.any(
            active
            & (
                (self.bifurcation_manning <= 0)
                | ~torch.isfinite(self.bifurcation_manning)
            )
        ).item():
            raise ValueError(
                "active bifurcation levels require finite positive roughness"
            )
        if torch.any(active & (self.bifurcation_width <= 0)).item():
            raise ValueError("active bifurcation levels require positive width")
        return self
