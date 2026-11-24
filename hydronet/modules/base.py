"""Catchment forest, geometry, roughness, routing state, forcing and the Newton solve."""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
from hydroforge.model import (
    AbstractModule,
    CoordinateField,
    ErrorField,
    ReferenceField,
    ReferenceIndexField,
    SelectionField,
    TensorField,
    computed_cache,
    computed_field,
    computed_tensor_field,
    kernel_field,
)
from pydantic import model_validator

from hydronet.config import (
    ADJOINT_NOT_CONVERGED,
    INVALID_ROUGHNESS,
    INVALID_STATE,
    NEWTON_NOT_CONVERGED,
    NEWTON_ROUND_LIMIT,
    PARALLEL_PATH_MAX,
    PARALLEL_PATH_MIN,
)
from hydronet.modules._forest import decompose_tensor

ADJOINT = "options.differentiable"
JACOBIAN_TERMS = 17


def BaseField(
    description: str,
    shape: tuple[str, ...] = ("num_catchments",),
    dtype: Literal["float", "int", "idx", "bool"] = "float",
    dim_coords: str | None = "catchment_id",
    category: Literal["topology", "param", "forcing", "init_state", "state"] = "param",
    **kwargs,
):
    return TensorField(
        description=description,
        shape=shape,
        dtype=dtype,
        dim_coords=dim_coords,
        category=category,
        **kwargs,
    )


def computed_layout_field(
    description: str, shape: tuple[str, ...], dim_coords: str | None = None
):
    return computed_tensor_field(
        description=description,
        shape=shape,
        dtype="idx",
        dim_coords=dim_coords,
        category="topology",
        output="disabled",
    )


class BaseModule(AbstractModule):
    """The catchment forest and its geometry, roughness, storage and discharge, and the
    implicit solve."""

    module_name: ClassVar[str] = "base"
    description: ClassVar[str] = (
        "Catchment forest, geometry, roughness, routing state, forcing and Newton solve"
    )

    @kernel_field
    def slope_regularization(self) -> float:
        return self.options.value("slope_regularization")

    @kernel_field
    def newton_tolerance(self) -> float:
        return self.options.value("newton_tolerance")

    @kernel_field
    def newton_max_iterations(self) -> int:
        return self.options.value("newton_max_iterations")

    # ------------------------------------------------------------------ #
    # Topology
    # ------------------------------------------------------------------ #
    catchment_id: torch.Tensor = CoordinateField(
        description="Unique catchment identifier",
        dtype="idx",
        shape=("num_catchments",),
    )
    downstream_id: torch.Tensor = ReferenceField(
        description="Downstream catchment identifier (itself at an outlet)",
        dtype="idx",
        shape=("num_catchments",),
        dim_coords="catchment_id",
        references="catchment_id",
    )
    catchment_basin_id: torch.Tensor = BaseField(
        "Basin identifier; whole basins share a partition",
        dtype="idx",
        category="topology",
        mode="cpu",
    )
    output_catchment_id: torch.Tensor = SelectionField(
        description="Catchments saved in output",
        shape=("num_output_catchments",),
        dtype="idx",
        selects="catchment_id",
    )
    downstream_idx = ReferenceIndexField("downstream_id")
    output_catchment_idx = ReferenceIndexField("output_catchment_id")

    # ------------------------------------------------------------------ #
    # Forest layout of the implicit solve: computed online from downstream_idx
    # ------------------------------------------------------------------ #
    @computed_cache
    def _forest(self) -> dict[str, torch.Tensor]:
        return decompose_tensor(self.downstream_idx)

    @computed_layout_field(
        "Outlet catchment index of each catchment's tree",
        ("num_catchments",),
        "catchment_id",
    )
    def outlet_idx(self) -> torch.Tensor:
        return self._forest["outlet"]

    @computed_layout_field(
        "Most downstream catchment index of each catchment's heavy path",
        ("num_catchments",),
        "catchment_id",
    )
    def heavy_path_idx(self) -> torch.Tensor:
        return self._forest["head"]

    @computed_tensor_field(
        description="Light links between the catchment's heavy path and its outlet: the path's elimination round",
        shape=("num_catchments",),
        dtype="int",
        dim_coords="catchment_id",
        category="topology",
        output="disabled",
    )
    def heavy_path_round(self) -> torch.Tensor:
        return self._forest["round"].to(torch.int64)

    @computed_tensor_field(
        description="Longest chain of upstream links above the catchment: its level in a block solve",
        shape=("num_catchments",),
        dtype="int",
        dim_coords="catchment_id",
        category="topology",
        output="disabled",
    )
    def source_height(self) -> torch.Tensor:
        return self._forest["height"].to(torch.int64)

    @model_validator(mode="after")
    def validate_forest_decomposition(self):
        """Catchments must run upstream before downstream."""
        down = self.downstream_idx.long()
        n = down.numel()
        index = torch.arange(n, device=down.device)
        outlet = (down < 0) | (down == index)
        link = torch.nonzero(~outlet).squeeze(1)
        if link.numel() and torch.any(down[link] <= link).item():
            raise ValueError("catchments must be ordered upstream before downstream")
        return self

    @computed_layout_field(
        "Tree of each catchment", ("num_catchments",), "catchment_id"
    )
    def tree_index(self) -> torch.Tensor:
        # Trees are numbered in the order of their outlets.
        outlet = self.outlet_idx.long()
        is_outlet = outlet == torch.arange(outlet.numel(), device=outlet.device)
        return (torch.cumsum(is_outlet, 0) - 1)[outlet].to(torch.int32)

    @computed_layout_field("Number of catchments of each tree", ("num_trees",))
    def tree_size(self) -> torch.Tensor:
        return torch.bincount(
            self.tree_index.long(), minlength=self.tree_root.numel()
        ).to(torch.int32)

    @computed_layout_field("Outlet catchment of each tree", ("num_trees",))
    def tree_root(self) -> torch.Tensor:
        outlet = self.outlet_idx
        index = torch.arange(outlet.numel(), device=outlet.device)
        return torch.nonzero(outlet == index).squeeze(1).to(torch.int32)

    @computed_layout_field(
        "Start of each catchment's upstream list", ("num_upstream_offsets",)
    )
    def upstream_offset(self) -> torch.Tensor:
        down = self.downstream_idx.long()
        index = torch.arange(down.numel(), device=down.device)
        counts = torch.bincount(
            down[(down >= 0) & (down != index)], minlength=down.numel()
        )
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    @computed_layout_field(
        "Upstream catchments, grouped by downstream catchment", ("num_upstream_links",)
    )
    def upstream_index(self) -> torch.Tensor:
        down = self.downstream_idx.long()
        index = torch.arange(down.numel(), device=down.device)
        link = torch.nonzero((down >= 0) & (down != index)).squeeze(1)
        return link[torch.argsort(down[link], stable=True)].to(torch.int32)

    @computed_layout_field(
        "Catchments by heavy path, upstream to downstream", ("num_catchments",)
    )
    def path_unit(self) -> torch.Tensor:
        # By round, then path, then catchment (upstream before downstream): stable sorts
        # from the last key to the first.
        order = torch.argsort(self.heavy_path_idx.long(), stable=True)
        return order[torch.argsort(self.heavy_path_round[order], stable=True)].to(
            torch.int32
        )

    @computed_layout_field(
        "Start of each heavy path in path_unit", ("num_path_offsets",)
    )
    def path_offset(self) -> torch.Tensor:
        head = self.heavy_path_idx[self.path_unit.long()]
        starts = torch.ones_like(head, dtype=torch.bool)
        starts[1:] = head[1:] != head[:-1]
        end = torch.full((1,), head.numel(), device=head.device)
        return torch.cat([torch.nonzero(starts).squeeze(1), end]).to(torch.int32)

    @computed_layout_field("Start of each round of heavy paths", ("num_round_offsets",))
    def round_offset(self) -> torch.Tensor:
        first = self.path_unit.long()[self.path_offset.long()[:-1]]
        counts = torch.bincount(self.heavy_path_round[first])
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    @computed_layout_field(
        "Catchments by tree and level (height above the sources)", ("num_catchments",)
    )
    def tree_unit(self) -> torch.Tensor:
        # By tree, then level, then catchment.
        order = torch.argsort(self.source_height, stable=True)
        return order[torch.argsort(self.tree_index[order], stable=True)].to(torch.int32)

    @computed_layout_field("Start of each level in tree_unit", ("num_level_offsets",))
    def level_offset(self) -> torch.Tensor:
        unit = self.tree_unit.long()
        tree, height = self.tree_index[unit], self.source_height[unit]
        starts = torch.ones_like(height, dtype=torch.bool)
        starts[1:] = (tree[1:] != tree[:-1]) | (height[1:] != height[:-1])
        end = torch.full((1,), unit.numel(), device=unit.device)
        return torch.cat([torch.nonzero(starts).squeeze(1), end]).to(torch.int32)

    @computed_layout_field("First level of each tree", ("num_tree_offsets",))
    def tree_level(self) -> torch.Tensor:
        first = self.tree_unit.long()[self.level_offset.long()[:-1]]
        counts = torch.bincount(self.tree_index.long()[first], minlength=self.num_trees)
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    # Split of the solve by options.grid_tree_size.
    @computed_layout_field("Trees solved by the whole device", ("num_grid_trees",))
    def grid_tree(self) -> torch.Tensor:
        # Trees of at least grid_tree_size units, when together they hold at least half of
        # the units. Otherwise the block solves of the other trees take as long, and a grid
        # pass after them only adds time.
        sizes = self.tree_size.long()
        large = sizes >= self.options.value("grid_tree_size")
        if 2 * int(sizes[large].sum()) < int(sizes.sum()):
            large = torch.zeros_like(large)
        return torch.nonzero(large).squeeze(1).to(torch.int32)

    @computed_layout_field("Trees solved by one block each", ("num_block_trees",))
    def block_tree(self) -> torch.Tensor:
        grid = torch.zeros(
            self.num_trees, dtype=torch.bool, device=self.tree_size.device
        )
        grid[self.grid_tree.long()] = True
        return torch.nonzero(~grid).squeeze(1).to(torch.int32)

    @computed_layout_field("Catchments of grid-solved trees", ("num_grid_units",))
    def grid_unit(self) -> torch.Tensor:
        grid = torch.zeros(
            self.num_trees, dtype=torch.bool, device=self.tree_size.device
        )
        grid[self.grid_tree.long()] = True
        unit = self.tree_unit.long()
        return unit[grid[self.tree_index.long()[unit]]].to(torch.int32)

    @computed_layout_field(
        "Heavy paths of grid-solved trees, by round", ("num_grid_paths",)
    )
    def grid_path(self) -> torch.Tensor:
        # Inside a round, the paths solved in parallel (PARALLEL_PATH_MIN..MAX units) come
        # first, each group longest first; the CUDA binary searches rely on this order.
        grid = torch.zeros(
            self.num_trees, dtype=torch.bool, device=self.tree_size.device
        )
        grid[self.grid_tree.long()] = True
        offset = self.path_offset.long()
        first = self.path_unit.long()[offset[:-1]]
        paths = torch.nonzero(grid[self.tree_index.long()[first]]).squeeze(1)
        lengths = (offset[1:] - offset[:-1])[paths]
        serial = ((lengths < PARALLEL_PATH_MIN) | (lengths > PARALLEL_PATH_MAX)).long()
        order = torch.argsort(-lengths, stable=True)
        order = order[torch.argsort(serial[order], stable=True)]
        order = order[
            torch.argsort(self.heavy_path_round[first][paths][order], stable=True)
        ]
        return paths[order].to(torch.int32)

    @computed_layout_field("Start of each round in grid_path", ("num_round_offsets",))
    def grid_round_offset(self) -> torch.Tensor:
        first = self.path_unit.long()[self.path_offset.long()[:-1]]
        rounds = self.heavy_path_round[first][self.grid_path.long()]
        counts = torch.bincount(rounds, minlength=self.num_round_offsets - 1)
        return torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    # ------------------------------------------------------------------ #
    # Channel and floodplain geometry (CaMa-Flood map) and roughness
    # ------------------------------------------------------------------ #
    river_length: torch.Tensor = BaseField("Channel length", units="m", gt=0)
    river_width: torch.Tensor = BaseField("Channel width", units="m", gt=0)
    river_height: torch.Tensor = BaseField("Bankfull channel depth", units="m", ge=0)
    catchment_area: torch.Tensor = BaseField("Local catchment area", units="m2", gt=0)
    catchment_elevation: torch.Tensor = BaseField("Bank-top elevation", units="m")
    downstream_distance: torch.Tensor = BaseField(
        "Distance to the downstream catchment", units="m"
    )
    flood_depth_table: torch.Tensor = BaseField(
        "Floodplain surface above the bank top at the end of each equal-area band",
        units="m",
        shape=("num_catchments", "num_flood_levels"),
        ge=0,
    )
    river_manning: torch.Tensor = BaseField(
        "Channel Manning roughness (HydroNetModel.set_parameters)",
        units="s m-1/3",
        default=0.03,
    )
    flood_manning: torch.Tensor = BaseField(
        "Floodplain Manning roughness (HydroNetModel.set_parameters)",
        units="s m-1/3",
        default=0.10,
    )

    @computed_tensor_field(
        description="Catchment drains through a channel to its downstream catchment",
        shape=("num_catchments",),
        dtype="bool",
        dim_coords="catchment_id",
        category="derived_param",
        output="disabled",
    )
    def downstream_edge(self) -> torch.Tensor:
        down = self.downstream_idx
        index = torch.arange(down.numel(), device=down.device)
        return (down >= 0) & (down != index) & (self.downstream_distance > 0)

    @computed_tensor_field(
        description="Floodplain bed length from the bank top to the end of each band; band perimeters of the floodplain conveyance",
        units="m",
        shape=("num_catchments", "num_flood_levels"),
        dim_coords="catchment_id",
        category="derived_param",
        output="disabled",
    )
    def flood_bed_length(self) -> torch.Tensor:
        table = self.flood_depth_table
        previous = torch.cat([torch.zeros_like(table[:, :1]), table[:, :-1]], 1)
        width = self.catchment_area / self.river_length / table.shape[1]
        return torch.cumsum(
            torch.sqrt(width[:, None].square() + (table - previous).square()), 1
        )

    # Parameter adjoints: every reverse step adds its own (zero_parameter_grads resets them).
    grad_river_manning: torch.Tensor | None = BaseField(
        "Adjoint of the channel roughness",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_flood_manning: torch.Tensor | None = BaseField(
        "Adjoint of the floodplain roughness",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_river_width: torch.Tensor | None = BaseField(
        "Adjoint of the channel width",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_river_height: torch.Tensor | None = BaseField(
        "Adjoint of the bankfull depth",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )

    # ------------------------------------------------------------------ #
    # State and forcing
    # ------------------------------------------------------------------ #
    storage: torch.Tensor = BaseField(
        "Channel and floodplain storage",
        units="m3",
        category="init_state",
        default=0,
        ge=0,
    )
    outflow: torch.Tensor = BaseField(
        "Signed discharge to the downstream catchment; negative in reverse flow",
        units="m3 s-1",
        category="init_state",
        default=0,
    )
    inflow: torch.Tensor = BaseField(
        "Discharge from upstream catchments plus reverse flow from downstream",
        units="m3 s-1",
        category="state",
        default=0,
    )
    water_surface: torch.Tensor = BaseField(
        "Water surface above the bank top",
        units="m",
        category="state",
        default=0,
    )
    runoff: torch.Tensor = BaseField(
        "Local runoff",
        units="m3 s-1",
        category="forcing",
        default=0,
    )

    # ------------------------------------------------------------------ #
    # Newton solve
    # ------------------------------------------------------------------ #
    newton_record: torch.Tensor = BaseField(
        "Per tree: residual, Newton steps, status, line-search trials, Newton steps, 0, "
        "evaluations, 0, 0, 0, 0, 0, damped Newton steps, last phase + 2 (1 Newton; 23 "
        "fixed-weight restart; 24 damped Newton; 2-22 graph continuation), fallback phase used, "
        "residual of the first phase",
        shape=("num_trees", 16),
        dim_coords=None,
        category="state",
        default=0,
        output="disabled",
    )
    newton_status = ErrorField(
        "Implicit solve status",
        {
            NEWTON_NOT_CONVERGED: "Newton iteration of a tree did not reach the residual tolerance",
            NEWTON_ROUND_LIMIT: "solve exceeded its round limit",
            INVALID_ROUGHNESS: "Manning roughness is not positive and finite",
            INVALID_STATE: "converged storage or discharge is negative or not finite",
            ADJOINT_NOT_CONVERGED: "coupled graph adjoint did not reach its linear residual tolerance",
        },
        info=("index",),
    )

    # ------------------------------------------------------------------ #
    # Adjoint (options.differentiable)
    # ------------------------------------------------------------------ #
    jacobian_snapshot: torch.Tensor | None = BaseField(
        "Converged derivatives, evaluated by step_adjoint at converged_surface: Q, S, dQ/deta, "
        "dQ/deta_down, area, dry bound, surface, dS/dW, dS/dH, dQ/dW, dQ/dH, dQ/dH_down, "
        "dQ/dQ_previous, dQ/dn_channel, dQ/dn_flood, reverse, time step",
        shape=("num_catchments", JACOBIAN_TERMS),
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    converged_surface: torch.Tensor | None = BaseField(
        "Converged end-of-step surface as an error-free pair (reference, offset)",
        units="m",
        shape=("num_catchments", 2),
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_outflow: torch.Tensor | None = BaseField(
        "Adjoint of end-of-step outflow; the adjoint step replaces it by that of start-of-step "
        "outflow",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_storage: torch.Tensor | None = BaseField(
        "Adjoint of end-of-step storage; the adjoint step replaces it by that of start-of-step "
        "storage",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_water_surface: torch.Tensor | None = BaseField(
        "Adjoint of end-of-step water surface",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )
    grad_runoff: torch.Tensor | None = BaseField(
        "Adjoint of runoff",
        category="state",
        default=0,
        output="disabled",
        depends_on=ADJOINT,
    )

    # ------------------------------------------------------------------ #
    # Dimensions
    # ------------------------------------------------------------------ #
    @computed_field(description="Number of catchments")
    def num_catchments(self) -> int:
        return self.catchment_id.numel()

    @computed_field(description="Number of saved catchments")
    def num_output_catchments(self) -> int:
        return self.output_catchment_id.numel()

    @computed_field(description="Number of floodplain bands")
    def num_flood_levels(self) -> int:
        return self.flood_depth_table.shape[-1]

    @computed_field(description="Number of trees (outlets)")
    def num_trees(self) -> int:
        down = self.downstream_idx
        index = torch.arange(down.numel(), device=down.device, dtype=down.dtype)
        return int(((down < 0) | (down == index)).sum().item())

    @computed_field(description="Number of upstream links")
    def num_upstream_links(self) -> int:
        return self.upstream_index.numel()

    @computed_field(description="Length of upstream_offset")
    def num_upstream_offsets(self) -> int:
        return self.num_catchments + 1

    @computed_field(description="Length of path_offset")
    def num_path_offsets(self) -> int:
        return self.path_offset.numel()

    @computed_field(description="Length of round_offset")
    def num_round_offsets(self) -> int:
        return self.round_offset.numel()

    @computed_field(description="Length of tree_level")
    def num_tree_offsets(self) -> int:
        return self.num_trees + 1

    @computed_field(description="Length of level_offset")
    def num_level_offsets(self) -> int:
        return self.level_offset.numel()

    @computed_field(description="Number of trees solved by one block each")
    def num_block_trees(self) -> int:
        return self.block_tree.numel()

    @computed_field(description="Number of trees solved by the whole device")
    def num_grid_trees(self) -> int:
        return self.grid_tree.numel()

    @computed_field(description="Number of catchments of grid-solved trees")
    def num_grid_units(self) -> int:
        return self.grid_unit.numel()

    @computed_field(description="Number of heavy paths of grid-solved trees")
    def num_grid_paths(self) -> int:
        return self.grid_path.numel()

    @computed_field(description="Number of heavy-path rounds")
    def num_path_rounds(self) -> int:
        return self.round_offset.numel() - 1

    @computed_field(description="Largest number of heavy paths in one round")
    def max_round_paths(self) -> int:
        return int(torch.diff(self.round_offset).max())
