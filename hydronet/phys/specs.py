"""Kernel ABIs of the implicit local-inertia step and its adjoint."""

from hydroforge.contracts import step_field
from hydroforge.kernels import KernelSpec, KernelWorkspace

from hydronet.config import TREE_INTEGERS, TREE_SCALARS


def _scratch(name: str, shape=("num_catchments",), dtype="float32") -> KernelWorkspace:
    return KernelWorkspace(key=f"hydronet.newton.{name}", dtype=dtype, shape=shape)


# Per-catchment Newton iterate (reference surface + offset, kept as an
# error-free pair), its trial and the evaluated geometry, flux and residual.
_UNIT_SCRATCH = (
    "newton_offset",
    "trial_offset",
    "newton_reference",
    "trial_reference",
    "trial_storage",
    "trial_area",
    "channel_conveyance",
    "flood_conveyance",
    "channel_conveyance_slope",
    "flood_conveyance_slope",
    "trial_outflow",
    "outflow_own_slope",
    "outflow_down_slope",
    "residual",
    "balance_weight",
    "accepted_weight",
)
# Tree elimination: pivots and right-hand sides (shared with the adjoint), step, coupling.
# Grid PCR stores dh = rhs - coupling * dh_parent with a unit pivot.
# After substitution these arrays can cache the bank-relative volume coordinate and slope.
_ELIMINATION = ("pivot", "pivot_rhs", "newton_step", "coupling")
WORKSPACE = {
    **{f"{name}_ptr": _scratch(name) for name in _UNIT_SCRATCH + _ELIMINATION},
    # Surface of the last continuation phase: total, reference, offset.
    "phase_trace_ptr": _scratch("phase_trace", (3, "num_catchments")),
    # Per tree: control integers, step/line-search scalars, merit of the round.
    "tree_control_ptr": _scratch("tree_control", ("num_trees", TREE_INTEGERS), "int32"),
    "tree_scalars_ptr": _scratch("tree_scalars", ("num_trees", TREE_SCALARS)),
    "tree_merit_ptr": _scratch("tree_merit", ("num_trees",), "int32"),
    # Residual of the round weighted by the accepted iterate's balance scale.
    "tree_line_ptr": _scratch("tree_line", ("num_trees",), "int32"),
    # Two parities of active and solving tree counts, error, plus coupled-component counters.
    "grid_control_ptr": _scratch("grid_control", (7,), "int32"),
}
ADJOINT_WORKSPACE = {
    "pivot_ptr": WORKSPACE["pivot_ptr"],
    "pivot_rhs_ptr": WORKSPACE["pivot_rhs_ptr"],
    "adjoint_continuity_ptr": _scratch("adjoint_continuity"),
    "adjoint_flux_ptr": _scratch("adjoint_flux"),
}

_GEOMETRY = (
    "river_length_ptr",
    "river_width_ptr",
    "river_height_ptr",
    "catchment_area_ptr",
    "flood_depth_table_ptr",
    "flood_bed_length_ptr",
    "catchment_elevation_ptr",
    "downstream_distance_ptr",
)
_FOREST = (
    "downstream_idx_ptr",
    "downstream_edge_ptr",
    "upstream_offset_ptr",
    "upstream_index_ptr",
    "path_unit_ptr",
    "path_offset_ptr",
    "round_offset_ptr",
)

_SOLVE_LAYOUT = (
    "tree_index_ptr",
    "tree_size_ptr",
    "tree_root_ptr",
    "grid_unit_ptr",
    "grid_tree_ptr",
    "grid_path_ptr",
    "grid_round_offset_ptr",
    "block_tree_ptr",
    "tree_unit_ptr",
    "level_offset_ptr",
    "tree_level_ptr",
)

# One implicit step of the model state; with options.differentiable it also
# saves the converged surface, from which the adjoint rebuilds the converged
# Jacobian (the reverse step has the step's time step).
INERTIA_STEP = KernelSpec(
    name="solve_inertia_step",
    size="num_catchments",
    parameters={
        "storage_ptr": "read_write",
        "outflow_ptr": "read_write",
        "runoff_ptr": "read",
        "river_manning_ptr": "read",
        "flood_manning_ptr": "read",
        **dict.fromkeys(_GEOMETRY + _FOREST, "read"),
        **dict.fromkeys(_SOLVE_LAYOUT, "read"),
        "inflow_ptr": "write",
        "water_surface_ptr": "write",
        "newton_record_ptr": "write",
        "newton_status_ptr": "read_write",
        "converged_surface_ptr": "write",
        **dict.fromkeys(WORKSPACE, "read_write"),
        "time_step_ptr": step_field("step_seconds"),
        "slope_regularization": "float32",
        "newton_tolerance": "float32",
        "newton_max_iterations": "int32",
        "num_catchments": "int32",
        "num_trees": "int32",
        "num_flood_levels": "int32",
        "num_path_rounds": "int32",
        "num_grid_units": "int32",
        "num_grid_trees": "int32",
        "num_block_trees": "int32",
    },
    optional={"converged_surface_ptr": None},
    workspace=WORKSPACE,
    block_sizes={"cuda": 512, "triton": 512, "metal": 256},
)

_EVAL_WORKSPACE = {
    f"{name}_ptr": WORKSPACE[f"{name}_ptr"]
    for name in (
        "newton_offset",
        "trial_offset",
        "newton_reference",
        "trial_reference",
        "trial_storage",
        "trial_area",
        "channel_conveyance",
        "flood_conveyance",
        "channel_conveyance_slope",
        "flood_conveyance_slope",
        "trial_outflow",
        "outflow_own_slope",
        "outflow_down_slope",
        "residual",
        "balance_weight",
        "accepted_weight",
    )
}
# The converged snapshot again, from the taped surface and the step's inputs.
INERTIA_REBUILD = KernelSpec(
    name="solve_inertia_rebuild",
    size="num_catchments",
    parameters={
        **dict.fromkeys(
            (
                "storage_ptr",
                "outflow_ptr",
                "runoff_ptr",
                "river_manning_ptr",
                "flood_manning_ptr",
                "converged_surface_ptr",
            ),
            "read",
        ),
        "time_step_ptr": step_field("step_seconds"),
        **dict.fromkeys(_GEOMETRY + _FOREST, "read"),
        "jacobian_snapshot_ptr": "write",
        **dict.fromkeys(_EVAL_WORKSPACE, "read_write"),
        "slope_regularization": "float32",
        "num_catchments": "int32",
        "num_flood_levels": "int32",
    },
    workspace=_EVAL_WORKSPACE,
    block_sizes={"cuda": 256, "triton": 256, "metal": 256},
)

# The transposed converged Jacobian of one step: adjoints of the step's
# inputs from adjoints of its results.
INERTIA_ADJOINT = KernelSpec(
    name="solve_inertia_adjoint",
    size="num_catchments",
    parameters={
        "jacobian_snapshot_ptr": "read",
        # Result adjoints in, start adjoints out (in place).
        "grad_outflow_ptr": "read_write",
        "grad_storage_ptr": "read_write",
        "grad_water_surface_ptr": "read",
        **dict.fromkeys(_FOREST, "read"),
        "grad_runoff_ptr": "write",
        # Parameter adjoints: added to (summed over reverse steps).
        "grad_river_width_ptr": "read_write",
        "grad_river_height_ptr": "read_write",
        "grad_river_manning_ptr": "read_write",
        "grad_flood_manning_ptr": "read_write",
        **dict.fromkeys(ADJOINT_WORKSPACE, "read_write"),
        "num_catchments": "int32",
        "num_path_rounds": "int32",
        "max_round_paths": "int32",
    },
    workspace=ADJOINT_WORKSPACE,
    block_sizes={"cuda": 256, "triton": 256, "metal": 256},
)

# The graph reuses the native geometry, main-channel derivatives and workspaces.
# Each Newton linearization assembles and factors the equilibrated Jacobian of
# the coupled pathways once (static pattern, hydronet.modules._bifurcation); it
# preconditions BiCGStab on each active coupled component in forward mode (full
# graph in adjoint mode).
_GRAPH_VECTORS = (
    "rhs",
    "temporary",
    "bound",
    "scaling",
    "balance_scale",
    "solution_low",
    "lu_forward",
    "lu_backward",
    "r",
    "rhat",
    "p",
    "v",
    "s",
    "t",
    "delta",
    "precondition_rhs",
)
GRAPH_WORKSPACE = {
    **WORKSPACE,
    **{
        f"graph_{name}_ptr": KernelWorkspace(
            key=f"hydronet.graph.{name}", dtype="float32", shape=("num_catchments",)
        )
        for name in _GRAPH_VECTORS
    },
    "graph_reduction_ptr": KernelWorkspace(
        key="hydronet.graph.reduction", dtype="float32", shape=(4, "num_catchments")
    ),
    "graph_factor_ptr": KernelWorkspace(
        key="hydronet.graph.factor", dtype="float32", shape=("max_factor_entries",)
    ),
    "graph_path_values_ptr": KernelWorkspace(
        key="hydronet.graph.path_values",
        dtype="float32",
        shape=("num_bifurcation_values", 7),
    ),
    "graph_path_sum_ptr": KernelWorkspace(
        key="hydronet.graph.path_sum",
        dtype="float32",
        shape=("num_bifurcation_paths", 3),
    ),
}
_GRAPH_INPUTS = (
    "bifurcation_catchment_idx_ptr",
    "bifurcation_downstream_idx_ptr",
    "incident_offset_ptr",
    "incident_path_ptr",
    "factor_layout_ptr",
    "bifurcation_width_ptr",
    "bifurcation_length_ptr",
    "bifurcation_elevation_ptr",
    "bifurcation_manning_ptr",
    "pure_block_tree_ptr",
    "graph_unit_ptr",
    "unit_comp_ptr",
    "path_comp_ptr",
    "coupled_trees_ptr",
    "coupled_tree_comp_ptr",
    "comp_rep_tree_ptr",
    "core_factor_layout_ptr",
    "core_entry_comp_ptr",
)
_GRAPH_PARAMETERS = {
    **INERTIA_STEP.parameters,
    **dict.fromkeys(_GRAPH_INPUTS, "read"),
    "bifurcation_outflow_ptr": "read_write",
    "linear_record_ptr": "read_write",
    **dict.fromkeys(GRAPH_WORKSPACE, "read_write"),
    "num_pure_trees": "int32",
    "num_graph_comps": "int32",
    "num_graph_units": "int32",
    "num_coupled_trees": "int32",
    "num_core_factor_entries": "int32",
    "num_core_factor_rounds": "int32",
    "num_bifurcation_paths": "int32",
    "num_bifurcation_levels": "int32",
    "num_bifurcation_values": "int32",
    "num_factor_entries": "int32",
    "num_factor_rounds": "int32",
}
INERTIA_GRAPH_STEP = KernelSpec(
    name="solve_inertia_graph_step",
    size="num_catchments",
    parameters=_GRAPH_PARAMETERS,
    optional={"converged_surface_ptr": None},
    workspace=GRAPH_WORKSPACE,
    block_sizes={"cuda": 256, "triton": 256, "metal": 256},
)
INERTIA_GRAPH_ADJOINT = KernelSpec(
    name="solve_inertia_graph_adjoint",
    size="num_catchments",
    parameters={
        **{
            k: v
            for k, v in _GRAPH_PARAMETERS.items()
            if k not in ("time_step_ptr", "converged_surface_ptr")
        },
        "jacobian_snapshot_ptr": "read",
        "bifurcation_jacobian_ptr": "read",
        **dict.fromkeys(
            (
                "storage_ptr",
                "outflow_ptr",
                "bifurcation_outflow_ptr",
                "inflow_ptr",
                "water_surface_ptr",
                "newton_record_ptr",
            ),
            "read",
        ),
        "grad_water_surface_ptr": "read",
        # Result adjoints in, start adjoints out (in place).
        **dict.fromkeys(
            ("grad_outflow_ptr", "grad_storage_ptr", "grad_bifurcation_outflow_ptr"),
            "read_write",
        ),
        "grad_runoff_ptr": "write",
        **dict.fromkeys(
            (
                "grad_river_width_ptr",
                "grad_river_height_ptr",
                "grad_river_manning_ptr",
                "grad_flood_manning_ptr",
                "grad_bifurcation_manning_ptr",
            ),
            "read_write",
        ),
    },
    workspace=GRAPH_WORKSPACE,
    block_sizes={"cuda": 256, "triton": 256, "metal": 256},
)

# The converged graph snapshot again, with the pathway Jacobians.
INERTIA_GRAPH_REBUILD = KernelSpec(
    name="solve_inertia_graph_rebuild",
    size="num_catchments",
    parameters={
        **_GRAPH_PARAMETERS,
        **dict.fromkeys(
            (
                "storage_ptr",
                "outflow_ptr",
                "bifurcation_outflow_ptr",
                "converged_surface_ptr",
            ),
            "read",
        ),
        "jacobian_snapshot_ptr": "write",
        "bifurcation_jacobian_ptr": "write",
    },
    workspace=GRAPH_WORKSPACE,
    block_sizes={"cuda": 256, "triton": 256, "metal": 256},
)
