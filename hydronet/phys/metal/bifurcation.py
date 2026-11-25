# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal (MSL) backend for HydroNet coupled bifurcation graph kernels."""

from __future__ import annotations

import math
import struct
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydroforge.kernels.metal import MetalProgram

from hydronet.config import (
    ADJOINT_NOT_CONVERGED,
    FACTOR_LAYOUT,
    GRAPH_CLOSED_ELEVATION,
    GRAPH_COUPLING_FULL_PHASE,
    GRAPH_DRY_DEPTH,
    GRAPH_LINEAR_ITERATIONS,
    GRAPH_LINEAR_RESTARTS,
    GRAPH_LINEAR_TOLERANCE,
    GRAPH_SLOPE_LIMIT,
    NEWTON_ROUND_LIMIT,
)
from hydronet.phys.metal.implicit_inertia import (
    UNIT_THREADS,
    build_commit_launch,
    build_tree_launch,
    forest_constants_prelude,
    forest_metal_library,
)
from hydronet.phys.metal.runtime import (
    MetalLibrary,
    MetalMultiStepKernel,
    SubKernelSpec,
    buf,
    close_launches,
    read_metal_helpers,
    scl,
)

METAL_DIR = Path(__file__).parent
HEAVY_NUM_THREADS = 256
HEAVY_SOLVE_THREADS = 32
LIGHT_THREADS = 64


def _round_up(n: int, group: int) -> int:
    return ((n + group - 1) // group) * group


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<i", bits))[0]


def _dispatch_once(prog: MetalProgram, values: dict[str, Any], group_size: int) -> None:
    launch = prog.launch(values, group_size)
    try:
        launch()
    finally:
        close = getattr(launch, "close", None)
        if callable(close):
            close()


def _bind(
    owned: list[Any],
    program: MetalProgram,
    values: dict[str, Any],
    group_size: int,
):
    launch = program.launch(values, group_size)
    owned.append(launch)
    return launch


def _graph_constants_prelude() -> str:
    factor_defs = "\n".join(
        f"constant constexpr int FACTOR_{name.upper()} = {i};"
        for i, name in enumerate(FACTOR_LAYOUT)
    )
    return (
        forest_constants_prelude()
        + f"constant constexpr int GRAPH_LINEAR_RESTARTS = {int(GRAPH_LINEAR_RESTARTS)};\n"
        + f"constant constexpr int GRAPH_LINEAR_ITERATIONS = {int(GRAPH_LINEAR_ITERATIONS)};\n"
        + f"constant constexpr int GRAPH_COUPLING_FULL_PHASE = {int(GRAPH_COUPLING_FULL_PHASE)};\n"
        + f"constant constexpr float GRAPH_LINEAR_TOLERANCE = {float(GRAPH_LINEAR_TOLERANCE)!r}f;\n"
        + f"constant constexpr float GRAPH_SLOPE_LIMIT = {float(GRAPH_SLOPE_LIMIT)!r}f;\n"
        + f"constant constexpr float GRAPH_CLOSED_ELEVATION = {float(GRAPH_CLOSED_ELEVATION)!r}f;\n"
        + f"constant constexpr float GRAPH_DRY_DEPTH = {float(GRAPH_DRY_DEPTH)!r}f;\n"
        + factor_defs
        + "\n"
        + read_metal_helpers(METAL_DIR / "implicit_inertia.metal")
        + "\n"
    )


_SUBKERNELS: tuple[SubKernelSpec, ...] = (
    SubKernelSpec(
        name="graph_coupled_start_units",
        extent=("num_graph_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("storage_ptr", "read", "float"),
            buf("river_manning_ptr", "read", "float"),
            buf("flood_manning_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("newton_reference_ptr", "write", "float"),
            buf("newton_offset_ptr", "write", "float"),
            buf("trial_reference_ptr", "write", "float"),
            buf("trial_offset_ptr", "write", "float"),
            buf("graph_bound_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("num_graph_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_start_comps",
        extent=("num_graph_comps",),
        arguments=(
            buf("tree_control_ptr", "write", "int"),
            buf("tree_scalars_ptr", "write", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            buf("time_step_ptr", "read", "float"),
            scl("num_graph_comps", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_stage",
        extent=("num_graph_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("flood_bed_length_ptr", "read", "float"),
            buf("trial_storage_ptr", "write", "float"),
            buf("trial_area_ptr", "write", "float"),
            buf("channel_conveyance_ptr", "write", "float"),
            buf("flood_conveyance_ptr", "write", "float"),
            buf("channel_conveyance_slope_ptr", "write", "float"),
            buf("flood_conveyance_slope_ptr", "write", "float"),
            scl("num_graph_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_main_flux",
        extent=("num_graph_units",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("outflow_ptr", "read", "float"),
            buf("river_manning_ptr", "read", "float"),
            buf("flood_manning_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("flood_bed_length_ptr", "read", "float"),
            buf("catchment_elevation_ptr", "read", "float"),
            buf("downstream_distance_ptr", "read", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("channel_conveyance_ptr", "read", "float"),
            buf("flood_conveyance_ptr", "read", "float"),
            buf("channel_conveyance_slope_ptr", "read", "float"),
            buf("flood_conveyance_slope_ptr", "read", "float"),
            buf("trial_outflow_ptr", "write", "float"),
            buf("outflow_own_slope_ptr", "write", "float"),
            buf("outflow_down_slope_ptr", "write", "float"),
            scl("slope_regularization", "float32"),
            scl("num_graph_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_bif_flux",
        extent=("num_bifurcation_paths",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("path_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("catchment_elevation_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("bifurcation_width_ptr", "read", "float"),
            buf("bifurcation_length_ptr", "read", "float"),
            buf("bifurcation_elevation_ptr", "read", "float"),
            buf("bifurcation_manning_ptr", "read", "float"),
            buf("bifurcation_outflow_ptr", "read", "float"),
            buf("graph_path_values_ptr", "write", "float"),
            buf("graph_path_sum_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("num_bifurcation_paths", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_residual",
        extent=("total_threads",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("storage_ptr", "read", "float"),
            buf("runoff_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("graph_path_sum_ptr", "read", "float"),
            buf("residual_ptr", "read_write", "float"),
            buf("balance_weight_ptr", "write", "float"),
            buf("graph_rhs_ptr", "write", "float"),
            buf("graph_balance_scale_ptr", "read", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            scl("num_graph_units", "int32"),
            scl("single_comp", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_control",
        extent=("num_graph_comps",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("tree_control_ptr", "read_write", "int"),
            buf("tree_scalars_ptr", "read_write", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            buf("comp_rep_tree_ptr", "read", "int"),
            buf("grid_control_ptr", "atomic_add", "atomic_int"),
            buf("newton_record_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("newton_tolerance", "float32"),
            scl("newton_max_iterations", "int32"),
            scl("num_graph_comps", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_apply",
        extent=("num_graph_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("storage_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("newton_reference_ptr", "read_write", "float"),
            buf("newton_offset_ptr", "read_write", "float"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("balance_weight_ptr", "read", "float"),
            buf("accepted_weight_ptr", "write", "float"),
            buf("phase_trace_ptr", "read_write", "float"),
            scl("num_graph_units", "int32"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_prep_rhs",
        extent=("num_graph_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("residual_ptr", "read", "float"),
            buf("graph_rhs_ptr", "read_write", "float"),
            buf("graph_balance_scale_ptr", "write", "float"),
            buf("graph_bound_ptr", "write", "float"),
            scl("num_graph_units", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_coupled_trial",
        extent=("num_graph_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("trial_reference_ptr", "write", "float"),
            buf("trial_offset_ptr", "write", "float"),
            buf("newton_step_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("pivot_ptr", "read_write", "float"),
            buf("pivot_rhs_ptr", "read_write", "float"),
            buf("coupling_ptr", "read_write", "float"),
            scl("num_graph_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_commit_paths",
        extent=("num_bifurcation_values",),
        arguments=(
            buf("graph_path_values_ptr", "read", "float"),
            buf("bifurcation_outflow_ptr", "write", "float"),
            scl("num_bifurcation_values", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_commit_inflow",
        extent=("num_catchments",),
        arguments=(
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("graph_path_values_ptr", "read", "float"),
            buf("inflow_ptr", "read_write", "float"),
            scl("num_catchments", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_matrix_and_init",
        extent=("total_threads",),
        arguments=(
            buf("dt_ptr", "read", "float"),
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("graph_path_sum_ptr", "read", "float"),
            buf("graph_bound_ptr", "read", "float"),
            buf("graph_scaling_ptr", "write", "float"),
            buf("graph_rhs_ptr", "read_write", "float"),
            buf("newton_step_ptr", "write", "float"),
            buf("graph_solution_low_ptr", "write", "float"),
            buf("norm_buf_ptr", "atomic_max", "atomic_uint"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("use_damping", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_assemble",
        extent=("num_entries",),
        arguments=(
            buf("dt_ptr", "read", "float"),
            buf("plan_ptr", "read", "int"),
            buf("entry_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("graph_bound_ptr", "read", "float"),
            buf("graph_scaling_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("graph_path_sum_ptr", "read", "float"),
            buf("graph_factor_ptr", "write", "float"),
            scl("num_entries", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_numeric_heavy",
        extent=("total_threads",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read_write", "float"),
            scl("r_begin", "int32"),
            scl("num_heavy", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_numeric_light",
        extent=("count",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read_write", "float"),
            scl("split", "int32"),
            scl("count", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_solve_fwd_heavy",
        extent=("total_threads",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read", "float"),
            buf("c_vec_ptr", "read", "float"),
            buf("lu_forward_ptr", "read_write", "float"),
            scl("r_begin", "int32"),
            scl("num_heavy", "int32"),
            scl("trans", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_solve_fwd_light",
        extent=("count",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read", "float"),
            buf("c_vec_ptr", "read", "float"),
            buf("lu_forward_ptr", "read_write", "float"),
            scl("split", "int32"),
            scl("count", "int32"),
            scl("trans", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_solve_bwd_heavy",
        extent=("total_threads",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read", "float"),
            buf("graph_scaling_ptr", "read", "float"),
            buf("lu_forward_ptr", "read", "float"),
            buf("lu_backward_ptr", "read_write", "float"),
            buf("out_vec_ptr", "write", "float"),
            scl("r_begin", "int32"),
            scl("num_heavy", "int32"),
            scl("trans", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_factor_solve_bwd_light",
        extent=("count",),
        arguments=(
            buf("plan_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_factor_ptr", "read", "float"),
            buf("graph_scaling_ptr", "read", "float"),
            buf("lu_forward_ptr", "read", "float"),
            buf("lu_backward_ptr", "read_write", "float"),
            buf("out_vec_ptr", "write", "float"),
            scl("split", "int32"),
            scl("count", "int32"),
            scl("trans", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_matvec",
        extent=("num_units",),
        arguments=(
            buf("dt_ptr", "read", "float"),
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("x_ptr", "read", "float"),
            buf("graph_solution_low_ptr", "read", "float"),
            buf("out_ptr", "write", "float"),
            buf("graph_bound_ptr", "read", "float"),
            buf("graph_scaling_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("graph_path_sum_ptr", "read", "float"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("use_damping", "int32"),
            scl("trans", "int32"),
            scl("compensated", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_vec_copy_dh",
        extent=("num_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("graph_delta_ptr", "read", "float"),
            buf("newton_step_ptr", "write", "float"),
            buf("graph_solution_low_ptr", "write", "float"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_vec_norm",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("x_ptr", "read", "float"),
            buf("norm_buf_ptr", "atomic_max", "atomic_uint"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_vec_dot",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("a_ptr", "read", "float"),
            buf("b_ptr", "read", "float"),
            buf("red_ptr", "write", "float"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_vec_dot_pair",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("t_ptr", "read", "float"),
            buf("s_ptr", "read", "float"),
            buf("red_ptr", "write", "float"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_init",
        extent=("num_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tmp_ptr", "read", "float"),
            buf("r_ptr", "write", "float"),
            buf("rhat_ptr", "write", "float"),
            buf("p_ptr", "write", "float"),
            buf("v_ptr", "write", "float"),
            buf("delta_ptr", "write", "float"),
            scl("initial", "float32"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_upd_p",
        extent=("num_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("r_ptr", "read", "float"),
            buf("p_ptr", "read_write", "float"),
            buf("v_ptr", "read", "float"),
            scl("beta", "float32"),
            scl("omega", "float32"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_upd_s",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("r_ptr", "read", "float"),
            buf("v_ptr", "read", "float"),
            buf("tmp_ptr", "read", "float"),
            buf("s_ptr", "write", "float"),
            buf("delta_ptr", "read_write", "float"),
            buf("norm_buf_ptr", "atomic_max", "atomic_uint"),
            scl("alpha", "float32"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_upd_r",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("s_ptr", "read", "float"),
            buf("t_ptr", "read", "float"),
            buf("tmp_ptr", "read", "float"),
            buf("r_ptr", "write", "float"),
            buf("delta_ptr", "read_write", "float"),
            buf("norm_buf_ptr", "atomic_max", "atomic_uint"),
            scl("omega", "float32"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_accum",
        extent=("num_units",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("newton_step_ptr", "read_write", "float"),
            buf("graph_solution_low_ptr", "read_write", "float"),
            buf("delta_ptr", "read", "float"),
            scl("initial", "float32"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_bicg_res_diff",
        extent=("total_threads",),
        arguments=(
            buf("graph_unit_ptr", "read", "int"),
            buf("unit_comp_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("rhs_ptr", "read", "float"),
            buf("tmp_ptr", "read_write", "float"),
            buf("norm_buf_ptr", "atomic_max", "atomic_uint"),
            scl("num_units", "int32"),
            scl("full_domain", "int32"),
            scl("all_active", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_scale_div",
        extent=("num_catchments",),
        arguments=(
            buf("in_ptr", "read", "float"),
            buf("scaling_ptr", "read", "float"),
            buf("out_ptr", "write", "float"),
            scl("num_catchments", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_report_error",
        extent=("one",),
        arguments=(
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("code", "int32"),
            scl("info", "int32"),
            scl("one", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_rebuild_stage",
        extent=("num_catchments",),
        arguments=(
            buf("converged_surface_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("flood_bed_length_ptr", "read", "float"),
            buf("newton_reference_ptr", "write", "float"),
            buf("newton_offset_ptr", "write", "float"),
            buf("trial_reference_ptr", "write", "float"),
            buf("trial_offset_ptr", "write", "float"),
            buf("trial_storage_ptr", "write", "float"),
            buf("trial_area_ptr", "write", "float"),
            buf("channel_conveyance_ptr", "write", "float"),
            buf("flood_conveyance_ptr", "write", "float"),
            buf("channel_conveyance_slope_ptr", "write", "float"),
            buf("flood_conveyance_slope_ptr", "write", "float"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_rebuild_main_flux",
        extent=("num_catchments",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("outflow_ptr", "read", "float"),
            buf("river_manning_ptr", "read", "float"),
            buf("flood_manning_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("flood_bed_length_ptr", "read", "float"),
            buf("catchment_elevation_ptr", "read", "float"),
            buf("downstream_distance_ptr", "read", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("channel_conveyance_ptr", "read", "float"),
            buf("flood_conveyance_ptr", "read", "float"),
            buf("channel_conveyance_slope_ptr", "read", "float"),
            buf("flood_conveyance_slope_ptr", "read", "float"),
            buf("trial_outflow_ptr", "write", "float"),
            buf("outflow_own_slope_ptr", "write", "float"),
            buf("outflow_down_slope_ptr", "write", "float"),
            scl("slope_regularization", "float32"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_rebuild_bif_flux",
        extent=("num_bifurcation_paths",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("catchment_elevation_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("trial_reference_ptr", "read", "float"),
            buf("trial_offset_ptr", "read", "float"),
            buf("bifurcation_width_ptr", "read", "float"),
            buf("bifurcation_length_ptr", "read", "float"),
            buf("bifurcation_elevation_ptr", "read", "float"),
            buf("bifurcation_manning_ptr", "read", "float"),
            buf("bifurcation_outflow_ptr", "read", "float"),
            buf("graph_path_values_ptr", "write", "float"),
            buf("graph_path_sum_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("num_bifurcation_paths", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_rebuild_residual",
        extent=("num_catchments",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("storage_ptr", "read", "float"),
            buf("runoff_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("graph_path_sum_ptr", "read", "float"),
            buf("residual_ptr", "write", "float"),
            buf("balance_weight_ptr", "write", "float"),
            buf("graph_rhs_ptr", "write", "float"),
            scl("num_catchments", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_rebuild_snapshot",
        extent=("total_threads",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("outflow_ptr", "read", "float"),
            buf("river_manning_ptr", "read", "float"),
            buf("flood_manning_ptr", "read", "float"),
            buf("river_length_ptr", "read", "float"),
            buf("river_width_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("catchment_area_ptr", "read", "float"),
            buf("flood_depth_table_ptr", "read", "float"),
            buf("flood_bed_length_ptr", "read", "float"),
            buf("catchment_elevation_ptr", "read", "float"),
            buf("downstream_distance_ptr", "read", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("residual_ptr", "read", "float"),
            buf("jacobian_snapshot_ptr", "write", "float"),
            buf("graph_path_values_ptr", "read", "float"),
            buf("bifurcation_jacobian_ptr", "write", "float"),
            scl("slope_regularization", "float32"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
            scl("num_bifurcation_values", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_adjoint_prep_cells",
        extent=("num_catchments",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("bifurcation_jacobian_ptr", "read", "float"),
            buf("grad_outflow_ptr", "read", "float"),
            buf("grad_storage_ptr", "read", "float"),
            buf("grad_water_surface_ptr", "read", "float"),
            buf("grad_bifurcation_outflow_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("outflow_own_slope_ptr", "write", "float"),
            buf("outflow_down_slope_ptr", "write", "float"),
            buf("trial_area_ptr", "write", "float"),
            buf("graph_bound_ptr", "write", "float"),
            buf("graph_rhs_ptr", "write", "float"),
            scl("num_catchments", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_adjoint_prep_paths",
        extent=("num_bifurcation_paths",),
        arguments=(
            buf("bifurcation_jacobian_ptr", "read", "float"),
            buf("graph_path_sum_ptr", "write", "float"),
            scl("num_bifurcation_paths", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_adjoint_step1",
        extent=("num_catchments",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("newton_step_ptr", "read", "float"),
            buf("graph_solution_low_ptr", "read", "float"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("grad_outflow_ptr", "read_write", "float"),
            buf("grad_storage_ptr", "read", "float"),
            buf("graph_temporary_ptr", "write", "float"),
            buf("grad_runoff_ptr", "write", "float"),
            buf("grad_river_width_ptr", "read_write", "float"),
            buf("grad_river_manning_ptr", "read_write", "float"),
            buf("grad_flood_manning_ptr", "read_write", "float"),
            scl("num_catchments", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_adjoint_step2",
        extent=("num_catchments",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("bifurcation_jacobian_ptr", "read", "float"),
            buf("newton_step_ptr", "read", "float"),
            buf("graph_solution_low_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("incident_offset_ptr", "read", "int"),
            buf("incident_path_ptr", "read", "int"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("graph_temporary_ptr", "read", "float"),
            buf("grad_storage_ptr", "read_write", "float"),
            buf("grad_bifurcation_outflow_ptr", "read", "float"),
            buf("grad_river_height_ptr", "read_write", "float"),
            scl("num_catchments", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="graph_adjoint_step3",
        extent=("num_bifurcation_paths",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("bifurcation_jacobian_ptr", "read", "float"),
            buf("newton_step_ptr", "read", "float"),
            buf("graph_solution_low_ptr", "read", "float"),
            buf("bifurcation_catchment_idx_ptr", "read", "int"),
            buf("bifurcation_downstream_idx_ptr", "read", "int"),
            buf("grad_bifurcation_outflow_ptr", "read_write", "float"),
            buf("grad_bifurcation_manning_ptr", "read_write", "float"),
            scl("num_bifurcation_paths", "int32"),
            scl("num_bifurcation_levels", "int32"),
        ),
    ),
)


@cache
def graph_metal_library() -> MetalLibrary:
    """Compile-time validated MSL library for all bifurcation graph sub-kernels."""
    return MetalLibrary(
        METAL_DIR / "bifurcation.metal",
        _SUBKERNELS,
        prelude=_graph_constants_prelude(),
    )


def _unpack_round_info(
    plan_tensor: torch.Tensor, rounds: int
) -> tuple[tuple[int, int, int, int, int], ...]:
    n_names = len(FACTOR_LAYOUT)
    offsets = plan_tensor[:n_names].cpu().tolist()
    total = int(plan_tensor.numel())
    ro_idx = FACTOR_LAYOUT.index("round_offset")
    rh_idx = FACTOR_LAYOUT.index("round_heavy")
    ro_end = offsets[ro_idx + 1] if ro_idx + 1 < n_names else total
    rh_end = offsets[rh_idx + 1] if rh_idx + 1 < n_names else total
    ro = plan_tensor[offsets[ro_idx] : ro_end].cpu().tolist()
    rh = plan_tensor[offsets[rh_idx] : rh_end].cpu().tolist()
    info = []
    for r in range(rounds):
        rb, re_idx, hv = int(ro[r]), int(ro[r + 1]), int(rh[r])
        split = rb + hv
        light = re_idx - split
        info.append((rb, split, re_idx, hv, light))
    return tuple(info)


def _make_linear_solver(
    owned: list[Any],
    programs: dict[str, MetalProgram],
    arguments: dict[str, Any],
    *,
    trans: bool,
    full_domain: bool,
    all_active: bool,
    use_damping: bool,
    dt_ptr: torch.Tensor,
    plan_ptr: torch.Tensor,
    entry_comp_ptr: torch.Tensor,
    num_units: int,
    entries: int,
    rounds: int,
):
    a = arguments
    device = a["storage_ptr"].device
    blk = UNIT_THREADS
    total_threads = _round_up(max(num_units, 1), blk)
    n_blocks = _cdiv(max(num_units, 1), blk)
    norm_buf = torch.zeros(1, dtype=torch.int32, device=device)

    full_i = 1 if full_domain else 0
    all_i = 1 if all_active else 0
    damp_i = 1 if use_damping else 0
    trans_i = 1 if trans else 0

    gunit = a["graph_unit_ptr"]
    u_comp = a["unit_comp_ptr"]
    ti = a["tree_control_ptr"]
    tf = a["tree_scalars_ptr"]
    r_vec = a["graph_r_ptr"]
    rhat = a["graph_rhat_ptr"]
    p_vec = a["graph_p_ptr"]
    v_vec = a["graph_v_ptr"]
    s_vec = a["graph_s_ptr"]
    t_vec = a["graph_t_ptr"]
    delta = a["graph_delta_ptr"]
    tmp = a["graph_temporary_ptr"]
    dh = a["newton_step_ptr"]
    sol_low = a["graph_solution_low_ptr"]
    rhs = a["graph_rhs_ptr"]
    pre = a["graph_precondition_rhs_ptr"]
    red = a["graph_reduction_ptr"].view(-1)

    round_info = _unpack_round_info(plan_ptr, rounds) if rounds > 0 else ()

    l_mat_init = _bind(
        owned,
        programs["graph_matrix_and_init"],
        {
            "dt_ptr": dt_ptr,
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "tree_scalars_ptr": tf,
            "trial_area_ptr": a["trial_area_ptr"],
            "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
            "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "incident_offset_ptr": a["incident_offset_ptr"],
            "incident_path_ptr": a["incident_path_ptr"],
            "graph_path_sum_ptr": a["graph_path_sum_ptr"],
            "graph_bound_ptr": a["graph_bound_ptr"],
            "graph_scaling_ptr": a["graph_scaling_ptr"],
            "graph_rhs_ptr": rhs,
            "newton_step_ptr": dh,
            "graph_solution_low_ptr": sol_low,
            "norm_buf_ptr": norm_buf,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "use_damping": damp_i,
            "total_threads": total_threads,
        },
        blk,
    )

    l_assemble = (
        _bind(
            owned,
            programs["graph_factor_assemble"],
            {
                "dt_ptr": dt_ptr,
                "plan_ptr": plan_ptr,
                "entry_comp_ptr": entry_comp_ptr,
                "tree_control_ptr": ti,
                "downstream_edge_ptr": a["downstream_edge_ptr"],
                "downstream_idx_ptr": a["downstream_idx_ptr"],
                "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
                "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
                "graph_bound_ptr": a["graph_bound_ptr"],
                "graph_scaling_ptr": a["graph_scaling_ptr"],
                "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
                "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
                "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                "graph_factor_ptr": a["graph_factor_ptr"],
                "num_entries": entries,
                "all_active": all_i,
            },
            blk,
        )
        if entries > 0
        else None
    )

    num_rounds_launches = []
    for rb, split, _, hv, light in round_info:
        lh = (
            _bind(
                owned,
                programs["graph_factor_numeric_heavy"],
                {
                    "plan_ptr": plan_ptr,
                    "unit_comp_ptr": u_comp,
                    "tree_control_ptr": ti,
                    "graph_factor_ptr": a["graph_factor_ptr"],
                    "r_begin": rb,
                    "num_heavy": hv,
                    "all_active": all_i,
                    "total_threads": hv * HEAVY_NUM_THREADS,
                },
                HEAVY_NUM_THREADS,
            )
            if hv > 0
            else None
        )
        ll = (
            _bind(
                owned,
                programs["graph_factor_numeric_light"],
                {
                    "plan_ptr": plan_ptr,
                    "unit_comp_ptr": u_comp,
                    "tree_control_ptr": ti,
                    "graph_factor_ptr": a["graph_factor_ptr"],
                    "split": split,
                    "count": light,
                    "all_active": all_i,
                },
                LIGHT_THREADS,
            )
            if light > 0
            else None
        )
        num_rounds_launches.append((lh, ll))

    def _build_solve_pair(c_in: torch.Tensor, out_vec: torch.Tensor):
        fwd_list = []
        bwd_list = []
        for rb, split, _, hv, light in round_info:
            fh = (
                _bind(
                    owned,
                    programs["graph_factor_solve_fwd_heavy"],
                    {
                        "plan_ptr": plan_ptr,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "graph_factor_ptr": a["graph_factor_ptr"],
                        "c_vec_ptr": c_in,
                        "lu_forward_ptr": a["graph_lu_forward_ptr"],
                        "r_begin": rb,
                        "num_heavy": hv,
                        "trans": trans_i,
                        "all_active": all_i,
                        "total_threads": hv * HEAVY_SOLVE_THREADS,
                    },
                    HEAVY_SOLVE_THREADS,
                )
                if hv > 0
                else None
            )
            fl = (
                _bind(
                    owned,
                    programs["graph_factor_solve_fwd_light"],
                    {
                        "plan_ptr": plan_ptr,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "graph_factor_ptr": a["graph_factor_ptr"],
                        "c_vec_ptr": c_in,
                        "lu_forward_ptr": a["graph_lu_forward_ptr"],
                        "split": split,
                        "count": light,
                        "trans": trans_i,
                        "all_active": all_i,
                    },
                    LIGHT_THREADS,
                )
                if light > 0
                else None
            )
            bh = (
                _bind(
                    owned,
                    programs["graph_factor_solve_bwd_heavy"],
                    {
                        "plan_ptr": plan_ptr,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "graph_factor_ptr": a["graph_factor_ptr"],
                        "graph_scaling_ptr": a["graph_scaling_ptr"],
                        "lu_forward_ptr": a["graph_lu_forward_ptr"],
                        "lu_backward_ptr": a["graph_lu_backward_ptr"],
                        "out_vec_ptr": out_vec,
                        "r_begin": rb,
                        "num_heavy": hv,
                        "trans": trans_i,
                        "all_active": all_i,
                        "total_threads": hv * HEAVY_SOLVE_THREADS,
                    },
                    HEAVY_SOLVE_THREADS,
                )
                if hv > 0
                else None
            )
            bl = (
                _bind(
                    owned,
                    programs["graph_factor_solve_bwd_light"],
                    {
                        "plan_ptr": plan_ptr,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "graph_factor_ptr": a["graph_factor_ptr"],
                        "graph_scaling_ptr": a["graph_scaling_ptr"],
                        "lu_forward_ptr": a["graph_lu_forward_ptr"],
                        "lu_backward_ptr": a["graph_lu_backward_ptr"],
                        "out_vec_ptr": out_vec,
                        "split": split,
                        "count": light,
                        "trans": trans_i,
                        "all_active": all_i,
                    },
                    LIGHT_THREADS,
                )
                if light > 0
                else None
            )
            fwd_list.append((fh, fl))
            bwd_list.append((bh, bl))

        def run_lu() -> None:
            for r in range(rounds - 1, -1, -1):
                fh, fl = fwd_list[r]
                if fh is not None:
                    fh()
                if fl is not None:
                    fl()
            for r in range(rounds):
                bh, bl = bwd_list[r]
                if bh is not None:
                    bh()
                if bl is not None:
                    bl()

        return run_lu

    if trans:
        l_scale_rhs = _bind(
            owned,
            programs["graph_scale_div"],
            {
                "in_ptr": rhs,
                "scaling_ptr": a["graph_scaling_ptr"],
                "out_ptr": pre,
                "num_catchments": num_units,
            },
            blk,
        )
        l_scale_p = _bind(
            owned,
            programs["graph_scale_div"],
            {
                "in_ptr": p_vec,
                "scaling_ptr": a["graph_scaling_ptr"],
                "out_ptr": pre,
                "num_catchments": num_units,
            },
            blk,
        )
        l_scale_s = _bind(
            owned,
            programs["graph_scale_div"],
            {
                "in_ptr": s_vec,
                "scaling_ptr": a["graph_scaling_ptr"],
                "out_ptr": pre,
                "num_catchments": num_units,
            },
            blk,
        )
        lu_pre_delta = _build_solve_pair(pre, delta)
        lu_pre_tmp = _build_solve_pair(pre, tmp)

        def prec_rhs_delta() -> None:
            l_scale_rhs()
            lu_pre_delta()

        def prec_p_tmp() -> None:
            l_scale_p()
            lu_pre_tmp()

        def prec_s_tmp() -> None:
            l_scale_s()
            lu_pre_tmp()

    else:
        prec_rhs_delta = _build_solve_pair(rhs, delta)
        prec_p_tmp = _build_solve_pair(p_vec, tmp)
        prec_s_tmp = _build_solve_pair(s_vec, tmp)

    def _build_matvec(x_vec: torch.Tensor, out_vec: torch.Tensor, compensated: bool):
        return _bind(
            owned,
            programs["graph_matvec"],
            {
                "dt_ptr": dt_ptr,
                "graph_unit_ptr": gunit,
                "unit_comp_ptr": u_comp,
                "tree_control_ptr": ti,
                "tree_scalars_ptr": tf,
                "x_ptr": x_vec,
                "graph_solution_low_ptr": sol_low,
                "out_ptr": out_vec,
                "graph_bound_ptr": a["graph_bound_ptr"],
                "graph_scaling_ptr": a["graph_scaling_ptr"],
                "trial_area_ptr": a["trial_area_ptr"],
                "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
                "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
                "downstream_idx_ptr": a["downstream_idx_ptr"],
                "downstream_edge_ptr": a["downstream_edge_ptr"],
                "upstream_offset_ptr": a["upstream_offset_ptr"],
                "upstream_index_ptr": a["upstream_index_ptr"],
                "incident_offset_ptr": a["incident_offset_ptr"],
                "incident_path_ptr": a["incident_path_ptr"],
                "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
                "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
                "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                "num_units": num_units,
                "full_domain": full_i,
                "all_active": all_i,
                "use_damping": damp_i,
                "trans": trans_i,
                "compensated": 1 if compensated else 0,
            },
            blk,
        )

    l_mv_dh_comp = _build_matvec(dh, tmp, True)
    l_mv_tmp_v = _build_matvec(tmp, v_vec, False)
    l_mv_tmp_t = _build_matvec(tmp, t_vec, False)

    l_copy_dh = _bind(
        owned,
        programs["graph_vec_copy_dh"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "graph_delta_ptr": delta,
            "newton_step_ptr": dh,
            "graph_solution_low_ptr": sol_low,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
        },
        blk,
    )
    l_res_diff = _bind(
        owned,
        programs["graph_bicg_res_diff"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "rhs_ptr": rhs,
            "tmp_ptr": tmp,
            "norm_buf_ptr": norm_buf,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "total_threads": total_threads,
        },
        blk,
    )
    l_norm_tmp = _bind(
        owned,
        programs["graph_vec_norm"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "x_ptr": tmp,
            "norm_buf_ptr": norm_buf,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "total_threads": total_threads,
        },
        blk,
    )
    l_dot_rhat_r = _bind(
        owned,
        programs["graph_vec_dot"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "a_ptr": rhat,
            "b_ptr": r_vec,
            "red_ptr": red,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "total_threads": total_threads,
        },
        blk,
    )
    l_dot_rhat_v = _bind(
        owned,
        programs["graph_vec_dot"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "a_ptr": rhat,
            "b_ptr": v_vec,
            "red_ptr": red,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "total_threads": total_threads,
        },
        blk,
    )
    l_dot_ts = _bind(
        owned,
        programs["graph_vec_dot_pair"],
        {
            "graph_unit_ptr": gunit,
            "unit_comp_ptr": u_comp,
            "tree_control_ptr": ti,
            "t_ptr": t_vec,
            "s_ptr": s_vec,
            "red_ptr": red,
            "num_units": num_units,
            "full_domain": full_i,
            "all_active": all_i,
            "total_threads": total_threads,
        },
        blk,
    )

    p_init = programs["graph_bicg_init"]
    p_upd_p = programs["graph_bicg_upd_p"]
    p_upd_s = programs["graph_bicg_upd_s"]
    p_upd_r = programs["graph_bicg_upd_r"]
    p_accum = programs["graph_bicg_accum"]

    def _read_norm() -> float:
        return _bits_to_f32(int(norm_buf.item()))

    def solve() -> tuple[int, int, float]:
        if num_units == 0:
            return 0, 0, 0.0
        norm_buf.zero_()
        l_mat_init()
        rhs_norm = _read_norm()
        if not (rhs_norm > 0.0):
            return 0, 0, 0.0
        if not math.isfinite(rhs_norm):
            return 2, 0, float("inf")

        if l_assemble is not None:
            l_assemble()
            for r in range(rounds - 1, -1, -1):
                lh, ll = num_rounds_launches[r]
                if lh is not None:
                    lh()
                if ll is not None:
                    ll()

        tol = float(GRAPH_LINEAR_TOLERANCE)
        prec_rhs_delta()
        solves_used = 1
        l_copy_dh()
        l_mv_dh_comp()
        norm_buf.zero_()
        l_res_diff()
        error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
        if math.isfinite(error) and error < tol:
            return 0, solves_used, error

        for _ in range(GRAPH_LINEAR_RESTARTS):
            norm_buf.zero_()
            l_norm_tmp()
            initial = _read_norm()
            if not (initial > 0.0) or not math.isfinite(initial):
                return (2, solves_used, error) if trans else (
                    1 if math.isfinite(error) else 2,
                    solves_used,
                    error,
                )
            _dispatch_once(
                p_init,
                {
                    "graph_unit_ptr": gunit,
                    "unit_comp_ptr": u_comp,
                    "tree_control_ptr": ti,
                    "tmp_ptr": tmp,
                    "r_ptr": r_vec,
                    "rhat_ptr": rhat,
                    "p_ptr": p_vec,
                    "v_ptr": v_vec,
                    "delta_ptr": delta,
                    "initial": float(initial),
                    "num_units": num_units,
                    "full_domain": full_i,
                    "all_active": all_i,
                },
                blk,
            )
            rho0 = np.float32(1.0)
            alpha = np.float32(1.0)
            omega = np.float32(1.0)
            scale_rel = float(np.float32(initial) / np.float32(rhs_norm))
            for _ in range(GRAPH_LINEAR_ITERATIONS):
                l_dot_rhat_r()
                rho = np.float32(red[:n_blocks].sum(dtype=torch.float32).item())
                if not math.isfinite(float(rho)) or abs(float(rho)) < 1e-30:
                    break
                beta = np.float32((rho / rho0) * (alpha / omega))
                _dispatch_once(
                    p_upd_p,
                    {
                        "graph_unit_ptr": gunit,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "r_ptr": r_vec,
                        "p_ptr": p_vec,
                        "v_ptr": v_vec,
                        "beta": float(beta),
                        "omega": float(omega),
                        "num_units": num_units,
                        "full_domain": full_i,
                        "all_active": all_i,
                    },
                    blk,
                )
                prec_p_tmp()
                solves_used += 1
                l_mv_tmp_v()
                l_dot_rhat_v()
                den = np.float32(red[:n_blocks].sum(dtype=torch.float32).item())
                if not math.isfinite(float(den)) or abs(float(den)) < 1e-30:
                    break
                alpha = np.float32(rho / den)
                norm_buf.zero_()
                _dispatch_once(
                    p_upd_s,
                    {
                        "graph_unit_ptr": gunit,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "r_ptr": r_vec,
                        "v_ptr": v_vec,
                        "tmp_ptr": tmp,
                        "s_ptr": s_vec,
                        "delta_ptr": delta,
                        "norm_buf_ptr": norm_buf,
                        "alpha": float(alpha),
                        "num_units": num_units,
                        "full_domain": full_i,
                        "all_active": all_i,
                        "total_threads": total_threads,
                    },
                    blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                prec_s_tmp()
                solves_used += 1
                l_mv_tmp_t()
                l_dot_ts()
                pair = (
                    red[: 2 * n_blocks]
                    .view(n_blocks, 2)
                    .sum(dim=0, dtype=torch.float32)
                    .tolist()
                )
                tt_f = float(np.float32(pair[0]))
                ts_f = float(np.float32(pair[1]))
                if not (tt_f > 1e-30) or not math.isfinite(tt_f):
                    break
                omega = np.float32(np.float32(ts_f) / np.float32(tt_f))
                if not math.isfinite(float(omega)) or abs(float(omega)) < 1e-20:
                    break
                norm_buf.zero_()
                _dispatch_once(
                    p_upd_r,
                    {
                        "graph_unit_ptr": gunit,
                        "unit_comp_ptr": u_comp,
                        "tree_control_ptr": ti,
                        "s_ptr": s_vec,
                        "t_ptr": t_vec,
                        "tmp_ptr": tmp,
                        "r_ptr": r_vec,
                        "delta_ptr": delta,
                        "norm_buf_ptr": norm_buf,
                        "omega": float(omega),
                        "num_units": num_units,
                        "full_domain": full_i,
                        "all_active": all_i,
                        "total_threads": total_threads,
                    },
                    blk,
                )
                if _read_norm() * scale_rel < tol:
                    break
                rho0 = rho

            _dispatch_once(
                p_accum,
                {
                    "graph_unit_ptr": gunit,
                    "unit_comp_ptr": u_comp,
                    "tree_control_ptr": ti,
                    "newton_step_ptr": dh,
                    "graph_solution_low_ptr": sol_low,
                    "delta_ptr": delta,
                    "initial": float(initial),
                    "num_units": num_units,
                    "full_domain": full_i,
                    "all_active": all_i,
                },
                blk,
            )
            l_mv_dh_comp()
            norm_buf.zero_()
            l_res_diff()
            error = float(np.float32(np.float32(_read_norm()) / np.float32(rhs_norm)))
            if math.isfinite(error) and error < tol:
                return 0, solves_used, error
            if not math.isfinite(error):
                return 2, solves_used, error
        return (1 if math.isfinite(error) else 2), solves_used, error

    return solve


def _prepare_graph_step(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    device = a["storage_ptr"].device
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    npt = int(a["num_pure_trees"])
    ngc = int(a["num_graph_comps"])
    ngu = int(a["num_graph_units"])
    nct = int(a["num_coupled_trees"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    n_vals = int(a["num_bifurcation_values"])
    eps = float(a["slope_regularization"])
    tol = float(a["newton_tolerance"])
    maxit = int(a["newton_max_iterations"])

    owned: list[Any] = []
    forest_progs = forest_metal_library().programs()
    pure_launch = (
        build_tree_launch(owned, forest_progs, a, a["pure_block_tree_ptr"], npt)
        if npt > 0
        else None
    )

    coupled_plan = None
    if ngc > 0 and ngu > 0:
        single_comp = 1 if ngc == 1 else 0
        res_threads = _round_up(ngu, UNIT_THREADS) if single_comp else ngu
        c_trees_long = a["coupled_trees_ptr"].long()
        c_rep_for_tree = a["comp_rep_tree_ptr"].long()[
            a["coupled_tree_comp_ptr"].long()
        ]
        linear_solve = _make_linear_solver(
            owned,
            programs,
            a,
            trans=False,
            full_domain=False,
            all_active=(ngc == 1),
            use_damping=True,
            dt_ptr=a["time_step_ptr"],
            plan_ptr=a["core_factor_layout_ptr"],
            entry_comp_ptr=a["core_entry_comp_ptr"],
            num_units=ngu,
            entries=int(a["num_core_factor_entries"]),
            rounds=int(a["num_core_factor_rounds"]),
        )
        coupled_plan = dict(
            linear_solve=linear_solve,
            start_u=_bind(
                owned,
                programs["graph_coupled_start_units"],
                {
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "storage_ptr": a["storage_ptr"],
                    "river_manning_ptr": a["river_manning_ptr"],
                    "flood_manning_ptr": a["flood_manning_ptr"],
                    "river_length_ptr": a["river_length_ptr"],
                    "river_width_ptr": a["river_width_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "catchment_area_ptr": a["catchment_area_ptr"],
                    "flood_depth_table_ptr": a["flood_depth_table_ptr"],
                    "newton_reference_ptr": a["newton_reference_ptr"],
                    "newton_offset_ptr": a["newton_offset_ptr"],
                    "trial_reference_ptr": a["trial_reference_ptr"],
                    "trial_offset_ptr": a["trial_offset_ptr"],
                    "graph_bound_ptr": a["graph_bound_ptr"],
                    "newton_status_ptr": a["newton_status_ptr"],
                    "num_graph_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            start_c=_bind(
                owned,
                programs["graph_coupled_start_comps"],
                {
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "time_step_ptr": a["time_step_ptr"],
                    "num_graph_comps": ngc,
                },
                32,
            ),
            stage=_bind(
                owned,
                programs["graph_coupled_stage"],
                {
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "trial_reference_ptr": a["trial_reference_ptr"],
                    "trial_offset_ptr": a["trial_offset_ptr"],
                    "river_length_ptr": a["river_length_ptr"],
                    "river_width_ptr": a["river_width_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "catchment_area_ptr": a["catchment_area_ptr"],
                    "flood_depth_table_ptr": a["flood_depth_table_ptr"],
                    "flood_bed_length_ptr": a["flood_bed_length_ptr"],
                    "trial_storage_ptr": a["trial_storage_ptr"],
                    "trial_area_ptr": a["trial_area_ptr"],
                    "channel_conveyance_ptr": a["channel_conveyance_ptr"],
                    "flood_conveyance_ptr": a["flood_conveyance_ptr"],
                    "channel_conveyance_slope_ptr": a["channel_conveyance_slope_ptr"],
                    "flood_conveyance_slope_ptr": a["flood_conveyance_slope_ptr"],
                    "num_graph_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            mflux=_bind(
                owned,
                programs["graph_coupled_main_flux"],
                {
                    "time_step_ptr": a["time_step_ptr"],
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "outflow_ptr": a["outflow_ptr"],
                    "river_manning_ptr": a["river_manning_ptr"],
                    "flood_manning_ptr": a["flood_manning_ptr"],
                    "river_length_ptr": a["river_length_ptr"],
                    "river_width_ptr": a["river_width_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "catchment_area_ptr": a["catchment_area_ptr"],
                    "flood_depth_table_ptr": a["flood_depth_table_ptr"],
                    "flood_bed_length_ptr": a["flood_bed_length_ptr"],
                    "catchment_elevation_ptr": a["catchment_elevation_ptr"],
                    "downstream_distance_ptr": a["downstream_distance_ptr"],
                    "downstream_idx_ptr": a["downstream_idx_ptr"],
                    "downstream_edge_ptr": a["downstream_edge_ptr"],
                    "trial_reference_ptr": a["trial_reference_ptr"],
                    "trial_offset_ptr": a["trial_offset_ptr"],
                    "trial_storage_ptr": a["trial_storage_ptr"],
                    "trial_area_ptr": a["trial_area_ptr"],
                    "channel_conveyance_ptr": a["channel_conveyance_ptr"],
                    "flood_conveyance_ptr": a["flood_conveyance_ptr"],
                    "channel_conveyance_slope_ptr": a["channel_conveyance_slope_ptr"],
                    "flood_conveyance_slope_ptr": a["flood_conveyance_slope_ptr"],
                    "trial_outflow_ptr": a["trial_outflow_ptr"],
                    "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
                    "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
                    "slope_regularization": eps,
                    "num_graph_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            bflux=(
                _bind(
                    owned,
                    programs["graph_coupled_bif_flux"],
                    {
                        "time_step_ptr": a["time_step_ptr"],
                        "path_comp_ptr": a["path_comp_ptr"],
                        "tree_control_ptr": a["tree_control_ptr"],
                        "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
                        "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
                        "catchment_elevation_ptr": a["catchment_elevation_ptr"],
                        "river_height_ptr": a["river_height_ptr"],
                        "trial_reference_ptr": a["trial_reference_ptr"],
                        "trial_offset_ptr": a["trial_offset_ptr"],
                        "bifurcation_width_ptr": a["bifurcation_width_ptr"],
                        "bifurcation_length_ptr": a["bifurcation_length_ptr"],
                        "bifurcation_elevation_ptr": a["bifurcation_elevation_ptr"],
                        "bifurcation_manning_ptr": a["bifurcation_manning_ptr"],
                        "bifurcation_outflow_ptr": a["bifurcation_outflow_ptr"],
                        "graph_path_values_ptr": a["graph_path_values_ptr"],
                        "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                        "newton_status_ptr": a["newton_status_ptr"],
                        "num_bifurcation_paths": m_paths,
                        "num_bifurcation_levels": blevels,
                    },
                    UNIT_THREADS,
                )
                if m_paths > 0
                else None
            ),
            res=_bind(
                owned,
                programs["graph_coupled_residual"],
                {
                    "time_step_ptr": a["time_step_ptr"],
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "storage_ptr": a["storage_ptr"],
                    "runoff_ptr": a["runoff_ptr"],
                    "trial_storage_ptr": a["trial_storage_ptr"],
                    "trial_outflow_ptr": a["trial_outflow_ptr"],
                    "upstream_offset_ptr": a["upstream_offset_ptr"],
                    "upstream_index_ptr": a["upstream_index_ptr"],
                    "incident_offset_ptr": a["incident_offset_ptr"],
                    "incident_path_ptr": a["incident_path_ptr"],
                    "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                    "residual_ptr": a["residual_ptr"],
                    "balance_weight_ptr": a["balance_weight_ptr"],
                    "graph_rhs_ptr": a["graph_rhs_ptr"],
                    "graph_balance_scale_ptr": a["graph_balance_scale_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "num_graph_units": ngu,
                    "single_comp": single_comp,
                    "total_threads": res_threads,
                },
                UNIT_THREADS,
            ),
            ctrl=_bind(
                owned,
                programs["graph_coupled_control"],
                {
                    "time_step_ptr": a["time_step_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "comp_rep_tree_ptr": a["comp_rep_tree_ptr"],
                    "grid_control_ptr": a["grid_control_ptr"],
                    "newton_record_ptr": a["newton_record_ptr"],
                    "newton_status_ptr": a["newton_status_ptr"],
                    "newton_tolerance": tol,
                    "newton_max_iterations": maxit,
                    "num_graph_comps": ngc,
                },
                32,
            ),
            apply=_bind(
                owned,
                programs["graph_coupled_apply"],
                {
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "storage_ptr": a["storage_ptr"],
                    "river_length_ptr": a["river_length_ptr"],
                    "river_width_ptr": a["river_width_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "catchment_area_ptr": a["catchment_area_ptr"],
                    "flood_depth_table_ptr": a["flood_depth_table_ptr"],
                    "newton_reference_ptr": a["newton_reference_ptr"],
                    "newton_offset_ptr": a["newton_offset_ptr"],
                    "trial_reference_ptr": a["trial_reference_ptr"],
                    "trial_offset_ptr": a["trial_offset_ptr"],
                    "balance_weight_ptr": a["balance_weight_ptr"],
                    "accepted_weight_ptr": a["accepted_weight_ptr"],
                    "phase_trace_ptr": a["phase_trace_ptr"],
                    "num_graph_units": ngu,
                    "num_catchments": n,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            prep_rhs=_bind(
                owned,
                programs["graph_coupled_prep_rhs"],
                {
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "newton_reference_ptr": a["newton_reference_ptr"],
                    "newton_offset_ptr": a["newton_offset_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "residual_ptr": a["residual_ptr"],
                    "graph_rhs_ptr": a["graph_rhs_ptr"],
                    "graph_balance_scale_ptr": a["graph_balance_scale_ptr"],
                    "graph_bound_ptr": a["graph_bound_ptr"],
                    "num_graph_units": ngu,
                    "all_active": single_comp,
                },
                UNIT_THREADS,
            ),
            trial=_bind(
                owned,
                programs["graph_coupled_trial"],
                {
                    "graph_unit_ptr": a["graph_unit_ptr"],
                    "unit_comp_ptr": a["unit_comp_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "river_length_ptr": a["river_length_ptr"],
                    "river_width_ptr": a["river_width_ptr"],
                    "river_height_ptr": a["river_height_ptr"],
                    "catchment_area_ptr": a["catchment_area_ptr"],
                    "flood_depth_table_ptr": a["flood_depth_table_ptr"],
                    "newton_reference_ptr": a["newton_reference_ptr"],
                    "newton_offset_ptr": a["newton_offset_ptr"],
                    "trial_reference_ptr": a["trial_reference_ptr"],
                    "trial_offset_ptr": a["trial_offset_ptr"],
                    "newton_step_ptr": a["newton_step_ptr"],
                    "trial_area_ptr": a["trial_area_ptr"],
                    "pivot_ptr": a["pivot_ptr"],
                    "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                    "coupling_ptr": a["coupling_ptr"],
                    "num_graph_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            c_trees_long=c_trees_long,
            c_rep_for_tree=c_rep_for_tree,
            need_meta_copy=(nct > ngc),
        )

    commit_l = build_commit_launch(owned, forest_progs, a)
    gcommit_p = (
        _bind(
            owned,
            programs["graph_commit_paths"],
            {
                "graph_path_values_ptr": a["graph_path_values_ptr"],
                "bifurcation_outflow_ptr": a["bifurcation_outflow_ptr"],
                "num_bifurcation_values": n_vals,
            },
            UNIT_THREADS,
        )
        if n_vals > 0
        else None
    )
    gcommit_i = (
        _bind(
            owned,
            programs["graph_commit_inflow"],
            {
                "incident_offset_ptr": a["incident_offset_ptr"],
                "incident_path_ptr": a["incident_path_ptr"],
                "graph_path_values_ptr": a["graph_path_values_ptr"],
                "inflow_ptr": a["inflow_ptr"],
                "num_catchments": n,
                "num_bifurcation_levels": blevels,
            },
            UNIT_THREADS,
        )
        if m_paths > 0
        else None
    )

    ctl = a["grid_control_ptr"]
    lin_rec = a["linear_record_ptr"]
    meta = a["newton_record_ptr"]

    def launch() -> None:
        ctl.zero_()
        if pure_launch is not None:
            pure_launch()
        if coupled_plan is None:
            lin_rec.zero_()
        else:
            cp = coupled_plan
            cp["start_u"]()
            cp["start_c"]()
            total_solves = 0
            last_lin = (0.0, 0.0, 0.0)
            converged_loop = False
            for _ in range(1 << 22):
                cp["stage"]()
                cp["mflux"]()
                if cp["bflux"] is not None:
                    cp["bflux"]()
                cp["res"]()
                ctl[5:7].zero_()
                cp["ctrl"]()
                cp["apply"]()
                active_c, solving_c = ctl[5:7].tolist()
                if active_c == 0:
                    if cp["need_meta_copy"]:
                        meta[cp["c_trees_long"]] = meta[cp["c_rep_for_tree"]]
                    lin_rec.copy_(
                        torch.tensor(
                            [
                                last_lin[0],
                                last_lin[1],
                                last_lin[2],
                                float(total_solves),
                            ],
                            dtype=torch.float32,
                            device=device,
                        )
                    )
                    converged_loop = True
                    break
                if solving_c > 0:
                    cp["prep_rhs"]()
                    res_code, iters, err = cp["linear_solve"]()
                    total_solves += iters
                    last_lin = (float(res_code), float(iters), float(err))
                cp["trial"]()
            if not converged_loop:
                ctl[4] = int(NEWTON_ROUND_LIMIT)

        commit_l()
        if gcommit_p is not None:
            gcommit_p()
        if gcommit_i is not None:
            gcommit_i()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


def _prepare_graph_rebuild(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    n_vals = int(a["num_bifurcation_values"])
    eps = float(a["slope_regularization"])
    owned: list[Any] = []

    l_stage = _bind(
        owned,
        programs["graph_rebuild_stage"],
        {
            "converged_surface_ptr": a["converged_surface_ptr"],
            "river_length_ptr": a["river_length_ptr"],
            "river_width_ptr": a["river_width_ptr"],
            "river_height_ptr": a["river_height_ptr"],
            "catchment_area_ptr": a["catchment_area_ptr"],
            "flood_depth_table_ptr": a["flood_depth_table_ptr"],
            "flood_bed_length_ptr": a["flood_bed_length_ptr"],
            "newton_reference_ptr": a["newton_reference_ptr"],
            "newton_offset_ptr": a["newton_offset_ptr"],
            "trial_reference_ptr": a["trial_reference_ptr"],
            "trial_offset_ptr": a["trial_offset_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_area_ptr": a["trial_area_ptr"],
            "channel_conveyance_ptr": a["channel_conveyance_ptr"],
            "flood_conveyance_ptr": a["flood_conveyance_ptr"],
            "channel_conveyance_slope_ptr": a["channel_conveyance_slope_ptr"],
            "flood_conveyance_slope_ptr": a["flood_conveyance_slope_ptr"],
            "num_catchments": n,
            "num_flood_levels": levels,
        },
        UNIT_THREADS,
    )
    l_mflux = _bind(
        owned,
        programs["graph_rebuild_main_flux"],
        {
            "time_step_ptr": a["time_step_ptr"],
            "outflow_ptr": a["outflow_ptr"],
            "river_manning_ptr": a["river_manning_ptr"],
            "flood_manning_ptr": a["flood_manning_ptr"],
            "river_length_ptr": a["river_length_ptr"],
            "river_width_ptr": a["river_width_ptr"],
            "river_height_ptr": a["river_height_ptr"],
            "catchment_area_ptr": a["catchment_area_ptr"],
            "flood_depth_table_ptr": a["flood_depth_table_ptr"],
            "flood_bed_length_ptr": a["flood_bed_length_ptr"],
            "catchment_elevation_ptr": a["catchment_elevation_ptr"],
            "downstream_distance_ptr": a["downstream_distance_ptr"],
            "downstream_idx_ptr": a["downstream_idx_ptr"],
            "downstream_edge_ptr": a["downstream_edge_ptr"],
            "trial_reference_ptr": a["trial_reference_ptr"],
            "trial_offset_ptr": a["trial_offset_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_area_ptr": a["trial_area_ptr"],
            "channel_conveyance_ptr": a["channel_conveyance_ptr"],
            "flood_conveyance_ptr": a["flood_conveyance_ptr"],
            "channel_conveyance_slope_ptr": a["channel_conveyance_slope_ptr"],
            "flood_conveyance_slope_ptr": a["flood_conveyance_slope_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
            "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
            "slope_regularization": eps,
            "num_catchments": n,
            "num_flood_levels": levels,
        },
        UNIT_THREADS,
    )
    l_bflux = (
        _bind(
            owned,
            programs["graph_rebuild_bif_flux"],
            {
                "time_step_ptr": a["time_step_ptr"],
                "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
                "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
                "catchment_elevation_ptr": a["catchment_elevation_ptr"],
                "river_height_ptr": a["river_height_ptr"],
                "trial_reference_ptr": a["trial_reference_ptr"],
                "trial_offset_ptr": a["trial_offset_ptr"],
                "bifurcation_width_ptr": a["bifurcation_width_ptr"],
                "bifurcation_length_ptr": a["bifurcation_length_ptr"],
                "bifurcation_elevation_ptr": a["bifurcation_elevation_ptr"],
                "bifurcation_manning_ptr": a["bifurcation_manning_ptr"],
                "bifurcation_outflow_ptr": a["bifurcation_outflow_ptr"],
                "graph_path_values_ptr": a["graph_path_values_ptr"],
                "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                "newton_status_ptr": a["newton_status_ptr"],
                "num_bifurcation_paths": m_paths,
                "num_bifurcation_levels": blevels,
            },
            UNIT_THREADS,
        )
        if m_paths > 0
        else None
    )
    l_res = _bind(
        owned,
        programs["graph_rebuild_residual"],
        {
            "time_step_ptr": a["time_step_ptr"],
            "storage_ptr": a["storage_ptr"],
            "runoff_ptr": a["runoff_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "incident_offset_ptr": a["incident_offset_ptr"],
            "incident_path_ptr": a["incident_path_ptr"],
            "graph_path_sum_ptr": a["graph_path_sum_ptr"],
            "residual_ptr": a["residual_ptr"],
            "balance_weight_ptr": a["balance_weight_ptr"],
            "graph_rhs_ptr": a["graph_rhs_ptr"],
            "num_catchments": n,
        },
        UNIT_THREADS,
    )
    l_snap = _bind(
        owned,
        programs["graph_rebuild_snapshot"],
        {
            "time_step_ptr": a["time_step_ptr"],
            "outflow_ptr": a["outflow_ptr"],
            "river_manning_ptr": a["river_manning_ptr"],
            "flood_manning_ptr": a["flood_manning_ptr"],
            "river_length_ptr": a["river_length_ptr"],
            "river_width_ptr": a["river_width_ptr"],
            "river_height_ptr": a["river_height_ptr"],
            "catchment_area_ptr": a["catchment_area_ptr"],
            "flood_depth_table_ptr": a["flood_depth_table_ptr"],
            "flood_bed_length_ptr": a["flood_bed_length_ptr"],
            "catchment_elevation_ptr": a["catchment_elevation_ptr"],
            "downstream_distance_ptr": a["downstream_distance_ptr"],
            "downstream_idx_ptr": a["downstream_idx_ptr"],
            "downstream_edge_ptr": a["downstream_edge_ptr"],
            "newton_reference_ptr": a["newton_reference_ptr"],
            "newton_offset_ptr": a["newton_offset_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_area_ptr": a["trial_area_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
            "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
            "residual_ptr": a["residual_ptr"],
            "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
            "graph_path_values_ptr": a["graph_path_values_ptr"],
            "bifurcation_jacobian_ptr": a["bifurcation_jacobian_ptr"],
            "slope_regularization": eps,
            "num_catchments": n,
            "num_flood_levels": levels,
            "num_bifurcation_values": n_vals,
            "total_threads": max(n, n_vals),
        },
        UNIT_THREADS,
    )

    def launch() -> None:
        l_stage()
        l_mflux()
        if l_bflux is not None:
            l_bflux()
        l_res()
        l_snap()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


def _prepare_graph_adjoint(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    m_paths = int(a["num_bifurcation_paths"])
    blevels = int(a["num_bifurcation_levels"])
    owned: list[Any] = []

    dt_ptr = a["jacobian_snapshot_ptr"].view(-1)[16:17]
    linear_solve = _make_linear_solver(
        owned,
        programs,
        a,
        trans=True,
        full_domain=True,
        all_active=True,
        use_damping=False,
        dt_ptr=dt_ptr,
        plan_ptr=a["factor_layout_ptr"],
        entry_comp_ptr=a["unit_comp_ptr"],
        num_units=n,
        entries=int(a["num_factor_entries"]),
        rounds=int(a["num_factor_rounds"]),
    )

    l_prep_c = _bind(
        owned,
        programs["graph_adjoint_prep_cells"],
        {
            "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
            "bifurcation_jacobian_ptr": a["bifurcation_jacobian_ptr"],
            "grad_outflow_ptr": a["grad_outflow_ptr"],
            "grad_storage_ptr": a["grad_storage_ptr"],
            "grad_water_surface_ptr": a["grad_water_surface_ptr"],
            "grad_bifurcation_outflow_ptr": a["grad_bifurcation_outflow_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "incident_offset_ptr": a["incident_offset_ptr"],
            "incident_path_ptr": a["incident_path_ptr"],
            "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
            "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
            "trial_area_ptr": a["trial_area_ptr"],
            "graph_bound_ptr": a["graph_bound_ptr"],
            "graph_rhs_ptr": a["graph_rhs_ptr"],
            "num_catchments": n,
            "num_bifurcation_levels": blevels,
        },
        UNIT_THREADS,
    )
    l_prep_p = (
        _bind(
            owned,
            programs["graph_adjoint_prep_paths"],
            {
                "bifurcation_jacobian_ptr": a["bifurcation_jacobian_ptr"],
                "graph_path_sum_ptr": a["graph_path_sum_ptr"],
                "num_bifurcation_paths": m_paths,
                "num_bifurcation_levels": blevels,
            },
            UNIT_THREADS,
        )
        if m_paths > 0
        else None
    )
    l_report = _bind(
        owned,
        programs["graph_report_error"],
        {
            "newton_status_ptr": a["newton_status_ptr"],
            "code": int(ADJOINT_NOT_CONVERGED),
            "info": 0,
            "one": 1,
        },
        1,
    )
    l_s1 = _bind(
        owned,
        programs["graph_adjoint_step1"],
        {
            "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
            "newton_step_ptr": a["newton_step_ptr"],
            "graph_solution_low_ptr": a["graph_solution_low_ptr"],
            "downstream_edge_ptr": a["downstream_edge_ptr"],
            "downstream_idx_ptr": a["downstream_idx_ptr"],
            "grad_outflow_ptr": a["grad_outflow_ptr"],
            "grad_storage_ptr": a["grad_storage_ptr"],
            "graph_temporary_ptr": a["graph_temporary_ptr"],
            "grad_runoff_ptr": a["grad_runoff_ptr"],
            "grad_river_width_ptr": a["grad_river_width_ptr"],
            "grad_river_manning_ptr": a["grad_river_manning_ptr"],
            "grad_flood_manning_ptr": a["grad_flood_manning_ptr"],
            "num_catchments": n,
        },
        UNIT_THREADS,
    )
    l_s2 = _bind(
        owned,
        programs["graph_adjoint_step2"],
        {
            "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
            "bifurcation_jacobian_ptr": a["bifurcation_jacobian_ptr"],
            "newton_step_ptr": a["newton_step_ptr"],
            "graph_solution_low_ptr": a["graph_solution_low_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "incident_offset_ptr": a["incident_offset_ptr"],
            "incident_path_ptr": a["incident_path_ptr"],
            "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
            "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
            "graph_temporary_ptr": a["graph_temporary_ptr"],
            "grad_storage_ptr": a["grad_storage_ptr"],
            "grad_bifurcation_outflow_ptr": a["grad_bifurcation_outflow_ptr"],
            "grad_river_height_ptr": a["grad_river_height_ptr"],
            "num_catchments": n,
            "num_bifurcation_levels": blevels,
        },
        UNIT_THREADS,
    )
    l_s3 = (
        _bind(
            owned,
            programs["graph_adjoint_step3"],
            {
                "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
                "bifurcation_jacobian_ptr": a["bifurcation_jacobian_ptr"],
                "newton_step_ptr": a["newton_step_ptr"],
                "graph_solution_low_ptr": a["graph_solution_low_ptr"],
                "bifurcation_catchment_idx_ptr": a["bifurcation_catchment_idx_ptr"],
                "bifurcation_downstream_idx_ptr": a["bifurcation_downstream_idx_ptr"],
                "grad_bifurcation_outflow_ptr": a["grad_bifurcation_outflow_ptr"],
                "grad_bifurcation_manning_ptr": a["grad_bifurcation_manning_ptr"],
                "num_bifurcation_paths": m_paths,
                "num_bifurcation_levels": blevels,
            },
            UNIT_THREADS,
        )
        if m_paths > 0
        else None
    )
    lin_rec = a["linear_record_ptr"]

    def launch() -> None:
        l_prep_c()
        if l_prep_p is not None:
            l_prep_p()
        res_code, iters, err = linear_solve()
        lin_rec[:3].copy_(
            torch.tensor(
                [float(res_code), float(iters), float(err)],
                dtype=torch.float32,
                device=lin_rec.device,
            )
        )
        if res_code != 0:
            l_report()
        l_s1()
        l_s2()
        if l_s3 is not None:
            l_s3()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


metal_inertia_graph_step = MetalMultiStepKernel(
    library_factory=graph_metal_library,
    prepare=_prepare_graph_step,
    primary_kernel="graph_coupled_stage",
    capturable=False,
)
metal_inertia_graph_rebuild = MetalMultiStepKernel(
    library_factory=graph_metal_library,
    prepare=_prepare_graph_rebuild,
    primary_kernel="graph_rebuild_stage",
    capturable=True,
)
metal_inertia_graph_adjoint = MetalMultiStepKernel(
    library_factory=graph_metal_library,
    prepare=_prepare_graph_adjoint,
    primary_kernel="graph_adjoint_prep_cells",
    capturable=False,
)
