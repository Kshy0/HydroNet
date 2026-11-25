# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Metal (MSL) backend for HydroNet implicit local-inertia forest kernels."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

import torch
from hydroforge.kernels.metal import MetalProgram

from hydronet.config import (
    ADJOINT_NOT_CONVERGED,
    INVALID_ROUGHNESS,
    INVALID_STATE,
    NEWTON_DAMPED_GROWTH,
    NEWTON_DAMPED_SIGMA,
    NEWTON_DAMPED_STEPS,
    NEWTON_NOT_CONVERGED,
    NEWTON_ROUND_LIMIT,
    PARALLEL_PATH_MAX,
    PARALLEL_PATH_MIN,
    PARALLEL_PATH_TOLERANCE_MIN,
    TREE_INTEGERS,
    TREE_SCALARS,
)
from hydronet.phys.metal.runtime import (
    MetalLibrary,
    MetalMultiStepKernel,
    SubKernelSpec,
    buf,
    close_launches,
    scl,
)

METAL_DIR = Path(__file__).parent
TREE_THREADS = 256
UNIT_THREADS = 256
PATH_THREADS = 64
PCR_THREADS = 256
PCR_SUB_THREADS = 64


def _round_up(n: int, group: int) -> int:
    return ((n + group - 1) // group) * group


def forest_constants_prelude() -> str:
    """MSL compile-time constants matching hydronet.config (strictly typed)."""
    return (
        f"constant constexpr int PARALLEL_PATH_MIN = {int(PARALLEL_PATH_MIN)};\n"
        f"constant constexpr int PARALLEL_PATH_MAX = {int(PARALLEL_PATH_MAX)};\n"
        f"constant constexpr float PARALLEL_PATH_TOLERANCE_MIN = "
        f"{float(PARALLEL_PATH_TOLERANCE_MIN)!r}f;\n"
        f"constant constexpr int NEWTON_DAMPED_STEPS = {int(NEWTON_DAMPED_STEPS)};\n"
        f"constant constexpr float NEWTON_DAMPED_SIGMA = "
        f"{float(NEWTON_DAMPED_SIGMA)!r}f;\n"
        f"constant constexpr float NEWTON_DAMPED_GROWTH = "
        f"{float(NEWTON_DAMPED_GROWTH)!r}f;\n"
        f"constant constexpr int TREE_INTEGERS = {int(TREE_INTEGERS)};\n"
        f"constant constexpr int TREE_SCALARS = {int(TREE_SCALARS)};\n"
        f"constant constexpr int NEWTON_NOT_CONVERGED = {int(NEWTON_NOT_CONVERGED)};\n"
        f"constant constexpr int NEWTON_ROUND_LIMIT = {int(NEWTON_ROUND_LIMIT)};\n"
        f"constant constexpr int INVALID_ROUGHNESS = {int(INVALID_ROUGHNESS)};\n"
        f"constant constexpr int INVALID_STATE = {int(INVALID_STATE)};\n"
        f"constant constexpr int ADJOINT_NOT_CONVERGED = {int(ADJOINT_NOT_CONVERGED)};\n"
    )


_SUBKERNELS: tuple[SubKernelSpec, ...] = (
    SubKernelSpec(
        name="inertia_solve_trees",
        extent=("total_threads",),
        arguments=(
            buf("storage_ptr", "read", "float"),
            buf("runoff_ptr", "read", "float"),
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
            buf("time_step_ptr", "read", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("block_tree_ptr", "read", "int"),
            buf("tree_unit_ptr", "read", "int"),
            buf("level_offset_ptr", "read", "int"),
            buf("tree_level_ptr", "read", "int"),
            buf("newton_offset_ptr", "read_write", "float"),
            buf("trial_offset_ptr", "read_write", "float"),
            buf("newton_reference_ptr", "read_write", "float"),
            buf("trial_reference_ptr", "read_write", "float"),
            buf("trial_storage_ptr", "read_write", "float"),
            buf("trial_area_ptr", "read_write", "float"),
            buf("channel_conveyance_ptr", "read_write", "float"),
            buf("flood_conveyance_ptr", "read_write", "float"),
            buf("channel_conveyance_slope_ptr", "read_write", "float"),
            buf("flood_conveyance_slope_ptr", "read_write", "float"),
            buf("trial_outflow_ptr", "read_write", "float"),
            buf("outflow_own_slope_ptr", "read_write", "float"),
            buf("outflow_down_slope_ptr", "read_write", "float"),
            buf("residual_ptr", "read_write", "float"),
            buf("balance_weight_ptr", "read_write", "float"),
            buf("accepted_weight_ptr", "read_write", "float"),
            buf("pivot_ptr", "read_write", "float"),
            buf("pivot_rhs_ptr", "read_write", "float"),
            buf("newton_step_ptr", "read_write", "float"),
            buf("coupling_ptr", "read_write", "float"),
            buf("phase_trace_ptr", "read_write", "float"),
            buf("tree_scalars_ptr", "read_write", "float"),
            buf("newton_record_ptr", "write", "float"),
            buf("tree_control_ptr", "read_write", "int"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            scl("slope_regularization", "float32"),
            scl("newton_tolerance", "float32"),
            scl("newton_max_iterations", "int32"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
            scl("num_block_trees", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_start_units",
        extent=("num_grid_units",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
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
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            scl("num_grid_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_start_trees",
        extent=("num_grid_trees",),
        arguments=(
            buf("grid_tree_ptr", "read", "int"),
            buf("tree_control_ptr", "write", "int"),
            buf("tree_scalars_ptr", "write", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            buf("time_step_ptr", "read", "float"),
            scl("num_grid_trees", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_stage",
        extent=("num_grid_units",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
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
            scl("num_grid_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_flux",
        extent=("num_grid_units",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
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
            scl("num_grid_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_residual",
        extent=("total_threads",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
            buf("grid_tree_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("storage_ptr", "read", "float"),
            buf("runoff_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("residual_ptr", "write", "float"),
            buf("balance_weight_ptr", "write", "float"),
            buf("accepted_weight_ptr", "read", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            scl("num_grid_units", "int32"),
            scl("single_tree", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_control",
        extent=("num_grid_trees",),
        arguments=(
            buf("grid_tree_ptr", "read", "int"),
            buf("tree_control_ptr", "read_write", "int"),
            buf("tree_scalars_ptr", "read_write", "float"),
            buf("tree_merit_ptr", "atomic_max", "atomic_uint"),
            buf("tree_line_ptr", "atomic_max", "atomic_uint"),
            buf("grid_control_ptr", "atomic_add", "atomic_int"),
            buf("newton_record_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            buf("time_step_ptr", "read", "float"),
            scl("newton_tolerance", "float32"),
            scl("newton_max_iterations", "int32"),
            scl("num_grid_trees", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_apply",
        extent=("num_grid_units",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
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
            scl("num_catchments", "int32"),
            scl("num_grid_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_eliminate_serial",
        extent=("num_paths",),
        arguments=(
            buf("paths_ptr", "read", "int"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("residual_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("pivot_ptr", "read_write", "float"),
            buf("pivot_rhs_ptr", "read_write", "float"),
            buf("coupling_ptr", "read_write", "float"),
            scl("num_paths", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_substitute_serial",
        extent=("num_paths",),
        arguments=(
            buf("paths_ptr", "read", "int"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("pivot_ptr", "read", "float"),
            buf("pivot_rhs_ptr", "read", "float"),
            buf("coupling_ptr", "read", "float"),
            buf("newton_step_ptr", "read_write", "float"),
            scl("num_paths", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_eliminate_pcr",
        extent=("total_threads",),
        arguments=(
            buf("pcr_paths_ptr", "read", "int"),
            buf("pcr_slots_ptr", "read", "int"),
            buf("pcr_scratch_ptr", "read_write", "float"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("tree_scalars_ptr", "read", "float"),
            buf("river_height_ptr", "read", "float"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("residual_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("pivot_ptr", "read_write", "float"),
            buf("pivot_rhs_ptr", "read_write", "float"),
            buf("coupling_ptr", "read_write", "float"),
            scl("num_pcr_paths", "int32"),
            scl("pcr_stride", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_substitute_pcr",
        extent=("total_threads",),
        arguments=(
            buf("pcr_paths_ptr", "read", "int"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
            buf("tree_control_ptr", "read", "int"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("pivot_rhs_ptr", "read", "float"),
            buf("coupling_ptr", "read", "float"),
            buf("newton_step_ptr", "read_write", "float"),
            scl("num_pcr_paths", "int32"),
            scl("total_threads", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_grid_trial",
        extent=("num_grid_units",),
        arguments=(
            buf("grid_unit_ptr", "read", "int"),
            buf("tree_index_ptr", "read", "int"),
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
            scl("num_grid_units", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_commit",
        extent=("num_catchments",),
        arguments=(
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("grid_control_ptr", "read", "int"),
            buf("storage_ptr", "write", "float"),
            buf("outflow_ptr", "write", "float"),
            buf("water_surface_ptr", "write", "float"),
            buf("inflow_ptr", "write", "float"),
            buf("newton_status_ptr", "atomic_add", "atomic_int"),
            buf("converged_surface_ptr", "write", "float", optional=True),
            scl("num_catchments", "int32"),
            scl("save_surface", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_rebuild_stage",
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
        name="inertia_rebuild_flux",
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
        name="inertia_rebuild_residual",
        extent=("num_catchments",),
        arguments=(
            buf("time_step_ptr", "read", "float"),
            buf("storage_ptr", "read", "float"),
            buf("runoff_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("residual_ptr", "write", "float"),
            buf("balance_weight_ptr", "write", "float"),
            scl("num_catchments", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_rebuild_snapshot",
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
            buf("newton_reference_ptr", "read", "float"),
            buf("newton_offset_ptr", "read", "float"),
            buf("trial_storage_ptr", "read", "float"),
            buf("trial_area_ptr", "read", "float"),
            buf("trial_outflow_ptr", "read", "float"),
            buf("outflow_own_slope_ptr", "read", "float"),
            buf("outflow_down_slope_ptr", "read", "float"),
            buf("residual_ptr", "read", "float"),
            buf("jacobian_snapshot_ptr", "write", "float"),
            scl("slope_regularization", "float32"),
            scl("num_catchments", "int32"),
            scl("num_flood_levels", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_adjoint_up",
        extent=("count",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("grad_outflow_ptr", "read", "float"),
            buf("grad_storage_ptr", "read", "float"),
            buf("grad_water_surface_ptr", "read", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("pivot_ptr", "read_write", "float"),
            buf("pivot_rhs_ptr", "read_write", "float"),
            scl("r_begin", "int32"),
            scl("count", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_adjoint_down",
        extent=("count",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("grad_outflow_ptr", "read_write", "float"),
            buf("downstream_idx_ptr", "read", "int"),
            buf("downstream_edge_ptr", "read", "uchar"),
            buf("path_unit_ptr", "read", "int"),
            buf("path_offset_ptr", "read", "int"),
            buf("pivot_ptr", "read", "float"),
            buf("pivot_rhs_ptr", "read", "float"),
            buf("adjoint_continuity_ptr", "read_write", "float"),
            buf("adjoint_flux_ptr", "write", "float"),
            buf("grad_runoff_ptr", "write", "float"),
            scl("r_begin", "int32"),
            scl("count", "int32"),
        ),
    ),
    SubKernelSpec(
        name="inertia_adjoint_finish",
        extent=("num_catchments",),
        arguments=(
            buf("jacobian_snapshot_ptr", "read", "float"),
            buf("grad_storage_ptr", "read_write", "float"),
            buf("upstream_offset_ptr", "read", "int"),
            buf("upstream_index_ptr", "read", "int"),
            buf("adjoint_continuity_ptr", "read", "float"),
            buf("adjoint_flux_ptr", "read", "float"),
            buf("grad_river_width_ptr", "read_write", "float"),
            buf("grad_river_height_ptr", "read_write", "float"),
            buf("grad_river_manning_ptr", "read_write", "float"),
            buf("grad_flood_manning_ptr", "read_write", "float"),
            scl("num_catchments", "int32"),
        ),
    ),
)


@cache
def forest_metal_library() -> MetalLibrary:
    """Compile-time validated MSL library for all forest physics sub-kernels."""
    return MetalLibrary(
        METAL_DIR / "implicit_inertia.metal",
        _SUBKERNELS,
        prelude=forest_constants_prelude(),
    )


def _bind(
    owned: list[Any],
    program: MetalProgram,
    values: dict[str, Any],
    group_size: int,
):
    launch = program.launch(values, group_size)
    owned.append(launch)
    return launch


def build_tree_launch(
    owned: list[Any],
    programs: dict[str, MetalProgram],
    arguments: dict[str, Any],
    btree_tensor: torch.Tensor,
    num_trees: int,
):
    """Build a specialized inertia_solve_trees launch for a block-tree list."""
    a = arguments
    return _bind(
        owned,
        programs["inertia_solve_trees"],
        {
            "storage_ptr": a["storage_ptr"],
            "runoff_ptr": a["runoff_ptr"],
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
            "time_step_ptr": a["time_step_ptr"],
            "downstream_idx_ptr": a["downstream_idx_ptr"],
            "downstream_edge_ptr": a["downstream_edge_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "tree_index_ptr": a["tree_index_ptr"],
            "block_tree_ptr": btree_tensor,
            "tree_unit_ptr": a["tree_unit_ptr"],
            "level_offset_ptr": a["level_offset_ptr"],
            "tree_level_ptr": a["tree_level_ptr"],
            "newton_offset_ptr": a["newton_offset_ptr"],
            "trial_offset_ptr": a["trial_offset_ptr"],
            "newton_reference_ptr": a["newton_reference_ptr"],
            "trial_reference_ptr": a["trial_reference_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_area_ptr": a["trial_area_ptr"],
            "channel_conveyance_ptr": a["channel_conveyance_ptr"],
            "flood_conveyance_ptr": a["flood_conveyance_ptr"],
            "channel_conveyance_slope_ptr": a["channel_conveyance_slope_ptr"],
            "flood_conveyance_slope_ptr": a["flood_conveyance_slope_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
            "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
            "residual_ptr": a["residual_ptr"],
            "balance_weight_ptr": a["balance_weight_ptr"],
            "accepted_weight_ptr": a["accepted_weight_ptr"],
            "pivot_ptr": a["pivot_ptr"],
            "pivot_rhs_ptr": a["pivot_rhs_ptr"],
            "newton_step_ptr": a["newton_step_ptr"],
            "coupling_ptr": a["coupling_ptr"],
            "phase_trace_ptr": a["phase_trace_ptr"],
            "tree_scalars_ptr": a["tree_scalars_ptr"],
            "newton_record_ptr": a["newton_record_ptr"],
            "tree_control_ptr": a["tree_control_ptr"],
            "newton_status_ptr": a["newton_status_ptr"],
            "tree_merit_ptr": a["tree_merit_ptr"],
            "tree_line_ptr": a["tree_line_ptr"],
            "slope_regularization": float(a["slope_regularization"]),
            "newton_tolerance": float(a["newton_tolerance"]),
            "newton_max_iterations": int(a["newton_max_iterations"]),
            "num_catchments": int(a["num_catchments"]),
            "num_flood_levels": int(a["num_flood_levels"]),
            "num_block_trees": int(num_trees),
            "total_threads": int(num_trees) * TREE_THREADS,
        },
        TREE_THREADS,
    )


def build_commit_launch(
    owned: list[Any],
    programs: dict[str, MetalProgram],
    arguments: dict[str, Any],
):
    """Build a specialized inertia_commit launch."""
    a = arguments
    save_surface = 1 if a["converged_surface_ptr"] is not None else 0
    return _bind(
        owned,
        programs["inertia_commit"],
        {
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "newton_reference_ptr": a["newton_reference_ptr"],
            "newton_offset_ptr": a["newton_offset_ptr"],
            "downstream_edge_ptr": a["downstream_edge_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "grid_control_ptr": a["grid_control_ptr"],
            "storage_ptr": a["storage_ptr"],
            "outflow_ptr": a["outflow_ptr"],
            "water_surface_ptr": a["water_surface_ptr"],
            "inflow_ptr": a["inflow_ptr"],
            "newton_status_ptr": a["newton_status_ptr"],
            "converged_surface_ptr": a["converged_surface_ptr"],
            "num_catchments": int(a["num_catchments"]),
            "save_surface": save_surface,
        },
        UNIT_THREADS,
    )


def _prepare_inertia_step(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    device = a["storage_ptr"].device
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    ngu = int(a["num_grid_units"])
    ngt = int(a["num_grid_trees"])
    nbt = int(a["num_block_trees"])
    rounds = int(a["num_path_rounds"])
    eps = float(a["slope_regularization"])
    tol = float(a["newton_tolerance"])
    maxit = int(a["newton_max_iterations"])

    owned: list[Any] = []
    tree_launch = None
    if nbt > 0:
        btree_cpu = a["block_tree_ptr"].cpu()
        tsize_cpu = a["tree_size_ptr"].cpu()
        order = torch.argsort(tsize_cpu[btree_cpu.long()], descending=True, stable=True)
        btree_sorted = (
            btree_cpu[order].to(device=device, dtype=torch.int32).contiguous()
        )
        tree_launch = build_tree_launch(owned, programs, a, btree_sorted, nbt)

    grid_plan = None
    if ngt > 0 and ngu > 0:
        pptr_cpu = a["path_offset_ptr"].cpu()
        gpath_cpu = a["grid_path_ptr"].cpu()
        grr_cpu = a["grid_round_offset_ptr"].cpu().tolist()
        use_pcr = tol >= PARALLEL_PATH_TOLERANCE_MIN

        round_data: list[tuple[list[int], list[int], list[int]]] = []
        max_pcr_nodes = 0
        for r in range(rounds):
            rb, re_idx = grr_cpu[r], grr_cpu[r + 1]
            r_paths = gpath_cpu[rb:re_idx].tolist()
            if not use_pcr:
                round_data.append((r_paths, [], []))
            else:
                serial_list: list[int] = []
                pcr_list: list[int] = []
                pcr_slots: list[int] = []
                offset = 0
                for p_id in r_paths:
                    plen = int(pptr_cpu[p_id + 1].item() - pptr_cpu[p_id].item())
                    if PARALLEL_PATH_MIN <= plen <= PARALLEL_PATH_MAX:
                        pcr_list.append(p_id)
                        pcr_slots.append(offset)
                        offset += plen
                    else:
                        serial_list.append(p_id)
                max_pcr_nodes = max(max_pcr_nodes, offset)
                round_data.append((serial_list, pcr_list, pcr_slots))

        pcr_scratch = (
            torch.empty(10 * max_pcr_nodes, dtype=torch.float32, device=device)
            if max_pcr_nodes > 0
            else None
        )

        round_launches = []
        for serial_list, pcr_list, pcr_slots in round_data:
            elim_s = sub_s = elim_p = sub_p = None
            if serial_list:
                sp_dev = torch.tensor(serial_list, dtype=torch.int32, device=device)
                n_sp = len(serial_list)
                elim_s = _bind(
                    owned,
                    programs["inertia_grid_eliminate_serial"],
                    {
                        "paths_ptr": sp_dev,
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "tree_index_ptr": a["tree_index_ptr"],
                        "tree_control_ptr": a["tree_control_ptr"],
                        "tree_scalars_ptr": a["tree_scalars_ptr"],
                        "river_height_ptr": a["river_height_ptr"],
                        "downstream_edge_ptr": a["downstream_edge_ptr"],
                        "upstream_offset_ptr": a["upstream_offset_ptr"],
                        "upstream_index_ptr": a["upstream_index_ptr"],
                        "newton_reference_ptr": a["newton_reference_ptr"],
                        "newton_offset_ptr": a["newton_offset_ptr"],
                        "residual_ptr": a["residual_ptr"],
                        "trial_area_ptr": a["trial_area_ptr"],
                        "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
                        "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
                        "pivot_ptr": a["pivot_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "coupling_ptr": a["coupling_ptr"],
                        "num_paths": n_sp,
                    },
                    PATH_THREADS,
                )
                sub_s = _bind(
                    owned,
                    programs["inertia_grid_substitute_serial"],
                    {
                        "paths_ptr": sp_dev,
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "tree_index_ptr": a["tree_index_ptr"],
                        "tree_control_ptr": a["tree_control_ptr"],
                        "downstream_idx_ptr": a["downstream_idx_ptr"],
                        "downstream_edge_ptr": a["downstream_edge_ptr"],
                        "pivot_ptr": a["pivot_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "coupling_ptr": a["coupling_ptr"],
                        "newton_step_ptr": a["newton_step_ptr"],
                        "num_paths": n_sp,
                    },
                    PATH_THREADS,
                )
            if pcr_list:
                bp_dev = torch.tensor(pcr_list, dtype=torch.int32, device=device)
                bs_dev = torch.tensor(pcr_slots, dtype=torch.int32, device=device)
                n_pcr = len(pcr_list)
                elim_p = _bind(
                    owned,
                    programs["inertia_grid_eliminate_pcr"],
                    {
                        "pcr_paths_ptr": bp_dev,
                        "pcr_slots_ptr": bs_dev,
                        "pcr_scratch_ptr": pcr_scratch,
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "tree_index_ptr": a["tree_index_ptr"],
                        "tree_control_ptr": a["tree_control_ptr"],
                        "tree_scalars_ptr": a["tree_scalars_ptr"],
                        "river_height_ptr": a["river_height_ptr"],
                        "downstream_edge_ptr": a["downstream_edge_ptr"],
                        "upstream_offset_ptr": a["upstream_offset_ptr"],
                        "upstream_index_ptr": a["upstream_index_ptr"],
                        "newton_reference_ptr": a["newton_reference_ptr"],
                        "newton_offset_ptr": a["newton_offset_ptr"],
                        "residual_ptr": a["residual_ptr"],
                        "trial_area_ptr": a["trial_area_ptr"],
                        "outflow_own_slope_ptr": a["outflow_own_slope_ptr"],
                        "outflow_down_slope_ptr": a["outflow_down_slope_ptr"],
                        "pivot_ptr": a["pivot_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "coupling_ptr": a["coupling_ptr"],
                        "num_pcr_paths": n_pcr,
                        "pcr_stride": max_pcr_nodes,
                        "total_threads": n_pcr * PCR_THREADS,
                    },
                    PCR_THREADS,
                )
                sub_p = _bind(
                    owned,
                    programs["inertia_grid_substitute_pcr"],
                    {
                        "pcr_paths_ptr": bp_dev,
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "tree_index_ptr": a["tree_index_ptr"],
                        "tree_control_ptr": a["tree_control_ptr"],
                        "downstream_idx_ptr": a["downstream_idx_ptr"],
                        "downstream_edge_ptr": a["downstream_edge_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "coupling_ptr": a["coupling_ptr"],
                        "newton_step_ptr": a["newton_step_ptr"],
                        "num_pcr_paths": n_pcr,
                        "total_threads": n_pcr * PCR_SUB_THREADS,
                    },
                    PCR_SUB_THREADS,
                )
            round_launches.append((elim_s, elim_p, sub_s, sub_p))

        single_tree = 1 if ngt == 1 else 0
        res_threads = _round_up(ngu, UNIT_THREADS) if single_tree else ngu
        grid_plan = dict(
            start_u=_bind(
                owned,
                programs["inertia_grid_start_units"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
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
                    "newton_status_ptr": a["newton_status_ptr"],
                    "num_grid_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            start_t=_bind(
                owned,
                programs["inertia_grid_start_trees"],
                {
                    "grid_tree_ptr": a["grid_tree_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "time_step_ptr": a["time_step_ptr"],
                    "num_grid_trees": ngt,
                },
                32,
            ),
            stage=_bind(
                owned,
                programs["inertia_grid_stage"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
                    "tree_index_ptr": a["tree_index_ptr"],
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
                    "num_grid_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            flux=_bind(
                owned,
                programs["inertia_grid_flux"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
                    "tree_index_ptr": a["tree_index_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
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
                    "num_grid_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            residual=_bind(
                owned,
                programs["inertia_grid_residual"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
                    "grid_tree_ptr": a["grid_tree_ptr"],
                    "tree_index_ptr": a["tree_index_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "storage_ptr": a["storage_ptr"],
                    "runoff_ptr": a["runoff_ptr"],
                    "trial_storage_ptr": a["trial_storage_ptr"],
                    "trial_outflow_ptr": a["trial_outflow_ptr"],
                    "upstream_offset_ptr": a["upstream_offset_ptr"],
                    "upstream_index_ptr": a["upstream_index_ptr"],
                    "residual_ptr": a["residual_ptr"],
                    "balance_weight_ptr": a["balance_weight_ptr"],
                    "accepted_weight_ptr": a["accepted_weight_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "num_grid_units": ngu,
                    "single_tree": single_tree,
                    "total_threads": res_threads,
                },
                UNIT_THREADS,
            ),
            control=_bind(
                owned,
                programs["inertia_grid_control"],
                {
                    "grid_tree_ptr": a["grid_tree_ptr"],
                    "tree_control_ptr": a["tree_control_ptr"],
                    "tree_scalars_ptr": a["tree_scalars_ptr"],
                    "tree_merit_ptr": a["tree_merit_ptr"],
                    "tree_line_ptr": a["tree_line_ptr"],
                    "grid_control_ptr": a["grid_control_ptr"],
                    "newton_record_ptr": a["newton_record_ptr"],
                    "newton_status_ptr": a["newton_status_ptr"],
                    "time_step_ptr": a["time_step_ptr"],
                    "newton_tolerance": tol,
                    "newton_max_iterations": maxit,
                    "num_grid_trees": ngt,
                },
                32,
            ),
            apply=_bind(
                owned,
                programs["inertia_grid_apply"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
                    "tree_index_ptr": a["tree_index_ptr"],
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
                    "num_catchments": n,
                    "num_grid_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            trial=_bind(
                owned,
                programs["inertia_grid_trial"],
                {
                    "grid_unit_ptr": a["grid_unit_ptr"],
                    "tree_index_ptr": a["tree_index_ptr"],
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
                    "num_grid_units": ngu,
                    "num_flood_levels": levels,
                },
                UNIT_THREADS,
            ),
            rounds=tuple(round_launches),
        )

    commit_launch = build_commit_launch(owned, programs, a)
    ctl = a["grid_control_ptr"]

    def launch() -> None:
        ctl.zero_()
        if tree_launch is not None:
            tree_launch()
        if grid_plan is not None:
            gp = grid_plan
            gp["start_u"]()
            gp["start_t"]()
            finished = False
            for _ in range(1 << 22):
                gp["stage"]()
                gp["flux"]()
                gp["residual"]()
                ctl[:4].zero_()
                gp["control"]()
                gp["apply"]()
                ctl_vals = ctl[:3].tolist()
                n_active, n_solving = ctl_vals[0], ctl_vals[2]
                if n_active == 0:
                    finished = True
                    break
                if n_solving > 0:
                    for r in range(rounds - 1, -1, -1):
                        elim_s, elim_p, _, _ = gp["rounds"][r]
                        if elim_s is not None:
                            elim_s()
                        if elim_p is not None:
                            elim_p()
                    for r in range(rounds):
                        _, _, sub_s, sub_p = gp["rounds"][r]
                        if sub_s is not None:
                            sub_s()
                        if sub_p is not None:
                            sub_p()
                gp["trial"]()
            if not finished:
                ctl[4] = NEWTON_ROUND_LIMIT
        commit_launch()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


def _prepare_inertia_rebuild(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    levels = int(a["num_flood_levels"])
    eps = float(a["slope_regularization"])
    owned: list[Any] = []

    stage_l = _bind(
        owned,
        programs["inertia_rebuild_stage"],
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
    flux_l = _bind(
        owned,
        programs["inertia_rebuild_flux"],
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
    res_l = _bind(
        owned,
        programs["inertia_rebuild_residual"],
        {
            "time_step_ptr": a["time_step_ptr"],
            "storage_ptr": a["storage_ptr"],
            "runoff_ptr": a["runoff_ptr"],
            "trial_storage_ptr": a["trial_storage_ptr"],
            "trial_outflow_ptr": a["trial_outflow_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "residual_ptr": a["residual_ptr"],
            "balance_weight_ptr": a["balance_weight_ptr"],
            "num_catchments": n,
        },
        UNIT_THREADS,
    )
    snap_l = _bind(
        owned,
        programs["inertia_rebuild_snapshot"],
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
            "slope_regularization": eps,
            "num_catchments": n,
            "num_flood_levels": levels,
        },
        UNIT_THREADS,
    )

    def launch() -> None:
        stage_l()
        flux_l()
        res_l()
        snap_l()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


def _prepare_inertia_adjoint(
    arguments: dict[str, Any],
    buffer_dtypes: Any,
    programs: dict[str, MetalProgram],
):
    del buffer_dtypes
    a = arguments
    n = int(a["num_catchments"])
    rounds = int(a["num_path_rounds"])
    rptr = a["round_offset_ptr"].cpu().tolist()
    owned: list[Any] = []

    up_launches = []
    down_launches = []
    for r in range(rounds):
        rb, re_idx = int(rptr[r]), int(rptr[r + 1])
        cnt = re_idx - rb
        if cnt > 0:
            up_launches.append(
                _bind(
                    owned,
                    programs["inertia_adjoint_up"],
                    {
                        "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
                        "grad_outflow_ptr": a["grad_outflow_ptr"],
                        "grad_storage_ptr": a["grad_storage_ptr"],
                        "grad_water_surface_ptr": a["grad_water_surface_ptr"],
                        "upstream_offset_ptr": a["upstream_offset_ptr"],
                        "upstream_index_ptr": a["upstream_index_ptr"],
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "pivot_ptr": a["pivot_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "r_begin": rb,
                        "count": cnt,
                    },
                    PATH_THREADS,
                )
            )
            down_launches.append(
                _bind(
                    owned,
                    programs["inertia_adjoint_down"],
                    {
                        "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
                        "grad_outflow_ptr": a["grad_outflow_ptr"],
                        "downstream_idx_ptr": a["downstream_idx_ptr"],
                        "downstream_edge_ptr": a["downstream_edge_ptr"],
                        "path_unit_ptr": a["path_unit_ptr"],
                        "path_offset_ptr": a["path_offset_ptr"],
                        "pivot_ptr": a["pivot_ptr"],
                        "pivot_rhs_ptr": a["pivot_rhs_ptr"],
                        "adjoint_continuity_ptr": a["adjoint_continuity_ptr"],
                        "adjoint_flux_ptr": a["adjoint_flux_ptr"],
                        "grad_runoff_ptr": a["grad_runoff_ptr"],
                        "r_begin": rb,
                        "count": cnt,
                    },
                    PATH_THREADS,
                )
            )
        else:
            up_launches.append(None)
            down_launches.append(None)

    finish_l = _bind(
        owned,
        programs["inertia_adjoint_finish"],
        {
            "jacobian_snapshot_ptr": a["jacobian_snapshot_ptr"],
            "grad_storage_ptr": a["grad_storage_ptr"],
            "upstream_offset_ptr": a["upstream_offset_ptr"],
            "upstream_index_ptr": a["upstream_index_ptr"],
            "adjoint_continuity_ptr": a["adjoint_continuity_ptr"],
            "adjoint_flux_ptr": a["adjoint_flux_ptr"],
            "grad_river_width_ptr": a["grad_river_width_ptr"],
            "grad_river_height_ptr": a["grad_river_height_ptr"],
            "grad_river_manning_ptr": a["grad_river_manning_ptr"],
            "grad_flood_manning_ptr": a["grad_flood_manning_ptr"],
            "num_catchments": n,
        },
        UNIT_THREADS,
    )

    def launch() -> None:
        for r in range(rounds - 1, -1, -1):
            up_l = up_launches[r]
            if up_l is not None:
                up_l()
        for r in range(rounds):
            dn_l = down_launches[r]
            if dn_l is not None:
                dn_l()
        finish_l()

    launch.close = lambda: close_launches(owned)  # type: ignore[attr-defined]
    return launch


metal_inertia_step = MetalMultiStepKernel(
    library_factory=forest_metal_library,
    prepare=_prepare_inertia_step,
    primary_kernel="inertia_solve_trees",
    capturable=False,
)
metal_inertia_rebuild = MetalMultiStepKernel(
    library_factory=forest_metal_library,
    prepare=_prepare_inertia_rebuild,
    primary_kernel="inertia_rebuild_stage",
    capturable=True,
)
metal_inertia_adjoint = MetalMultiStepKernel(
    library_factory=forest_metal_library,
    prepare=_prepare_inertia_adjoint,
    primary_kernel="inertia_adjoint_up",
    capturable=True,
)
