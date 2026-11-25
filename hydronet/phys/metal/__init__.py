# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Apple Metal (MSL) backend for HydroNet physics kernels."""

from hydronet.phys.metal.bifurcation import (
    metal_inertia_graph_adjoint,
    metal_inertia_graph_rebuild,
    metal_inertia_graph_step,
)
from hydronet.phys.metal.implicit_inertia import (
    metal_inertia_adjoint,
    metal_inertia_rebuild,
    metal_inertia_step,
)

__all__ = [
    "metal_inertia_adjoint",
    "metal_inertia_graph_adjoint",
    "metal_inertia_graph_rebuild",
    "metal_inertia_graph_step",
    "metal_inertia_rebuild",
    "metal_inertia_step",
]
