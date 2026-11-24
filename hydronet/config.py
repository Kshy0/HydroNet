"""HydroNet options and device status codes."""

from typing import Final

from hydroforge.contracts import OptionsConfig
from pydantic import Field

# Codes of BaseModule.newton_status.
NEWTON_NOT_CONVERGED: Final = 1
NEWTON_ROUND_LIMIT: Final = 3
INVALID_ROUGHNESS: Final = 4
INVALID_STATE: Final = 5
ADJOINT_NOT_CONVERGED: Final = 6


# Grid linear-solve scheduling. These constants also define CUDA compilation.
GRID_SOLVE_THREADS: Final = 512
PARALLEL_PATH_MIN: Final = 4
PARALLEL_PATH_MAX: Final = 1024
# Tighter requests keep the original ordering to avoid a changed FP32 floor.
PARALLEL_PATH_TOLERANCE_MIN: Final = 1e-5
GRAPH_SOLVE_THREADS: Final = 256
GRAPH_LINEAR_TOLERANCE: Final = 1e-6
# BiCGStab preconditioned by the factored graph: iterations per
# restart; each restart ends with a compensated true-residual audit.
GRAPH_LINEAR_ITERATIONS: Final = 64
GRAPH_LINEAR_RESTARTS: Final = 4
GRAPH_COUPLING_FULL_PHASE: Final = 7
GRAPH_SLOPE_LIMIT: Final = 0.005
GRAPH_CLOSED_ELEVATION: Final = 1e20
GRAPH_DRY_DEPTH: Final = 1e-7

# Forest trees whose line search fails: a fixed-weight line search from the
# start surface, then damped Newton, (J + sigma D) dh = -F, from where it stopped.
# D is each row's diagonal magnitude; sigma starts at NEWTON_DAMPED_SIGMA, follows
# the fixed-weight residual ratio (at most NEWTON_DAMPED_GROWTH-fold per step).
NEWTON_DAMPED_STEPS: Final = 200
NEWTON_DAMPED_SIGMA: Final = 0.3
NEWTON_DAMPED_GROWTH: Final = 10.0
# Per-tree control integers and scalars of the Newton kernels (tree_control, tree_scalars).
TREE_INTEGERS: Final = 15
TREE_SCALARS: Final = 6

# Named arrays of the packed graph factor plan (hydronet.modules._bifurcation).
FACTOR_LAYOUT: Final = (
    "order",
    "urow_ptr",
    "urow_entry",
    "urow_index",
    "lcol_ptr",
    "lcol_entry",
    "lcol_index",
    "lrow_ptr",
    "lrow_entry",
    "lrow_index",
    "ucol_ptr",
    "ucol_entry",
    "ucol_index",
    "diag",
    "product_ptr",
    "product_lower",
    "product_upper",
    "source_ptr",
    "source_code",
    "path_unit",
    "path_offset",
    "round_offset",
    "round_heavy",
)
# Paths of the elimination tree with more products and entries than this are
# factored by a block and substituted by a warp; the others by one thread.
FACTOR_HEAVY_WORK: Final = 256


class HydroNetOptions(OptionsConfig):
    """Numerical settings of the implicit step."""

    differentiable: bool = Field(
        default=False,
        description="Save the converged surface of every step for the adjoint step",
    )
    bifurcation: bool = Field(
        default=False,
        description="Include the map's bifurcation paths in the implicit graph solve",
    )
    slope_regularization: float = Field(
        default=1.0e-7,
        gt=0,
        allow_inf_nan=False,
        description="Water-surface slope below which the friction law is regularized",
    )
    newton_tolerance: float = Field(
        default=2.0e-5,
        gt=0,
        le=2.0e-5,
        allow_inf_nan=False,
        description="Largest relative continuity residual of a converged step",
    )
    newton_max_iterations: int = Field(
        default=120,
        ge=1,
        description="Newton iterations of one line-search phase",
    )
    grid_tree_size: int = Field(
        default=16384,
        ge=1,
        description="Trees of at least this many catchments are solved by all blocks of the "
        "device together when they hold at least half of the catchments; other "
        "trees by one block each (same equations and residual tolerance)",
    )
