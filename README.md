# HydroNet

**HydroNet** is a differentiable, fully implicit local-inertia river hydrodynamics model built on the [HydroForge](https://github.com/Kshy0/hydroforge) framework. Each step solves continuity and local-inertial momentum on a [CaMa-Flood](https://github.com/Kshy0/CaMa-Flood-GPU) river network (including backwater, floodplain inundation, reverse flow, and optional multi-level bifurcation pathways with cycles) using GPU-accelerated **Triton** and **native CUDA** backends. When `options.differentiable=True`, the forward step records only its converged water surface and `step_adjoint` computes exact first-order gradients via a single transposed solve of the converged Jacobian — without unrolling Newton iterations.

The `scripts/` workflow couples HydroNet to an ERA5 runoff LSTM (`scripts/s05_train.py`) and trains runoff generation end to end from river gauge discharge through the physical adjoint, alongside the lumped LSTM (`scripts/s07_lumped_baseline.py`) and graph neural network (`scripts/s08_gnn_baseline.py`) baselines.

---

## Installation

Python 3.11+ and a CUDA-enabled PyTorch installation are required. Install [HydroForge](https://github.com/Kshy0/hydroforge) and HydroNet in the same environment (and optionally [CaMa-Flood-GPU](https://github.com/Kshy0/CaMa-Flood-GPU) for gauge allocation, spin-up restarts, and map cropping):

```bash
git clone https://github.com/Kshy0/HydroNet.git
cd HydroNet
pip install -e ".[test]"          # core package + pytest
pip install -e ".[scripts,plot]"  # optional: cmfgpu, matplotlib for workflow scripts
```

Select the kernel backend before running (`triton` is the default on CUDA when unset; `cuda` uses native CUDA kernels via JIT compilation):

```bash
export HYDROFORGE_BACKEND=triton          # or cuda
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Quickstart on the committed tiny fixture

```bash
pytest -q
python -m scripts.s05_train configs/tiny.yaml
python -m scripts.s06_evaluate configs/tiny.yaml
```

---

## Package Layout

| Path | Contents |
|---|---|
| `hydronet/config.py` | `HydroNetOptions` and device status codes |
| `hydronet/models` | `HydroNetModel`: `set_inputs`, `set_parameters`, `step_advance`, and `step_adjoint` |
| `hydronet/modules` | `base` (forest topology, channel/floodplain geometry, roughness, state, forcing, Newton record and adjoints), optional `bifurcation` (multi-level pathways, recurrent discharge, and adjoints), and pure GPU online topology helpers (`_forest.py`, `_bifurcation.py`) |
| `hydronet/phys` | Kernel ABIs (`specs.py`), launch plans (`implicit_inertia.py`, `bifurcation.py`), native CUDA kernels (`cuda/`), and pure `@triton.jit` kernels (`triton/`) |
| `configs/` | Global 15′ templates (`glb_15min.yaml`, `lumped_glb_15min.yaml`, `gnn_glb_15min.yaml`) and the tiny test fixture config (`tiny.yaml`) |
| `tests/` | CUDA/Triton parity, conservation, adjoint, baseline, and end-to-end training tests |

---

## Workflow Scripts (`scripts/`)

Run commands from the repository root (`python -m scripts.<module>`). Shared training utilities live in `scripts/training/`.

| Step | Script | Purpose |
|---|---|---|
| **1** | `scripts/s01_prepare_gauges.py` | Allocate GRDC daily discharge archives onto the CaMa-Flood map (`gauges.nc`) |
| **2** | `scripts/s02_assign_folds.py` | Assign whole river groups to `k` spatially balanced cross-validation folds |
| **3** | `scripts/s03_run_cama_spinup.py` | Spin up CaMa-Flood-GPU to generate the initial storage/outflow restart (`cama_restart.nc`) |
| **4** | `scripts/s04_prepare_lumped.py` | Aggregate area-weighted upstream forcing and attributes for the lumped LSTM baseline |
| **5** | `scripts/s05_train.py` | Train HydroNet (runoff LSTM two-way coupled to the implicit river model) |
| **6** | `scripts/s06_evaluate.py` | Replay trained HydroNet continuously over the full period (`results.json`, `daily_rank*.npz`) |
| **7** | `scripts/s07_lumped_baseline.py` | Train and evaluate the lumped LSTM baseline (`train` / `evaluate`) |
| **8** | `scripts/s08_gnn_baseline.py` | Train and evaluate the recurrent graph neural network baseline (`train` / `evaluate`) |
| **9** | `scripts/s09_crossval.py` | Combine `k`-fold runs (single-seed or multi-seed `--ensemble`) into space-only, time-only, and space-time metrics & CDF plots |
| **10** | `scripts/s10_benchmark.py` | Measure forward/adjoint step timing and compare bitwise reference outputs across resolutions |

### Step-by-step usage

```bash
# Steps 1–2: gauges and 5-fold whole-river spatial split (reads CaMa-Flood parameters.nc directly)
python -m scripts.s01_prepare_gauges GRDC_DIR CAMA_MAP_DIR inp/glb_15min/parameters.nc inp/glb_15min/gauges.nc
python -m scripts.s02_assign_folds inp/glb_15min/gauges.nc inp/glb_15min/parameters.nc --folds 5

# Step 3: CaMa-Flood-GPU spin-up restart (add --bifurcation when training with bifurcation paths)
python -m scripts.s03_run_cama_spinup --params inp/glb_15min/parameters.nc \
    --runoff-dir RUNOFF_DIR --runoff-mapping cama/runoff_mapping_bin.npz \
    --start 1979-01-01 --end 1979-12-31 --output out/cama_spinup

# Step 4 (for lumped baseline): upstream area-weighted inputs
python -m scripts.s04_prepare_lumped inp/glb_15min/parameters.nc inp/glb_15min/gauges.nc \
    inp/glb_15min/era5 inp/glb_15min/attributes.nc inp/glb_15min/lumped

# Steps 5–6: train and evaluate HydroNet (one run per held-out fold K and seed S)
python -m scripts.s05_train configs/glb_15min.yaml --check-config
torchrun --standalone --nproc-per-node=4 -m scripts.s05_train configs/glb_15min.yaml \
    --set gauges.heldout_fold=0 --set training.seed=42 --set paths.checkpoints_dir=out/fold0-s42
torchrun --standalone --nproc-per-node=4 -m scripts.s06_evaluate configs/glb_15min.yaml \
    --set gauges.heldout_fold=0 --set training.seed=42 --set paths.checkpoints_dir=out/fold0-s42

# Steps 7–8: neural baselines (same protocol and output format)
python -m scripts.s07_lumped_baseline train configs/lumped_glb_15min.yaml --set gauges.heldout_fold=0
python -m scripts.s07_lumped_baseline evaluate configs/lumped_glb_15min.yaml --set gauges.heldout_fold=0
torchrun --standalone --nproc-per-node=4 -m scripts.s08_gnn_baseline train configs/gnn_glb_15min.yaml --set gauges.heldout_fold=0
torchrun --standalone --nproc-per-node=4 -m scripts.s08_gnn_baseline evaluate configs/gnn_glb_15min.yaml --set gauges.heldout_fold=0

# Step 9: combine folds (single seed, or multi-seed daily-discharge ensemble)
python -m scripts.s09_crossval out/fold0 out/fold1 out/fold2 out/fold3 out/fold4 --out crossval.json --plot cdf.png
python -m scripts.s09_crossval --ensemble --gauges inp/glb_15min/gauges.nc \
    --params inp/glb_15min/parameters.nc out/fold*-s42 out/fold*-s43 out/fold*-s44 --out crossval_s3.json

# Step 10: solver timing benchmark
python -m scripts.s10_benchmark warm --case glb15 --maps MAP_ROOT --runoff RUNOFF_DIR
python -m scripts.s10_benchmark time --case glb15 --tag triton --adjoint-days 30 --maps MAP_ROOT --runoff RUNOFF_DIR
```

Resume training with `--resume OUT/last.pt` in the same output directory; the checkpoint's validated configuration must match the current one. Paths in YAML configurations are resolved relative to the repository root unless absolute.

---

## Model Formulation

### Implicit local-inertia solver and adjoint

- **State and equations**: Each daily step solves continuity and implicit local-inertial momentum for the end-of-step water-surface elevation of every unit catchment, using CaMa-Flood channel geometry and piecewise-linear floodplain conveyance summed over flood-table bands. Backwater and reverse flow follow directly from the water-surface slope.
- **Nonlinear solve**: Each river tree runs Newton's method with Armijo line search. If line search stalls, the tree restarts with fixed residual weights and, if needed, switches to switched-evolution-relaxation damped Newton `(J + sigma D) dh = -F` until the full-step continuity residual drops below `newton_tolerance`. No physical sub-stepping is used.
- **Direct tree linear solver**: Heavy-path elimination solves each river tree directly in at most `log2(n) + 1` rounds. Small trees run one thread block per tree; large trees (`>= grid_tree_size` catchments holding at least half the domain) span all device blocks with serial, warp cyclic-reduction, and block cyclic-reduction paths.
- **Exact implicit adjoint**: Each forward step stores only the converged water surface (`base.converged_surface`). `step_adjoint` rebuilds the converged Jacobian at that surface and performs one transposed tree/graph solve to propagate gradients to runoff, start-of-step storage and outflow, Manning roughness, and channel geometry. Arithmetic is FP32 with compensated pairs in sensitive differences.
- **Error handling**: Non-convergence or invalid inputs are flagged on device in `base.newton_status` and raise `DeviceStatusError` without per-step host synchronization.

```python
from datetime import timedelta
import torch

model.set_parameters(river_manning=n_channel, flood_manning=n_flood)
model.assign_state({"base.storage": s0, "base.outflow": q0})
model.set_inputs(runoff=r)
model.step_advance(time_step=timedelta(days=1))
surface = model.capture_state(("base.converged_surface",))

# Reverse step from the end of that day back to its start:
model.zero_parameter_grads()
model.set_inputs(runoff=r)
model.assign_state({
    "base.storage": s0,
    "base.outflow": q0,
    **surface,
    "base.grad_outflow": gq,
    "base.grad_storage": gs,
    "base.grad_water_surface": torch.zeros_like(gq),
})
model.step_adjoint(time_step=timedelta(days=1))
g_runoff = model.base.grad_runoff
```

### Coupled bifurcation routing

Enable multi-level bifurcation pathways with `opened_modules=("base", "bifurcation")` (or `model.bifurcation: true` in training configs). Pure-forest components are solved directly by the forest kernels while bifurcation-coupled components are solved with a condensed static sparse LU preconditioner and BiCGStab in compensated FP32, certified by a true residual audit for both forward and transposed adjoint solves. Bifurcation training requires a restart containing `bifurcation_path_id` and `bifurcation_outflow` (generated via `scripts.s03_run_cama_spinup --bifurcation`).

### Coupled runoff LSTM and training inputs

- **Runoff frontend**: A shared single-layer LSTM (256 hidden units) reads daily forcing, seasonal `sin`/`cos` of day-of-year, static catchment attributes, and start-of-day hydraulic state (log-depth storage and `asinh` specific outflow per upstream area, standardized by `frontend.hydraulic`). Nonnegative runoff is `R = sigma * softplus(z + b)` mm/day, initialized at each catchment's climatological prior mean.
- **Training protocol**: River groups are partitioned into `k` spatial folds; within each river group, pooled station records are split chronologically (`gauges.roles`, default 60/20/20 train/validation/test). Training reads only non-held-out folds, runs truncated BPTT over `chunk_days` routing windows with `lstm_memory_days` recurrent memory, and optimizes `(1 - w) * zMSE + w * (1 - r)` (`w = training.correlation_weight`).
- **Required NetCDF inputs** (`paths.*`):
  - `params`: CaMa-Flood `parameters.nc` (forest heavy-path decomposition and bifurcation LU plans are computed online at model initialization).
  - `cama_restart`: CaMa-Flood spin-up restart (`scripts.s03_run_cama_spinup`).
  - `forcing`: directory of `{variable}_{year}.nc` HydroForge `ExportedDataset` files (`(time, saved_points)` with `catchment_id`).
  - `attributes`: standardized static attributes plus `runoff_mean` and `runoff_std` (mm/day) on `catchment`.
  - `gauges`: `catchment_id(station)`, `discharge(time, station)` (m³/s, NaN missing), and `fold(station)`.

---

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
