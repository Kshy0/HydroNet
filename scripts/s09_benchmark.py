"""Step 9: timing and bitwise reference outputs of HydroNet's daily implicit step (FP32).

    python -m scripts.s09_benchmark warm    --case amazon03 --maps MAP_ROOT --runoff RUNOFF_DIR
    python -m scripts.s09_benchmark time    --case glb15 --tag mine --maps MAP_ROOT --runoff RUNOFF_DIR
    python -m scripts.s09_benchmark golden  --case glb15 --tag mine --maps MAP_ROOT --runoff RUNOFF_DIR
    python -m scripts.s09_benchmark compare --case glb15 --tag mine --other base

Cases: ``amazon15``/``amazon06``/``amazon03`` (Amazon cropped from the global 15'/6'/3'
CaMa-Flood maps: 7,767 / 48,539 / 194,115 catchments) and ``glb15``/``glb06`` (global
15'/6' maps: 252,383 / 1,562,463 catchments). Forcing is 1-degree daily-bin runoff of
2000 mapped with each map's area weights. ``--roughness default`` uses CaMa-Flood
defaults (0.03 channel, 0.10 floodplain); ``--roughness random`` draws a fixed seeded
field inside the physical bounds.

Set ``--maps`` / ``HYDRONET_MAPS`` to the directory holding ``glb_{15,06,03}min/{parameters.nc,
runoff_mapping_bin.npz}``, ``--runoff`` / ``HYDRONET_RUNOFF`` to the daily-bin runoff
directory, and ``--bench-dir`` / ``HYDRONET_BENCH_DIR`` (default ``out/benchmark``) for
cropped maps, warm states, timings, and golden references.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

MOUTHS = {15: 372601, 6: 2326504, 3: 9307808}
CASES = {
    "amazon15": (15, True),
    "amazon06": (6, True),
    "amazon03": (3, True),
    "glb15": (15, False),
    "glb06": (6, False),
}


def _require_path(value: str | None, env_var: str, label: str) -> Path:
    resolved = value or os.environ.get(env_var)
    if not resolved:
        raise ValueError(f"{label} is required: pass its CLI flag or set {env_var}")
    return Path(resolved)


def cama_map(case: str, *, root: Path, maps_dir: Path) -> Path:
    """The CaMa-Flood map of a case: a global map, or the Amazon cropped from it."""
    res, crop = CASES[case]
    if not crop:
        return maps_dir / f"glb_{res:02d}min/parameters.nc"
    target = root / "maps" / f"amazon_{res:02d}min.nc"
    if not target.exists():
        from cmfgpu.params.utils import crop_parameters_nc

        target.parent.mkdir(parents=True, exist_ok=True)
        crop_parameters_nc(
            maps_dir / f"glb_{res:02d}min/parameters.nc",
            target,
            {"catchments": [MOUTHS[res]]},
            only_save_pois=True,
            split_bif_basins=True,
        )
    return target


def parameters(case: str, *, root: Path, maps_dir: Path) -> Path:
    """The CaMa-Flood parameter file of a case (HydroNet computes solver plans online)."""
    return cama_map(case, root=root, maps_dir=maps_dir)


def forcing(case: str, ids, days: int, *, maps_dir: Path, runoff_dir: Path) -> torch.Tensor:
    from hydroforge.data.datasets import DailyBinDataset
    from torch.utils.data import DataLoader

    res = CASES[case][0]
    ds = DailyBinDataset(
        base_dir=str(runoff_dir),
        shape=(180, 360),
        start_date=datetime(2000, 1, 1),
        end_date=datetime(2000, 12, 31),
        time_interval=timedelta(days=1),
        model_step=timedelta(days=1),
        source_units="mm day-1",
        target_units="m s-1",
        bin_dtype="float32",
        prefix="Roff____",
        suffix=".one",
        lat_south_to_north=False,
        lon_0_to_360=False,
    )
    local, mapping = ds.build_local_mapping(
        maps_dir / f"glb_{res:02d}min/runoff_mapping_bin.npz", ids, device=torch.device("cuda")
    )
    blocks = []
    for chunk in DataLoader(local, batch_size=None, num_workers=0):
        blocks.append(local.shard_forcing(chunk.to("cuda"), mapping))
        if sum(len(b) for b in blocks) >= days:
            break
    return torch.cat(blocks)[:days].contiguous()


class Case:
    """The differentiable model of a case and its one-day step."""

    def __init__(self, case: str, *, root: Path, maps_dir: Path, differentiable=True, bifurcation=False):
        from hydronet.config import HydroNetOptions
        from scripts.training.model import build_model

        options = HydroNetOptions(differentiable=differentiable, bifurcation=bifurcation).model_dump()
        self.model, self.proxy = build_model(
            parameters(case, root=root, maps_dir=maps_dir), torch.device("cuda"), options=options
        )
        self.ids = self.model.base.catchment_id.cpu().numpy()
        bif = self.model.bifurcation
        self.has_bifurcation = bif is not None and bif.num_bifurcation_paths > 0

    def step(self, s, r, q, qb=None):
        """(Q, S[, Qb]) after one day; differentiable as in the training coupling."""
        from scripts.training.coupling import DAY, PhysicsStep

        q, s, qb_out = PhysicsStep.apply(self.model, DAY, s, q, r, qb)
        return (q, s, qb_out) if qb is not None else (q, s)

    @property
    def record(self):
        return self.model.base.newton_record

    def close(self):
        self.model.close()
        self.proxy.close()


def roughness(ids, mode: str = "default"):
    if mode == "default":
        return (
            torch.full((len(ids),), 0.03, device="cuda", dtype=torch.float32),
            torch.full((len(ids),), 0.10, device="cuda", dtype=torch.float32),
        )
    rng = np.random.default_rng(20261114)
    u = rng.random((len(ids), 2)).astype(np.float32)
    lo, hi = np.log([0.025, 0.05]), np.log([0.06, 0.15])
    n = np.exp(lo + (hi - lo) * u).astype(np.float32)
    n = torch.tensor(n, device="cuda")
    return n[:, 0].contiguous(), n[:, 1].contiguous()


def warm_path(root: Path, case: str, *, bifurcation: bool = False) -> Path:
    suffix = "_bif" if bifurcation else ""
    return root / f"warm_{case}{suffix}.npz"


def load_warm(root: Path, case: str, ids, *, bifurcation: bool = False):
    z = np.load(warm_path(root, case, bifurcation=bifurcation))
    np.testing.assert_array_equal(z["ids"], ids)
    s = torch.tensor(z["s"], device="cuda")
    q = torch.tensor(z["q"], device="cuda")
    qb = torch.tensor(z["qb"], device="cuda") if bifurcation else None
    return (s, q, qb) if bifurcation else (s, q)


@torch.no_grad()
def warm(case: str, *, root: Path, maps_dir: Path, runoff_dir: Path, days=30,
         roughness_mode="default", bifurcation=False):
    """Spin up 30 days from CaMa-Flood's initial state of the map."""
    from cmfgpu.models import CaMaFlood
    from hydroforge.contracts import SimulationSchedule
    from hydroforge.data import InputProxy
    from hydroforge.model import OutputConfig

    bench = Case(case, root=root, maps_dir=maps_dir, differentiable=False, bifurcation=bifurcation)
    ids = bench.ids
    proxy = InputProxy.from_nc(cama_map(case, root=root, maps_dir=maps_dir), lazy=False)
    modules = ("base", "bifurcation") if bifurcation else ("base",)
    native = CaMaFlood(
        device=torch.device("cuda"),
        input_proxy=proxy,
        opened_modules=modules,
        simulation_schedule=SimulationSchedule.regular(
            start=datetime(2000, 1, 1), end=datetime(2000, 1, 2), step=timedelta(days=1)
        ),
        precision="float32",
        mixed_precision=False,
        output=OutputConfig(sink="memory", workers=0, variables={}),
    )
    native.materialize()
    np.testing.assert_array_equal(native.base.catchment_id.cpu().numpy(), ids)
    b = native.base
    s = (b.river_storage + b.flood_storage).float().contiguous()
    q = (b.river_outflow + b.flood_outflow).float().contiguous()
    qb = native.bifurcation.bifurcation_outflow.float().contiguous() if bifurcation else None
    native.close()
    proxy.close()
    r = forcing(case, ids, days, maps_dir=maps_dir, runoff_dir=runoff_dir)
    bench.model.set_parameters(**dict(zip(("river_manning", "flood_manning"), roughness(ids, roughness_mode))))
    for d in range(days):
        if bifurcation:
            q, s, qb = bench.step(s, r[d], q, qb)
        else:
            q, s = bench.step(s, r[d], q)
    payload = dict(ids=ids, s=s.cpu().numpy(), q=q.cpu().numpy())
    if bifurcation:
        payload["qb"] = qb.cpu().numpy()
    np.savez(warm_path(root, case, bifurcation=bifurcation), **payload)
    print("WARM", case, len(ids), float(s.sum()), flush=True)
    bench.close()


def golden(case: str, tag: str, *, root: Path, maps_dir: Path, runoff_dir: Path, days=30,
           grad_days=5, roughness_mode="random"):
    bench = Case(case, root=root, maps_dir=maps_dir)
    ids = bench.ids
    s, q = load_warm(root, case, ids)
    r = forcing(case, ids, 30 + days, maps_dir=maps_dir, runoff_dir=runoff_dir)[30:]
    bench.model.set_parameters(**dict(zip(("river_manning", "flood_manning"), roughness(ids, roughness_mode))))
    out = dict(q=[], s=[], meta=[])
    with torch.no_grad():
        for d in range(days):
            q, s = bench.step(s, r[d], q)
            out["q"].append(q.cpu().numpy())
            out["s"].append(s.cpu().numpy())
            out["meta"].append(bench.record.cpu().numpy())
    s0, q0 = load_warm(root, case, ids)
    bench.model.reset_clock()
    leaves = [t.clone().requires_grad_(True) for t in (s0, r[0], q0)]
    S, R, Q = leaves
    bench.model.zero_parameter_grads()
    g = torch.Generator(device="cuda").manual_seed(7)
    wq = torch.randn(len(ids), device="cuda", generator=g) / len(ids)
    ws = torch.randn(len(ids), device="cuda", generator=g) / len(ids)
    loss = 0.0
    for d in range(grad_days):
        Q, S = bench.step(S, R if d == 0 else r[d], Q)
        loss = loss + (wq * Q).sum() + (ws * S).sum() * 1e-6
    loss.backward()
    grads = {f"g_{k}": t.grad.cpu().numpy() for k, t in zip(("s0", "r0", "q0"), leaves)}
    base = bench.model.base
    grads.update(g_nc=base.grad_river_manning.cpu().numpy(), g_nf=base.grad_flood_manning.cpu().numpy())
    path = root / f"golden_{case}_{tag}.npz"
    np.savez_compressed(path, ids=ids, loss=float(loss), **{k: np.stack(v) for k, v in out.items()}, **grads)
    print("GOLDEN", case, tag, path, float(loss), flush=True)
    bench.close()


def compare(root: Path, case: str, a: str, b: str):
    x, y = np.load(root / f"golden_{case}_{a}.npz"), np.load(root / f"golden_{case}_{b}.npz")
    rows = {}
    for k in x.files:
        if k == "ids":
            continue
        u, v = x[k], y[k]
        if k == "meta":
            n = min(u.shape[-1], v.shape[-1])
            u, v = u[..., :n], v[..., :n]
        diff = np.abs(u.astype(np.float64) - v.astype(np.float64))
        scale = np.abs(u).max() if u.size else 0
        rows[k] = dict(
            bitwise=bool(np.array_equal(u, v)),
            max_abs=float(diff.max()) if diff.size else 0.0,
            rel_to_max=float(diff.max() / scale) if scale else 0.0,
        )
    print(json.dumps({"case": case, "a": a, "b": b, **rows}, indent=1))


def timing(case: str, tag: str, *, root: Path, maps_dir: Path, runoff_dir: Path, days=30,
           repeats=3, adjoint_days=0, roughness_mode="default", bifurcation=False):
    bench = Case(case, root=root, maps_dir=maps_dir, differentiable=False, bifurcation=bifurcation)
    ids = bench.ids
    warm_state = load_warm(root, case, ids, bifurcation=bifurcation)
    r = forcing(case, ids, 30 + days + 1, maps_dir=maps_dir, runoff_dir=runoff_dir)[30:]
    bench.model.set_parameters(**dict(zip(("river_manning", "flood_manning"), roughness(ids, roughness_mode))))
    with torch.no_grad():
        if bifurcation:
            bench.step(warm_state[0], r[0], warm_state[1], warm_state[2])
        else:
            bench.step(warm_state[0], r[0], warm_state[1])
        seconds, per_day, meta = [], [], []
        for rep in range(repeats):
            s, q = warm_state[0].clone(), warm_state[1].clone()
            qb = warm_state[2].clone() if bifurcation else None
            torch.cuda.synchronize()
            tick = time.perf_counter()
            for d in range(days):
                if rep == 0:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                if bifurcation:
                    q, s, qb = bench.step(s, r[d + 1], q, qb)
                else:
                    q, s = bench.step(s, r[d + 1], q)
                if rep == 0:
                    torch.cuda.synchronize()
                    per_day.append(time.perf_counter() - t0)
                    m = bench.record
                    meta.append([float(m[:, 1].max()), float(m[:, 6].max()), float(m[:, 13].max()),
                                 float(m[:, 1].sum()), float(m[:, 6].sum())])
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
    trees = int(bench.model.base.num_trees)
    bench.close()

    result = dict(
        case=case,
        tag=tag,
        bifurcation=bifurcation,
        roughness=roughness_mode,
        nodes=len(ids),
        trees=trees,
        days=days,
        ms_per_day_median_repeat=float(np.median(seconds)) * 1000 / days,
        ms_per_day_each_day=[x * 1000 for x in per_day],
        meta_cols="max_global_steps max_full_evals max_phase sum_global_steps sum_full_evals",
        meta=meta,
        device=torch.cuda.get_device_name(),
    )

    if adjoint_days > 0:
        torch.cuda.empty_cache()
        diff_bench = Case(case, root=root, maps_dir=maps_dir, differentiable=True, bifurcation=bifurcation)
        diff_bench.model.set_parameters(
            **dict(zip(("river_manning", "flood_manning"), roughness(ids, roughness_mode)))
        )
        fw, bw = [], []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            diff_bench.model.reset_clock()
            diff_bench.model.zero_parameter_grads()
            S = warm_state[0].clone().requires_grad_(True)
            Q = warm_state[1].clone()
            Qb = warm_state[2].clone() if bifurcation else None
            R = r[1 : adjoint_days + 1].clone().requires_grad_(True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            total_loss = 0.0
            for d in range(adjoint_days):
                if bifurcation:
                    Q, S, Qb = diff_bench.step(S, R[d], Q, Qb)
                else:
                    Q, S = diff_bench.step(S, R[d], Q)
                total_loss = total_loss + Q.sum() * 1e-6
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            total_loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            fw.append(t1 - t0)
            bw.append(t2 - t1)
        result.update(
            adjoint_days=adjoint_days,
            forward_diff_ms_per_day=float(np.median(fw)) * 1000 / adjoint_days,
            adjoint_ms_per_day=float(np.median(bw)) * 1000 / adjoint_days,
            peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20,
        )
        diff_bench.close()

    path = root / f"time_{case}_{tag}.json"
    path.write_text(json.dumps(result, indent=1) + "\n")
    print("TIME", case, tag, f"{result['ms_per_day_median_repeat']:.2f} ms/day", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["warm", "golden", "compare", "time"])
    p.add_argument("--case", required=True, choices=list(CASES))
    p.add_argument("--tag", default="base")
    p.add_argument("--other", default="base")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--adjoint-days", type=int, default=0,
                   help="time mode: also measure forward+adjoint ms/day and peak memory over this many days")
    p.add_argument("--roughness", choices=["default", "random"], default="default")
    p.add_argument("--bifurcation", action="store_true")
    p.add_argument("--maps", help="directory of glb_*min maps (or set HYDRONET_MAPS)")
    p.add_argument("--runoff", help="directory of 1-degree daily-bin runoff (or set HYDRONET_RUNOFF)")
    p.add_argument("--bench-dir", default=os.environ.get("HYDRONET_BENCH_DIR", "out/benchmark"))
    a = p.parse_args()
    root = Path(a.bench_dir)
    root.mkdir(parents=True, exist_ok=True)
    if a.mode == "compare":
        compare(root, a.case, a.other, a.tag)
        return
    maps_dir = _require_path(a.maps, "HYDRONET_MAPS", "--maps")
    runoff_dir = _require_path(a.runoff, "HYDRONET_RUNOFF", "--runoff")
    from hydroforge.parallel import setup_distributed

    setup_distributed(allowed_devices=("cuda",))
    torch.set_num_threads(2)
    if a.mode == "warm":
        warm(a.case, root=root, maps_dir=maps_dir, runoff_dir=runoff_dir,
             roughness_mode=a.roughness, bifurcation=a.bifurcation)
    elif a.mode == "golden":
        golden(a.case, a.tag, root=root, maps_dir=maps_dir, runoff_dir=runoff_dir,
               days=a.days, roughness_mode=a.roughness)
    else:
        timing(a.case, a.tag, root=root, maps_dir=maps_dir, runoff_dir=runoff_dir,
               days=a.days, adjoint_days=a.adjoint_days, roughness_mode=a.roughness,
               bifurcation=a.bifurcation)


if __name__ == "__main__":
    main()
