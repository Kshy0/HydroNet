"""Independent CPU64 diagnostics; no tensors here participate in training."""

import numpy as np


def mass_balance(src, dst, initial_storage, q, storage, runoff, *, limit, dt=86400.0, bifurcation=None):
    """Largest daily continuity residual relative to the Newton merit's scale; raises at ``limit``
    (callers pass the Newton tolerance with headroom for FP32 rounding of the stored state)."""
    old, q, storage, runoff = [
        v.detach().cpu().numpy().astype(float)
        for v in (initial_storage, q, storage, runoff)
    ]
    src, dst = src.cpu().numpy(), dst.cpu().numpy()
    incoming, absolute = np.zeros_like(q), np.zeros_like(q)
    for t in range(len(q)):
        incoming[t] = np.bincount(dst, weights=q[t, src], minlength=q.shape[1])
        absolute[t] = np.bincount(dst, weights=abs(q[t, src]), minlength=q.shape[1])
    bif_net = np.zeros_like(q)
    bif_absolute = np.zeros_like(q)
    if bifurcation is not None:
        bs, bd, bq = (v.detach().cpu().numpy() for v in bifurcation)
        bq = bq.astype(float).sum(axis=-1)
        for t in range(len(q)):
            bif_net[t] = (np.bincount(bs, weights=bq[t], minlength=q.shape[1])
                          - np.bincount(bd, weights=bq[t], minlength=q.shape[1]))
            bif_absolute[t] = (np.bincount(bs, weights=abs(bq[t]), minlength=q.shape[1])
                               + np.bincount(bd, weights=abs(bq[t]), minlength=q.shape[1]))
    previous = np.concatenate([old[None], storage[:-1]])
    residual = storage - previous - dt * (runoff + incoming - q - bif_net)
    scale = previous + dt * (abs(runoff) + absolute + abs(q) + bif_absolute) + 1000
    value = float(np.max(abs(residual) / scale))
    if (
        value >= limit
        or not np.isfinite(q).all()
        or not np.isfinite(storage).all()
        or storage.min() < 0
    ):
        raise RuntimeError(
            f"Independent mass/convergence audit failed: residual={value:.8g}"
        )
    return value


def gauge_metrics(prediction, observation, cids):
    """NSE, KGE and percent bias of every gauge with at least 30 observed days.

    Every observed day must have a finite prediction (a model that fails
    somewhere is not scored on its other days)."""
    unpredicted = np.isfinite(observation) & ~np.isfinite(prediction)
    if unpredicted.any():
        bad = np.flatnonzero(unpredicted.any(0))
        raise ValueError(f"{int(unpredicted.sum())} observed days of {len(bad)} gauges have no finite "
                         f"prediction, e.g. gauge {int(np.asarray(cids)[bad[0]])}")
    rows = []
    for j, cid in enumerate(cids):
        valid = np.isfinite(observation[:, j])
        p, o = prediction[valid, j].astype(float), observation[valid, j].astype(float)
        if len(o) < 30 or np.var(o) == 0 or o.mean() == 0:
            continue
        nse = 1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2)
        corr = np.corrcoef(p, o)[0, 1] if np.std(p) > 0 else 0.0
        # KGE of Kling et al. (2012): variability as the ratio of coefficients of variation.
        # (A simulation with zero mean has no coefficient of variation; it counts as gamma 0.)
        gamma = (np.std(p) / p.mean()) / (np.std(o) / o.mean()) if p.mean() != 0 else 0.0
        kge = 1 - np.sqrt((corr - 1) ** 2 + (gamma - 1) ** 2 + (p.mean() / o.mean() - 1) ** 2)
        rows.append(
            dict(
                cid=int(cid),
                n=int(valid.sum()),
                nse=float(nse),
                kge=float(kge),
                pbias_percent=float(100 * (p.sum() - o.sum()) / o.sum()),
            )
        )
    return rows
