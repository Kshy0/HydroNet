"""HydroNetModel construction for the drivers."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import torch


def build_model(params_nc, device, *, options=None, output_cids=None, initial_time=None, keep=None):
    """Materialized FP32 ``HydroNetModel`` of a CaMa-Flood ``parameters.nc``.

    The model has no schedule: drivers advance it one day at a time
    (``step_advance(time_step=...)``) and restore its state for recomputation.
    ``output_cids`` replaces the map's saved points; ``keep`` (a catchment
    mask of whole trees, see scripts.training.domain) restricts the map.
    Returns the model and its input proxy; close both.
    """
    from hydroforge.core import cleanup_on_exit, find_indices_in
    from hydroforge.data import InputProxy
    from hydroforge.model import OutputConfig

    from hydronet.config import HydroNetOptions
    from hydronet.models import HydroNetModel

    proxy = InputProxy.from_nc(params_nc, lazy=True)
    releases = [proxy.close]
    with cleanup_on_exit("HydroNet model construction", releases):
        parsed_options = HydroNetOptions.model_validate(options or {})
        if keep is not None:
            from scripts.training.domain import restrict_proxy

            proxy = restrict_proxy(proxy, params_nc, keep, bifurcation=parsed_options.bifurcation)
        if output_cids is not None:
            selected = np.asarray(output_cids, dtype=np.int64)
            if (find_indices_in(selected, np.asarray(proxy["catchment_id"])) < 0).any():
                raise ValueError("output_cids contains catchments outside the map")
            proxy = proxy.updated(
                values={"output_catchment_id": selected},
                dimensions={name: selected.size for name in ("saved_points", "saved_catchment")
                            if name in proxy.dims},
            )
        opened = ("base", "bifurcation") if parsed_options.bifurcation else ("base",)
        model = HydroNetModel(
            input_proxy=proxy, device=device, opened_modules=opened,
            output=OutputConfig(variables={}, sink="memory"),
            initial_time=initial_time or datetime(2000, 1, 1),
            options=parsed_options,
            progress=False,
        )
        releases.append(model.close)
        model.materialize()
        releases.clear()
    return model, proxy


def routing_edges(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream and downstream catchment of every link of the forest."""
    down = model.base.downstream_idx.long()
    index = torch.arange(down.numel(), device=down.device)
    source = torch.nonzero((down >= 0) & (down != index)).squeeze(1)
    return source, down[source]


def check_newton_status(model) -> None:
    """Raise if any step since construction reported a solve failure.

    The model's device status surfaces on its own only at output windows and
    ``close``; a training driver without outputs checks it at chunk boundaries.
    """
    status = model.base.newton_status.cpu().tolist()
    if status[0]:
        raise RuntimeError(f"HydroNet step failed: code {status[1]} at index {status[2]} "
                           f"({status[0]} failures); see hydronet.config")
