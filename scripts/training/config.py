# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The training configuration: one schema, every key checked and defaults filled.

A resumed run compares the validated configuration of its checkpoint with the
current one, so a key deleted from either side means its default on both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from hydronet.config import HydroNetOptions

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str) -> str:
    return str(path) if Path(path).is_absolute() else str(PROJECT_ROOT / path)


FilePath = Annotated[str, AfterValidator(_resolve)]
Years = tuple[int, int]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LumpedPaths(Section):
    params: FilePath
    # Directory of ``{variable}_{year}.nc`` daily ERA5 inputs per catchment (data.Forcing).
    forcing: FilePath
    # Per-catchment LSTM attributes and the runoff prior (data.load_attributes).
    attributes: FilePath
    # Every station: catchment_id, discharge(time, station) and the cross-validation fold
    # (scripts.s01_prepare_gauges, scripts.s02_assign_folds).
    gauges: FilePath
    checkpoints_dir: FilePath


class Paths(LumpedPaths):
    cama_restart: FilePath


class GaugeSplit(Section):
    """The pooled valid days of each river group's stations, in time order, split into train,
    validation and test days by ``roles`` (one set of dates per group); the stations of ``heldout_fold`` (whole river groups) are not trained
    on, nor read by training (none: train on every station)."""

    roles: tuple[float, float, float] = (0.6, 0.2, 0.2)
    heldout_fold: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _roles(self):
        if min(self.roles) < 0 or abs(sum(self.roles) - 1) > 1e-9:
            raise ValueError("gauges.roles are three nonnegative fractions summing to 1")
        return self


class ForcingInputs(Section):
    """Daily forcing variables in LSTM input order, standardized by fixed training-period
    statistics; the ``log1p`` variables are log1p-transformed first."""

    variables: list[str]
    log1p: list[str] = []
    mean: list[float]
    std: list[float]

    @model_validator(mode="after")
    def _shapes(self):
        n = len(self.variables)
        if len(self.mean) != n or len(self.std) != n or min(self.std) <= 0:
            raise ValueError("frontend.forcing needs a mean and a positive std per variable")
        if not set(self.log1p) <= set(self.variables):
            raise ValueError("frontend.forcing.log1p lists variables outside variables")
        return self


class HydraulicInputs(Section):
    """Fixed standardization of the start-of-day storage and outflow LSTM inputs."""

    mean: tuple[float, float]
    std: tuple[float, float]


class LumpedFrontend(Section):
    hidden: int = Field(256, ge=1)
    spatial_checkpoint_cells: int = Field(4096, ge=1)
    temporal_checkpoint_days: int = Field(0, ge=0)
    attributes: list[str]
    forcing: ForcingInputs


class Frontend(LumpedFrontend):
    hydraulic: HydraulicInputs


class Training(Section):
    epochs: int = Field(ge=1)
    seed: int
    lr: float = Field(gt=0)
    lr_min: float = Field(ge=0)
    grad_clip: float = Field(gt=0)
    warmup_days: int = Field(ge=0)
    # Routed days after the warm-up of every river group per epoch: its training days, and
    # in validation its validation days (scripts.training.windows).
    window_days: int = Field(ge=1)
    validation_days: int = Field(ge=1)
    chunk_days: int = Field(ge=1)
    lstm_memory_days: int
    correlation_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _memory(self):
        if self.lstm_memory_days < self.chunk_days:
            raise ValueError("training.lstm_memory_days must cover a window")
        return self


class LumpedConfig(Section):
    """``scripts.s07_lumped_baseline``: the LSTM alone on each station's upstream inputs, no routing."""

    paths: LumpedPaths
    # Whole years of forcing and gauges; evaluation routes all of them.
    period: Years
    gauges: GaugeSplit = GaugeSplit()
    frontend: LumpedFrontend
    training: Training

    @model_validator(mode="after")
    def _windows(self):
        years = self.period[1] - self.period[0] + 1
        longest = self.training.warmup_days + max(self.training.window_days, self.training.validation_days)
        if longest > 365 * years:
            raise ValueError("training.warmup_days plus a window exceed the period")
        return self


class Graph(Section):
    """``scripts.s08_gnn_baseline``: width of the graph state and upstream hops a day."""

    hidden: int = Field(64, ge=1)
    hops: int = Field(8, ge=1)


class GNNConfig(LumpedConfig):
    """``scripts.s08_gnn_baseline``: the LSTM encoder on the catchment inputs and a recurrent graph network
    on the river network instead of HydroNet's routing."""

    graph: Graph = Graph()


class Config(LumpedConfig):
    paths: Paths
    frontend: Frontend
    model: HydroNetOptions = HydroNetOptions()


def _override(cfg: dict, spec: str) -> None:
    key, value = spec.split("=", 1)
    *parents, last = [part.strip() for part in key.split(".") if part.strip()]
    for part in parents:
        cfg = cfg.setdefault(part, {})
    cfg[last] = yaml.safe_load(value)


def add_config_override_args(parser):
    """Add repeated ``--set KEY=VALUE`` overrides (VALUE parsed as YAML)."""
    parser.add_argument("--set", dest="config_overrides", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="Override a YAML value after loading, e.g. --set paths.checkpoints_dir=out/x")
    return parser


def load_config(path, overrides=(), schema=None):
    """The validated configuration (``Config`` by default) of a YAML file after ``--set``
    overrides."""
    cfg = yaml.safe_load(Path(path).read_text())
    for spec in overrides:
        _override(cfg, spec)
    return (schema or Config).model_validate(cfg)


def _differences(a, b, prefix=""):
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [] if a == b else [prefix.rstrip(".")]
    return [key for k in sorted(set(a) | set(b))
            for key in _differences(a.get(k), b.get(k), f"{prefix}{k}.")]


def training_differences(saved, cfg) -> list[str]:
    """Keys whose training meaning differs between a checkpoint's configuration and ``cfg``.

    Both are validated first, so a key deleted on either side takes its default on both.
    """

    def view(config):
        return type(cfg).model_validate(config).model_dump(mode="json")

    return _differences(view(saved), view(cfg))
