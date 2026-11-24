"""Forcing/static LSTM runoff with a nonnegative head."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class RunoffLSTM(nn.Module):
    """Daily local runoff (mm/day) of every unit.

    The LSTM reads ``inputs`` daily features (forcing and the hydraulic state)
    and the static attributes through a linear projection, LayerNorm and tanh.
    Runoff is ``sigma * softplus(head + b)`` with the global runoff prior: the
    scale ``sigma`` is its standard deviation and ``b`` places a zero head at its
    mean (``frontend.runoff_prior``, mm/day, the same for every unit).
    Checkpoint tiles bound the fused cuDNN reserve: ``spatial_checkpoint_cells``
    units at a time and, with ``temporal_checkpoint_days``, that many days of
    their sequences at a time (the recurrent state carries across tiles, so
    the recurrence is unchanged; zero runs whole sequences).
    """

    def __init__(
        self,
        d_static,
        prior_std,
        prior_mean,
        *,
        hidden=256,
        spatial_checkpoint_cells=2048,
        temporal_checkpoint_days=0,
        inputs,
        features=None,
    ):
        super().__init__()
        self.hidden, self.block_cells = hidden, spatial_checkpoint_cells
        self.block_days = temporal_checkpoint_days
        if self.block_cells < 1:
            raise ValueError("LSTM spatial checkpoint size must be positive")
        if self.block_days < 0:
            raise ValueError("LSTM temporal checkpoint size must be nonnegative")
        if not (prior_std > 0 and prior_mean > 0):
            raise ValueError("The runoff prior needs a positive mean and std")
        bias = math.log(math.expm1(2.0))
        self.prior_std_mm_day = float(prior_std)
        # softplus(offset + bias) * std = mean for a zero head.
        self.head_offset = math.log(math.expm1(prior_mean / prior_std)) - bias
        self.projection = nn.Sequential(nn.Linear(inputs + d_static, hidden), nn.LayerNorm(hidden),
                                        nn.Tanh())
        self.lstm = nn.LSTM(hidden, hidden)
        # With ``features`` the head gives that many features per unit and day instead of
        # runoff (an encoder, e.g. for scripts.s07_gnn_baseline).
        self.features = features
        self.head = nn.Linear(hidden, features or 1)
        if features is None:
            nn.init.normal_(self.head.weight, std=0.01)
            nn.init.constant_(self.head.bias, bias)

    def _block(self, forcing, t0, t1, a, b, static, h, c, scale, offset):
        # Sliced here so a checkpoint keeps only the full forcing, not a copy of its tile.
        forcing = forcing[t0:t1, a:b]
        features = torch.cat([forcing, static[None].expand(len(forcing), -1, -1)], -1)
        y, (h, c) = self.lstm(self.projection(features), (h, c))
        if self.features:
            return self.head(y), h, c
        return torch.nn.functional.softplus(self.head(y).squeeze(-1) + offset) * scale, h, c

    def forward(self, forcing, static, hidden, cell):
        days = len(forcing)
        span = self.block_days or days
        rates, hs, cs = [], [], []
        for a in range(0, forcing.shape[1], self.block_cells):
            b = min(a + self.block_cells, forcing.shape[1])
            h, c = hidden[:, a:b].contiguous(), cell[:, a:b].contiguous()
            tiles = []
            for t0 in range(0, days, span):
                args = (forcing, t0, min(t0 + span, days), a, b, static[a:b], h, c,
                        self.prior_std_mm_day, self.head_offset)
                r, h, c = (
                    checkpoint(self._block, *args, use_reentrant=False)
                    if torch.is_grad_enabled()
                    else self._block(*args)
                )
                tiles.append(r)
            rates.append(tiles[0] if len(tiles) == 1 else torch.cat(tiles, 0))
            hs.append(h)
            cs.append(c)
        return torch.cat(rates, 1), torch.cat(hs, 1), torch.cat(cs, 1)
