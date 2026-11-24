"""Calendar windows of the river groups: every group routes its own stretch of days.

Groups route independently (``domain.river_groups``), so in one routed day
every group may sit at a different calendar day. A phase routes ``warmup +
length`` days; each group starts ``warmup`` days before its loss window, which
covers as much of its stations' role days as ``length`` allows. Longer role
spans are cut (training draws the window anew every epoch, so all of them are
used over the epochs); shorter ones are followed by days without observations.
The cost of an epoch is then ``warmup + length`` days, whatever the spread of
the stations' records over the calendar.
"""

from __future__ import annotations

import numpy as np


def group_windows(groups, lo, hi, valid, length, warmup, total, *, seed=None):
    """First routed calendar day of every group, ``{group: day}``.

    ``groups`` (S,) is each station's group and ``lo``/``hi`` (S,) its role's
    day span; ``valid`` (total, S) marks its valid role days. Without ``seed``
    the loss window holds the most role days of the group's stations; with
    ``seed = (seed, epoch)`` it starts at a uniformly drawn day of the span
    (per group, so the draw does not depend on the ranks' layout).
    """
    starts = {}
    counts = np.cumsum(np.r_[np.zeros((1, valid.shape[1]), np.int64), valid], 0)
    for group in np.unique(groups).tolist():
        mine = np.flatnonzero(groups == group)
        first, last = int(lo[mine].min()), int(hi[mine].max())
        if last - first <= length:
            start = first
        elif seed is None:
            # Role days of the group's stations in [s, s + length) for every start s.
            inside = counts[:, mine].sum(1)
            candidates = np.arange(first, last - length + 1)
            start = int(candidates[np.argmax(inside[candidates + length] - inside[candidates])])
        else:
            start = int(np.random.default_rng([*seed, group]).integers(first, last - length + 1))
        starts[group] = int(np.clip(start - warmup, 0, total - warmup - length))
    return starts


def unit_shift(starts, unit_groups):
    """The first routed day of every unit, from its group's window."""
    return np.array([starts[int(g)] for g in unit_groups], np.int64)
