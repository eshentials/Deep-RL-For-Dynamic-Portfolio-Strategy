"""Baseline policies on the shared engine (protocol Section 6).

A policy is `policy(sim, t) -> (exposure, horizon)`. Year-dependent policies take a
`params_for_year(year)` callable, so each test year uses the parameters its own fold
selected on validation. The year that a decision at close t governs is the year of
session t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from env.engine import MarketData

ANN = np.sqrt(252.0)


def governed_year(data: MarketData, t: int) -> int:
    return int(data.dates[min(t + 1, len(data.dates) - 1)].year)


def fixed(e: float, h: int):
    return lambda sim, t: (e, h)


def rp_vol(data: MarketData, t: int, window: int) -> float:
    r = data.rp_ret[t - window + 1:t + 1]
    return float(np.std(r, ddof=1) * ANN)


def vol_target(data: MarketData, target: float, horizon: int, floor: float, cap: float = 1.0,
               window: int = 60):
    def pol(sim, t):
        s = rp_vol(data, t, window)
        e = cap if s <= 0 else float(np.clip(target / s, floor, cap))
        return e, horizon
    return pol


def per_year(data: MarketData, make_policy, params_for_year):
    """Policy that rebuilds itself with the parameters selected for each test year."""
    cache: dict[int, object] = {}

    def pol(sim, t):
        y = governed_year(data, t)
        if y not in cache:
            cache[y] = make_policy(**params_for_year(y))
        return cache[y](sim, t)
    return pol


def random_policy(exposures, horizons, seed: int):
    rng = np.random.default_rng(seed)
    return lambda sim, t: (float(rng.choice(exposures)), int(rng.choice(horizons)))


def month_end_horizon(data: MarketData):
    """Rebalance at each month-end close: horizon = sessions until the next month-end."""
    m = data.dates.month.to_numpy()
    is_me = np.zeros(len(m), bool)
    is_me[:-1] = m[1:] != m[:-1]
    is_me[-1] = True
    nxt = np.full(len(m), len(m) - 1)
    last = len(m) - 1
    for t in range(len(m) - 1, -1, -1):
        nxt[t] = last
        if is_me[t]:
            last = t

    def pol(sim, t):
        return 1.0, max(1, int(nxt[t] - t))
    return pol


def benchmark_buy_and_hold(bench_tr: pd.Series, data: MarketData, start: int, end: int,
                           cost_bps: float) -> np.ndarray:
    """Daily returns of buying the benchmark at close `start` (one entry cost) and holding."""
    lv = bench_tr.reindex(data.dates).to_numpy()
    r = lv[start + 1:end + 1] / lv[start:end] - 1.0
    r = r.copy()
    r[0] = (1 + r[0]) * (1 - cost_bps / 1e4) - 1.0
    return r
