"""Per-fold validation selection for the tuned baselines (protocol Section 6).

Each candidate runs on each validation year from cash; the score is the mean of the
two annualised Sharpe ratios, exactly as for PPO checkpoint selection.
"""
from __future__ import annotations

import itertools

import numpy as np

from baselines import rules
from config import load
from env.engine import MarketData, run
from eval.metrics import sharpe
from train.common import Fold


def _score(data: MarketData, policy_factory, fold: Fold, cost_bps: float) -> float:
    shs = []
    for (_, s, e) in fold.val:
        res = run(data, policy_factory(), s, e, cost_bps=cost_bps)
        shs.append(sharpe(res.returns, res.rf))
    return float(np.mean(shs))


def select_vol_target(data: MarketData, fold: Fold, floor: float) -> dict:
    vt = load()["baselines"]["vol_target"]
    cost = load()["portfolio"]["cost_bps"]
    grid = []
    for tgt, hz in itertools.product(vt["targets"], vt["horizons"]):
        def make(tgt=tgt, hz=hz):
            return rules.vol_target(data, tgt, hz, floor, vt["cap"], vt["vol_window"])
        grid.append(({"target": tgt, "horizon": hz, "floor": floor, "cap": vt["cap"],
                      "window": vt["vol_window"]}, _score(data, make, fold, cost)))
    best = max(grid, key=lambda g: g[1])
    return {"params": best[0], "val_sharpe": best[1], "grid": grid}


def select_best_fixed(data: MarketData, fold: Fold) -> dict:
    pf = load()["portfolio"]
    grid = []
    for e, h in itertools.product(pf["exposure_grid"], pf["horizon_grid"]):
        grid.append(({"e": e, "h": h}, _score(data, lambda e=e, h=h: rules.fixed(e, h), fold, pf["cost_bps"])))
    best = max(grid, key=lambda g: g[1])
    return {"params": best[0], "val_sharpe": best[1], "grid": grid}


def select_vix_rule(data: MarketData, fold: Fold) -> dict:
    """Protocol v2: threshold x horizon grid of the VIX rule, chosen on the fold's validation years."""
    vr = load()["baselines"]["vix_rule"]
    cost = load()["portfolio"]["cost_bps"]
    grid = []
    for th, hz in itertools.product(vr["thresholds"], vr["horizons"]):
        def make(th=th, hz=hz):
            return rules.vix_threshold(data, th, hz, vr["low"], vr["high"])
        grid.append(({"threshold": th, "horizon": hz, "low": vr["low"], "high": vr["high"]},
                     _score(data, make, fold, cost)))
    best = max(grid, key=lambda g: g[1])
    return {"params": best[0], "val_sharpe": best[1], "grid": grid}
