"""Assemble engine arrays from a returns panel, a membership mask and market series.

Timing rules (protocol Sections 2-3):
- eligible[t, i]: stock i is a member for session t+1 (NSE announces changes weeks
  ahead and index funds trade at the close of t), its price exists at t, and its
  trailing `window` sessions have at most `max_missing` missing returns.
- exit_mask[t, i]: i is a member on t and either leaves the index at t+1 (last member
  session) or has no price after t while still a member (delisting: exit at the last
  traded price; flagged in delist_mask).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from env import features, riskparity
from env.engine import MarketData


def eligibility(member: np.ndarray, R: np.ndarray, window: int, max_missing: int = 0) -> np.ndarray:
    T = member.shape[0]
    nxt = np.vstack([member[1:], member[-1:]])
    has = ~np.isnan(R)
    cnt = pd.DataFrame(has.astype(float)).rolling(window, min_periods=window).sum().to_numpy()
    ok_window = np.nan_to_num(cnt, nan=0.0) >= window - max_missing
    return nxt & has & ok_window


def exits(member: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T, N = member.shape
    has = ~np.isnan(R)
    # last session with a price, per stock
    last_px = np.full(N, -1)
    for i in range(N):
        idx = np.flatnonzero(has[:, i])
        if idx.size:
            last_px[i] = idx[-1]
    leave = np.zeros((T, N), bool)
    leave[:-1] = member[:-1] & ~member[1:]
    delist = np.zeros((T, N), bool)
    for i in range(N):
        t = last_px[i]
        # still a member after its last traded day (and the data has not simply ended)
        if 0 <= t < T - 1 and member[t, i] and member[t + 1, i]:
            delist[t, i] = True
    return leave | delist, delist


def assemble(dates: pd.DatetimeIndex, market: pd.DataFrame, R: np.ndarray, member: np.ndarray,
             rf: np.ndarray, *, window: int = 120, max_missing: int = 0,
             symbols: list[str] | None = None) -> MarketData:
    exit_mask, delist_mask = exits(member, R)
    # A stock exits at its last traded price, so it gets no new weight that close.
    elig = eligibility(member, R, window, max_missing) & ~delist_mask
    W = riskparity.precompute(R, elig, window)
    feats = features.market_features(market, R, elig).to_numpy()
    rp = features.rp_portfolio_returns(W, R)
    data = MarketData(dates=dates, R=R, W=W, exit_mask=exit_mask, delist_mask=delist_mask,
                      rf=rf, feats=feats, rp_ret=rp, symbols=symbols or [])
    data.eligible = elig
    return data


def equal_weights(eligible: np.ndarray) -> np.ndarray:
    n = eligible.sum(axis=1, keepdims=True)
    return np.where(n > 0, eligible / np.maximum(n, 1), 0.0)
