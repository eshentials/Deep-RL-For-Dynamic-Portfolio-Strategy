"""State features, all computed from data up to and including close t.

Market features (10) come from the index, India VIX, USD/INR and the eligible
constituents. Path features (exposure, own drawdown, own volatility) depend on the
strategy's own NAV and are produced by env/engine.py.

Every window here is trailing (pandas rolling / shift with positive lags), so no
value at t can depend on data after t. tests/test_no_lookahead.py checks this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252.0)
MARKET_FEATURES = ["idx_vol20", "idx_vol60", "idx_vol_ratio", "idx_mom21", "idx_mom63",
                   "idx_dd252", "avg_corr60", "vix", "vix_chg5", "fx_chg21_lag1"]


def avg_pairwise_corr(returns: np.ndarray, eligible: np.ndarray, window: int = 60) -> np.ndarray:
    T = returns.shape[0]
    out = np.full(T, np.nan)
    X = np.nan_to_num(returns)
    for t in range(window - 1, T):
        win = X[t - window + 1:t + 1]
        cols = np.flatnonzero(eligible[t])
        if cols.size < 2:
            continue
        x = win[:, cols]
        sd = x.std(axis=0)
        keep = sd > 0
        if keep.sum() < 2:
            continue
        c = np.corrcoef(x[:, keep], rowvar=False)
        n = c.shape[0]
        out[t] = (c.sum() - n) / (n * (n - 1))
    return out


def market_features(market: pd.DataFrame, returns: np.ndarray, eligible: np.ndarray) -> pd.DataFrame:
    """`market` is indexed by session with columns vix, usdinr and the universe's index:
    `index` when the study has one (Nifty Next 50), else `nsei` (Nifty 50, protocol v1)."""
    px = market["index"] if "index" in market.columns else market["nsei"]
    lp = np.log(px)
    lr = lp.diff()
    f = pd.DataFrame(index=market.index)
    f["idx_vol20"] = lr.rolling(20).std() * ANN
    f["idx_vol60"] = lr.rolling(60).std() * ANN
    f["idx_vol_ratio"] = f["idx_vol20"] / f["idx_vol60"]
    f["idx_mom21"] = lp - lp.shift(21)
    f["idx_mom63"] = lp - lp.shift(63)
    f["idx_dd252"] = px / px.rolling(252, min_periods=1).max() - 1.0
    f["avg_corr60"] = avg_pairwise_corr(returns, eligible, 60)
    f["vix"] = market["vix"] / 100.0
    f["vix_chg5"] = np.log(market["vix"]).diff(5)
    lfx = np.log(market["usdinr"])
    f["fx_chg21_lag1"] = lfx.shift(1) - lfx.shift(22)
    return f[MARKET_FEATURES]


def zscore_stats(feats: pd.DataFrame, lo: int, hi: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std over rows lo..hi inclusive (a fold's training window only)."""
    x = feats.iloc[lo:hi + 1].to_numpy()
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def rp_portfolio_returns(W: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Daily return of the unlevered risk-parity portfolio: weights set at close t-1."""
    r = np.zeros(len(R))
    r[1:] = np.nansum(W[:-1] * np.nan_to_num(R[1:]), axis=1)
    return r
