"""Performance metrics on daily returns (protocol Section 7)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def sharpe(ret: np.ndarray, rf: np.ndarray) -> float:
    ex = np.asarray(ret) - np.asarray(rf)
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan")


def nav(ret: np.ndarray) -> np.ndarray:
    return np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(ret))])


def max_drawdown(ret: np.ndarray) -> float:
    v = nav(ret)
    return float((v / np.maximum.accumulate(v) - 1.0).min())


def cagr(ret: np.ndarray) -> float:
    n = len(ret)
    return float(nav(ret)[-1] ** (ANN / n) - 1.0) if n else float("nan")


def vol(ret: np.ndarray) -> float:
    return float(np.std(ret, ddof=1) * np.sqrt(ANN))


def calmar(ret: np.ndarray) -> float:
    mdd = max_drawdown(ret)
    return cagr(ret) / abs(mdd) if mdd < 0 else float("nan")


def summary(ret: np.ndarray, rf: np.ndarray, **extra) -> dict:
    out = {"sharpe": sharpe(ret, rf), "cagr": cagr(ret), "vol": vol(ret),
           "max_drawdown": max_drawdown(ret), "calmar": calmar(ret),
           "total_return": float(nav(ret)[-1] - 1.0), "sessions": int(len(ret))}
    out.update(extra)
    return out


def by_year(dates: pd.DatetimeIndex, ret: np.ndarray, rf: np.ndarray) -> pd.DataFrame:
    rows = {}
    years = dates.year
    for y in np.unique(years):
        m = years == y
        rows[int(y)] = summary(ret[m], rf[m])
    return pd.DataFrame(rows).T
