"""Probabilistic and Deflated Sharpe Ratios (Bailey and Lopez de Prado, 2012; 2014).

All Sharpe ratios here are per period (daily), not annualised. Kurtosis is the raw
(non-excess) fourth standardised moment, as in the papers.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis, norm, skew

EULER = 0.5772156649015329


def psr(sr: float, T: int, sk: float, ku: float, sr0: float = 0.0) -> float:
    denom = np.sqrt(1.0 - sk * sr + (ku - 1.0) / 4.0 * sr**2)
    return float(norm.cdf((sr - sr0) * np.sqrt(T - 1.0) / denom))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected maximum Sharpe of n independent zero-skill trials (False Strategy Theorem)."""
    if n_trials <= 1:
        return 0.0
    return float(np.sqrt(var_sr) * ((1 - EULER) * norm.ppf(1 - 1.0 / n_trials)
                                    + EULER * norm.ppf(1 - 1.0 / (n_trials * np.e))))


def dsr_from_stats(sr: float, T: int, sk: float, ku: float, n_trials: int,
                   var_sr: float | None = None) -> dict:
    v = 1.0 / (T - 1.0) if var_sr is None else var_sr
    sr0 = expected_max_sr(n_trials, v)
    return {"sr": sr, "T": T, "skew": sk, "kurt": ku, "n_trials": n_trials, "var_sr": v,
            "sr0": sr0, "dsr": psr(sr, T, sk, ku, sr0), "psr": psr(sr, T, sk, ku, 0.0)}


def dsr(excess_returns: np.ndarray, n_trials: int, var_sr: float | None = None) -> dict:
    x = np.asarray(excess_returns)
    sr = float(x.mean() / x.std(ddof=1))
    return dsr_from_stats(sr, len(x), float(skew(x)), float(kurtosis(x, fisher=False)),
                          n_trials, var_sr)
