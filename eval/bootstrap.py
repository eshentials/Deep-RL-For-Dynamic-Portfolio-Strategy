"""Paired circular block bootstrap (Politis and Romano 1992) for metric differences.

The same resampled session indices are applied to both strategies, so their
dependence (they trade the same market on the same days) is preserved.
"""
from __future__ import annotations

import numpy as np

ANN = np.sqrt(252.0)


def block_indices(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(draws, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(draws, -1)[:, :n]


def _sharpe_rows(x: np.ndarray) -> np.ndarray:
    sd = x.std(axis=1, ddof=1)
    return np.where(sd > 0, x.mean(axis=1) / sd * ANN, np.nan)


def paired_sharpe_diff(ret_a, ret_b, rf, block: int = 20, draws: int = 10000, seed: int = 0,
                       chunk: int = 1000) -> dict:
    ea = np.asarray(ret_a) - np.asarray(rf)
    eb = np.asarray(ret_b) - np.asarray(rf)
    n = len(ea)
    obs = float(_sharpe_rows(ea[None])[0] - _sharpe_rows(eb[None])[0])
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for k in range(0, draws, chunk):
        m = min(chunk, draws - k)
        idx = block_indices(n, block, m, rng)
        out[k:k + m] = _sharpe_rows(ea[idx]) - _sharpe_rows(eb[idx])
    return {"observed": obs, "draws": out, "p05": float(np.quantile(out, 0.05)),
            "ci95": (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))),
            "share_le_0": float((out <= 0).mean())}


def paired_metric_diff(ret_a, ret_b, fn, block: int = 20, draws: int = 2000, seed: int = 0) -> dict:
    """Generic paired bootstrap for any per-path metric fn(returns) (e.g. max drawdown)."""
    a, b = np.asarray(ret_a), np.asarray(ret_b)
    rng = np.random.default_rng(seed)
    idx = block_indices(len(a), block, draws, rng)
    out = np.array([fn(a[i]) - fn(b[i]) for i in idx])
    return {"observed": float(fn(a) - fn(b)), "draws": out,
            "ci95": (float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975)))}
