"""Equal-risk-contribution weights, precomputed once per session.

ERC solves Spinu's (2013) convex problem
    min_y  0.5 y' S y - (1/n) sum_i log y_i,   y > 0,    w = y / sum(y)
whose optimum gives every asset the same risk contribution w_i (S w)_i. Newton's
method with a positivity-preserving line search converges in a few iterations.

The covariance is Ledoit-Wolf shrinkage on the trailing `window` sessions of daily
returns of the assets eligible on that session (complete window required). Weights
depend only on returns up to and including the session, never on the agent.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf


def erc_weights(cov: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    b = 1.0 / n
    y = np.sqrt(b / np.diag(cov))  # exact for a diagonal covariance
    for _ in range(max_iter):
        sy = cov @ y
        g = sy - b / y
        hess = cov + np.diag(b / y**2)
        step = np.linalg.solve(hess, g)
        # Largest step that keeps y strictly positive, then Armijo backtracking.
        neg = step > 0
        t = 1.0 if not neg.any() else min(1.0, 0.99 * float(np.min(y[neg] / step[neg])))
        f0 = 0.5 * y @ sy - b * np.log(y).sum()
        while True:
            y_new = y - t * step
            f1 = 0.5 * y_new @ cov @ y_new - b * np.log(y_new).sum()
            if f1 <= f0 - 1e-4 * t * (g @ step) or t < 1e-10:
                break
            t *= 0.5
        y = y_new
        if np.abs(t * step).max() < tol * max(1.0, np.abs(y).max()):
            break
    return y / y.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance contributed by each asset (sums to 1)."""
    rc = w * (cov @ w)
    return rc / rc.sum()


def precompute(returns: np.ndarray, eligible: np.ndarray, window: int = 120) -> np.ndarray:
    """T x N ERC weights; row t uses returns[t-window+1 .. t] of assets eligible at t.

    `returns[t, i]` is asset i's simple return from close t-1 to close t (NaN if no
    price; treated as 0). `eligible` already encodes the window-coverage rule
    (env/build.py), so an asset enters row t exactly when eligible[t, i].
    """
    T, N = returns.shape
    W = np.zeros((T, N))
    lw = LedoitWolf(assume_centered=False)
    X = np.nan_to_num(returns)
    for t in range(window - 1, T):
        win = X[t - window + 1:t + 1]
        cols = np.flatnonzero(eligible[t])
        if cols.size == 0:
            continue
        if cols.size == 1:
            W[t, cols] = 1.0
            continue
        cov = lw.fit(win[:, cols]).covariance_
        W[t, cols] = erc_weights(cov)
    return W
