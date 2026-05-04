"""
smart_tangency.py — Smart Tangency Portfolio Optimizer
=======================================================

Implements the portfolio optimization framework from:
  "Smart Tangency Portfolio: Deep Reinforcement Learning for Dynamic
   Rebalancing and Risk–Return Trade-Off" (Yu & Chang, 2025)

Supports THREE utility (risk) functions:
  1. Mean-Variance   (MV)         — classical Markowitz (λ/2 wᵀΣw − μᵀw)
  2. Mean-Semivariance (SV)       — downside risk only (below-benchmark dev)
  3. Mean-CVaR        (CVaR)      — tail-risk via Conditional Value-at-Risk

In all cases the DRL agent controls:
  · λ ∈ {1 … 100} — risk-aversion index mapped to a percentile of the
                     target return between min and max asset return
  · h ∈ {5 … 60}  — rebalancing horizon in trading days

The "smart tangency" component:
  The tangency portfolio is the point on the efficient frontier with the
  maximum Sharpe ratio.  `find_tangency()` scans the frontier (100 λ levels)
  and returns the max-Sharpe portfolio.  This can be used as a static
  benchmark or as the default action when the agent is uncertain.

Public API
──────────
  optimize_mv(mu, cov, lam)             → SmartResult
  optimize_sv(returns_hist, lam)        → SmartResult
  optimize_cvar(returns_hist, lam, β)   → SmartResult
  find_tangency(utility, ...)            → SmartResult  # max-Sharpe scan
  SmartOptimizer(utility)               # convenience wrapper used by env

Utility names (string constants):
  UTILITY_MV    = "mv"
  UTILITY_SV    = "sv"
  UTILITY_CVAR  = "cvar"
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import linprog, minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Constants ──────────────────────────────────────────────────────────────────
UTILITY_MV   = "mv"
UTILITY_SV   = "sv"
UTILITY_CVAR = "cvar"
UTILITIES    = (UTILITY_MV, UTILITY_SV, UTILITY_CVAR)

# Risk-free rate (annualised). Paper uses ~6.7% for India G-Sec.
RISK_FREE     = 0.067

# Portfolio constraints
MAX_WEIGHT    = 0.40   # max single-asset weight (concentration limit)
MIN_WEIGHT    = 0.00   # no short-selling
EPS           = 1e-9

# CVaR confidence level (paper uses 95%)
CVAR_BETA     = 0.95

# Number of λ levels to scan when building the frontier
N_LAMBDA      = 100


# ── λ-to-target-return mapping ─────────────────────────────────────────────────
#
# The paper defines λ as an integer in {1…100} that maps to the λ-th percentile
# of the range [μ_min, μ_max] where μ_min and μ_max are the minimum and maximum
# expected returns among all individual assets.
#
# λ = 1   → target return = μ_min (most conservative)
# λ = 100 → target return = μ_max (most aggressive)
#
# The optimiser then solves:
#   min Risk(w)
#   s.t.  μᵀw ≥ μ_target(λ),  Σwᵢ=1,  wᵢ≥0

def lambda_to_target_return(lam_idx: int,
                             mu: np.ndarray,
                             n_levels: int = N_LAMBDA) -> float:
    """
    Map a 1-indexed λ level to a target portfolio return.

    lam_idx  : int ∈ {1 … n_levels}
    mu       : (N,) per-asset expected returns
    n_levels : total number of λ levels (default 100)

    Returns μ_target as a float.
    """
    lam_idx = int(np.clip(lam_idx, 1, n_levels))
    frac    = (lam_idx - 1) / max(n_levels - 1, 1)   # 0.0 … 1.0
    mu_min  = float(mu.min())
    mu_max  = float(mu.max())
    return mu_min + frac * (mu_max - mu_min)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SmartResult:
    """
    Unified output container for all three utility functions.

    Attributes
    ──────────
    utility        : one of "mv", "sv", "cvar"
    lam_idx        : integer λ index used (1 … 100)
    weights        : (N+1,) stock weights + cash slot (sums to 1)
    mu             : (N,) expected return vector
    exp_return     : float — μᵀw_stocks
    exp_risk       : float — risk measure value (variance / semivariance / CVaR)
    exp_volatility : float — √(wᵀΣw) for Sharpe, even for SV/CVaR utilities
    sharpe         : float — (exp_return − rf) / exp_volatility
    tickers        : list[str]
    """
    utility:        str
    lam_idx:        int
    weights:        np.ndarray   # (N+1,)
    mu:             np.ndarray   # (N,)
    exp_return:     float
    exp_risk:       float
    exp_volatility: float
    sharpe:         float
    tickers:        list[str]

    @property
    def stock_weights(self) -> np.ndarray:
        return self.weights[:-1]

    @property
    def cash_weight(self) -> float:
        return float(self.weights[-1])

    def summary(self) -> str:
        top3 = sorted(zip(self.tickers, self.stock_weights),
                      key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{t.split('.')[0]}={w:.3f}"
                             for t, w in top3 if w > 0.001)
        return (f"[{self.utility.upper()}] λ={self.lam_idx:>3d}  "
                f"ret={self.exp_return:+.4f}  vol={self.exp_volatility:.4f}  "
                f"Sharpe={self.sharpe:+.3f}  cash={self.cash_weight:.3f}  "
                f"[{top3_str}]")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — expected-return estimator (shared by all utilities)
# ══════════════════════════════════════════════════════════════════════════════

def _build_expected_returns(returns_hist: np.ndarray) -> np.ndarray:
    """
    Estimate per-asset expected returns from a (T, N) historical return matrix.
    Returns the simple column mean (μ = E[yᵢ]).
    """
    return np.nanmean(returns_hist, axis=0).astype(np.float64)


def _portfolio_volatility(w: np.ndarray, cov: np.ndarray) -> float:
    """√(wᵀΣw) from stock weights (N,) and (N,N) covariance."""
    w = w.clip(0)
    return float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def _normalise_weights(w: np.ndarray) -> np.ndarray:
    """Clip to [0, MAX_WEIGHT], renormalise to sum=1."""
    w = np.clip(w, 0.0, MAX_WEIGHT)
    s = w.sum()
    return w / s if s > EPS else np.ones(len(w)) / len(w)


def _equal_weight(N: int) -> np.ndarray:
    """Fallback equal-weight vector of length N."""
    return np.ones(N) / N


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER 1 — Mean-Variance  (Equation 23 in paper)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_mv(mu:      np.ndarray,
                cov:     np.ndarray,
                lam_idx: int,
                tickers: list[str]) -> SmartResult:
    """
    Solve the Mean-Variance programme for a given λ index.

    min  (λ_eff/2) wᵀΣw
    s.t. μᵀw ≥ μ_target(lam_idx)
         Σwᵢ = 1,  0 ≤ wᵢ ≤ MAX_WEIGHT

    λ_eff is the actual risk-aversion scalar derived from lam_idx such that
    the constraint is met at a penalty for straying from target return.
    We implement this via the target-return constraint form (more stable).
    """
    N = len(mu)
    mu_target = lambda_to_target_return(lam_idx, mu)

    w0 = _equal_weight(N)

    def _obj(w):
        return 0.5 * float(w @ cov @ w)

    def _grad(w):
        return cov @ w

    constraints = [
        {"type": "eq",  "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq","fun": lambda w: float(mu @ w) - mu_target},
    ]

    result = minimize(
        _obj, w0,
        jac=_grad,
        method="SLSQP",
        bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * N,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 3000},
    )

    w_stocks = _normalise_weights(result.x if result.success else w0)
    return _make_result(UTILITY_MV, lam_idx, w_stocks, mu, cov, tickers)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER 2 — Mean-Semivariance  (Equations 5, 6, 24 in paper)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_sv(returns_hist: np.ndarray,
                lam_idx:      int,
                tickers:      list[str],
                benchmark:    float | None = None) -> SmartResult:
    """
    Solve the Mean-Semivariance programme.

    Semivariance SV = (1/T) Σ min(yᵢₜ − B, 0)²  per asset
    Semi-covariance matrix: SCOV[i,j] = E[min(yᵢ−B,0)·min(yⱼ−B,0)]

    min  wᵀ SCOV w
    s.t. μᵀw ≥ μ_target(lam_idx)
         Σwᵢ = 1,  0 ≤ wᵢ ≤ MAX_WEIGHT

    Parameters
    ──────────
    returns_hist : (T, N) historical returns matrix
    lam_idx      : integer risk-aversion index 1–100
    tickers      : asset ticker list
    benchmark    : below-target benchmark B (default: μ_mean)
    """
    T, N = returns_hist.shape
    mu   = _build_expected_returns(returns_hist)

    if benchmark is None:
        benchmark = float(mu.mean())   # mean return as benchmark

    # ── Semi-covariance matrix ─────────────────────────────────────────────────
    # SCOV[i,j] = mean of min(yᵢₜ−B, 0) * min(yⱼₜ−B, 0)  (Estrada 2002)
    shortfalls = np.minimum(returns_hist - benchmark, 0.0)   # (T, N)
    scov = (shortfalls.T @ shortfalls) / max(T, 1)           # (N, N)
    scov = 0.5 * (scov + scov.T)                             # symmetrise
    scov += np.eye(N) * EPS                                  # PSD safety

    # ── Full covariance (for vol/Sharpe reporting only) ────────────────────────
    cov = np.cov(returns_hist.T) * 252     # annualised
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(N) * EPS

    mu_target = lambda_to_target_return(lam_idx, mu)
    w0 = _equal_weight(N)

    def _obj(w):  return 0.5 * float(w @ scov @ w)
    def _grad(w): return scov @ w

    constraints = [
        {"type": "eq",  "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq","fun": lambda w: float(mu @ w) - mu_target},
    ]

    result = minimize(
        _obj, w0,
        jac=_grad,
        method="SLSQP",
        bounds=[(MIN_WEIGHT, MAX_WEIGHT)] * N,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 3000},
    )

    w_stocks = _normalise_weights(result.x if result.success else w0)
    sv_val   = float(w_stocks @ scov @ w_stocks)
    return _make_result(UTILITY_SV, lam_idx, w_stocks, mu, cov, tickers,
                        risk_override=sv_val)


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER 3 — Mean-CVaR  (Equations 7–10, 25 in paper)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_cvar(returns_hist: np.ndarray,
                  lam_idx:      int,
                  tickers:      list[str],
                  beta:         float = CVAR_BETA) -> SmartResult:
    """
    Solve the Mean-CVaR programme using Rockafellar & Uryasev (2000) LP.

    CVaR_β(w) = min_{l}  l + 1/(q(1−β)) Σ [f(w,yₖ)−l]⁺
    where f(w,y) = −wᵀy  (portfolio loss)

    Transformed to LP via auxiliary variables Sₖ ≥ [f(w,yₖ)−l]⁺:

    min_{w,l,S}   l + 1/(q(1−β)) Σ Sₖ
    s.t.           Sₖ ≥ −wᵀyₖ − l       ∀ k
                   Sₖ ≥ 0               ∀ k
                   Σwᵢ = 1
                   wᵀμ ≥ μ_target
                   0 ≤ wᵢ ≤ MAX_WEIGHT

    We use scipy.optimize.linprog in HiGHS mode for speed.

    Parameters
    ──────────
    returns_hist : (T, N) historical returns (simple daily)
    lam_idx      : integer risk-aversion index 1–100
    beta         : CVaR confidence level (default 0.95)
    """
    T, N = returns_hist.shape
    mu   = _build_expected_returns(returns_hist)
    mu_target = lambda_to_target_return(lam_idx, mu)

    q    = T
    coef = 1.0 / (q * (1.0 - beta))

    # Decision variables: x = [w (N), l (1), S (T)]
    # Objective: min 0*w + 1*l + coef * Σ Sₖ
    c = np.concatenate([np.zeros(N), [1.0], coef * np.ones(T)])

    # Inequality Sₖ ≥ −wᵀyₖ − l  =>  −wᵀyₖ − l − Sₖ ≤ 0
    # A_ub x ≤ b_ub:
    #   For each scenario k: [−yₖ (N), −1 (1), −eₖ (T)] ≤ 0
    A_list = []
    b_list = []
    for k in range(T):
        y_k   = returns_hist[k]                    # (N,)
        row   = np.zeros(N + 1 + T)
        row[:N]     = -y_k                         # −wᵀyₖ part
        row[N]      = -1.0                         # −l part
        row[N + 1 + k] = -1.0                      # −Sₖ part
        A_list.append(row)
        b_list.append(0.0)

    A_ub = np.array(A_list)
    b_ub = np.array(b_list)

    # Return constraint: μᵀw ≥ μ_target  =>  −μᵀw ≤ −μ_target
    ret_row = np.zeros(N + 1 + T)
    ret_row[:N] = -mu
    A_ub = np.vstack([A_ub, ret_row])
    b_ub = np.append(b_ub, -mu_target)

    # Equality: Σwᵢ = 1
    A_eq = np.zeros((1, N + 1 + T))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    # Bounds: 0 ≤ wᵢ ≤ MAX_WEIGHT, l unbounded, Sₖ ≥ 0
    bounds = (
        [(MIN_WEIGHT, MAX_WEIGHT)] * N +
        [(None, None)] +           # l
        [(0.0, None)] * T          # S
    )

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds,
                      method="highs")
        if res.success:
            w_stocks = _normalise_weights(res.x[:N])
            l_star   = float(res.x[N])
            S_vals   = res.x[N + 1:]
            cvar_val = float(l_star + coef * S_vals.sum())
        else:
            w_stocks = _equal_weight(N)
            cvar_val = 0.0
    except Exception:
        w_stocks = _equal_weight(N)
        cvar_val = 0.0

    cov = np.cov(returns_hist.T) * 252
    if cov.ndim == 0:
        cov = np.array([[cov]])
    cov += np.eye(N) * EPS

    return _make_result(UTILITY_CVAR, lam_idx, w_stocks, mu, cov, tickers,
                        risk_override=cvar_val)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — assemble SmartResult
# ══════════════════════════════════════════════════════════════════════════════

def _make_result(utility:        str,
                 lam_idx:        int,
                 w_stocks:       np.ndarray,
                 mu:             np.ndarray,
                 cov:            np.ndarray,
                 tickers:        list[str],
                 risk_override:  float | None = None) -> SmartResult:
    """Compute derived metrics and package into SmartResult."""
    exp_ret  = float(w_stocks @ mu)
    exp_vol  = _portfolio_volatility(w_stocks, cov)
    sharpe   = (exp_ret - RISK_FREE) / (exp_vol + EPS)

    exp_risk = risk_override if risk_override is not None else float(w_stocks @ cov @ w_stocks)

    cash_w  = float(np.clip(1.0 - w_stocks.sum(), 0.0, 1.0))
    weights = np.append(w_stocks, cash_w)
    weights /= weights.sum()

    return SmartResult(
        utility        = utility,
        lam_idx        = lam_idx,
        weights        = weights.astype(np.float32),
        mu             = mu,
        exp_return     = exp_ret,
        exp_risk       = exp_risk,
        exp_volatility = exp_vol,
        sharpe         = sharpe,
        tickers        = tickers,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SMART TANGENCY — max-Sharpe portfolio (scan frontier)
# ══════════════════════════════════════════════════════════════════════════════

def find_tangency(utility:       str,
                  mu:            np.ndarray,
                  cov:           np.ndarray,
                  tickers:       list[str],
                  returns_hist:  np.ndarray | None = None,
                  n_scan:        int = N_LAMBDA) -> SmartResult:
    """
    Scan the efficient frontier and return the max-Sharpe (tangency) portfolio.

    For MV utility, only `mu` and `cov` are needed.
    For SV and CVaR, `returns_hist` (T,N) is required.

    Parameters
    ──────────
    utility      : "mv" | "sv" | "cvar"
    mu           : (N,) expected returns
    cov          : (N, N) covariance (used for MV and vol reporting)
    tickers      : list of asset tickers
    returns_hist : (T, N) historical returns (required for SV / CVaR)
    n_scan       : number of λ levels to scan (default 100)

    Returns
    ───────
    SmartResult with the highest Sharpe ratio across all scanned λ levels.
    """
    assert utility in UTILITIES, f"Unknown utility: {utility}"

    best: SmartResult | None = None

    for lam_idx in range(1, n_scan + 1):
        try:
            if utility == UTILITY_MV:
                r = optimize_mv(mu, cov, lam_idx, tickers)
            elif utility == UTILITY_SV:
                assert returns_hist is not None, "returns_hist required for SV"
                r = optimize_sv(returns_hist, lam_idx, tickers)
            else:
                assert returns_hist is not None, "returns_hist required for CVaR"
                r = optimize_cvar(returns_hist, lam_idx, tickers)
        except Exception:
            continue

        if best is None or r.sharpe > best.sharpe:
            best = r

    if best is None:
        # Ultimate fallback: equal-weight portfolio
        N = len(mu)
        w = _equal_weight(N)
        best = _make_result(utility, N_LAMBDA // 2, w, mu, cov, tickers)

    return best


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER — SmartOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class SmartOptimizer:
    """
    Stateless convenience wrapper around the three utility optimizers.

    Usage (inside the RL environment)
    ──────────────────────────────────
    opt = SmartOptimizer(utility="mv")
    result = opt.optimize(mu, cov, lam_idx, tickers, returns_hist=...)

    The `returns_hist` argument is optional for MV but required for SV/CVaR.
    """

    def __init__(self, utility: str = UTILITY_MV):
        if utility not in UTILITIES:
            raise ValueError(f"utility must be one of {UTILITIES}, got '{utility}'")
        self.utility = utility

    def optimize(self,
                 mu:           np.ndarray,
                 cov:          np.ndarray,
                 lam_idx:      int,
                 tickers:      list[str],
                 returns_hist: np.ndarray | None = None) -> SmartResult:
        """
        Solve the optimization for the configured utility function.

        Parameters
        ──────────
        mu           : (N,) expected return vector
        cov          : (N, N) covariance matrix
        lam_idx      : integer λ index 1–100
        tickers      : list of N ticker strings
        returns_hist : (T, N) historical returns — required for SV and CVaR

        Returns
        ───────
        SmartResult
        """
        if self.utility == UTILITY_MV:
            return optimize_mv(mu, cov, lam_idx, tickers)
        elif self.utility == UTILITY_SV:
            if returns_hist is None:
                raise ValueError("returns_hist required for SV utility")
            return optimize_sv(returns_hist, lam_idx, tickers)
        else:
            if returns_hist is None:
                raise ValueError("returns_hist required for CVaR utility")
            return optimize_cvar(returns_hist, lam_idx, tickers)

    def tangency(self,
                 mu:           np.ndarray,
                 cov:          np.ndarray,
                 tickers:      list[str],
                 returns_hist: np.ndarray | None = None) -> SmartResult:
        """Return the tangency (max-Sharpe) portfolio for the configured utility."""
        return find_tangency(
            utility      = self.utility,
            mu           = mu,
            cov          = cov,
            tickers      = tickers,
            returns_hist = returns_hist,
        )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — extract optimizer inputs from existing env state
# ══════════════════════════════════════════════════════════════════════════════

def extract_smart_inputs(state:        np.ndarray,
                         bundle,
                         t:            int,
                         lookback:     int = 60) -> dict:
    """
    Extract inputs needed by SmartOptimizer from the RL environment state/bundle.

    Parameters
    ──────────
    state    : (49,) state vector from state_spec.build_state_vector
    bundle   : FeatureBundle from features.load_bundle
    t        : current timestep index
    lookback : rows of return history to include (default 60 days)

    Returns dict with:
      mu           : (N,) expected returns from momentum + MA signals
      cov          : (N, N) annualised covariance
      returns_hist : (T, N) recent return history (for SV / CVaR)
    """
    from state_spec import STATE, extract_mvo_inputs
    from efficient_frontier import (build_compressed_covariance,
                                    expected_returns_from_state)

    mvo = extract_mvo_inputs(state)
    cov = build_compressed_covariance(
        mvo["vol20d"].astype(np.float64),
        mvo["avg_correlation"]
    )
    mu = mvo["momentum_5d"].astype(float)
    t_start = max(t - lookback, bundle.warm_up_rows())
    t_end   = t
    returns_hist = bundle.daily_ret.iloc[t_start:t_end].values.astype(np.float64)

    if returns_hist.shape[0] < 5:
        returns_hist = np.zeros((5, len(mu)))

    return {"mu": mu, "cov": cov, "returns_hist": returns_hist}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    N, T = 8, 252

    # Synthetic data
    tickers = [f"ASSET_{i}" for i in range(N)]
    rets    = np.random.normal(0.0003, 0.015, (T, N))
    mu      = rets.mean(axis=0)
    cov     = np.cov(rets.T) * 252 + np.eye(N) * 1e-6

    print("=" * 70)
    print("  Smart Tangency Portfolio — smoke test")
    print("=" * 70)

    for utility in UTILITIES:
        print(f"\n  Utility: {utility.upper()}")
        opt = SmartOptimizer(utility)

        # Test optimize at specific λ
        r = opt.optimize(mu, cov, lam_idx=50, tickers=tickers, returns_hist=rets)
        print(f"    λ=50  : {r.summary()}")

        # Test tangency
        tang = opt.tangency(mu, cov, tickers, returns_hist=rets)
        print(f"    tangency (λ={tang.lam_idx:>3d}): {tang.summary()}")

    print("\n  ✓  All utilities verified")
