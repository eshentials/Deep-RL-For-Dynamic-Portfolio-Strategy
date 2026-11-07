"""
efficient_frontier.py — Mean-Variance Optimizer (MVO)

Solves the parametric Markowitz programme for a caller-supplied λ:

    max  μᵀw  −  (λ / 2) wᵀ Σ̂ w
    s.t. Σwᵢ = 1,   0 ≤ wᵢ ≤ MAX_WEIGHT

λ (risk-aversion) is an INPUT — this module never picks it autonomously.
The caller (ef_optimizer.py, env.py, or any strategy) decides which λ to use
and passes it to `optimize()`.  This keeps the math separate from the policy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Public API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  optimize(state, tickers, lam)            → MVOResult
      Solve MVO for a single λ value.

  scan_frontier(state, tickers, lambdas)   → list[MVOResult]
      Solve MVO for every λ in an array — returns the full frontier curve.
      Use this for visualisation or if the caller wants to pick across λ.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compressed covariance  (constant-correlation model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Σ̂[i, i] = σᵢ²                         (diagonal — per-asset variance)
  Σ̂[i, j] = ρ̄ × σᵢ × σⱼ   (i ≠ j)      (off-diagonal — avg correlation)

  σᵢ  from STATE.VOL20D  (8 annualised volatilities)
  ρ̄   from STATE.AVG_CORR (1 scalar — mean pairwise correlation)

  9 numbers instead of 36 (full 8×8 lower-triangle).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected-return signal  μ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μᵢ = α_mom × momentum_5d[i]  +  α_trend × ma_spread[i]

  Both signals come from the state vector (STATE.MOM5D, STATE.MA_SPREAD).
  Tunable weights α_mom=0.60, α_trend=0.40.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reference λ grid  (20 values, for scanning / analysis)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LAMBDA_VALUES = np.linspace(0.1, 2.0, 20)

  λ → 0    : return-maximising (ignores variance)
  λ = 1    : balanced (Sharpe-like tradeoff)
  λ → ∞   : minimum-variance (ignores expected return)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from state_spec import STATE, extract_mvo_inputs

# ── Reference λ grid (for callers who want to scan the frontier) ──────────────
LAMBDA_VALUES: np.ndarray = np.linspace(0.1, 2.0, 20)

# ── Optimisation constraints ──────────────────────────────────────────────────
MAX_WEIGHT = 0.40    # maximum single-asset weight (concentration limit)
MIN_WEIGHT = 0.00    # no short-selling
RISK_FREE  = 0.067   # India 10-yr G-Sec (annualised), for Sharpe computation

# ── Expected-return signal weights ────────────────────────────────────────────
ALPHA_MOM   = 0.60   # weight on 5-day momentum
ALPHA_TREND = 0.40   # weight on MA 50/200 spread

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "features")
RESULTS_DIR = os.path.join(DATA_DIR, "ef_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}

EPS = 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — COMPRESSED COVARIANCE  (constant-correlation model)
# ══════════════════════════════════════════════════════════════════════════════

def build_compressed_covariance(vol20d: np.ndarray,
                                 avg_corr: float) -> np.ndarray:
    """
    Build an (N, N) annualised covariance matrix from N volatilities + 1 scalar.

    Σ̂ = D × C × D
      D     = diag(vol20d)         — per-asset volatility diagonal
      C[i,j] = ρ̄  for i ≠ j       — constant off-diagonal correlation
      C[i,i] = 1

    PSD guarantee
    ─────────────
    Constant-correlation C is PSD iff ρ̄ ≥ −1/(N−1).
    For N=8 this means ρ̄ ≥ −0.143.
    ρ̄ is clipped to [−0.14, 0.99] before construction.

    Parameters
    ──────────
    vol20d   : (N,) annualised per-asset volatilities  (from STATE.VOL20D)
    avg_corr : scalar average pairwise correlation      (from STATE.AVG_CORR)

    Returns
    ───────
    (N, N) symmetric PSD matrix.
    """
    N       = len(vol20d)
    min_rho = -1.0 / (N - 1) + 1e-4
    rho     = float(np.clip(avg_corr, min_rho, 0.99))

    C       = np.full((N, N), rho)
    np.fill_diagonal(C, 1.0)

    D       = np.diag(vol20d.clip(min=EPS))
    cov     = D @ C @ D
    cov     = 0.5 * (cov + cov.T)          # symmetrise
    cov    += np.eye(N) * EPS              # numerical safety
    return cov


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EXPECTED RETURNS  μ  from state vector signals
# ══════════════════════════════════════════════════════════════════════════════

def expected_returns_from_state(mvo_inputs: dict,
                                 alpha_mom:   float = ALPHA_MOM,
                                 alpha_trend: float = ALPHA_TREND) -> np.ndarray:
    """
    Estimate per-asset expected returns from two state-vector signals.

    μᵢ = α_mom × momentum_5d[i]  +  α_trend × ma_spread[i]

    Signals are used as relative rankings (not absolute forecasts), so
    intentionally not annualised — annualising 5-day momentum by ×50
    produces extreme values that destabilise the solver.

    Parameters
    ──────────
    mvo_inputs  : dict from state_spec.extract_mvo_inputs(state)
    alpha_mom   : weight on 5-day momentum signal   (default 0.60)
    alpha_trend : weight on MA 50/200 spread signal (default 0.40)

    Returns
    ───────
    (N,) float64 array — per-asset expected return proxies.
    """
    mom   = mvo_inputs["momentum_5d"].astype(np.float64)
    trend = mvo_inputs["ma_spread"].astype(np.float64)
    return alpha_mom * mom + alpha_trend * trend


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CORE MVO FORMULA
# ══════════════════════════════════════════════════════════════════════════════

def solve_mvo(mu: np.ndarray,
              cov: np.ndarray,
              lam: float,
              max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """
    Solve the mean-variance programme for the given risk-aversion λ.

    Objective (to minimise):
        f(w) = (λ/2) wᵀ Σ̂ w  −  μᵀ w

    Constraints:
        Σwᵢ = 1      (fully invested)
        0 ≤ wᵢ ≤ max_weight   (no shorting, concentration cap)

    Solver: scipy SLSQP — handles equality + box constraints directly.
    Gradient is supplied analytically: ∇f = λ Σ̂ w − μ

    Parameters
    ──────────
    mu         : (N,) expected return vector
    cov        : (N, N) covariance matrix  (compressed or full)
    lam        : risk-aversion scalar  (caller's responsibility to supply)
    max_weight : per-asset upper bound

    Returns
    ───────
    (N,) stock weight vector, sums to 1, all entries in [0, max_weight].
    Falls back to equal-weight if solver fails.
    """
    N  = len(mu)
    w0 = np.ones(N) / N                      # warm start: equal weight

    def _obj(w):  return 0.5 * lam * float(w @ cov @ w) - float(mu @ w)
    def _grad(w): return lam * (cov @ w) - mu

    result = minimize(
        _obj, w0,
        jac=_grad,
        method="SLSQP",
        bounds=[(MIN_WEIGHT, max_weight)] * N,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not result.success:
        return w0.copy()                      # fallback: equal weight

    w     = np.clip(result.x, 0.0, max_weight)
    w_sum = w.sum()
    return w / w_sum if w_sum > EPS else w0


# ══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MVOResult:
    """
    Output of a single MVO solve.

    Attributes
    ──────────
    lam           : risk-aversion value used  (caller-supplied)
    weights       : (N+1,) stock weights + cash slot  (sums to 1)
    mu            : (N,) expected return vector used in the solve
    cov           : (N, N) compressed covariance used in the solve
    exp_return    : μᵀ w_stocks   (annualised proxy)
    exp_volatility: √(wᵀ Σ̂ w)   (annualised)
    sharpe        : (exp_return − risk_free) / exp_volatility
    tickers       : list of stock ticker strings (length N)
    """
    lam:           float
    weights:       np.ndarray    # (N+1,) stocks + cash
    mu:            np.ndarray    # (N,)
    cov:           np.ndarray    # (N, N)
    exp_return:    float
    exp_volatility: float
    sharpe:        float
    tickers:       list[str]

    @property
    def stock_weights(self) -> np.ndarray:
        return self.weights[:-1]

    @property
    def cash_weight(self) -> float:
        return float(self.weights[-1])

    def summary(self) -> str:
        top3 = sorted(zip(self.tickers, self.stock_weights),
                      key=lambda x: x[1], reverse=True)[:3]
        top3_str = "  ".join(f"{t.split('.')[0]}={w:.3f}" for t, w in top3 if w > 0.001)
        return (f"λ={self.lam:.3f}  ret={self.exp_return:+.4f}  "
                f"vol={self.exp_volatility:.4f}  Sharpe={self.sharpe:+.3f}  "
                f"cash={self.cash_weight:.3f}  [{top3_str}]")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def optimize(state:   np.ndarray,
             tickers: list[str],
             lam:     float) -> MVOResult:
    """
    Solve the MVO for a single caller-supplied λ.

    This is the primary entry point.  The caller decides λ; this function
    only executes the formula.

    Parameters
    ──────────
    state   : (49,) state vector from state_spec.build_state_vector
    tickers : list of N stock ticker strings  (must match STATE.VOL20D length)
    lam     : risk-aversion parameter  (e.g. from LAMBDA_VALUES[k])

    Returns
    ───────
    MVOResult with weights (N+1,) including cash slot.
    """
    assert state.shape == (STATE.DIM,), (
        f"Expected state shape ({STATE.DIM},), got {state.shape}"
    )
    mvo     = extract_mvo_inputs(state)
    vol20d  = mvo["vol20d"].astype(np.float64)
    avg_corr = mvo["avg_correlation"]

    cov     = build_compressed_covariance(vol20d, avg_corr)
    mu      = expected_returns_from_state(mvo)

    w_stocks = solve_mvo(mu, cov, lam)

    exp_ret  = float(w_stocks @ mu)
    exp_vol  = float(np.sqrt(np.maximum(w_stocks @ cov @ w_stocks, 0.0)))
    sharpe   = (exp_ret - RISK_FREE) / (exp_vol + EPS)

    cash_w   = float(np.clip(1.0 - w_stocks.sum(), 0.0, 1.0))
    weights  = np.append(w_stocks, cash_w)
    weights /= weights.sum()               # normalise for float safety

    return MVOResult(
        lam=lam,
        weights=weights.astype(np.float32),
        mu=mu,
        cov=cov,
        exp_return=exp_ret,
        exp_volatility=exp_vol,
        sharpe=sharpe,
        tickers=tickers,
    )


def scan_frontier(state:        np.ndarray,
                  tickers:      list[str],
                  lambda_values: np.ndarray = LAMBDA_VALUES) -> list[MVOResult]:
    """
    Solve MVO for every λ in `lambda_values` and return all results.

    Use this when the caller wants to inspect the full frontier before
    deciding which portfolio to use, e.g.:

        results = scan_frontier(state, tickers)
        best    = max(results, key=lambda r: r.sharpe)   # caller picks

    Parameters
    ──────────
    state         : (49,) state vector
    tickers       : list of N stock ticker strings
    lambda_values : 1-D array of λ values to evaluate  (default: LAMBDA_VALUES)

    Returns
    ───────
    list of MVOResult, one per λ, in the same order as lambda_values.
    """
    return [optimize(state, tickers, float(lam)) for lam in lambda_values]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — demonstrate the MVO on real feature data
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_snapshot(split: str) -> tuple[np.ndarray, list[str], pd.Timestamp]:
    """Load features for a split and return a state vector at the midpoint date."""
    from features import load_bundle
    from state_spec import build_state_vector, MACRO_PATH

    label  = SPLIT_LABELS[split]
    bundle = load_bundle(split)
    macro  = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)

    t      = len(bundle.dates) // 2          # midpoint timestep
    date   = bundle.dates[t]
    N      = bundle.n_assets
    w_eq   = np.ones(N + 1) / (N + 1)

    state = build_state_vector(
        daily_ret_t = bundle.daily_ret.iloc[t].values.astype(np.float32),
        momentum_t  = bundle.momentum.iloc[t].values.astype(np.float32),
        vol20d_t    = bundle.realized_vol.iloc[t].values.astype(np.float32),
        ma_spread_t = bundle.ma_spread.iloc[t].values.astype(np.float32),
        cov_t       = bundle.cov[t],
        weights     = w_eq[:-1].astype(np.float32),
        tickers     = bundle.tickers,
        macro_df    = macro,
        t_date      = date,
        nav_series  = np.ones(1),
        log_ret_t   = bundle.log_ret.iloc[t].values.astype(np.float32),
    )
    return state, bundle.tickers, date


if __name__ == "__main__":
    TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "NIFTYBEES.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    ]

    print("=" * 70)
    print("  Mean-Variance Optimizer  —  formula: max μᵀw − (λ/2) wᵀΣ̂w")
    print("=" * 70)
    print(f"  Constraints : 0 ≤ wᵢ ≤ {MAX_WEIGHT}   Σwᵢ = 1")
    print(f"  Covariance  : compressed (9 params = 8 σ + 1 ρ̄)")
    print(f"  λ grid      : {len(LAMBDA_VALUES)} values  "
          f"[{LAMBDA_VALUES[0]:.1f} … {LAMBDA_VALUES[-1]:.1f}]")

    for split in ("train", "test", "val"):
        print(f"\n{'─' * 70}")
        print(f"  Split: {split}  ({SPLIT_LABELS[split]})")
        print(f"{'─' * 70}")

        try:
            state, tickers, date = _load_split_snapshot(split)
            mvo = extract_mvo_inputs(state)

            print(f"  Date: {date.date()}  |  "
                  f"vol20d mean={mvo['vol20d'].mean():.4f}  "
                  f"avg_corr={mvo['avg_correlation']:.4f}  "
                  f"regime={mvo['regime']:+.0f}")

            # ── Show compression stats ─────────────────────────────────────────
            from features import load_bundle
            bundle = load_bundle(split)
            t_mid  = len(bundle.dates) // 2
            cov_full = bundle.cov[t_mid]
            cov_comp = build_compressed_covariance(
                mvo["vol20d"].astype(np.float64), mvo["avg_correlation"]
            )
            N_assets = len(tickers)
            full_params = (N_assets * (N_assets + 1)) // 2
            frob_err = np.linalg.norm(cov_full - cov_comp, "fro")
            frob_ref = np.linalg.norm(cov_full, "fro")
            print(f"  Covariance: {full_params} params → 9 params  "
                  f"(Frobenius error {frob_err/frob_ref*100:.1f}%)")

            # ── Scan the full frontier and print the table ─────────────────────
            results = scan_frontier(state, tickers)

            print(f"\n  {'λ':>6}  {'E[ret]':>8}  {'E[vol]':>8}  "
                  f"{'Sharpe':>7}  {'cash%':>6}  Top holdings")
            print(f"  {'─'*66}")
            for r in results:
                top3 = sorted(zip(tickers, r.stock_weights),
                              key=lambda x: x[1], reverse=True)[:3]
                top3_str = "  ".join(
                    f"{t.split('.')[0]}={w:.2f}" for t, w in top3 if w > 0.001
                )
                print(f"  {r.lam:>6.2f}  {r.exp_return:>+8.4f}  "
                      f"{r.exp_volatility:>8.4f}  {r.sharpe:>+7.3f}  "
                      f"{r.cash_weight*100:>5.1f}%  {top3_str}")

            # ── Show effect of caller-chosen λ ─────────────────────────────────
            print(f"\n  Caller-supplied λ examples:")
            for lam_example in [0.1, 1.0, 2.0]:
                r = optimize(state, tickers, lam_example)
                print(f"    λ = {lam_example:.1f}  →  {r.summary()}")

        except FileNotFoundError as exc:
            print(f"  ✗  {exc}")

    print(f"\n{'=' * 70}")
    print("  API:  from efficient_frontier import optimize, scan_frontier")
    print("        result = optimize(state, tickers, lam=0.8)  # caller picks λ")
    print("        results = scan_frontier(state, tickers)      # view full curve")
