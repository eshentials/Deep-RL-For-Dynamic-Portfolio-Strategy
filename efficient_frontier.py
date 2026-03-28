"""
efficient_frontier.py — Mean-variance efficient frontier via compressed
                        covariance and risk-aversion parameter λ.

Solves the parametric Markowitz problem for 20 values of λ ∈ [0.1, 2.0]:

    max  μᵀw  −  (λ / 2) wᵀ Σ̂ w
    s.t. Σwᵢ = 1,   0 ≤ wᵢ ≤ MAX_WEIGHT

All inputs are extracted from the 49-dim state vector defined in state_spec.py.
No external price data or full (T, N, N) covariance array is required.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compressed covariance  (constant-correlation model)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Σ̂[i, i] = σᵢ²                         (diagonal — per-asset variance)
  Σ̂[i, j] = ρ̄ × σᵢ × σⱼ   (i ≠ j)      (off-diagonal — avg correlation)

  σᵢ = vol20d[i]  from STATE.VOL20D     (annualised, already in state)
  ρ̄  = avg_corr   from STATE.AVG_CORR   (scalar — mean pairwise corr)

  → only 9 numbers (8 σ + 1 ρ̄) instead of 36 (full 8×8 lower-triangle).

  Why constant-correlation compression?
  ──────────────────────────────────────
  · The full LW covariance requires loading a (T, 8, 8) array (~2 000 frames
    × 64 floats per frame), which is heavy in a tight RL inner loop.
  · The constant-correlation model is a well-studied (Elton & Gruber 1973)
    structural simplification that retains the dominant risk structure
    (individual volatilities + a single market-wide correlation).
  · It is always PSD by construction when ρ̄ ∈ (−1/(N−1), 1].
  · For N = 8 liquid large-cap Indian stocks the pairwise correlations are
    indeed tightly clustered (avg ≈ 0.5–0.7), making the approximation
    particularly accurate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Expected-return signal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  μᵢ = α_mom × momentum_5d[i]  +  α_trend × ma_spread[i]

  Both signals come from the state vector (STATE.MOM5D, STATE.MA_SPREAD).
  · momentum_5d  → fast short-term reversal / momentum signal (~1 week)
  · ma_spread    → slow trend-following signal (~months)
  Tunable weights α_mom, α_trend default to 0.60 and 0.40.

  Regime adjustment
  ─────────────────
  effective_λ = λ_base × (1 + REGIME_SCALE × |regime|)
  · Bear (regime=−1) or Bull (regime=+1) both increase effective λ,
    making the optimizer more risk-averse at market extremes.
  · Neutral (regime=0) leaves λ unchanged.
  REGIME_SCALE = 0.30 by default.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
20 lambda values
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LAMBDA_VALUES = np.linspace(0.1, 2.0, 20)

  λ = 0.1  → near-aggressive / return-maximising
  λ = 1.0  → standard balanced (Sharpe-ish)
  λ = 2.0  → near minimum-variance / conservative
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from state_spec import STATE, extract_mvo_inputs

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LAMBDA_VALUES: np.ndarray = np.linspace(0.1, 2.0, 20)  # 20 risk-aversion levels

MAX_WEIGHT    = 0.40    # concentration limit per asset
MIN_WEIGHT    = 0.00    # no short-selling
RISK_FREE     = 0.067   # India 10-yr G-Sec yield (annualised)
EPS           = 1e-9    # numerical floor

# Expected-return signal weights (tunable)
ALPHA_MOM     = 0.60    # weight on 5-day momentum signal
ALPHA_TREND   = 0.40    # weight on MA-spread trend signal

# Regime-based λ scaling  (bear/bull → more conservative)
REGIME_SCALE  = 0.30

# Paths
DATA_DIR    = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "features")
RESULTS_DIR = os.path.join(DATA_DIR, "ef_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}


# ══════════════════════════════════════════════════════════════════════════════
# COMPRESSED COVARIANCE  (constant-correlation model)
# ══════════════════════════════════════════════════════════════════════════════

def build_compressed_covariance(vol20d: np.ndarray,
                                 avg_corr: float) -> np.ndarray:
    """
    Build an (N, N) annualised covariance matrix from N volatilities and
    a single average pairwise correlation (constant-correlation model).

    Parameters
    ──────────
    vol20d   : (N,) annualised per-asset volatilities from STATE.VOL20D.
    avg_corr : scalar — mean pairwise correlation from STATE.AVG_CORR.

    Construction
    ────────────
    Σ̂ = D × C × D
    where D = diag(vol20d),  C[i,j] = ρ̄ for i≠j,  C[i,i] = 1.

    PSD guarantee
    ─────────────
    The constant-correlation matrix C is PSD if and only if
        ρ̄ ≥ −1 / (N − 1).
    For N = 8 this means ρ̄ ≥ −0.143.  We clip ρ̄ to [−0.14, 0.99]
    before construction to guarantee PSD regardless of input noise.

    Returns
    ───────
    np.ndarray of shape (N, N), symmetric, guaranteed PSD.
    """
    N       = len(vol20d)
    min_rho = -1.0 / (N - 1) + 1e-4
    rho     = float(np.clip(avg_corr, min_rho, 0.99))

    # Constant-correlation matrix C
    C = np.full((N, N), rho)
    np.fill_diagonal(C, 1.0)

    # Scale by outer product of vols: Σ̂ = D C D
    D    = np.diag(vol20d.clip(min=EPS))
    cov  = D @ C @ D

    # Symmetrise and add tiny jitter to diagonal for numerical safety
    cov  = 0.5 * (cov + cov.T)
    cov += np.eye(N) * EPS

    return cov


# ══════════════════════════════════════════════════════════════════════════════
# EXPECTED RETURNS FROM STATE VECTOR
# ══════════════════════════════════════════════════════════════════════════════

def expected_returns_from_state(mvo_inputs: dict,
                                 alpha_mom:   float = ALPHA_MOM,
                                 alpha_trend: float = ALPHA_TREND) -> np.ndarray:
    """
    Estimate per-asset expected returns from two state-vector signals.

    Signal combination
    ──────────────────
    μᵢ = α_mom × momentum_5d[i]  +  α_trend × ma_spread[i]

    · momentum_5d[i] ∈ [−0.6, 0.6]  — already clipped in features.py
    · ma_spread[i]   ∈ [−0.5, 0.5]  — already clipped in features.py

    Both signals are dimensionless fractions.  The combined μ is therefore
    comparable in scale across assets and does not need re-normalisation.

    Why not annualise?
    ──────────────────
    Annualising 5-day momentum by ×(252/5) = ×50 would produce extreme μ
    (a 2% weekly move → 100% annual).  Keeping the signals in their natural
    units and letting λ control the risk-return tradeoff is more stable for
    the MVO solver.

    Returns
    ───────
    np.ndarray of shape (N,), unconstrained range (but typically ±0.3).
    """
    mom   = mvo_inputs["momentum_5d"].astype(np.float64)
    trend = mvo_inputs["ma_spread"].astype(np.float64)
    mu    = alpha_mom * mom + alpha_trend * trend
    return mu


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE-λ MVO SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_mvo_lambda(mu: np.ndarray,
                     cov: np.ndarray,
                     lam: float,
                     max_weight: float = MAX_WEIGHT) -> np.ndarray:
    """
    Solve the mean-variance programme for a given risk-aversion λ:

        min   (λ/2) wᵀ Σ̂ w  −  μᵀ w
        s.t.  Σwᵢ = 1,   0 ≤ wᵢ ≤ max_weight

    Solver: scipy SLSQP — handles box + equality constraints cleanly,
    typically converges in < 50 iterations for N = 8.

    Warm start: equal-weight initialisation is always feasible and close
    to the optimum for moderate λ.  For extreme λ (near 0 or very large),
    the solver may need a few extra iterations.

    Returns
    ───────
    np.ndarray of shape (N,) — stock-only weights, sum = 1.0.
    (The cash slot is computed externally as 1 − sum(w) if MAX_WEIGHT < 1.)
    """
    N   = len(mu)
    w0  = np.ones(N, dtype=np.float64) / N

    def objective(w: np.ndarray) -> float:
        return 0.5 * lam * float(w @ cov @ w) - float(mu @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return lam * (cov @ w) - mu

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds      = [(MIN_WEIGHT, max_weight)] * N

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not result.success:
        # Fallback: return equal-weight rather than a bad solution
        return w0.copy()

    w = np.clip(result.x, 0.0, max_weight)
    # Re-normalise for floating-point drift
    w_sum = w.sum()
    return w / w_sum if w_sum > EPS else w0


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def portfolio_stats(w: np.ndarray,
                    mu: np.ndarray,
                    cov: np.ndarray) -> tuple[float, float, float]:
    """
    Compute (expected_return, expected_volatility, Sharpe) for stock weights w.

    Returns annualised quantities (cov is already annualised by build_compressed_covariance).
    """
    exp_ret = float(w @ mu)
    exp_vol = float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))
    sharpe  = (exp_ret - RISK_FREE) / (exp_vol + EPS)
    return exp_ret, exp_vol, sharpe


# ══════════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER RESULT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EFPoint:
    """One point on the efficient frontier (one λ value)."""
    lambda_idx:      int
    lambda_base:     float          # raw λ from LAMBDA_VALUES
    lambda_eff:      float          # regime-adjusted effective λ
    weights:         np.ndarray     # (N,) stock-only weights
    cash_weight:     float          # 1 − sum(weights)
    exp_return:      float
    exp_volatility:  float
    sharpe:          float
    tickers:         list[str]

    def to_dict(self) -> dict:
        return {
            "lambda_idx":     self.lambda_idx,
            "lambda_base":    round(self.lambda_base,   4),
            "lambda_eff":     round(self.lambda_eff,    4),
            "exp_return":     round(self.exp_return,    6),
            "exp_volatility": round(self.exp_volatility, 6),
            "sharpe":         round(self.sharpe,        6),
            "cash_weight":    round(self.cash_weight,   6),
            "weights":        {t: round(float(w), 6)
                               for t, w in zip(self.tickers, self.weights)},
        }


@dataclass
class EfficientFrontierResult:
    """All 20 EF points computed from a single state vector."""
    tickers:       list[str]
    regime:        float
    avg_corr:      float
    points:        list[EFPoint] = field(default_factory=list)

    # ── Derived properties ────────────────────────────────────────────────────

    def weights_matrix(self) -> np.ndarray:
        """(20, N) array — one row per λ, stock-only weights."""
        return np.stack([p.weights for p in self.points])

    def returns_curve(self) -> np.ndarray:
        """(20,) expected return at each λ."""
        return np.array([p.exp_return for p in self.points])

    def volatility_curve(self) -> np.ndarray:
        """(20,) expected volatility at each λ."""
        return np.array([p.exp_volatility for p in self.points])

    def sharpe_curve(self) -> np.ndarray:
        """(20,) Sharpe ratio at each λ."""
        return np.array([p.sharpe for p in self.points])

    def best_sharpe(self) -> EFPoint:
        """The λ with the highest Sharpe ratio."""
        return max(self.points, key=lambda p: p.sharpe)

    def at_lambda(self, lam: float) -> EFPoint:
        """Return the EFPoint whose base λ is closest to `lam`."""
        return min(self.points, key=lambda p: abs(p.lambda_base - lam))

    def print_table(self) -> None:
        """Print a human-readable summary table."""
        print(f"\n  Efficient Frontier  "
              f"(regime={self.regime:+.0f}  avg_corr={self.avg_corr:.3f})")
        print(f"  {'λ_base':>7}  {'λ_eff':>7}  "
              f"{'E[ret]':>8}  {'E[vol]':>8}  {'Sharpe':>7}  "
              f"{'cash%':>6}  Top-3 holdings")
        print("  " + "─" * 80)
        for p in self.points:
            top3 = sorted(
                zip(self.tickers, p.weights),
                key=lambda x: x[1], reverse=True
            )[:3]
            top3_str = "  ".join(f"{t.split('.')[0]}={w:.2f}" for t, w in top3 if w > 0.001)
            star = " ◀" if p is self.best_sharpe() else ""
            print(f"  {p.lambda_base:>7.2f}  {p.lambda_eff:>7.2f}  "
                  f"{p.exp_return:>+8.4f}  {p.exp_volatility:>8.4f}  "
                  f"{p.sharpe:>+7.3f}  "
                  f"{p.cash_weight*100:>5.1f}%  {top3_str}{star}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_efficient_frontier(
    state:       np.ndarray,
    tickers:     list[str],
    lambda_values: np.ndarray = LAMBDA_VALUES,
    regime_scale:  float      = REGIME_SCALE,
    alpha_mom:     float      = ALPHA_MOM,
    alpha_trend:   float      = ALPHA_TREND,
    max_weight:    float      = MAX_WEIGHT,
) -> EfficientFrontierResult:
    """
    Build the full efficient frontier from a 49-dim state vector.

    Steps
    ─────
    1. Extract MVO inputs from the state vector (via state_spec.extract_mvo_inputs).
    2. Build compressed (N, N) covariance from vol20d and avg_correlation.
    3. Build expected-return vector μ from momentum_5d and ma_spread.
    4. For each of the 20 λ values, scale by regime and solve the MVO.
    5. Compute portfolio statistics and package into EFPoint objects.

    Parameters
    ──────────
    state        : (49,) float32 state vector — output of state_spec.build_state_vector
    tickers      : list of N stock ticker strings (length must equal len(STATE.VOL20D))
    lambda_values: (20,) array of base λ values  (default: np.linspace(0.1, 2.0, 20))
    regime_scale : how much |regime| inflates effective λ  (default 0.30)
    alpha_mom    : signal weight for 5-day momentum          (default 0.60)
    alpha_trend  : signal weight for MA-spread               (default 0.40)
    max_weight   : concentration cap per asset               (default 0.40)

    Returns
    ───────
    EfficientFrontierResult with 20 EFPoint objects.
    """
    assert state.shape == (STATE.DIM,), (
        f"Expected state shape ({STATE.DIM},), got {state.shape}"
    )

    # ── 1. Extract MVO inputs from state vector ───────────────────────────────
    mvo = extract_mvo_inputs(state)

    vol20d   = mvo["vol20d"].astype(np.float64)
    avg_corr = mvo["avg_correlation"]
    regime   = mvo["regime"]

    # ── 2. Compressed covariance ──────────────────────────────────────────────
    cov = build_compressed_covariance(vol20d, avg_corr)

    # ── 3. Expected returns ───────────────────────────────────────────────────
    mu = expected_returns_from_state(mvo, alpha_mom=alpha_mom, alpha_trend=alpha_trend)

    # ── 4. Solve for each λ ───────────────────────────────────────────────────
    result = EfficientFrontierResult(
        tickers=tickers, regime=regime, avg_corr=avg_corr,
    )

    for idx, lam_base in enumerate(lambda_values):
        # Regime adjustment: bear or bull → more conservative
        lam_eff = float(lam_base * (1.0 + regime_scale * abs(regime)))

        w         = solve_mvo_lambda(mu, cov, lam_eff, max_weight=max_weight)
        exp_r, exp_v, sharpe = portfolio_stats(w, mu, cov)

        # Cash is anything not allocated to stocks
        cash = float(np.clip(1.0 - w.sum(), 0.0, 1.0))

        result.points.append(EFPoint(
            lambda_idx=idx,
            lambda_base=float(lam_base),
            lambda_eff=lam_eff,
            weights=w,
            cash_weight=cash,
            exp_return=exp_r,
            exp_volatility=exp_v,
            sharpe=sharpe,
            tickers=tickers,
        ))

    return result


# ══════════════════════════════════════════════════════════════════════════════
# STEP-LEVEL CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def optimize_from_state(
    state:      np.ndarray,
    tickers:    list[str],
    lambda_idx: int   = 10,
) -> np.ndarray:
    """
    Single-call interface for the RL env and MVO strategy:
    given a state vector and a lambda index [0, 19], return (N+1,) portfolio
    weights including a cash slot.

    This is the function env.py and the MVO strategy should call at each step.

    Parameters
    ──────────
    state      : (49,) state vector from state_spec.build_state_vector
    tickers    : list of N stock tickers (len = 8 for this project)
    lambda_idx : which of the 20 EF points to use  (0 = aggressive, 19 = conservative)

    Returns
    ───────
    np.ndarray of shape (N+1,) — stock weights + cash weight, sum = 1.0.
    """
    ef     = build_efficient_frontier(state, tickers)
    pt     = ef.points[lambda_idx]
    cash_w = np.array([pt.cash_weight], dtype=np.float32)
    return np.concatenate([pt.weights.astype(np.float32), cash_w])


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / LOAD
# ══════════════════════════════════════════════════════════════════════════════

def save_ef_result(result: EfficientFrontierResult,
                   label: str) -> None:
    """Save the 20-point EF result to disk (NPZ + human-readable CSV)."""
    npz_path = os.path.join(RESULTS_DIR, f"ef_portfolios_{label}.npz")
    csv_path = os.path.join(RESULTS_DIR, f"ef_portfolios_{label}.csv")

    np.savez(
        npz_path,
        weights=result.weights_matrix(),
        lambdas_base=np.array([p.lambda_base for p in result.points]),
        lambdas_eff=np.array([p.lambda_eff   for p in result.points]),
        exp_returns=result.returns_curve(),
        exp_vols=result.volatility_curve(),
        sharpes=result.sharpe_curve(),
        cash_weights=np.array([p.cash_weight for p in result.points]),
    )

    rows = []
    for p in result.points:
        row = {
            "lambda_base": p.lambda_base,
            "lambda_eff":  p.lambda_eff,
            "exp_return":  p.exp_return,
            "exp_vol":     p.exp_volatility,
            "sharpe":      p.sharpe,
            "cash_pct":    round(p.cash_weight * 100, 2),
        }
        row.update({t: round(float(w), 6) for t, w in zip(p.tickers, p.weights)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"  Saved → {npz_path}")
    print(f"  Saved → {csv_path}")


def load_ef_weights(label: str) -> np.ndarray:
    """Load the (20, N) weight matrix for a given split label."""
    path = os.path.join(RESULTS_DIR, f"ef_portfolios_{label}.npz")
    return np.load(path)["weights"]


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run on all three splits using real feature data
# ══════════════════════════════════════════════════════════════════════════════

def _load_split_features(split: str) -> dict:
    """
    Load feature CSVs for a split and return a dict of DataFrames.
    This is used in __main__ to build realistic state vectors without
    needing the full RL episode machinery.
    """
    label = SPLIT_LABELS[split]

    def _csv(name: str) -> pd.DataFrame:
        path = os.path.join(FEATURE_DIR, f"{name}_{label}.csv")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _npy(name: str) -> np.ndarray:
        path = os.path.join(FEATURE_DIR, f"{name}_{label}.npy")
        return np.load(path)

    return {
        "daily_ret":    _csv("daily_returns"),
        "momentum":     _csv("momentum_5d"),
        "vol20d":       _csv("realized_vol_20d"),
        "ma_spread":    _csv("ma_spread_50_200"),
        "cov_array":    _npy("covariance_90d"),       # (T, N, N) — used only for avg_corr
        "tickers":      list(_csv("daily_returns").columns),
    }


def _state_from_features(feats: dict, t: int) -> np.ndarray:
    """
    Construct a minimal 49-dim state vector at row index t from feature dicts.

    For the standalone demo we synthesise the cross-asset, macro, and portfolio
    groups from the available feature data so that STATE slice indices are
    respected exactly.  The RL environment will supply these via
    state_spec.build_state_vector() at every step.
    """
    N = len(feats["tickers"])

    # Per-asset group (32 scalars)
    daily_ret  = feats["daily_ret"].iloc[t].values.astype(np.float32)
    momentum   = feats["momentum"].iloc[t].values.astype(np.float32)
    vol20d     = feats["vol20d"].iloc[t].values.astype(np.float32)
    ma_spread  = feats["ma_spread"].iloc[t].values.astype(np.float32)
    per_asset  = np.concatenate([daily_ret, momentum, vol20d, ma_spread])

    # Cross-asset group (3 scalars) — derive avg_corr from snapshot cov matrix
    cov_t   = feats["cov_array"][t]
    std_vec = np.sqrt(np.diag(cov_t)).clip(min=EPS)
    corr    = cov_t / np.outer(std_vec, std_vec)
    np.fill_diagonal(corr, 0.0)
    iu       = np.triu_indices(N, k=1)
    avg_corr = float(corr[iu].mean())

    w_eq     = np.ones(N) / N
    port_vol = float(np.sqrt(w_eq @ cov_t @ w_eq))

    # Niftybees correlation
    from state_spec import NIFTYBEES_TICKER
    tickers = feats["tickers"]
    if NIFTYBEES_TICKER in tickers:
        idx = tickers.index(NIFTYBEES_TICKER)
        np.fill_diagonal(corr, 1.0)
        nifty_corr = float(np.delete(corr[idx], idx).mean())
    else:
        np.fill_diagonal(corr, 1.0)
        nifty_corr = avg_corr

    cross_asset = np.array([avg_corr, port_vol, nifty_corr], dtype=np.float32)

    # Macro group (4 scalars) — neutral placeholders for standalone demo
    macro = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Portfolio group (10 scalars) — equal-weight, no history
    weights   = w_eq.astype(np.float32)
    pnl       = np.array([0.0], dtype=np.float32)
    drawdown  = np.array([0.0], dtype=np.float32)
    portfolio = np.concatenate([weights, pnl, drawdown])

    state = np.concatenate([per_asset, cross_asset, macro, portfolio])
    assert state.shape == (STATE.DIM,), f"Expected {STATE.DIM}, got {state.shape}"
    return state


if __name__ == "__main__":
    TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "NIFTYBEES.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    ]

    print("=" * 75)
    print("  Efficient Frontier  —  compressed covariance, 20 λ values")
    print("=" * 75)
    print(f"\n  λ grid : {LAMBDA_VALUES.round(3).tolist()}")
    print(f"  N      : {len(TICKERS)} assets")
    print(f"  Model  : constant-correlation Σ̂  "
          f"(9 params instead of 36)")
    print(f"  Regime : effective_λ = λ × (1 + {REGIME_SCALE} × |regime|)")

    for split in ("train", "test", "val"):
        print(f"\n{'─' * 75}")
        print(f"  Split: {split}  ({SPLIT_LABELS[split]})")
        print(f"{'─' * 75}")

        try:
            feats = _load_split_features(split)
            tickers = feats["tickers"]
            T = len(feats["daily_ret"])

            # Pick a representative mid-point timestep
            t_mid = T // 2
            date  = feats["daily_ret"].index[t_mid]
            print(f"\n  Representative date : {date.date()}  (row {t_mid} / {T})")

            state = _state_from_features(feats, t_mid)
            mvo   = extract_mvo_inputs(state)

            print(f"  vol20d (mean)       : {mvo['vol20d'].mean():.4f}  (annualised)")
            print(f"  avg_correlation     : {mvo['avg_correlation']:.4f}")
            print(f"  regime              : {mvo['regime']:+.0f}")

            # Build compressed covariance and show compression stats
            cov_full   = feats["cov_array"][t_mid]
            cov_comp   = build_compressed_covariance(
                mvo["vol20d"].astype(np.float64), mvo["avg_correlation"]
            )
            full_params = (len(tickers) * (len(tickers) + 1)) // 2
            print(f"\n  Covariance compression:")
            print(f"    Full LW matrix  : {full_params} independent params  "
                  f"(lower-tri of {len(tickers)}×{len(tickers)})")
            print(f"    Compressed model: {len(tickers) + 1} params  "
                  f"({len(tickers)} σᵢ + 1 ρ̄)")
            frob_err = np.linalg.norm(cov_full - cov_comp, 'fro')
            frob_ref = np.linalg.norm(cov_full, 'fro')
            print(f"    Frobenius error : {frob_err:.6f}  "
                  f"(relative {frob_err/frob_ref*100:.2f}%)")

            # Run the frontier
            ef = build_efficient_frontier(state, tickers)
            ef.print_table()

            # Best-Sharpe summary
            best = ef.best_sharpe()
            print(f"\n  Best Sharpe → λ={best.lambda_base:.2f}  "
                  f"(idx={best.lambda_idx})  "
                  f"Sharpe={best.sharpe:.3f}  "
                  f"ret={best.exp_return:+.4f}  "
                  f"vol={best.exp_volatility:.4f}")

            # Save
            save_ef_result(ef, SPLIT_LABELS[split])

        except FileNotFoundError as exc:
            print(f"  ✗  {exc}")

    print(f"\n{'=' * 75}")
    print(f"  Results saved to ./{RESULTS_DIR}/")
    print(f"  Import via:  from efficient_frontier import build_efficient_frontier")
    print(f"               from efficient_frontier import optimize_from_state")
