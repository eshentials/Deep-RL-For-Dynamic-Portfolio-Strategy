"""
efficient_frontier.py — Mean-Variance Optimizer (MVO)

Solves the parametric Markowitz programme for a caller-supplied λ:

    max  μᵀw  −  (λ / 2) wᵀ Σ̂ w − γ·turnover
    s.t. Σwᵢ ≤ max_equity,   0 ≤ wᵢ ≤ MAX_WEIGHT

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
MAX_WEIGHT = 0.25    # maximum single-asset weight (concentration limit)
MIN_WEIGHT = 0.00    # no short-selling
RISK_FREE  = 0.067   # India 10-yr G-Sec (annualised), for Sharpe computation
TURNOVER_PENALTY = 0.005  # soft objective penalty; realised TC is charged in env.py
DIVERSIFICATION_PENALTY = 0.008  # discourages single-name corner portfolios

# ── Expected-return blending (smooth + short-horizon) ─────────────────────────
ALPHA_5_MOM = 0.25
ALPHA_20_MOM = 0.35
ALPHA_60_MOM = 0.20
ALPHA_TREND = 0.20
ALPHA_RESIDUAL = 0.10   # tilt away from overcrowded momentum

SHRINK_MU_TARGET = 0.0           # shrink expected returns toward this (risk-free proxy)
VOL_PEN_MU_FACTOR = 0.35          # attenuate µ when vol spikes vs cross-section median
EQUITY_RISK_PREMIUM = 0.08        # broad annual equity return prior
MU_ALPHA_SCALE = 0.08           # rank signal maps to roughly +/-5% annual alpha

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

def _compute_long_horizon_signals(
    *,
    momentum_5d: np.ndarray,
    momentum_20d: np.ndarray,
    momentum_60d: np.ndarray,
    ma_spread: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend multi-horizon momentum and crude mean-reversion after spikes."""
    base = (
        ALPHA_5_MOM * momentum_5d
        + ALPHA_20_MOM * momentum_20d
        + ALPHA_60_MOM * momentum_60d
        + ALPHA_TREND * ma_spread
    )
    spike = momentum_20d > np.quantile(momentum_20d, 0.85)
    residual = ALPHA_RESIDUAL * (-momentum_5d)
    tilt = np.where(spike, residual, 0.0)
    return base, tilt


def expected_returns_advanced(
    mvo_inputs: dict,
    *,
    mom20d_t: np.ndarray,
    mom60d_t: np.ndarray,
    daily_ret_t: np.ndarray,
    shrink: float = 0.35,
    rank_normalize: bool = True,
) -> np.ndarray:
    """
    Robust expected-return vector:
      • multi-horizon momentum + MA trend
      • anti-chase residual when 20d momentum spikes
      • shrink toward neutral
      • down-weight noisy names via relative vol
      • rank-normalize cross-section before returning
    """
    mom5 = mvo_inputs["momentum_5d"].astype(np.float64)
    trend = mvo_inputs["ma_spread"].astype(np.float64)
    vol20_ann = mvo_inputs["vol20d"].astype(np.float64)

    mom20 = np.asarray(mom20d_t, dtype=np.float64)
    mom60 = np.asarray(mom60d_t, dtype=np.float64)
    drift = np.asarray(daily_ret_t, dtype=np.float64)

    base, tilt = _compute_long_horizon_signals(
        momentum_5d=mom5,
        momentum_20d=mom20,
        momentum_60d=mom60,
        ma_spread=trend,
    )
    mu = base + tilt

    # ── NEW: Sharpe-adjusted momentum ──
    sharpe_signal = mom20 / (vol20_ann + EPS)
    mu += 0.3 * sharpe_signal

    median_vol = float(np.median(vol20_ann))
    rel_vol = vol20_ann / max(median_vol, EPS)
    vol_discount = np.exp(-0.6 * np.clip(rel_vol - 1.0, -0.5, 4.0))
    mu *= vol_discount

    mu = SHRINK_MU_TARGET + (1.0 - shrink) * (mu - SHRINK_MU_TARGET)

    if rank_normalize and len(mu) > 1:
        order = np.argsort(mu)
        ranks = np.empty_like(order)
        ranks[order] = np.linspace(0.0, 1.0, num=len(mu), endpoint=True)
        mu = EQUITY_RISK_PREMIUM + (2.0 * ranks - 1.0) * MU_ALPHA_SCALE

    mu = mu + 0.25 * drift
        # ── NEW: drawdown filter (penalize recent losers) ──
    drawdown_proxy = -np.minimum(drift, 0.0)
    mu -= 0.2 * drawdown_proxy
    mu = np.clip(mu, 0.0, EQUITY_RISK_PREMIUM + MU_ALPHA_SCALE * 1.5)
    return mu


def ensure_psd_cov(cov: np.ndarray) -> np.ndarray:
    """Symmetrize and add jitter if needed for numerical PSD."""
    cov = 0.5 * (cov + cov.T)
    jitter = EPS
    for _ in range(5):
        w, _ = np.linalg.eigh(cov + jitter * np.eye(cov.shape[0]))
        if float(w.min()) > -1e-8:
            return cov + jitter * np.eye(cov.shape[0])
        jitter *= 10.0
    return cov + 1e-6 * np.eye(cov.shape[0])


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CORE MVO FORMULA
# ══════════════════════════════════════════════════════════════════════════════

def solve_mvo(mu: np.ndarray,
              cov: np.ndarray,
              lam: float,
              max_weight: float = MAX_WEIGHT,
              current_weights: np.ndarray | None = None,
              max_equity: float = 1.0,
              min_equity: float = 0.0,
              turnover_penalty: float = TURNOVER_PENALTY) -> np.ndarray:
    """
    Solve the mean-variance programme for the given risk-aversion λ.

    Objective (to minimise):
        f(w) = (λ/2) wᵀ Σ̂ w  −  μᵀ w + γΣ|w − w_old|

    Constraints:
        min_equity ≤ Σwᵢ ≤ max_equity
        0 ≤ wᵢ ≤ max_weight   (no shorting, concentration cap)

    Solver: scipy SLSQP — handles inequality + box constraints directly.

    Parameters
    ──────────
    mu         : (N,) expected return vector
    cov        : (N, N) covariance matrix  (compressed or full)
    lam        : risk-aversion scalar  (caller's responsibility to supply)
    max_weight       : per-asset upper bound
    current_weights  : current (N+1,) or (N,) weights, used for turnover penalty
    max_equity       : maximum stock exposure; leftover is cash
    turnover_penalty : objective penalty on stock turnover

    Returns
    ───────
    (N,) stock weight vector, sums to <= max_equity, all entries in [0, max_weight].
    Falls back to equal-weight if solver fails.
    """
    N  = len(mu)
    max_equity = float(np.clip(max_equity, 0.0, 1.0))
    min_equity = float(np.clip(min_equity, 0.0, max_equity))

    if current_weights is None:
        w_prev = np.zeros(N, dtype=np.float64)
    else:
        w_prev = np.asarray(current_weights, dtype=np.float64)[:N]
    w_prev = np.clip(w_prev, 0.0, max_weight)

    if w_prev.sum() > max_equity + EPS:
        w_prev *= max_equity / max(w_prev.sum(), EPS)
    if w_prev.sum() < min_equity - EPS:
        w_prev = np.ones(N) * min(max(min_equity / N, MIN_WEIGHT), max_weight)

    if w_prev.sum() > EPS:
        w0 = w_prev.copy()
    else:
        start_equity = max(min_equity, min(max_equity, 1.0))
        w0 = np.ones(N) * min(start_equity / N, max_weight)

    def _obj(w):
        risk = 0.5 * lam * float(w @ cov @ w)
        ret = float(mu @ w)
        turnover = float(np.abs(w - w_prev).sum())
        concentration = float(w @ w)
        return risk - ret + turnover_penalty * turnover + DIVERSIFICATION_PENALTY * concentration

    result = minimize(
        _obj, w0,
        method="SLSQP",
        bounds=[(MIN_WEIGHT, max_weight)] * N,
        constraints=[
            {"type": "ineq", "fun": lambda w: max_equity - w.sum()},
            {"type": "ineq", "fun": lambda w: w.sum() - min_equity},
        ],
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    if not result.success:
        return w0.copy()

    w     = np.clip(result.x, 0.0, max_weight)
    if w.sum() > max_equity + EPS:
        w *= max_equity / max(w.sum(), EPS)
    if w.sum() < min_equity - EPS and w.sum() > EPS:
        w *= min_equity / max(w.sum(), EPS)
        w = np.clip(w, 0.0, max_weight)
    return w


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
             lam:     float,
             current_weights: np.ndarray | None = None,
             max_equity: float = 1.0,
             min_equity: float = 0.0,
             turnover_penalty: float = TURNOVER_PENALTY,
             cov_matrix: np.ndarray | None = None,
             mom20d_t: np.ndarray | None = None,
             mom60d_t: np.ndarray | None = None,
             daily_ret_t: np.ndarray | None = None) -> MVOResult:
    """
    Solve the MVO for a single caller-supplied λ.

    This is the primary entry point.  The caller decides λ; this function
    only executes the formula.

    Parameters
    ──────────
    state   : state vector from state_spec.build_state_vector
    tickers : list of N stock ticker strings  (must match STATE.VOL20D length)
    lam     : risk-aversion parameter  (e.g. from LAMBDA_VALUES[k])
    current_weights : current portfolio weights, used to avoid costly turnover
    max_equity : maximum stock exposure; leftover becomes cash

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

    if cov_matrix is not None:
        cov = ensure_psd_cov(np.asarray(cov_matrix, dtype=np.float64))
    else:
        cov = build_compressed_covariance(vol20d, avg_corr)

    if (
        mom20d_t is not None
        and mom60d_t is not None
        and daily_ret_t is not None
    ):
        mu = expected_returns_advanced(
            mvo,
            mom20d_t=np.asarray(mom20d_t, dtype=np.float64),
            mom60d_t=np.asarray(mom60d_t, dtype=np.float64),
            daily_ret_t=np.asarray(daily_ret_t, dtype=np.float64),
        )
    else:
        mom20 = np.zeros_like(vol20d)
        mom60 = np.zeros_like(vol20d)
        mu = expected_returns_advanced(
            mvo,
            mom20d_t=mom20,
            mom60d_t=mom60,
            daily_ret_t=np.zeros_like(vol20d),
        )

    if current_weights is None:
        current_weights = mvo["current_weights"]

    w_stocks = solve_mvo(
        mu,
        cov,
        lam,
        current_weights=current_weights,
        max_equity=max_equity,
        min_equity=min_equity,
        turnover_penalty=turnover_penalty,
    )

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


def horizon_momentum_pct(prices: pd.DataFrame, t_mid: int, window: int) -> np.ndarray:
    """Rolling simple return over `window` days ending at row t."""
    px = prices.to_numpy(dtype=np.float64)
    if t_mid <= 0:
        return np.zeros(prices.shape[1], dtype=np.float64)
    lag = max(t_mid - window, 0)
    prev = px[lag].copy()
    prev[prev <= 0] = EPS
    return (px[t_mid] / prev - 1.0).astype(np.float64)


def scan_frontier(
    state:        np.ndarray,
    tickers:      list[str],
    lambda_values: np.ndarray = LAMBDA_VALUES,
    *,
    cov_matrix: np.ndarray | None = None,
    mom20d_t: np.ndarray | None = None,
    mom60d_t: np.ndarray | None = None,
    daily_ret_t: np.ndarray | None = None,
    current_weights: np.ndarray | None = None,
    max_equity: float = 1.0,
    min_equity: float = 0.0,
    turnover_penalty: float = TURNOVER_PENALTY,
) -> list[MVOResult]:
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
    return [
        optimize(
            state,
            tickers,
            float(lam),
            current_weights=current_weights,
            max_equity=max_equity,
            min_equity=min_equity,
            turnover_penalty=turnover_penalty,
            cov_matrix=cov_matrix,
            mom20d_t=mom20d_t,
            mom60d_t=mom60d_t,
            daily_ret_t=daily_ret_t,
        )
        for lam in lambda_values
    ]


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
    volume_path = os.path.join("data", f"volume_{SPLIT_LABELS[split]}.csv")
    volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
    volume = volume.reindex(index=bundle.dates, columns=bundle.tickers).ffill().fillna(0.0)
    adv20 = volume.rolling(20, min_periods=1).mean().replace(0.0, EPS)

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
        volume_t    = volume.iloc[t].values.astype(np.float32),
        adv_t       = adv20.iloc[t].values.astype(np.float32),
        prices_t    = bundle.prices.iloc[t].values.astype(np.float32),
    )
    return state, bundle.tickers, date, w_eq

if __name__ == "__main__":
    TICKERS = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "NIFTYBEES.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    ]

    print("=" * 70)
    print("  Mean-Variance Optimizer  —  formula: max μᵀw − (λ/2) wᵀΣ̂w")
    print("=" * 70)
    print(f"  Constraints : 0 ≤ wᵢ ≤ {MAX_WEIGHT}   Σwᵢ ≤ max_equity")
    print(f"  Covariance    : Ledoit-Wolf Σ from features (+ compressed fallback)")
    print(f"  λ grid      : {len(LAMBDA_VALUES)} values  "
          f"[{LAMBDA_VALUES[0]:.1f} … {LAMBDA_VALUES[-1]:.1f}]")

    for split in ("train", "test", "val"):
        print(f"\n{'─' * 70}")
        print(f"  Split: {split}  ({SPLIT_LABELS[split]})")
        print(f"{'─' * 70}")

        try:
            state, tickers, date, w_eq = _load_split_snapshot(split)   
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

            mom20 = horizon_momentum_pct(bundle.prices, t_mid, 20)
            mom60 = horizon_momentum_pct(bundle.prices, t_mid, 60)
            cov_full = bundle.cov[t_mid]
            daily = bundle.daily_ret.iloc[t_mid].values.astype(np.float64)

            # ── Scan the full frontier and print the table ─────────────────────
            results = scan_frontier(
                state,
                tickers,
                cov_matrix=cov_full,
                mom20d_t=mom20,
                mom60d_t=mom60,
                daily_ret_t=daily,
                current_weights=w_eq[:-1],
            )

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
                r = optimize(
                    state,
                    tickers,
                    lam_example,
                    cov_matrix=cov_full,
                    mom20d_t=mom20,
                    mom60d_t=mom60,
                    daily_ret_t=daily,
                    current_weights=w_eq[:-1],
                )
                print(f"    λ = {lam_example:.1f}  →  {r.summary()}")

        except FileNotFoundError as exc:
            print(f"  ✗  {exc}")

    print(f"\n{'=' * 70}")
    print("  API:  from efficient_frontier import optimize, scan_frontier")
    print("        result = optimize(state, tickers, lam=0.8)  # caller picks λ")
    print("        results = scan_frontier(state, tickers)      # view full curve")

def expected_returns_from_state(state, n_assets):
    """
    Extract expected returns vector from state.
    
    Assumes first n_assets elements of state correspond to returns.
    Adjust if your state structure is different.
    """
    import numpy as np

    if state is None:
        return np.zeros(n_assets, dtype=float)

    state = np.asarray(state)

    # simplest assumption: first n_assets = returns
    if len(state) >= n_assets:
        return state[:n_assets]

    # fallback (pad if needed)
    out = np.zeros(n_assets, dtype=float)
    out[:len(state)] = state
    return out