"""
state_spec.py — Canonical state-vector specification for the Deep-RL
                portfolio system.

This file is the single source of truth for:
  · What every element of the state vector means
  · Which slice indices correspond to which feature group
  · How each group is computed from raw feature artefacts

Import this module in env.py, mean_variance_optimizer.py, and evaluate.py
to guarantee every component reads from the same definition.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
State vector layout  (total = 49 scalars)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Slice      Group               Dim   Description
  ─────────────────────────────────────────────────────────────────────
  [0:8]      daily_returns        8    (P_t − P_{t-1}) / P_{t-1} per asset
  [8:16]     momentum_5d          8    (P_t / P_{t-5}) − 1 per asset
  [16:24]    volatility_20d       8    annualised std of 20-day returns per asset
  [24:32]    trend_MA_spread      8    (MA50 − MA200) / MA200 per asset
  ─────────────────────────────────────────────────────────────────────
  [32]       avg_correlation      1    mean pairwise corr across all asset pairs
  [33]       portfolio_volatility 1    √(wᵀ Σ w), annualised
  [34]       niftybees_corr       1    mean corr of each stock with NIFTYBEES
  ─────────────────────────────────────────────────────────────────────
  [35]       vix_change_5d        1    (VIX_t / VIX_{t-5}) − 1
  [36]       crude_change_5d      1    (Crude_t / Crude_{t-5}) − 1
  [37]       usdinr_change_5d     1    (USDINR_t / USDINR_{t-5}) − 1
  [38]       regime_label         1    −1 bear | 0 neutral | +1 bull  (see below)
  ─────────────────────────────────────────────────────────────────────
  [39:47]    current_weights      8    portfolio stock weights (sum ≤ 1)
  [47]       pnl_recent           1    5-day portfolio simple return
  [48]       drawdown             1    (NAV_t − max NAV) / max NAV ∈ [−1, 0]
  ─────────────────────────────────────────────────────────────────────
  TOTAL                          49

Regime label rule
─────────────────
  Bull  (+1) : NIFTYBEES MA-spread > +0.02  AND  VIX < 20
  Bear  (−1) : NIFTYBEES MA-spread < −0.02  OR   VIX > 25
  Neutral (0): otherwise

  Rationale: the 50/200 cross (golden/death cross) on the benchmark ETF
  captures the broad market trend; VIX > 25 signals elevated fear
  regardless of price trend.  The regime scalar allows the policy to
  conditionally weight its other inputs without needing the agent to
  re-discover this nonlinear interaction from scratch.

Mean-variance optimizer usage
──────────────────────────────
  The MVO only needs a subset of this vector:
    · volatility_20d[8]  → per-asset σ for the diagonal of Σ̂
    · avg_correlation     → to fill off-diagonal entries of Σ̂ simply
    · portfolio_volatility → risk constraint target
    · regime_label        → scale risk-aversion parameter κ
  Use the slice constants below (e.g. STATE.VOL20D) to extract them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Paths (mirrors features.py conventions) ───────────────────────────────────
DATA_DIR    = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "features")
MACRO_PATH  = os.path.join(DATA_DIR, "macro_2014_2024.csv")

# ── Transaction cost ──────────────────────────────────────────────────────────
# Flat 0.1% applied to total turnover (sum of |Δw| across stock positions).
# Covers brokerage + STT + exchange charges on NSE at a discount broker.
# Usage in env.step():
#   cost = transaction_cost(w_old, w_new)   # fraction of NAV
#   reward = gross_return - cost
TRANSACTION_COST_RATE: float = 0.001   # 0.1 %


def transaction_cost(w_old: np.ndarray, w_new: np.ndarray) -> float:
    """
    Flat-rate transaction cost as a fraction of portfolio NAV.

    turnover = Σ |w_new[i] − w_old[i]|   (stock positions only, cash excluded)
    cost     = TRANSACTION_COST_RATE × turnover

    The cash slot is the last element of both weight vectors and is excluded
    because moving money to/from cash has no direct brokerage or STT cost.

    Parameters
    ──────────
    w_old : (N+1,) weights before rebalancing  (N stocks + 1 cash)
    w_new : (N+1,) weights after  rebalancing

    Returns
    ───────
    float — cost as a fraction of NAV (subtract from gross return for net reward)
    """
    turnover = float(np.abs(w_new[:-1] - w_old[:-1]).sum())
    return TRANSACTION_COST_RATE * turnover

NIFTYBEES_TICKER = "NIFTYBEES.NS"
MACRO_WINDOW     = 5      # look-back for 5-day macro changes
EPS              = 1e-8


# ══════════════════════════════════════════════════════════════════════════════
# SLICE INDICES  — import these everywhere instead of hardcoding integers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _StateLayout:
    """
    Immutable container of named slice objects.
    Access as:  STATE.DAILY_RET, STATE.WEIGHTS, STATE.REGIME …
    """
    # Per-asset
    DAILY_RET:   slice = slice(0,  8)
    MOM5D:       slice = slice(8,  16)
    VOL20D:      slice = slice(16, 24)
    MA_SPREAD:   slice = slice(24, 32)

    # Cross-asset
    AVG_CORR:    slice = slice(32, 33)
    PORT_VOL:    slice = slice(33, 34)
    NIFTY_CORR:  slice = slice(34, 35)

    # Macro
    VIX5D:       slice = slice(35, 36)
    CRUDE5D:     slice = slice(36, 37)
    USDINR5D:    slice = slice(37, 38)
    REGIME:      slice = slice(38, 39)

    # Portfolio
    WEIGHTS:     slice = slice(39, 47)
    PNL_RECENT:  slice = slice(47, 48)
    DRAWDOWN:    slice = slice(48, 49)

    DIM: int = 49


STATE = _StateLayout()


# ══════════════════════════════════════════════════════════════════════════════
# GROUP BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def per_asset_features(
    daily_ret_t:  np.ndarray,   # (8,)  simple daily return
    momentum_t:   np.ndarray,   # (8,)  5-day momentum
    vol20d_t:     np.ndarray,   # (8,)  20-day annualised volatility
    ma_spread_t:  np.ndarray,   # (8,)  MA50/200 spread
) -> np.ndarray:
    """Concatenate and validate the 32 per-asset scalars."""
    block = np.concatenate([daily_ret_t, momentum_t, vol20d_t, ma_spread_t])
    assert block.shape == (32,), f"per_asset_features: expected (32,), got {block.shape}"
    assert np.isfinite(block).all(), "per_asset_features: non-finite values"
    return block.astype(np.float32)


def cross_asset_features(
    cov_t:   np.ndarray,    # (N, N)  LW-shrunken annualised covariance
    weights: np.ndarray,    # (N,)    stock-only weights (no cash slot here)
    tickers: list[str],
) -> np.ndarray:
    """
    Compute 3 cross-asset scalars.

    avg_correlation
    ────────────────
    Convert covariance → correlation, take the mean of the
    strictly upper-triangle entries (N*(N-1)/2 pairs).
    Values ∈ [−1, 1].

    portfolio_volatility
    ─────────────────────
    σ_p = √(wᵀ Σ w).  Uses only stock weights (cash earns 0 vol).
    Result is annualised because cov_t is already annualised.

    niftybees_corr
    ───────────────
    Row/column of the correlation matrix corresponding to NIFTYBEES.
    Mean of its off-diagonal entries = average correlation of every
    other asset with the benchmark ETF.
    """
    N = cov_t.shape[0]

    # Correlation matrix
    std_vec = np.sqrt(np.diag(cov_t)).clip(min=EPS)
    corr    = cov_t / np.outer(std_vec, std_vec)
    corr    = np.clip(corr, -1.0, 1.0)

    # avg pairwise correlation (upper triangle, no diagonal)
    iu     = np.triu_indices(N, k=1)
    avg_corr = float(corr[iu].mean())

    # portfolio volatility
    w      = weights[:N].clip(0.0)
    norm   = w.sum()
    w      = w / norm if norm > EPS else np.ones(N) / N
    port_vol = float(np.sqrt(w @ cov_t @ w))

    # NIFTYBEES correlation
    if NIFTYBEES_TICKER in tickers:
        idx          = tickers.index(NIFTYBEES_TICKER)
        corr_row     = np.delete(corr[idx], idx)   # remove self-correlation
        nifty_corr   = float(corr_row.mean())
    else:
        nifty_corr   = avg_corr   # fallback: use market-wide average

    block = np.array([avg_corr, port_vol, nifty_corr], dtype=np.float32)
    assert np.isfinite(block).all(), "cross_asset_features: non-finite values"
    return block


def macro_features(
    macro_df: pd.DataFrame,    # full macro DataFrame indexed by Date
    t_date:   pd.Timestamp,    # current step date
) -> np.ndarray:
    """
    Compute 4 macro scalars at date t_date.

    5-day changes (pct)
    ─────────────────────
    Captures the direction and magnitude of recent macro shocks:
    · VIX rise   → fear increasing → reduce equity exposure
    · Crude rise → cost-push inflation → sector-specific headwinds
    · INR depreciation → FII outflows → bearish for equities broadly

    Why 5-day and not 1-day?
    Single-day macro moves are noisy (flash crashes, data corrections).
    5 trading days ≈ 1 calendar week — long enough to confirm a shift,
    short enough to still be actionable.

    Regime label
    ─────────────
    Derived from NIFTYBEES MA-spread (already computed in features.py)
    AND the VIX level at t_date.  Rule:
      Bull (+1) : VIX < 20
      Bear (−1) : VIX > 25
      Neutral (0): 20 ≤ VIX ≤ 25
    (The MA-spread component is captured separately in trend_MA_spread.)
    """
    # Slice the macro series up to and including t_date
    hist = macro_df[macro_df.index <= t_date].iloc[-MACRO_WINDOW - 1:]

    def _pct_change(col: str) -> float:
        series = hist[col].dropna()
        if len(series) < 2:
            return 0.0
        return float((series.iloc[-1] / series.iloc[max(0, len(series) - MACRO_WINDOW)]) - 1.0)

    vix_5d    = _pct_change("INDIAVIX")
    crude_5d  = _pct_change("CRUDE_OIL")
    usdinr_5d = _pct_change("USDINR")

    # Regime based on current VIX level
    vix_now = float(hist["INDIAVIX"].dropna().iloc[-1]) if not hist["INDIAVIX"].dropna().empty else 20.0
    if vix_now < 20.0:
        regime = 1.0
    elif vix_now > 25.0:
        regime = -1.0
    else:
        regime = 0.0

    # Clip macro returns to ±0.5 (> 50% weekly macro move = data error)
    block = np.array([
        np.clip(vix_5d,    -0.5, 0.5),
        np.clip(crude_5d,  -0.5, 0.5),
        np.clip(usdinr_5d, -0.5, 0.5),
        regime,
    ], dtype=np.float32)

    assert np.isfinite(block).all(), "macro_features: non-finite values"
    return block


def portfolio_features(
    weights:    np.ndarray,   # (N,)   stock-only weights
    nav_series: np.ndarray,   # 1-D    NAV history up to and including t
    log_ret_t:  np.ndarray,   # (N,)   log returns at t (for pnl)
) -> np.ndarray:
    """
    Compute 10 portfolio scalars.

    current_weights (8)
    ─────────────────────
    Stock-only portion (cash weight is 1 − sum(weights)).
    The policy already knows total allocation; explicit weights let
    it track drift and concentration risk.

    pnl_recent (1)
    ─────────────────────
    5-day simple portfolio return: exp(sum of last 5 log NAV changes) − 1.
    Gives the agent a short-term P&L signal: "am I in a drawdown or a run?"

    drawdown (1)
    ─────────────────────
    (NAV_t − peak NAV) / peak NAV ∈ [−1, 0].
    Critical for risk-aware policy:  the agent should tighten risk tolerance
    when it is already in a deep drawdown (avoid doubling down on losses).
    """
    N = len(weights)

    # --- pnl_recent ---
    if len(nav_series) >= MACRO_WINDOW + 1:
        nav_window = nav_series[-(MACRO_WINDOW + 1):]
        pnl_recent = float(nav_window[-1] / nav_window[0] - 1.0)
    else:
        pnl_recent = 0.0
    pnl_recent = float(np.clip(pnl_recent, -1.0, 1.0))

    # --- drawdown ---
    peak     = float(np.max(nav_series)) if len(nav_series) > 0 else 1.0
    nav_now  = float(nav_series[-1])     if len(nav_series) > 0 else 1.0
    drawdown = float(np.clip((nav_now - peak) / max(peak, EPS), -1.0, 0.0))

    w_clipped = np.clip(weights[:N], 0.0, 1.0).astype(np.float32)

    block = np.concatenate([w_clipped, [pnl_recent, drawdown]]).astype(np.float32)
    assert block.shape == (N + 2,), f"portfolio_features: expected ({N+2},), got {block.shape}"
    assert np.isfinite(block).all(), "portfolio_features: non-finite values"
    return block


# ══════════════════════════════════════════════════════════════════════════════
# FULL STATE ASSEMBLER
# ══════════════════════════════════════════════════════════════════════════════

def build_state_vector(
    *,
    daily_ret_t:  np.ndarray,
    momentum_t:   np.ndarray,
    vol20d_t:     np.ndarray,
    ma_spread_t:  np.ndarray,
    cov_t:        np.ndarray,
    weights:      np.ndarray,
    tickers:      list[str],
    macro_df:     pd.DataFrame,
    t_date:       pd.Timestamp,
    nav_series:   np.ndarray,
    log_ret_t:    np.ndarray,
) -> np.ndarray:
    """
    Assemble the full 49-dimensional state vector.

    Keyword-only arguments prevent silent positional-order bugs.

    Returns
    ───────
    np.ndarray of shape (49,) dtype float32, all finite.
    Use STATE.<FIELD> slices to extract groups downstream.
    """
    s = np.concatenate([
        per_asset_features(daily_ret_t, momentum_t, vol20d_t, ma_spread_t),
        cross_asset_features(cov_t, weights, tickers),
        macro_features(macro_df, t_date),
        portfolio_features(weights, nav_series, log_ret_t),
    ]).astype(np.float32)

    assert s.shape == (STATE.DIM,), (
        f"build_state_vector: expected ({STATE.DIM},), got {s.shape}"
    )
    assert np.isfinite(s).all(), "build_state_vector: non-finite values in assembled state"
    return s


# ══════════════════════════════════════════════════════════════════════════════
# MEAN-VARIANCE OPTIMIZER HELPER
# ══════════════════════════════════════════════════════════════════════════════

def extract_mvo_inputs(state: np.ndarray) -> dict:
    """
    Extract the subset of state variables needed by the mean-variance optimizer.

    Returns a dict with:
      vol20d          (8,)   per-asset annualised volatility
      avg_correlation  float  mean pairwise correlation → off-diagonal of Σ̂
      port_vol         float  current portfolio volatility constraint reference
      regime           float  −1 / 0 / +1 → scales risk-aversion κ
      ma_spread        (8,)   trend signals → expected-return tilt
      momentum_5d      (8,)   short-term return signals → expected-return tilt

    How the MVO uses these
    ──────────────────────
    A simple constant-correlation Σ̂ can be constructed as:
      Σ̂[i,i] = vol20d[i]² / 252          (daily variance)
      Σ̂[i,j] = avg_corr × vol20d[i] × vol20d[j] / 252  (off-diagonal)
    This avoids a full LW estimation when only a quick re-optimisation
    is needed mid-episode.

    Risk-aversion scaling:
      κ = κ_base × (1 + 0.5 × |regime|)   (tighter in bear/bull extremes)
    """
    return {
        "vol20d":          state[STATE.VOL20D],
        "avg_correlation": float(state[STATE.AVG_CORR][0]),
        "port_vol":        float(state[STATE.PORT_VOL][0]),
        "regime":          float(state[STATE.REGIME][0]),
        "ma_spread":       state[STATE.MA_SPREAD],
        "momentum_5d":     state[STATE.MOM5D],
        "current_weights": state[STATE.WEIGHTS],
        "drawdown":        float(state[STATE.DRAWDOWN][0]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — smoke-test the spec with synthetic data
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    np.random.seed(0)
    N = 8
    tickers = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
        "ICICIBANK.NS", "NIFTYBEES.NS", "HINDUNILVR.NS", "KOTAKBANK.NS",
    ]

    # Synthetic inputs
    daily_ret  = np.random.normal(0.001, 0.02, N).astype(np.float32)
    momentum   = np.random.normal(0.005, 0.04, N).astype(np.float32)
    vol20d     = np.abs(np.random.normal(0.25, 0.05, N)).astype(np.float32) + 0.05
    ma_spread  = np.random.uniform(-0.2, 0.3, N).astype(np.float32)
    raw_cov    = np.random.randn(N, N)
    cov_t      = (raw_cov @ raw_cov.T) / N * 0.06      # PSD, annualised
    weights    = np.ones(N) / N
    nav_series = np.cumprod(1 + np.random.normal(0.0005, 0.01, 100))

    # Synthetic macro DataFrame
    dates = pd.date_range("2022-01-01", periods=20, freq="B")
    macro_df = pd.DataFrame({
        "INDIAVIX":  np.abs(np.random.normal(18, 4, 20)),
        "CRUDE_OIL": 80 + np.random.normal(0, 5, 20).cumsum(),
        "USDINR":    82 + np.random.normal(0, 0.3, 20).cumsum(),
    }, index=dates)
    t_date = dates[-1]

    state = build_state_vector(
        daily_ret_t=daily_ret,
        momentum_t=momentum,
        vol20d_t=vol20d,
        ma_spread_t=ma_spread,
        cov_t=cov_t,
        weights=weights,
        tickers=tickers,
        macro_df=macro_df,
        t_date=t_date,
        nav_series=nav_series,
        log_ret_t=daily_ret,
    )

    print(f"State vector shape : {state.shape}")
    print(f"All finite         : {np.isfinite(state).all()}")
    print(f"Min / Max / Mean   : {state.min():.4f} / {state.max():.4f} / {state.mean():.4f}")
    print()
    print("Group breakdown:")
    groups = [
        ("daily_returns  [0:8]  ", STATE.DAILY_RET),
        ("momentum_5d    [8:16] ", STATE.MOM5D),
        ("volatility_20d [16:24]", STATE.VOL20D),
        ("MA_spread      [24:32]", STATE.MA_SPREAD),
        ("avg_corr       [32]   ", STATE.AVG_CORR),
        ("port_vol       [33]   ", STATE.PORT_VOL),
        ("nifty_corr     [34]   ", STATE.NIFTY_CORR),
        ("vix_5d         [35]   ", STATE.VIX5D),
        ("crude_5d       [36]   ", STATE.CRUDE5D),
        ("usdinr_5d      [37]   ", STATE.USDINR5D),
        ("regime         [38]   ", STATE.REGIME),
        ("weights        [39:47]", STATE.WEIGHTS),
        ("pnl_recent     [47]   ", STATE.PNL_RECENT),
        ("drawdown       [48]   ", STATE.DRAWDOWN),
    ]
    for name, sl in groups:
        vals = state[sl]
        if vals.size == 1:
            print(f"  {name}  = {vals[0]:+.4f}")
        else:
            print(f"  {name}  = [{vals.min():+.4f} … {vals.max():+.4f}]  mean={vals.mean():+.4f}")

    print()
    mvo = extract_mvo_inputs(state)
    print("MVO inputs extracted:")
    for k, v in mvo.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:20s}: shape={v.shape}  mean={v.mean():+.4f}")
        else:
            print(f"  {k:20s}: {v:+.4f}")
