"""
features.py — Feature engineering for Deep-RL portfolio state space.

Computes and validates every feature the RL environment needs:

  Feature                   Shape per step        Notes
  ──────────────────────────────────────────────────────────────────────
  Log returns               (T, N)                window of LOOKBACK days
  Rolling volatility 30d    (T, N)                annualised σ (existing)
  Rolling covariance 90d    (T, N, N)             Ledoit-Wolf shrunken
  Portfolio weights         (T, N+1)              N stocks + cash, sums to 1
  Drawdown                  (T,)                  portfolio-level peak-to-trough
  ── Per-asset features (new) ──────────────────────────────────────────
  Daily simple returns      (T, N)                (P_t − P_{t-1}) / P_{t-1}
  5-day momentum            (T, N)                (P_t / P_{t-5}) − 1
  20-day realized vol       (T, N)                annualised std of daily returns
  MA 50/200 spread          (T, N)                (MA50 − MA200) / MA200

All outputs are finite (no NaN, no Inf) after the warm-up period.
The module can be run standalone to validate and cache features to disk.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
LOOKBACK_RETURN  = 20    # days of log-return history fed to the RL state
VOL_WINDOW       = 30    # rolling window for 30-day annualised volatility
COV_WINDOW       = 90    # rolling window for covariance / LW shrinkage
TRADING_DAYS     = 252   # annualisation factor
EPS              = 1e-8  # numerical epsilon (replace zero std, avoid /0)
EWMA_SPAN        = 5     # EWMA span for drawdown smoothing
COV_CLIP         = 1.0   # clip individual covariance entries to ± this
WEIGHT_TOL       = 1e-6  # floating-point drift threshold for weight sum

# ── Per-asset feature windows ─────────────────────────────────────────────────
MOM_WINDOW       = 5     # 5-day return momentum window
REALVOL_WINDOW   = 20    # 20-day realized volatility window
MA_SHORT         = 50    # short moving-average period
MA_LONG          = 200   # long moving-average period (drives warm-up on full series)

DATA_DIR    = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

SPLITS = ("train", "test", "val")

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_prices(split: str) -> pd.DataFrame:
    label = SPLIT_LABELS[split]
    path  = os.path.join(DATA_DIR, f"prices_{label}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def _load_full_prices() -> pd.DataFrame:
    """Load the complete 2014-2024 price series (needed for MA200 warm-up)."""
    path = os.path.join(DATA_DIR, "prices_2014_2024.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def _assert_finite(arr: np.ndarray, name: str, after_warmup: int = 0) -> None:
    """Assert no NaN / Inf after the warm-up rows."""
    check = arr[after_warmup:]
    n_nan = int(np.isnan(check).sum())
    n_inf = int(np.isinf(check).sum())
    assert n_nan == 0, f"{name}: {n_nan} NaN values after warm-up row {after_warmup}"
    assert n_inf == 0, f"{name}: {n_inf} Inf values after warm-up row {after_warmup}"


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — LOG RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1}).

    Why log returns (not simple returns)?
    ──────────────────────────────────────
    · Time-additive: multi-period return = sum of daily log returns.
    · More symmetric / closer to Gaussian → better for neural-net inputs.
    · Naturally bounded below at ln(0) = -∞  (but practically ≥ -40% after
      winsorisation in preprocess.py).

    Cleaning
    ────────
    · Row 0 set to 0.0 (no prior price to diff against).
    · Any remaining NaN from pre-alignment gaps is left as NaN here;
      they are eliminated by the warm-up drop in the RL environment.
    · Result is never forward-filled — no leakage.
    """
    log_ret = np.log(prices / prices.shift(1))

    # Row 0 has no predecessor — set to zero (neutral, not NaN)
    log_ret.iloc[0] = 0.0

    return log_ret


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — ROLLING VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_volatility(log_ret: pd.DataFrame,
                                window: int = VOL_WINDOW) -> pd.DataFrame:
    """
    Annualised rolling volatility: σ_t = std(r_{t-w+1:t}) × √252.

    Cleaning
    ────────
    · First (window-1) rows are NaN — filled with the first valid value
      (back-fill) so the RL state never sees NaN.
    · Zero std replaced by EPS to prevent divide-by-zero when normalising
      returns by volatility (e.g. Sharpe-like reward scaling).
    · Result is clipped to [EPS, 10] — values above 1000% annualised are
      data artefacts.
    """
    vol = log_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)

    # Back-fill warm-up NaNs with the first valid row
    vol = vol.bfill()

    # Replace exact zero std (e.g. zero-volume day) with epsilon
    vol = vol.replace(0.0, EPS).clip(lower=EPS, upper=10.0)

    return vol


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET FEATURES A — DAILY SIMPLE RETURNS
# ══════════════════════════════════════════════════════════════════════════════

def compute_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily simple returns: r_t = (P_t − P_{t-1}) / P_{t-1}.

    Why simple returns here (not log)?
    ────────────────────────────────────
    · Directly interpretable as percentage P&L per day.
    · MA-spread and momentum are price-ratio features — pairing them with
      simple returns keeps all four features in the same unit space (fraction).
    · Log returns are used separately for covariance / volatility inputs
      where time-additivity matters.

    Cleaning
    ────────
    · Row 0 set to 0.0 (no prior price).
    · Clipped to [−0.5, 0.5] for NN stability (pre-cleaning winsorises to ±40%).
    """
    ret = prices.pct_change(fill_method=None)
    ret.iloc[0] = 0.0
    return ret.clip(lower=-0.5, upper=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET FEATURES B — 5-DAY MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════

def compute_momentum(prices: pd.DataFrame,
                     window: int = MOM_WINDOW) -> pd.DataFrame:
    """
    5-day price momentum: mom_5 = (P_t / P_{t-5}) − 1.

    Why 5-day momentum?
    ─────────────────────
    · Captures the dominant short-term (1-week) mean-reversion / momentum
      signal that is well-documented in Indian equity markets.
    · Short enough to be actionable by a daily-rebalancing agent, long enough
      to filter single-day noise.
    · Complements the MA spread (200d trend) by providing a fast signal.

    Cleaning
    ────────
    · First (window) rows back-filled with the first valid value.
    · Clipped to [−0.6, 0.6] — a 5-day move beyond 60% signals a data error.
    """
    mom = prices.pct_change(periods=window, fill_method=None)
    mom = mom.bfill()
    return mom.clip(lower=-0.6, upper=0.6)


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET FEATURES C — 20-DAY REALIZED VOLATILITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_realized_volatility(simple_ret: pd.DataFrame,
                                 window: int = REALVOL_WINDOW) -> pd.DataFrame:
    """
    20-day realized (annualised) volatility: σ_20 = std(r_{t-19:t}) × √252.

    Why 20-day alongside the existing 30-day vol?
    ──────────────────────────────────────────────
    · 20-day ≈ 1 calendar month — the standard institutional risk window.
    · 30-day (existing) uses log returns for portfolio-level hedging/covariance.
    · 20-day uses simple returns → directly proportional to the daily P&L σ.
    · Together they give the agent two volatility time-scales per asset.

    Cleaning
    ────────
    · First (window-1) rows back-filled with the first valid value.
    · Zero σ replaced with EPS; clipped to [EPS, 10].
    """
    rvol = simple_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    rvol = rvol.bfill()
    return rvol.replace(0.0, EPS).clip(lower=EPS, upper=10.0)


# ══════════════════════════════════════════════════════════════════════════════
# PER-ASSET FEATURES D — 50/200-DAY MA SPREAD  (trend strength)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ma_spread(full_prices: pd.DataFrame,
                      split_index: pd.DatetimeIndex,
                      short: int = MA_SHORT,
                      long: int  = MA_LONG) -> pd.DataFrame:
    """
    50/200-day MA spread: (MA_50 − MA_200) / MA_200.

    Why this spread?
    ─────────────────
    · The 50/200 golden/death-cross is among the most cited trend indicators
      in equity markets and is universally available as a signal.
    · Positive → short MA above long MA → uptrend (over-weight equity).
    · Negative → short MA below long MA → downtrend (reduce exposure / hold cash).
    · Normalising by MA_200 makes the signal scale-free across assets with
      very different price levels (e.g. RELIANCE at ₹3000 vs NIFTYBEES at ₹200).

    Why compute on the FULL price series (not just the split)?
    ───────────────────────────────────────────────────────────
    MA_200 needs 200 days of price history.  The test split is only 248 rows —
    computing on the split alone would leave ≤ 48 valid rows.  Instead we
    compute on `prices_2014_2024.csv` (full history) and slice to `split_index`.
    This is NOT leakage: MA at day t only uses prices ≤ t (causal).

    Cleaning
    ────────
    · Warm-up rows (first 199 days of full series → 2014) are back-filled.
    · Clipped to [−0.5, 0.5] — a 50% deviation of MA50 from MA200 is extreme.
    · Result sliced to `split_index` before returning.
    """
    ma_short = full_prices.rolling(short).mean()
    ma_long  = full_prices.rolling(long).mean()

    spread = (ma_short - ma_long) / ma_long.replace(0.0, EPS)
    spread = spread.bfill()
    spread = spread.clip(lower=-0.5, upper=0.5)

    return spread.loc[split_index]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — ROLLING COVARIANCE  (Ledoit-Wolf shrunken)
# ══════════════════════════════════════════════════════════════════════════════

def _ledoit_wolf_cov(return_window: np.ndarray) -> np.ndarray:
    """
    Fit Ledoit-Wolf shrinkage covariance on a (T × N) return window.
    Returns an (N × N) annualised covariance matrix.

    Ledoit-Wolf shrinkage
    ─────────────────────
    The sample covariance Σ̂ is noisy when T ≈ N (small sample, many assets).
    LW shrinks toward a structured target (scaled identity):
        Σ_LW = (1 - α) Σ̂  +  α μ I
    where α is determined analytically (Oracle approximating shrinkage).
    This improves conditioning and stabilises Markowitz-style optimisation
    inside the RL agent without requiring matrix inversion tricks.
    """
    lw = LedoitWolf().fit(return_window)
    return lw.covariance_ * TRADING_DAYS   # annualise


def compute_rolling_covariance(log_ret: pd.DataFrame,
                                window: int = COV_WINDOW
                                ) -> np.ndarray:
    """
    Compute a (T, N, N) array of rolling Ledoit-Wolf covariance matrices.

    Cleaning
    ────────
    · Warm-up rows (first window-1) are filled by repeating the first valid
      matrix — the RL environment will mask these anyway via its warm-up skip,
      but a finite value prevents NaN propagation during batched tensor ops.
    · Individual covariance entries clipped to [−COV_CLIP, +COV_CLIP].
      Off-diagonal entries > 1 have no financial meaning (correlation > 1)
      and signal a numerical instability.
    · Diagonal (variances) further forced ≥ EPS.
    """
    T, N = log_ret.shape
    cov_array = np.full((T, N, N), np.nan)

    # Fit LW on each rolling window
    for t in range(window - 1, T):
        window_data = log_ret.iloc[t - window + 1 : t + 1].values
        if np.isnan(window_data).any():
            continue
        cov_array[t] = _ledoit_wolf_cov(window_data)

    # ── Clip off-diagonal to ±COV_CLIP, force positive diagonal ──────────────
    for t in range(T):
        if np.isnan(cov_array[t]).any():
            continue
        mat = cov_array[t]
        mat = np.clip(mat, -COV_CLIP, COV_CLIP)          # clip all entries
        np.fill_diagonal(mat, np.maximum(np.diag(mat), EPS))   # variances > 0
        cov_array[t] = mat

    # ── Back-fill warm-up NaN slices with the first valid matrix ──────────────
    first_valid = None
    for t in range(T):
        if not np.isnan(cov_array[t]).any():
            first_valid = cov_array[t].copy()
            break
    if first_valid is not None:
        for t in range(T):
            if np.isnan(cov_array[t]).any():
                cov_array[t] = first_valid

    return cov_array


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — PORTFOLIO WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioWeights:
    """
    Manages portfolio weight vectors for the RL environment.

    Layout: weights[i] for i in 0..N-1 are stock weights;
            weights[N] is the cash weight.
    Sum of all weights = 1.0 at all times.
    """
    n_assets: int
    weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> np.ndarray:
        """Equal-weight initialisation including cash."""
        n = self.n_assets + 1   # N stocks + 1 cash
        self.weights = np.ones(n, dtype=np.float64) / n
        return self.weights.copy()

    def update(self, new_weights: np.ndarray) -> np.ndarray:
        """
        Set weights from the agent's action output.

        Normalisation
        ─────────────
        · Clip to [0, 1] (no shorting in this formulation).
        · Divide by L1 norm — handles floating-point drift so weights
          always sum exactly to 1.
        · If the norm collapses to zero (degenerate action), fall back
          to equal weight.

        Drift check
        ───────────
        After normalisation the sum should be 1.0 ± WEIGHT_TOL.
        An assertion fires if it is not — catches silent numerical bugs
        before they corrupt the reward signal.
        """
        w = np.clip(new_weights, 0.0, 1.0).astype(np.float64)
        norm = w.sum()
        if norm < EPS:
            w = np.ones_like(w) / len(w)   # degenerate → equal weight
        else:
            w = w / norm

        # Floating-point drift guard
        drift = abs(w.sum() - 1.0)
        assert drift < WEIGHT_TOL, (
            f"Weight sum drift {drift:.2e} exceeds tolerance {WEIGHT_TOL}"
        )
        self.weights = w
        return self.weights.copy()

    @property
    def stock_weights(self) -> np.ndarray:
        return self.weights[:-1]

    @property
    def cash_weight(self) -> float:
        return float(self.weights[-1])


def initial_equal_weights(n_assets: int) -> np.ndarray:
    """Return a (n_assets + 1,) equal-weight vector including cash slot."""
    return np.ones(n_assets + 1, dtype=np.float64) / (n_assets + 1)


def normalize_weights(w: np.ndarray) -> np.ndarray:
    """Project any non-negative weight vector onto the unit simplex."""
    w = np.clip(w, 0.0, 1.0)
    s = w.sum()
    return w / s if s > EPS else np.ones_like(w) / len(w)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — DRAWDOWN
# ══════════════════════════════════════════════════════════════════════════════

def compute_drawdown(nav: pd.Series,
                     smooth: bool = True,
                     ewma_span: int = EWMA_SPAN) -> pd.Series:
    """
    Compute portfolio drawdown series from a NAV (Net Asset Value) series.

        DD_t = (NAV_t − CumMax_t) / CumMax_t

    Cleaning
    ────────
    · Division-by-zero guard: CumMax is forced ≥ EPS before division.
      (A zero NAV would mean total portfolio wipe-out — physically impossible
      in this environment but handled defensively.)
    · Optional EWMA smoothing removes micro-spikes caused by single-day
      data artefacts (same-day rebalancing cost noise).
      This does NOT introduce leakage — EWMA at step t only uses
      information from steps ≤ t.
    · Result clipped to [−1, 0]: drawdown ∈ {−100%, 0%}.

    Returns
    ───────
    pd.Series indexed identically to `nav`, values ∈ [−1, 0].
    """
    cum_max = nav.cummax().clip(lower=EPS)
    dd = (nav - cum_max) / cum_max

    if smooth:
        dd = dd.ewm(span=ewma_span, adjust=False).mean()

    dd = dd.clip(lower=-1.0, upper=0.0)
    return dd


def compute_drawdown_from_weights(adj_close: pd.DataFrame,
                                   weights_history: np.ndarray
                                   ) -> pd.Series:
    """
    Compute drawdown for a time-varying-weight portfolio.

    Parameters
    ──────────
    adj_close        : (T, N) DataFrame of adjusted close prices
    weights_history  : (T, N) array of stock-only weights at each step
                       (cash weight excluded — cash earns 0 here)

    Returns
    ───────
    pd.Series of drawdown values indexed by adj_close.index.
    """
    log_ret  = np.log(adj_close / adj_close.shift(1)).fillna(0.0)
    port_ret = (log_ret.values * weights_history).sum(axis=1)
    nav      = pd.Series(
        np.exp(np.cumsum(port_ret)),
        index=adj_close.index,
        name="NAV",
    )
    return compute_drawdown(nav)


# ══════════════════════════════════════════════════════════════════════════════
# BUNDLE — compute & validate everything for one split
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeatureBundle:
    """All pre-computed features for one data split, ready for the RL env."""
    split:        str
    tickers:      list[str]
    dates:        pd.DatetimeIndex
    prices:       pd.DataFrame        # (T, N)  Adj Close
    log_ret:      pd.DataFrame        # (T, N)  log returns
    volatility:   pd.DataFrame        # (T, N)  30-day annualised σ (log-return based)
    cov:          np.ndarray          # (T, N, N) LW-shrunken covariance
    daily_ret:    pd.DataFrame        # (T, N)  simple daily returns
    momentum:     pd.DataFrame        # (T, N)  5-day price momentum
    realized_vol: pd.DataFrame        # (T, N)  20-day realized vol (annualised)
    ma_spread:    pd.DataFrame        # (T, N)  (MA50 − MA200) / MA200
    n_assets:     int = field(init=False)

    def __post_init__(self) -> None:
        self.n_assets = len(self.tickers)

    def warm_up_rows(self) -> int:
        """Number of leading rows to skip in the RL episode (covariance needs most data)."""
        return COV_WINDOW

    def state_at(self, t: int, weights: np.ndarray) -> np.ndarray:
        """
        Assemble the full RL state vector at time step t.

        Components (concatenated)
        ─────────────────────────
        · log_returns[t-LOOKBACK_RETURN+1 : t+1]  flattened  (LOOKBACK × N)
        · volatility_30d[t]                                    (N,)
        · cov[t] lower-triangle (no redundancy)               (N*(N+1)//2,)
        · weights                                              (N+1,)
        · daily_returns[t]                                     (N,)
        · momentum_5d[t]                                       (N,)
        · realized_vol_20d[t]                                  (N,)
        · ma_spread[t]                                         (N,)

        Note: drawdown is computed episode-level (needs the NAV history),
        not assembled per-step here.  It is passed in by the environment.
        """
        assert t >= self.warm_up_rows(), (
            f"t={t} is inside the warm-up period ({self.warm_up_rows()} rows)."
        )
        ret_window = self.log_ret.iloc[t - LOOKBACK_RETURN + 1: t + 1].values
        vol_t      = self.volatility.iloc[t].values
        cov_t      = self.cov[t]
        tril_idx   = np.tril_indices(self.n_assets)
        cov_flat   = cov_t[tril_idx]

        state = np.concatenate([
            ret_window.flatten(),
            vol_t,
            cov_flat,
            weights,
            self.daily_ret.iloc[t].values,
            self.momentum.iloc[t].values,
            self.realized_vol.iloc[t].values,
            self.ma_spread.iloc[t].values,
        ]).astype(np.float32)

        assert np.isfinite(state).all(), (
            f"Non-finite values in state at t={t}"
        )
        return state

    def state_dim(self) -> int:
        N = self.n_assets
        # log-return window + vol30d + cov lower-tri + weights
        # + daily_ret + momentum_5d + realized_vol_20d + ma_spread
        return (LOOKBACK_RETURN * N) + N + (N * (N + 1) // 2) + (N + 1) + 4 * N


def build_features(split: str, verbose: bool = True) -> FeatureBundle:
    """
    Load prices for `split` and compute all features.
    Runs validation assertions before returning.

    MA-spread uses the full 2014-2024 price series to avoid the 200-day
    warm-up consuming most of the test/val splits.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log(f"\n{'='*60}")
    log(f"  Building features: {split}")
    log(f"{'='*60}")

    prices = _load_prices(split)
    T, N   = prices.shape
    log(f"  Prices loaded: {T} rows × {N} tickers")

    # ── 1. Log returns ────────────────────────────────────────────────────────
    log(f"\n  [1/8] Log returns  (window={LOOKBACK_RETURN}d) …")
    log_ret = compute_log_returns(prices)
    _assert_finite(log_ret.values, "log_ret", after_warmup=1)
    log(f"        Mean={log_ret.iloc[1:].mean().mean():+.5f}  "
        f"Std={log_ret.iloc[1:].std().mean():.5f}")

    # ── 2. Rolling volatility 30d ─────────────────────────────────────────────
    log(f"\n  [2/8] Rolling volatility 30d  (window={VOL_WINDOW}d, annualised) …")
    vol = compute_rolling_volatility(log_ret, window=VOL_WINDOW)
    _assert_finite(vol.values, "volatility_30d")
    log(f"        Min={vol.values.min():.4f}  "
        f"Mean={vol.values.mean():.4f}  "
        f"Max={vol.values.max():.4f}")

    # ── 3. Rolling covariance (LW) ────────────────────────────────────────────
    log(f"\n  [3/8] Rolling covariance  (LW shrinkage, window={COV_WINDOW}d) …")
    cov = compute_rolling_covariance(log_ret, window=COV_WINDOW)
    _assert_finite(cov, "cov")
    diag_vals = np.array([np.diag(cov[t]) for t in range(T)])
    log(f"        Diagonal (variance) — "
        f"Min={diag_vals.min():.4f}  "
        f"Mean={diag_vals.mean():.4f}  "
        f"Max={diag_vals.max():.4f}")

    # ── 4. Weights initialised ────────────────────────────────────────────────
    log(f"\n  [4/8] Portfolio weights  ({N} stocks + 1 cash = {N+1} slots) …")
    pw = PortfolioWeights(n_assets=N)
    w0 = pw.reset()
    assert abs(w0.sum() - 1.0) < WEIGHT_TOL
    log(f"        Initial equal weight = {w0[0]:.6f} each  (sum={w0.sum():.8f})")

    # ── 5. Daily simple returns ───────────────────────────────────────────────
    log(f"\n  [5/8] Daily simple returns …")
    daily_ret = compute_simple_returns(prices)
    _assert_finite(daily_ret.values, "daily_ret", after_warmup=1)
    log(f"        Mean={daily_ret.iloc[1:].mean().mean():+.5f}  "
        f"Std={daily_ret.iloc[1:].std().mean():.5f}")

    # ── 6. 5-day momentum ─────────────────────────────────────────────────────
    log(f"\n  [6/8] 5-day momentum …")
    momentum = compute_momentum(prices, window=MOM_WINDOW)
    _assert_finite(momentum.values, "momentum_5d")
    log(f"        Mean={momentum.mean().mean():+.5f}  "
        f"Std={momentum.std().mean():.5f}")

    # ── 7. 20-day realized volatility ─────────────────────────────────────────
    log(f"\n  [7/8] 20-day realized volatility …")
    realized_vol = compute_realized_volatility(daily_ret, window=REALVOL_WINDOW)
    _assert_finite(realized_vol.values, "realized_vol_20d")
    log(f"        Min={realized_vol.values.min():.4f}  "
        f"Mean={realized_vol.values.mean():.4f}  "
        f"Max={realized_vol.values.max():.4f}")

    # ── 8. MA 50/200 spread ───────────────────────────────────────────────────
    # Computed on FULL 2014-2024 prices to avoid 200-day warm-up consuming the
    # test/val splits (test has only 248 rows; 200-day warm-up → 48 usable).
    log(f"\n  [8/8] MA spread  (MA{MA_SHORT}/MA{MA_LONG}) …")
    full_prices = _load_full_prices()
    full_prices = full_prices[prices.columns]          # same ticker order
    ma_spread   = compute_ma_spread(full_prices, prices.index)
    _assert_finite(ma_spread.values, "ma_spread")
    log(f"        Min={ma_spread.values.min():.4f}  "
        f"Mean={ma_spread.values.mean():.4f}  "
        f"Max={ma_spread.values.max():.4f}")

    # ── State-vector dimension check ──────────────────────────────────────────
    bundle = FeatureBundle(
        split=split, tickers=list(prices.columns),
        dates=prices.index, prices=prices,
        log_ret=log_ret, volatility=vol, cov=cov,
        daily_ret=daily_ret, momentum=momentum,
        realized_vol=realized_vol, ma_spread=ma_spread,
    )
    N = bundle.n_assets
    log(f"\n  State vector dim = {bundle.state_dim()}")
    log(f"    {LOOKBACK_RETURN}d × {N} log-returns  "
        f"+  {N} vol30d  "
        f"+  {N*(N+1)//2} cov (lower-tri)  "
        f"+  {N+1} weights  "
        f"+  {N} daily-ret  "
        f"+  {N} mom5d  "
        f"+  {N} rvol20d  "
        f"+  {N} MA-spread")

    # ── Spot-check a live state ───────────────────────────────────────────────
    t_test = bundle.warm_up_rows()
    s = bundle.state_at(t_test, w0)
    log(f"\n  Spot-check state at t={t_test}:")
    log(f"    shape={s.shape}  dtype={s.dtype}  "
        f"finite={np.isfinite(s).all()}")
    log(f"    min={s.min():.4f}  max={s.max():.4f}  mean={s.mean():.4f}")

    log(f"\n  ✓  All checks passed for split='{split}'")
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
# PERSIST features to disk (optional caching for fast RL env startup)
# ══════════════════════════════════════════════════════════════════════════════

def save_bundle(bundle: FeatureBundle) -> None:
    """Save features as CSV / npy for fast loading."""
    s     = bundle.split
    label = SPLIT_LABELS[s]
    d     = FEATURE_DIR

    bundle.log_ret.to_csv(os.path.join(d,       f"log_returns_{label}.csv"))
    bundle.volatility.to_csv(os.path.join(d,    f"volatility_30d_{label}.csv"))
    np.save(os.path.join(d,                      f"covariance_90d_{label}.npy"), bundle.cov)
    bundle.daily_ret.to_csv(os.path.join(d,      f"daily_returns_{label}.csv"))
    bundle.momentum.to_csv(os.path.join(d,       f"momentum_5d_{label}.csv"))
    bundle.realized_vol.to_csv(os.path.join(d,   f"realized_vol_20d_{label}.csv"))
    bundle.ma_spread.to_csv(os.path.join(d,      f"ma_spread_50_200_{label}.csv"))

    print(f"  Saved features for split='{s}' to {d}/")


def load_bundle(split: str) -> FeatureBundle:
    """Re-load a previously saved FeatureBundle (skips recomputation)."""
    label    = SPLIT_LABELS[split]
    prices   = _load_prices(split)

    def _csv(name: str) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(FEATURE_DIR, f"{name}_{label}.csv"),
            index_col=0, parse_dates=True,
        )

    log_ret      = _csv("log_returns")
    vol          = _csv("volatility_30d")
    cov          = np.load(os.path.join(FEATURE_DIR, f"covariance_90d_{label}.npy"))
    daily_ret    = _csv("daily_returns")
    momentum     = _csv("momentum_5d")
    realized_vol = _csv("realized_vol_20d")
    ma_spread    = _csv("ma_spread_50_200")

    return FeatureBundle(
        split=split, tickers=list(prices.columns),
        dates=prices.index, prices=prices,
        log_ret=log_ret, volatility=vol, cov=cov,
        daily_ret=daily_ret, momentum=momentum,
        realized_vol=realized_vol, ma_spread=ma_spread,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_bundles: dict[str, FeatureBundle] = {}

    for split in SPLITS:
        try:
            bundle = build_features(split)
            save_bundle(bundle)
            all_bundles[split] = bundle
        except FileNotFoundError as exc:
            print(f"\n  ✗  {exc}")

    print("\n" + "=" * 60)
    print("  Summary across splits")
    print("=" * 60)
    for split, b in all_bundles.items():
        warm = b.warm_up_rows()
        usable = len(b.dates) - warm
        print(f"  {split:<6}  total={len(b.dates):>5}  "
              f"warm-up={warm}  usable={usable:>5}  "
              f"state_dim={b.state_dim()}")

    print("\n  Feature files saved to ./data/features/")
