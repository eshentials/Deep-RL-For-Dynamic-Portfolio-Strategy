"""
rl_preprocess.py — RL-specific preprocessing for Deep-RL portfolio.

Three components built here:

  1. Stacked Observation Window
     Shape: (T, LOOKBACK, N_ASSETS, N_FEATURES)
     Features per asset per day: [log_return, norm_volatility, norm_volume]
     Provides the CNN / LSTM-compatible tensor the RL policy network sees.

  2. Discrete Action Space — Efficient Frontier Portfolios
     K = 10 risk levels sampled uniformly along the mean-variance frontier.
     Each action is an (N+1,)-weight vector (stocks + cash).
     Computed with PyPortfolioOpt + Ledoit-Wolf covariance on the training set.
     Stored to disk so the RL environment never recomputes them.

  3. Transaction Cost Model
     Two-component model applied inside the reward function:
       · Proportional brokerage:  c_b  = BROKERAGE_RATE  × |Δw| × portfolio_value
       · Market-impact slippage:  c_s  = SLIPPAGE_RATE   × |Δw|² × portfolio_value
     The quadratic slippage term penalises large single-step rebalancing
     more aggressively than small incremental moves — consistent with
     realistic NSE market-impact models.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "data"
FEATURE_DIR = os.path.join(DATA_DIR, "features")
RL_DIR      = os.path.join(DATA_DIR, "rl")
os.makedirs(RL_DIR,      exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# ── Observation window ─────────────────────────────────────────────────────────
LOOKBACK   = 30        # days of history stacked per observation
N_FEATURES = 3         # features per (asset, day): return, volatility, volume

# ── Efficient frontier ─────────────────────────────────────────────────────────
K_PORTFOLIOS    = 10   # number of risk levels
COV_WINDOW_EF   = 252  # 1-year window for EF covariance estimation
RISK_FREE_RATE  = 0.067  # India 10-yr G-Sec yield (~6.7% as of 2024)
MIN_WEIGHT      = 0.0  # no short-selling
MAX_WEIGHT      = 0.40 # max single-asset weight (concentration limit)
CASH_TICKER     = "CASH"

# ── Transaction cost model ────────────────────────────────────────────────────
BROKERAGE_RATE = 0.0002   # 0.02% flat brokerage per trade
SLIPPAGE_RATE  = 0.001    # 0.1% base market-impact / slippage
SLIPPAGE_MAX   = 0.003    # cap slippage at 0.3% (large moves)

# ── Numerical constants ────────────────────────────────────────────────────────
EPS        = 1e-8
SPLITS     = ("train", "test", "val")
TRADING_DAYS = 252


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — STACKED OBSERVATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════

def _load_feature_csv(name: str, split: str) -> pd.DataFrame:
    path = os.path.join(FEATURE_DIR, f"{name}_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run features.py first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _load_volume(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"volume_clean_{split}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run preprocess.py first."
        )
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _normalise_volume(vol_df: pd.DataFrame,
                       window: int = LOOKBACK) -> pd.DataFrame:
    """
    Normalise volume to [0, 1] using a rolling min-max over `window` days.

    Why normalise volume?
    ─────────────────────
    Raw volume is in units of shares and varies by 6 orders of magnitude
    across assets (NIFTYBEES ETF vs KOTAKBANK).  Min-max over a rolling
    window makes it comparable across assets and stable over time.
    Rolling (not global) normalisation prevents look-ahead leakage.
    """
    rolling_min = vol_df.rolling(window, min_periods=1).min()
    rolling_max = vol_df.rolling(window, min_periods=1).max()
    denom = (rolling_max - rolling_min).replace(0, EPS)
    return (vol_df - rolling_min) / denom


def build_observation_stack(split: str,
                              verbose: bool = True) -> np.ndarray:
    """
    Build a (T, LOOKBACK, N_ASSETS, N_FEATURES) observation tensor.

    Feature channels (axis=-1)
    ──────────────────────────
    Channel 0 — log_return     (already computed by features.py)
    Channel 1 — norm_vol       (annualised σ, clipped to [EPS, 5])
    Channel 2 — norm_volume    (rolling min-max normalised turnover)

    Warm-up
    ───────
    The first (LOOKBACK - 1) rows cannot form a full window.
    These rows are assigned zero-filled tensors and flagged in
    `valid_mask` so the RL environment can skip them.

    Shape contract
    ──────────────
    obs[t]  has shape (LOOKBACK, N_ASSETS, N_FEATURES)
    obs[t, d, i, f] = feature f for asset i, d days before t
                      (obs[t, 0, :, :] = oldest; obs[t, -1, :, :] = most recent)
    """
    log_ret  = _load_feature_csv("log_ret",    split)
    roll_vol = _load_feature_csv("volatility", split)
    volume   = _load_volume(split)

    # Align all three DataFrames to the same index
    common_idx = log_ret.index.intersection(roll_vol.index).intersection(volume.index)
    log_ret    = log_ret.loc[common_idx]
    roll_vol   = roll_vol.loc[common_idx]
    volume     = volume.loc[common_idx]

    # Normalise volatility and volume for the neural network
    norm_vol    = roll_vol.clip(lower=EPS, upper=5.0) / 5.0   # scale to ~[0,1]
    norm_volume = _normalise_volume(volume, window=LOOKBACK)

    T, N = log_ret.shape
    obs  = np.zeros((T, LOOKBACK, N, N_FEATURES), dtype=np.float32)
    valid_mask = np.zeros(T, dtype=bool)

    for t in range(LOOKBACK - 1, T):
        window_slice = slice(t - LOOKBACK + 1, t + 1)
        obs[t, :, :, 0] = log_ret.iloc[window_slice].values
        obs[t, :, :, 1] = norm_vol.iloc[window_slice].values
        obs[t, :, :, 2] = norm_volume.iloc[window_slice].values
        valid_mask[t]   = True

    # Integrity checks
    valid_obs = obs[valid_mask]
    assert np.isfinite(valid_obs).all(), "Non-finite values in observation stack"
    assert valid_obs.shape[1:] == (LOOKBACK, N, N_FEATURES)

    if verbose:
        print(f"\n  Observation stack ({split})")
        print(f"    Shape      : {obs.shape}  "
              f"(T={T}, lookback={LOOKBACK}, N={N}, features={N_FEATURES})")
        print(f"    Valid rows : {valid_mask.sum()} / {T}")
        print(f"    Value range: [{valid_obs.min():.4f}, {valid_obs.max():.4f}]")
        print(f"    NaN count  : {np.isnan(valid_obs).sum()}")

    return obs, valid_mask


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DISCRETE ACTION SPACE (EFFICIENT FRONTIER PORTFOLIOS)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EFPortfolio:
    """A single efficient-frontier portfolio defining one RL action."""
    action_id:     int
    risk_label:    str          # "low" / "medium-low" / … / "high"
    target_return: float        # annualised target return used in optimisation
    weights:       np.ndarray   # (N+1,) including cash slot (index -1 = cash)
    tickers:       list[str]    # N stock tickers (+ "CASH" appended)
    exp_return:    float = 0.0  # portfolio expected return (annualised)
    exp_vol:       float = 0.0  # portfolio expected volatility (annualised)
    sharpe:        float = 0.0  # Sharpe ratio at risk-free = RISK_FREE_RATE

    def to_dict(self) -> dict:
        return {
            "action_id":     self.action_id,
            "risk_label":    self.risk_label,
            "target_return": round(self.target_return, 6),
            "exp_return":    round(self.exp_return, 6),
            "exp_vol":       round(self.exp_vol, 6),
            "sharpe":        round(self.sharpe, 6),
            "weights": {t: round(float(w), 6)
                        for t, w in zip(self.tickers, self.weights)},
        }


def _risk_labels(k: int) -> list[str]:
    """Generate K evenly spaced risk-level labels."""
    if k == 1:
        return ["medium"]
    labels = []
    for i in range(k):
        frac = i / (k - 1)
        if frac < 0.20:
            labels.append("low")
        elif frac < 0.40:
            labels.append("medium-low")
        elif frac < 0.60:
            labels.append("medium")
        elif frac < 0.80:
            labels.append("medium-high")
        else:
            labels.append("high")
    return labels


def _lw_covariance(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Fit Ledoit-Wolf on the full return matrix; return annualised DataFrame."""
    lw = LedoitWolf().fit(log_ret.values)
    cov = pd.DataFrame(lw.covariance_ * TRADING_DAYS,
                       index=log_ret.columns,
                       columns=log_ret.columns)
    return cov


def _portfolio_stats(weights_dict: dict[str, float],
                     mu: pd.Series,
                     cov: pd.DataFrame) -> tuple[float, float, float]:
    """Compute expected return, volatility and Sharpe for a weight dict."""
    tickers = list(weights_dict.keys())
    w       = np.array([weights_dict[t] for t in tickers])
    mu_arr  = mu.reindex(tickers).values
    cov_arr = cov.reindex(index=tickers, columns=tickers).values

    exp_ret = float(w @ mu_arr)
    exp_vol = float(np.sqrt(w @ cov_arr @ w))
    sharpe  = (exp_ret - RISK_FREE_RATE) / (exp_vol + EPS)
    return exp_ret, exp_vol, sharpe


def build_efficient_frontier_actions(split: str = "train",
                                      k: int = K_PORTFOLIOS,
                                      verbose: bool = True) -> list[EFPortfolio]:
    """
    Compute K portfolios along the mean-variance efficient frontier.

    Method
    ──────
    1. Estimate expected returns using CAPM mean (historical mean + shrinkage).
    2. Estimate covariance with Ledoit-Wolf shrinkage on the training returns.
    3. Solve EfficientFrontier for K target-return levels spaced uniformly
       between the minimum-variance portfolio return and the maximum-return
       portfolio (100 % in the highest-μ asset).
    4. Add a "cash-only" portfolio as action 0 (risk-free anchor).
    5. Each solution is projected to include a cash slot.

    Constraints
    ───────────
    · No short selling: all weights ∈ [0, 1].
    · Max single-asset weight: MAX_WEIGHT (0.40) — concentration limit.
    · Weights sum to ≤ 1; remainder allocated to cash.

    Why discrete actions?
    ─────────────────────
    The continuous simplex (infinite weight combinations) makes the policy
    network's output space extremely high-dimensional.  Discretising to K
    EF portfolios reduces the action space to K integers while retaining
    economically meaningful diversity (from near-cash to aggressive growth).
    """
    prices_path = os.path.join(DATA_DIR, f"adj_close_clean_{split}.csv")
    if not os.path.exists(prices_path):
        raise FileNotFoundError(
            f"{prices_path} not found. Run preprocess.py first."
        )

    prices  = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    log_ret = np.log(prices / prices.shift(1)).dropna()
    tickers = list(prices.columns)
    N       = len(tickers)

    # ── Expected returns (CAPM / historical mean) ──────────────────────────
    mu  = expected_returns.mean_historical_return(prices, frequency=TRADING_DAYS)
    cov = _lw_covariance(log_ret)

    # ── Determine feasible return range ───────────────────────────────────
    # Min-variance portfolio
    try:
        ef_min = EfficientFrontier(mu, cov,
                                   weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef_min.min_volatility()
        min_w  = ef_min.clean_weights()
        min_ret, _, _ = _portfolio_stats(min_w, mu, cov)
    except OptimizationError:
        min_ret = float(mu.min())

    max_ret = float(mu.max())

    if verbose:
        print(f"\n  Efficient Frontier ({split})")
        print(f"    Tickers       : {tickers}")
        print(f"    Return range  : [{min_ret:.4f}, {max_ret:.4f}] annualised")
        print(f"    K portfolios  : {k}")

    # ── Solve for K target returns ─────────────────────────────────────────
    # Stay well inside the feasible region given MAX_WEIGHT constraints
    # (constrained max-return < unconstrained mu.max())
    try:
        ef_max = EfficientFrontier(mu, cov, weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
        ef_max.max_sharpe(risk_free_rate=RISK_FREE_RATE)
        mw = ef_max.clean_weights()
        feasible_max_ret, _, _ = _portfolio_stats(mw, mu, cov)
    except OptimizationError:
        feasible_max_ret = max_ret * 0.85

    target_returns = np.linspace(min_ret, feasible_max_ret * 0.98, k)
    labels         = _risk_labels(k)
    portfolios: list[EFPortfolio] = []

    # Action 0 — pure cash (risk-free baseline)
    cash_w = np.zeros(N + 1, dtype=np.float64)
    cash_w[-1] = 1.0
    portfolios.append(EFPortfolio(
        action_id=0, risk_label="cash",
        target_return=RISK_FREE_RATE,
        weights=cash_w,
        tickers=tickers + [CASH_TICKER],
        exp_return=RISK_FREE_RATE, exp_vol=0.0,
        sharpe=0.0,
    ))

    for i, (target, label) in enumerate(zip(target_returns, labels), start=1):
        try:
            ef = EfficientFrontier(mu, cov,
                                   weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
            ef.efficient_return(target_return=target)
            raw_w = ef.clean_weights()
        except (OptimizationError, ValueError):
            # Fall back to max-Sharpe if target return is infeasible
            try:
                ef = EfficientFrontier(mu, cov,
                                       weight_bounds=(MIN_WEIGHT, MAX_WEIGHT))
                ef.max_sharpe(risk_free_rate=RISK_FREE_RATE)
                raw_w = ef.clean_weights()
            except OptimizationError:
                if verbose:
                    print(f"    Action {i:>2}  [{label}]  "
                          f"target={target:.4f}  → INFEASIBLE, skipped")
                continue

        # Build (N+1,) weight vector with cash slot
        stock_w = np.array([raw_w.get(t, 0.0) for t in tickers],
                            dtype=np.float64)
        cash    = max(0.0, 1.0 - stock_w.sum())
        full_w  = np.append(stock_w, cash)
        full_w  = full_w / full_w.sum()   # normalise for float safety

        exp_ret, exp_vol, sharpe = _portfolio_stats(raw_w, mu, cov)

        pfol = EFPortfolio(
            action_id=i, risk_label=label,
            target_return=target,
            weights=full_w,
            tickers=tickers + [CASH_TICKER],
            exp_return=exp_ret, exp_vol=exp_vol, sharpe=sharpe,
        )
        portfolios.append(pfol)

        if verbose:
            alloc = {t: f"{w:.3f}"
                     for t, w in zip(tickers, stock_w) if w > 0.001}
            print(f"    Action {i:>2}  [{label:<12}]  "
                  f"ret={exp_ret:.4f}  vol={exp_vol:.4f}  "
                  f"Sharpe={sharpe:.3f}  cash={cash:.3f}  "
                  f"alloc={alloc}")

    if verbose:
        print(f"\n  Total actions (incl. cash): {len(portfolios)}")

    return portfolios


def save_action_space(portfolios: list[EFPortfolio],
                       split: str = "train") -> None:
    """Persist action space as JSON (readable) and NPZ (fast load)."""
    # JSON — human-readable
    json_path = os.path.join(RL_DIR, f"action_space_{split}.json")
    with open(json_path, "w") as f:
        json.dump([p.to_dict() for p in portfolios], f, indent=2)

    # NPZ — fast numpy load for the RL environment
    npz_path = os.path.join(RL_DIR, f"action_space_{split}.npz")
    weight_matrix = np.stack([p.weights for p in portfolios])   # (K+1, N+1)
    np.savez(npz_path,
             weights=weight_matrix,
             action_ids=np.array([p.action_id  for p in portfolios]),
             exp_returns=np.array([p.exp_return for p in portfolios]),
             exp_vols=np.array([p.exp_vol    for p in portfolios]),
             sharpes=np.array([p.sharpe     for p in portfolios]))

    print(f"\n  Action space saved:")
    print(f"    {json_path}")
    print(f"    {npz_path}")


def load_action_space(split: str = "train") -> np.ndarray:
    """Load the (K+1, N+1) weight matrix from disk."""
    npz = np.load(os.path.join(RL_DIR, f"action_space_{split}.npz"))
    return npz["weights"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — TRANSACTION COST MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransactionCostModel:
    """
    Two-component transaction cost model for Indian equities.

    Component 1 — Proportional brokerage
    ──────────────────────────────────────
    India NSE typical all-in cost (brokerage + STT + SEBI charges):
        c_b = BROKERAGE_RATE × |Δw| × portfolio_value
    BROKERAGE_RATE = 0.02% per side (discount broker, e.g. Zerodha)

    Component 2 — Market-impact / slippage
    ────────────────────────────────────────
    Slippage is modelled as quadratic in trade size — consistent with
    Almgren-Chriss market-impact literature:
        c_s = SLIPPAGE_RATE × |Δw|² × portfolio_value
    Capped at SLIPPAGE_MAX × |Δw| × portfolio_value (linear cap for large trades).
    SLIPPAGE_RATE = 0.1%  (liquid large-cap Indian stocks)
    SLIPPAGE_MAX  = 0.3%  (upper bound for thin trading days)

    Total cost
    ──────────
        cost = (c_b + c_s) × portfolio_value
             = [BROKERAGE_RATE × |Δw|  +  min(SLIPPAGE_RATE × |Δw|², SLIPPAGE_MAX × |Δw|)]
               × portfolio_value

    Usage in reward function
    ────────────────────────
        reward = portfolio_return - transaction_cost_model.cost_fraction(Δw)
    """
    brokerage_rate: float = BROKERAGE_RATE
    slippage_rate:  float = SLIPPAGE_RATE
    slippage_max:   float = SLIPPAGE_MAX

    def turnover(self, w_old: np.ndarray, w_new: np.ndarray) -> float:
        """
        Half-turnover: sum of absolute weight changes for stock positions only.
        Cash changes are excluded (no cost to hold cash).
        Index -1 is the cash slot; exclude it.
        """
        delta = np.abs(w_new[:-1] - w_old[:-1])
        return float(delta.sum())

    def brokerage_cost(self, turnover: float) -> float:
        """Proportional brokerage as a fraction of portfolio value."""
        return self.brokerage_rate * turnover

    def slippage_cost(self, turnover: float) -> float:
        """
        Quadratic slippage, capped at linear maximum.
        Quadratic: penalises large trades more than proportional cost.
        Cap: prevents absurdly large penalties for 100%-rebalancing.
        """
        quadratic = self.slippage_rate * (turnover ** 2)
        linear_cap = self.slippage_max * turnover
        return float(min(quadratic, linear_cap))

    def cost_fraction(self, w_old: np.ndarray,
                       w_new: np.ndarray) -> float:
        """
        Total transaction cost as a *fraction* of portfolio value.
        Subtract this from the period return to get the net reward.

        Example
        ───────
        w_old = [0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.0, 0.0, 0.1]  (8 stocks + cash)
        w_new = [0.1, 0.3, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1]
        Δw    = [0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0]  (ignoring cash)
        turnover = 0.4
        brokerage = 0.0002 × 0.4 = 0.00008  (0.008%)
        slippage  = min(0.001 × 0.16, 0.003 × 0.4) = min(0.00016, 0.0012) = 0.00016
        total     = 0.024%  of portfolio value
        """
        to = self.turnover(w_old, w_new)
        return self.brokerage_cost(to) + self.slippage_cost(to)

    def cost_pnl(self, w_old: np.ndarray,
                  w_new: np.ndarray,
                  portfolio_value: float) -> float:
        """Absolute cost in rupees (or whatever currency portfolio_value is in)."""
        return self.cost_fraction(w_old, w_new) * portfolio_value

    def describe(self) -> None:
        print(f"  Transaction Cost Model")
        print(f"    Brokerage rate : {self.brokerage_rate*100:.4f}%  (per unit turnover)")
        print(f"    Slippage rate  : {self.slippage_rate*100:.4f}%  (quadratic in turnover)")
        print(f"    Slippage cap   : {self.slippage_max*100:.4f}%  (linear cap for large trades)")

        # Illustrative table
        print(f"\n    Turnover → Total cost (% of NAV)")
        print(f"    {'Turnover':>10}  {'Brokerage':>12}  {'Slippage':>10}  {'Total':>8}")
        print(f"    {'─'*10}  {'─'*12}  {'─'*10}  {'─'*8}")
        for to in [0.05, 0.10, 0.20, 0.40, 0.80, 1.0]:
            b  = self.brokerage_cost(to)
            s  = self.slippage_cost(to)
            print(f"    {to:>10.2f}  {b*100:>11.4f}%  {s*100:>9.4f}%  "
                  f"{(b+s)*100:>7.4f}%")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  RL Preprocessing Pipeline")
    print("=" * 60)

    # ── 1. Observation stacks ──────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SECTION 1 — Stacked Observation Windows")
    print("─" * 60)

    obs_stacks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        try:
            obs, mask = build_observation_stack(split)
            obs_stacks[split] = (obs, mask)
            # Save
            np.save(os.path.join(RL_DIR, f"obs_{split}.npy"),  obs)
            np.save(os.path.join(RL_DIR, f"mask_{split}.npy"), mask)
            print(f"    Saved obs_{split}.npy  mask_{split}.npy")
        except FileNotFoundError as exc:
            print(f"    ✗  {exc}")

    # ── 2. Efficient frontier action space ────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SECTION 2 — Efficient Frontier Action Space")
    print("─" * 60)

    try:
        portfolios = build_efficient_frontier_actions("train", k=K_PORTFOLIOS)
        save_action_space(portfolios, split="train")
        action_matrix = load_action_space("train")
        print(f"\n  Action weight matrix shape : {action_matrix.shape}")
        print(f"  Actions × (N_stocks + cash) = {action_matrix.shape[0]} × "
              f"{action_matrix.shape[1]}")
        # Verify all rows sum to 1
        row_sums = action_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), \
            f"Action weight rows don't sum to 1: {row_sums}"
        print(f"  ✓  All action weight vectors sum to 1.0")
    except FileNotFoundError as exc:
        print(f"  ✗  {exc}")

    # ── 3. Transaction cost model ─────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  SECTION 3 — Transaction Cost Model")
    print("─" * 60)

    tc = TransactionCostModel()
    tc.describe()

    # Worked example
    print("\n  Worked example — rebalance from equal-weight to action 3:")
    if "train" in [s for s in SPLITS if s in str(obs_stacks.keys())]:
        try:
            actions = load_action_space("train")
            N       = actions.shape[1] - 1   # number of stocks
            w_old   = np.ones(N + 1) / (N + 1)
            w_new   = actions[min(3, len(actions) - 1)]
            to      = tc.turnover(w_old, w_new)
            cost    = tc.cost_fraction(w_old, w_new)
            print(f"    Turnover           : {to:.4f}")
            print(f"    Total cost (% NAV) : {cost*100:.4f}%")
            print(f"    On ₹10,00,000 NAV  : ₹{tc.cost_pnl(w_old, w_new, 1_000_000):.2f}")
        except Exception as exc:
            print(f"    (skipped: {exc})")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Observation tensor  : (T, {LOOKBACK}, N, {N_FEATURES})  "
          f"— CNN/LSTM ready")
    print(f"  Action space        : {K_PORTFOLIOS + 1} discrete EF portfolios "
          f"(0 = cash)")
    print(f"  Cost model          : brokerage={BROKERAGE_RATE*100:.3f}%  "
          f"slippage={SLIPPAGE_RATE*100:.3f}% (quadratic, capped at "
          f"{SLIPPAGE_MAX*100:.2f}%)")
    print(f"\n  Files written to ./data/rl/")
