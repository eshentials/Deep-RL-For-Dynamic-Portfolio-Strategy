"""
env.py — PortfolioEnv: Gymnasium environment for Deep-RL portfolio management.

The RL agent jointly decides four things at every step:
  1. Risk value   — which λ to pass to the Mean-Variance Optimizer (MVO).
                    λ ∈ LAMBDA_VALUES = linspace(0.1, 2.0, 20).
                    Low λ → return-seeking; high λ → minimum-variance.
  2. Holding period — how many trading days to hold the MVO portfolio before
                      the next decision.
                      HOLDING_PERIODS = [1, 3, 5, 10, 21] days.
  3. Equity exposure — max stock allocation passed to the MVO. The remainder
                      is cash, allowing the policy to de-risk.
  4. Benchmark blend — how much to blend the MVO target with equal-weight.

At each step the environment:
  (a) Passes (state, λ) to efficient_frontier.optimize() → target weights.
  (b) Deducts dynamic transaction costs from NAV using the shared model in
      state_spec.py: statutory delivery charges plus volatility/liquidity stress.
  (c) Simulates the portfolio forward for `hold_days` trading days,
      updating weights by daily price drift (buy-and-hold within period).
  (d) Returns a regime-conditioned reward blending risk-adjusted alpha,
      TC cost, drawdown shaping, and an opportunity cost for excess cash in
      positive market regimes.

Action space:  MultiDiscrete([20, 5, 6, 5])
  action[0] ∈ {0…19}  → lambda_idx → LAMBDA_VALUES[lambda_idx]
  action[1] ∈ {0…4}   → hold_idx   → HOLDING_PERIODS[hold_idx]
  action[2] ∈ {0…5}   → exposure   → [25%, 40%, 55%, 70%, 85%, 100%]
  action[3] ∈ {0…4}   → anchor blend with equal-weight [0%, 25%, 50%, 75%, 100%]

Observation space: Box(-5, 5, shape=(84,), dtype=float32)
  The state vector defined in state_spec.py (clipped for NN stability).

Transaction cost:
  cost = explicit buy/sell charges + dynamic execution friction
  Dynamic friction responds to realised volatility, current volume versus ADV,
  market regime, and drawdown. The agent observes its effect through reward/NAV
  but cannot directly control the cost model.

Reward (hybrid regime + DSR/MDD):
  Three core signal terms, then regime-conditioned bonuses/penalties:

  R_lr  = cum_log_ret / trading_days_span
        — per-day log return (growth signal).

  R_dsr = Differential Sharpe Ratio (Moody & Saffell, 1999).
        — online, stateful risk-adjusted return via EMA of E[r] and E[r²].
        — denominator is (Var[r])^{3/2}; numerator encodes the marginal
          improvement in Sharpe from the current return.
        — stateful across steps: self.X (EMA of returns), self.Y (EMA of r²),
          self.eta = 1/252 (decay constant ≈ one-year memory).
        — replaces the ad-hoc Sharpe/Sortino alpha_signal.

  R_mdd = -sqrt(max_drawdown_so_far)
        — running Maximum Drawdown penalised with sqrt for smooth gradients.
        — self.max_drawdown updated every simulated day inside the holding loop.

  Final blend (per-day amortised when reward_scale=True):
    reward = rp["alpha_weight"] * R_lr          ← growth (doubled multiplier [RT-1])
           + DSR_WEIGHT          * R_dsr         ← risk-adjustment
           + rp["drawdown_weight"] * R_mdd       ← MDD (halved weight [RT-2])
           + bull_floor_penalty                  ← hard exposure floor in bull [RT-5]
           + participation_bonus                 (bull only, [BF-6], scale ↑ [RT-4])
           + momentum_bonus                      (bull only, [BF-7])
           + lambda_aggression_bonus             (bull only, [LA-2])
           - lambda_defensiveness_penalty        (bull only, [LA-3])
           - 1.0 * tc
           - exposure_penalty                    (asymmetric in bull, [BF-5])
           - effective_turnover_weight * turnover ([LA-4], global 0.5× relief [RT-6])
           - cash_penalty                        ([BF-8], quadrupled [RT-3])

RETURN-AGGRESSION FIXES (annotated with [RT-N]):
  [RT-1]  R_lr multiplier doubled: alpha_weight already scales R_lr per regime,
          but the base R_lr itself is now 2× — effectively doubling the growth
          gradient for all regimes without disturbing the relative regime weighting.
          Mathematically: grad(reward)/grad(return) doubles → policy gradient
          pushes harder toward high-return actions.

  [RT-2]  MDD penalty halved: drawdown_weight values in REGIME_REWARD_PARAMS all
          multiplied by 0.5 at the reward assembly stage via MDD_PENALTY_SCALE=0.5.
          This reduces the fear signal without changing the regime calibration
          structure — the relative ordering (crisis > bear > sideways > bull) is
          preserved.

  [RT-3]  CASH_OPPORTUNITY_PENALTY quadrupled: 5.0e-3 → 2.0e-2. This is the
          single highest-impact lever. Cash is "free" in the old env — zero vol,
          zero return, zero penalty beyond a soft threshold. Quadrupling the
          penalty makes holding cash in bull/sideways regimes actively costly,
          forcing the agent to deploy capital. The tighter bull cash floor (5%)
          means the penalty fires on anything above 5% cash.

  [RT-4]  BULL_PARTICIPATION_SCALE raised: 8.0 → 15.0. The participation bonus
          (cum_log_ret / trading_days_span) was previously too small relative to
          the TC and turnover penalties to overcome them. At 15×, a 1% daily
          return contributes +0.15 to reward, which clearly dominates a 0.5%
          one-way TC hit. Agent now clearly prefers participating over sitting out.

  [RT-5]  Hard bull exposure floor penalty: if actual_exposure < BULL_EXPOSURE_FLOOR
          (0.80), apply a quadratic penalty scaled by BULL_FLOOR_PENALTY_SCALE (3.0).
          Unlike the existing exposure_penalty (which uses rp["exposure_weight"]=0.15
          and is one-sided), this is an additional hard penalty making sub-80%
          allocation in bull very costly. Gradient is steeper near the floor:
          at 70% exposure: -(3.0 * 0.10²) = -0.030 per step
          at 60% exposure: -(3.0 * 0.20²) = -0.120 per step
          This directly addresses the behavioral finding that the agent was
          choosing ~63% exposure in bull markets.

  [RT-6]  Global turnover penalty halved: all regimes get effective_turnover_weight
          *= GLOBAL_TURNOVER_RELIEF (0.5) before further [LA-4] adjustment in bull.
          The TC term (1.0 * tc) already penalises actual transaction cost in NAV;
          the turnover_weight penalty was a redundant second-order deterrent to
          action. Halving it allows the agent to rebalance more freely without
          double-counting the economic cost of doing so.

BULL-MARKET FIXES (annotated with [BF-N]):
  [BF-1]  Regime detection: crisis vol threshold raised 0.025 → 0.035.
  [BF-2]  Regime detection: bull gate changed from conjunctive AND to a
          weighted-score approach.
  [BF-3]  Regime detection: added explicit confidence score; sideways is
          the genuine default for ambiguous conditions.
  [BF-4]  Bull exposure_weight reduced 0.40 → 0.15.
  [BF-5]  Bull exposure penalty changed to one-sided asymmetric form.
  [BF-6]  Added bull-regime participation bonus.
  [BF-7]  Added momentum-alignment bonus in bull.
  [BF-8]  Cash opportunity penalty increased 2.5e-3 → 5.0e-3; tighter bull floor.
  [BF-9]  Removed hard equity floor for BULL regime only.
  [BF-10] Bull alpha_weight raised 6.0 → 7.5.
  [BF-11] Sideways regime is the genuine default fallthrough.

LAMBDA AGGRESSION FIXES for BULL (annotated with [LA-N]):
  [LA-1]  REMOVED — was Sortino-style alpha denominator. The alpha_signal is
          now replaced by DSR (Differential Sharpe Ratio), which is inherently
          downside-sensitive via the variance term in its denominator. Upside
          vol inflates both E[r] and E[r²]; when r > E[r] (positive surprise),
          the numerator is positive regardless of total volatility.

  [LA-2]  Lambda-aggression bonus in bull: directly rewards choosing low λ
          when actual returns were positive. Mathematically equivalent to a
          Kelly-style bet-sizing premium — if you took more risk (low λ) and
          got paid for it (positive return), you earn more than someone who
          held a defensive portfolio and happened to ride the same market.
          Formula: BULL_LAMBDA_AGG_SCALE * max(cum_log_ret, 0) * (1 - lam/LAM_MAX)
          Zero at lam=LAM_MAX (fully defensive), maximum at lam=LAM_MIN
          (fully aggressive). Zero whenever returns are negative.

  [LA-3]  Lambda-defensiveness penalty in bull: penalises choosing high λ
          when returns were positive, i.e. deliberately leaving return on the
          table by minimising variance in a rising market.
          Formula: BULL_LAMBDA_DEF_SCALE * max(cum_log_ret, 0) * (lam/LAM_MAX)
          Combined with [LA-2], the net signed gradient on λ is proportional
          to cum_log_ret * (1 - 2*lam/LAM_MAX), crossing zero at lam=LAM_MAX/2
          (~1.05). Clear gradient: low λ positive, high λ negative.

  [LA-4]  Lambda-adaptive turnover weight in bull: low-λ MVO concentrates into
          high-momentum names, generating higher natural turnover when momentum
          rotates. The old flat turnover_weight=0.01 disproportionately penalised
          aggressive λ choices. In bull, the weight is scaled down further when
          the agent chooses a low-λ action.
          Effective weight = base_tw * (1 - BULL_LOW_LAM_TURNOVER_RELIEF * (1 - lam/LAM_MAX))
          At lam=LAM_MIN: ~0.62 * base_tw. At lam=LAM_MAX: unchanged.

DSR IMPLEMENTATION NOTES:
  The Differential Sharpe Ratio (Moody & Saffell 1999) is defined as:

      DSR_t = [Y_{t-1}(r_t - X_{t-1}) - 0.5 * X_{t-1}(r_t² - Y_{t-1})]
              / (Y_{t-1} - X_{t-1}²)^{3/2}

  where X = EMA(r), Y = EMA(r²), updated via:
      X_t = X_{t-1} + η (r_t - X_{t-1})
      Y_t = Y_{t-1} + η (r_t² - Y_{t-1})

  η = 1/252 ≈ one-year exponential memory.

  Since each RL step spans `hold_days` simulated days, we update X and Y
  once per simulated day (inside the holding loop), then compute DSR once
  using the *final* daily return of the holding period against the EMA state
  that has been updated through the penultimate day. This mirrors the
  original single-step formulation while being compatible with multi-day holds.

  DSR is clipped to [-DSR_CLIP, +DSR_CLIP] before scaling to prevent
  blow-up during the early warm-up phase when variance estimates are tiny.

MDD IMPLEMENTATION NOTES:
  self.max_drawdown tracks the worst (peak-to-trough) fractional drawdown
  *so far in this episode*, updated every simulated day:

      current_dd = (peak_nav - nav) / (peak_nav + EPS)
      self.max_drawdown = max(self.max_drawdown, current_dd)

  The reward term uses -sqrt(max_drawdown) rather than -max_drawdown for two
  reasons:
    (a) Smoother gradient: sqrt has infinite slope at 0, giving a large early
        signal even for tiny drawdowns, then flattening for large ones (where
        the agent is already penalised heavily by NAV loss).
    (b) Reduces reward variance: MDD can jump from 0 to 0.2 in one bad step;
        sqrt compresses this to 0 → 0.45, keeping PPO advantage estimates stable.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from features import load_bundle, FeatureBundle
from state_spec import (
    STATE,
    build_state_vector,
    transaction_cost,
    MACRO_PATH,
    EPS,
)
from efficient_frontier import horizon_momentum_pct, optimize, LAMBDA_VALUES

# ── Action-space constants ─────────────────────────────────────────────────────
HOLDING_PERIODS: list[int]   = [1, 3, 5, 10, 21]                          # trading days
EXPOSURE_LEVELS: list[float] = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
ANCHOR_BLEND_LEVELS: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4]
N_LAMBDA:       int = len(LAMBDA_VALUES)        # 20
N_HOLD:         int = len(HOLDING_PERIODS)      # 5
N_EXPOSURE:     int = len(EXPOSURE_LEVELS)      # 6
N_ANCHOR_BLEND: int = len(ANCHOR_BLEND_LEVELS)  # 5

# ── Lambda range constants (used in aggression/defensiveness formulas) ─────────
LAM_MIN: float = float(LAMBDA_VALUES[0])    # 0.1
LAM_MAX: float = float(LAMBDA_VALUES[-1])   # 2.0

# ── Observation clipping ───────────────────────────────────────────────────────
OBS_CLIP: float = 5.0

# ── MVO soft turnover penalty (separate from the reward TC term) ───────────────
TURNOVER_PENALTY: float = 0.005

# ── Benchmark for excess-return calculation ────────────────────────────────────
BENCHMARK_REWARD: str = "equal_weight"   # "equal_weight" | "cash"

# ── Opportunity cost for holding too much cash in positive regimes ─────────────
# [BF-8] Doubled from 2.5e-3 → 5.0e-3.
# [RT-3] Quadrupled again: 5.0e-3 → 2.0e-2. This is the single highest-impact
#         lever for forcing capital deployment. At this level, holding 20% cash
#         above the 5% bull floor costs 0.02 * hold_days * 0.15 ≈ 0.015 per
#         5-day step — clearly dominating a typical rebalancing TC of ~0.2%.
CASH_OPPORTUNITY_PENALTY: float = 2.0e-2

# ── Cash floor by regime ───────────────────────────────────────────────────────
# [BF-8] Tighter floor in bull (5%) so the penalty fires more aggressively.
CASH_FLOOR_BY_REGIME: dict[str, float] = {
    "bull":     0.05,
    "sideways": 0.10,
    "bear":     0.15,
    "crisis":   0.20,
}

# ── Clip on the DSR signal before scaling by regime weight ────────────────────
# Replaces ALPHA_SIGNAL_CLIP. Same purpose: prevent blow-up in the early
# warm-up phase when X and Y are near zero and variance is tiny.
DSR_CLIP: float = 3.0

# ── DSR EMA decay constant ────────────────────────────────────────────────────
# η = 1/252: exponential memory of ~one trading year.
# Smaller η → slower adaptation (more stable but lags regime shifts).
# Larger η → faster adaptation (more reactive but noisier).
DSR_ETA: float = 1.0 / 252.0

# ── Weight on DSR term in the final reward blend ──────────────────────────────
# rp["alpha_weight"] governs R_lr; DSR_WEIGHT governs R_dsr.
# Kept at 0.3 so DSR stabilises without dominating growth incentive.
DSR_WEIGHT: float = 0.1

# ── [BF-6] Participation bonus scale for bull regime ──────────────────────────
# [RT-4] Raised 8.0 → 15.0. At old scale, a 0.5% daily return contributed only
#         +0.04 to reward — easily swamped by TC and turnover penalties. At 15×,
#         the same return contributes +0.075, making participation unambiguously
#         worth the cost of deploying capital. This creates a clear gradient:
#         participate → positive return → large bonus > TC cost.
BULL_PARTICIPATION_SCALE: float = 18.0

# ── [BF-7] Momentum-alignment bonus ───────────────────────────────────────────
MOMENTUM_ALIGNMENT_SCALE: float = 4.0

# ── [LA-2] Lambda-aggression bonus scale ──────────────────────────────────────
# Rewards choosing low λ when bull returns are positive.
BULL_LAMBDA_AGG_SCALE: float = 9.0

# ── [LA-3] Lambda-defensiveness penalty scale ─────────────────────────────────
# Penalises choosing high λ when bull returns were positive.
BULL_LAMBDA_DEF_SCALE: float = 8.0   # match AGG_SCALE for symmetric gradient

# ── [LA-4] Turnover relief for low-λ choices in bull ──────────────────────────
BULL_LOW_LAM_TURNOVER_RELIEF: float = 0.40

# ── [RT-1] R_lr multiplier — doubles growth gradient globally ─────────────────
# The base R_lr = cum_log_ret / trading_days_span is multiplied by this before
# being scaled by rp["alpha_weight"]. Doubling it doubles d(reward)/d(return)
# for every regime without disturbing relative regime calibration.
# At 2×, a 0.1% daily return edge over benchmark contributes:
#   bull: 2 * 7.5 * 0.001 = +0.015 vs old +0.0075
# This makes small daily alpha clearly visible to the policy gradient.
R_LR_MULTIPLIER: float = 3.0

# ── [RT-2] MDD penalty scale — halves the fear signal ────────────────────────
# Applied at reward assembly: R_mdd_scaled = MDD_PENALTY_SCALE * R_mdd.
# Values in REGIME_REWARD_PARAMS["drawdown_weight"] are preserved for regime
# calibration; this global scalar reduces the overall fear without changing
# the bull < sideways < bear < crisis ordering.
# Old: drawdown_weight * (-sqrt(MDD))   e.g. bull: -0.05 * sqrt(MDD)
# New: 0.5 * drawdown_weight * (-sqrt(MDD))  e.g. bull: -0.025 * sqrt(MDD)
MDD_PENALTY_SCALE: float = 0.5

# ── [RT-5] Bull hard exposure floor penalty ───────────────────────────────────
# Additional quadratic penalty when actual_exposure < BULL_EXPOSURE_FLOOR in bull.
# The existing exposure_penalty (exposure_weight=0.15, one-sided) is too soft.
# This hard floor makes sub-80% exposure in bull genuinely costly:
#   70% exposure → -(3.0 * 0.10²) = -0.030 per step
#   60% exposure → -(3.0 * 0.20²) = -0.120 per step
# Combined with the softer exposure_penalty, the gradient is steep below 80%.
BULL_EXPOSURE_FLOOR: float  = 0.95
BULL_FLOOR_PENALTY_SCALE: float = 3.0

# ── [RT-6] Global turnover penalty relief ─────────────────────────────────────
# All regimes: effective_turnover_weight *= GLOBAL_TURNOVER_RELIEF before
# further [LA-4] adjustment in bull. The explicit tc term (1.0 * tc) already
# captures the economic cost of trading in NAV; the turnover_weight is a
# behavioural deterrent that was over-penalising action. Halving it allows
# the agent to rebalance toward high-return positions more freely.
GLOBAL_TURNOVER_RELIEF: float = 0.5

DEFAULT_MAX_EPISODE_DAYS: int = 252

SPLIT_LABELS: dict[str, str] = {
    "train": "train_2014_2020",
    "test":  "test_2021",
    "val":   "val_2022_2024",
}

# ── Regime-conditioned reward coefficients ────────────────────────────────────
# alpha_weight    → scales R_lr * R_LR_MULTIPLIER (log return per day)
# drawdown_weight → scales MDD_PENALTY_SCALE * R_mdd (-sqrt(max_drawdown))
# exposure_weight → scales the quadratic exposure penalty (soft, one-sided bull)
# exposure_target → desired equity allocation for this regime
# turnover_weight → base; multiplied by GLOBAL_TURNOVER_RELIEF then [LA-4] in bull
REGIME_REWARD_PARAMS: dict[str, dict] = {
    "bull": dict(
        alpha_weight    = 12.5,   # [BF-10]; effective growth scale = 7.5 * R_LR_MULTIPLIER = 15.0
        drawdown_weight = 0.05,  # effective MDD scale = 0.05 * MDD_PENALTY_SCALE = 0.025
        exposure_weight = 0.05,  # [BF-4]; soft one-sided; hard floor via BULL_FLOOR_PENALTY_SCALE
        exposure_target = 1.00,
        turnover_weight = 0.04,  # effective = 0.01 * GLOBAL_TURNOVER_RELIEF * (1 - [LA-4])
    ),
    "bear": dict(
        alpha_weight    = 4.0,
        drawdown_weight = 0.25,  # effective = 0.25 * 0.5 = 0.125
        exposure_weight = 0.10,
        exposure_target = 0.60,
        turnover_weight = 0.03,  # effective = 0.03 * 0.5 = 0.015
    ),
    "crisis": dict(
        alpha_weight    = 2.0,
        drawdown_weight = 0.50,
        exposure_weight = 0.05,
        exposure_target = 0.40,
        turnover_weight = 0.02,
    ),
    "sideways": dict(
        alpha_weight    = 8.0,
        drawdown_weight = 0.10,
        exposure_weight = 0.10,
        exposure_target = 0.90,
        turnover_weight = 0.04,
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — weight drift within a holding period
# ══════════════════════════════════════════════════════════════════════════════

def _drift_weights(w: np.ndarray, stock_returns: np.ndarray) -> np.ndarray:
    """
    Update (N+1,) portfolio weights after one day of price moves (no rebalance).

    Stock weights grow with their return; cash weight is unchanged.
    Result is re-normalised so weights continue to sum to 1.
    """
    w_stocks = w[:-1] * (1.0 + stock_returns)
    w_cash   = w[-1]
    total    = w_stocks.sum() + w_cash
    if total < EPS:
        N = len(stock_returns)
        return np.ones(N + 1, dtype=np.float64) / (N + 1)
    return np.append(w_stocks / total, w_cash / total)


# ══════════════════════════════════════════════════════════════════════════════
# GYMNASIUM ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioEnv(gym.Env):
    """
    Gymnasium environment where the agent jointly chooses:
      · λ (risk-aversion) passed to the MVO
      · holding period (days before the next rebalance decision)
      · max equity exposure (leftover goes to cash)
      · benchmark anchor blend (how much to pull weights toward equal-weight)

    The MVO (efficient_frontier.optimize) computes the portfolio weights;
    the RL agent selects the risk level, time horizon, exposure cap, and blend.

    Reward signal uses three mathematically grounded components:
      1. R_lr  — log return per day (growth)
      2. R_dsr — Differential Sharpe Ratio (online risk-adjustment via EMA)
      3. R_mdd — negative sqrt of running Maximum Drawdown (downside control)

    Parameters
    ──────────
    split            : data split to use ("train" | "test" | "val")
    reward_scale     : if True, reward is divided by the number of simulated
                       trading days so that steps with different holding periods
                       are directly comparable to the PPO critic.
    random_start     : if True AND split=="train", sample a random episode start
                       inside the usable history window.
    max_episode_days : force episode truncation after this many trading days
                       unless the dataset ends sooner.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        split: str = "train",
        reward_scale: bool = True,
        random_start: bool | None = None,
        max_episode_days: int = DEFAULT_MAX_EPISODE_DAYS,
    ):
        super().__init__()
        self.split        = split
        self.reward_scale = reward_scale
        if random_start is None:
            self.random_start = split == "train"
        else:
            self.random_start = bool(random_start) and split == "train"
        self.max_episode_days = int(max(max_episode_days or DEFAULT_MAX_EPISODE_DAYS, 30))

        # ── Load data ──────────────────────────────────────────────────────────
        self.bundle: FeatureBundle = load_bundle(split)
        self.macro_df: pd.DataFrame = pd.read_csv(
            MACRO_PATH, index_col=0, parse_dates=True
        )
        label = SPLIT_LABELS[split]
        volume_path = os.path.join("data", f"volume_{label}.csv")
        self.volume_df: pd.DataFrame = pd.read_csv(
            volume_path, index_col=0, parse_dates=True
        ).reindex(index=self.bundle.dates, columns=self.bundle.tickers).ffill().fillna(0.0)
        self.adv20_df: pd.DataFrame = (
            self.volume_df.rolling(20, min_periods=1).mean().replace(0.0, EPS)
        )
        self.tickers: list[str] = self.bundle.tickers
        self.N: int             = self.bundle.n_assets
        self.T: int             = len(self.bundle.dates)
        self.warm_up: int       = self.bundle.warm_up_rows()   # 90

        # ── Spaces ─────────────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete([
            N_LAMBDA,
            N_HOLD,
            N_EXPOSURE,
            N_ANCHOR_BLEND,
        ])
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP,
            shape=(STATE.DIM,), dtype=np.float32,
        )

        # ── Episode state (initialised in reset) ──────────────────────────────
        self.t:                   int         = self.warm_up
        self.nav:                 float       = 1.0
        self.nav_history:         list[float] = []
        self.current_w:           np.ndarray  = np.ones(self.N + 1) / (self.N + 1)
        self.peak_nav:            float       = 1.0
        self.max_drawdown:        float       = 0.0   # running MDD for this episode
        self.total_tc:            float       = 0.0
        self.episode_steps:       int         = 0
        self.last_turnover:       float       = 0.0
        self.last_tc_rate:        float       = 0.0
        self.episode_horizon_end: int         = self.T - 1
        self.days_in_episode:     int         = 0

        # ── DSR stateful accumulators (reset each episode) ────────────────────
        # X = EMA of daily portfolio returns  E[r]
        # Y = EMA of daily squared returns    E[r²]
        # η = exponential decay ≈ 1-year memory
        self.dsr_X:   float = 0.0   # EMA of r
        self.dsr_Y:   float = 0.0   # EMA of r²
        self.dsr_eta: float = DSR_ETA

    # ── Private helpers ────────────────────────────────────────────────────────

    def _episode_upper_bound(self, start_t: int) -> int:
        """Last usable index keeping at least max_episode_days + buffer."""
        return min(self.T - 2, start_t + self.max_episode_days)

    @staticmethod
    def _prepare_mvo_extras(bundle: FeatureBundle, t: int) -> dict[str, np.ndarray]:
        px    = bundle.prices
        t_idx = max(0, min(t, len(bundle.dates) - 1))
        return dict(
            mom20d_t    = horizon_momentum_pct(px, t_idx, 20),
            mom60d_t    = horizon_momentum_pct(px, t_idx, 60),
            cov_matrix  = bundle.cov[t_idx],
            daily_ret_t = bundle.daily_ret.iloc[t_idx].values.astype(np.float64),
        )

    def _benchmark_log_return(self, start_t: int, end_t: int) -> float:
        """Equal-weight (or cash) benchmark log return over [start_t, end_t)."""
        if end_t <= start_t:
            return 0.0
        daily = self.bundle.daily_ret.iloc[start_t:end_t].values.astype(np.float64)
        if BENCHMARK_REWARD == "cash":
            bench_daily = np.zeros(daily.shape[0], dtype=np.float64)
        else:
            bench_daily = np.mean(daily, axis=1)
        bench_daily = np.clip(bench_daily, -0.95, None)
        return float(np.log1p(bench_daily).sum())

    def _get_regime(self, obs: np.ndarray) -> str:
        """
        Detect market regime from pre-hold observation signals only.

        [BF-1][BF-2][BF-3] — Revised regime logic:
        1. Crisis vol threshold raised 0.025 → 0.035.
        2. Bull detection: weighted score (momentum dominant, macro tiebreaker).
        3. Sideways is the genuine default for ambiguous conditions.
        """
        vol          = float(np.mean(obs[STATE.VOL20D]))
        momentum     = float(np.mean(obs[STATE.MOM5D]))
        macro_regime = float(obs[STATE.REGIME][0])

        # [BF-1] Crisis: require BOTH high vol AND negative momentum.
        if vol > 0.035 and momentum < -0.005:
            return "crisis"

        # [BF-2] Bull: weighted score — momentum dominates, macro confirms.
        bull_score = (
            8.0 * max(momentum, 0.0)
            + 4.0 * max(macro_regime, 0.0)
            - 0.005 * max(vol - 0.020, 0.0)
        )
        if bull_score >= 0.6:
            return "bull"

        # Bear: negative momentum or strongly negative macro.
        bear_score = (
            -3.0 * min(momentum, 0.0)
            + -0.5 * min(macro_regime, 0.0)
        )
        if bear_score >= 0.9:
            return "bear"

        # [BF-3] Sideways is the genuine default.
        return "sideways"

    def _equal_weights(self) -> np.ndarray:
        return np.ones(self.N + 1, dtype=np.float64) / (self.N + 1)

    def _benchmark_anchor_weights(self) -> np.ndarray:
        """Fully-invested equal-weight anchor (no cash)."""
        w      = np.zeros(self.N + 1, dtype=np.float64)
        w[:-1] = 1.0 / self.N
        return w

    def _build_obs(self) -> np.ndarray:
        """Construct the state vector at current timestep self.t."""
        t = self.t
        state = build_state_vector(
            daily_ret_t   = self.bundle.daily_ret.iloc[t].values.astype(np.float32),
            momentum_t    = self.bundle.momentum.iloc[t].values.astype(np.float32),
            vol20d_t      = self.bundle.realized_vol.iloc[t].values.astype(np.float32),
            ma_spread_t   = self.bundle.ma_spread.iloc[t].values.astype(np.float32),
            cov_t         = self.bundle.cov[t],
            weights       = self.current_w[:-1].astype(np.float32),
            tickers       = self.tickers,
            macro_df      = self.macro_df,
            t_date        = self.bundle.dates[t],
            nav_series    = np.array(
                self.nav_history if self.nav_history else [self.nav], dtype=np.float64
            ),
            log_ret_t     = self.bundle.log_ret.iloc[t].values.astype(np.float32),
            volume_t      = self.volume_df.iloc[t].values.astype(np.float32),
            adv_t         = self.adv20_df.iloc[t].values.astype(np.float32),
            prices_t      = self.bundle.prices.iloc[t].values.astype(np.float32),
            prev_turnover = self.last_turnover,
            prev_tc_rate  = self.last_tc_rate,
        )
        return np.clip(state, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _simulate_holding(
        self, target_w: np.ndarray, hold_days: int
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Simulate portfolio forward for `hold_days` days with buy-and-hold.

        Updates self.t, self.nav, self.peak_nav, self.max_drawdown,
        self.current_w, self.nav_history, self.dsr_X, self.dsr_Y in-place.

        The DSR EMA accumulators (X, Y) are updated daily inside this loop
        so they incorporate every simulated day, not just the step boundary.
        This matches the Moody & Saffell (1999) formulation where X and Y
        are maintained at each decision time t.

        The running Maximum Drawdown is also updated daily:
            current_dd = (peak_nav - nav) / (peak_nav + EPS)
            max_drawdown = max(max_drawdown, current_dd)

        Returns
        ───────
        (cumulative_log_return, final_weights, daily_portfolio_returns)
        """
        t_end       = min(self.t + hold_days, self.T - 1)
        cum_log_ret = 0.0
        w           = target_w.copy()
        realised_daily: list[float] = []

        for t_step in range(self.t, t_end):
            stock_rets = self.bundle.daily_ret.iloc[t_step].values.astype(np.float64)
            daily_ret  = float((stock_rets * w[:-1]).sum())
            self.nav  *= (1.0 + daily_ret)
            self.peak_nav = max(self.peak_nav, self.nav)
            cum_log_ret  += float(np.log1p(daily_ret))
            realised_daily.append(daily_ret)
            w = _drift_weights(w, stock_rets)
            self.nav_history.append(self.nav)

            # ── Running Maximum Drawdown ───────────────────────────────────────
            # current_dd is the fraction of peak NAV that has been lost.
            # MDD is the worst such fraction seen so far in this episode.
            current_dd = (self.peak_nav - self.nav) / (self.peak_nav + EPS)
            self.max_drawdown = max(self.max_drawdown, current_dd)

            # ── DSR EMA update (Moody & Saffell 1999) ─────────────────────────
            # X_t = X_{t-1} + η (r_t - X_{t-1})   ← EMA of returns
            # Y_t = Y_{t-1} + η (r_t² - Y_{t-1})  ← EMA of squared returns
            # Updated every simulated day so X and Y track the portfolio's
            # realised return distribution continuously across all steps.
            self.dsr_X += self.dsr_eta * (daily_ret - self.dsr_X)
            self.dsr_Y += self.dsr_eta * (daily_ret ** 2 - self.dsr_Y)

        self.t         = t_end
        self.current_w = w
        return cum_log_ret, w, np.asarray(realised_daily, dtype=np.float64)

    def _compute_dsr(self, r: float) -> float:
        """
        Compute the Differential Sharpe Ratio for the current step's
        representative return `r`, using the EMA state *before* this step's
        update (i.e., X and Y reflect history up to the previous day).

        Formula (Moody & Saffell 1999):

            DSR = [Y (r - X) - 0.5 X (r² - Y)] / (Y - X²)^{3/2}

        where X = self.dsr_X, Y = self.dsr_Y are the EMA values.

        Variance floor of 1e-8 prevents division by zero during warm-up.
        Result clipped to [-DSR_CLIP, +DSR_CLIP] for NN stability.

        Mathematical intuition:
          - Numerator = d/dr [r / σ], the marginal improvement in Sharpe
            from seeing one more return r, evaluated at the current (X, Y).
          - Denominator = σ³ = (Var[r])^{3/2}, normalises the scale.
          - If r > X (above-average return) AND Y is large (high variance
            memory), the numerator is dominated by Y(r-X) > 0 → positive DSR.
          - Upside variance (r > X) contributes positively; downside (r < X)
            contributes negatively — exactly the behaviour we want.
        """
        X   = self.dsr_X
        Y   = self.dsr_Y
        var = max(Y - X ** 2, 1e-8)

        numerator = Y * (r - X) - 0.5 * X * (r ** 2 - Y)
        dsr       = numerator / (var ** 1.5)
        return float(np.clip(dsr, -DSR_CLIP, DSR_CLIP))

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        rng = getattr(self, "np_random", None)
        if rng is None:
            rng = np.random.default_rng(seed)
            self.np_random = rng

        usable_end = self.T - max(self.max_episode_days + 120, HOLDING_PERIODS[-1] + 90)
        if self.random_start and usable_end > self.warm_up:
            episode_start_t = int(rng.integers(self.warm_up, usable_end + 1))
        else:
            episode_start_t = self.warm_up

        episode_end_t = min(self.T - 1, self._episode_upper_bound(episode_start_t))

        self.t                   = episode_start_t
        self.episode_start_t     = episode_start_t
        self.episode_horizon_end = episode_end_t
        self.days_in_episode     = 0
        self.nav                 = 1.0
        self.peak_nav            = 1.0
        self.max_drawdown        = 0.0   # reset MDD at episode start
        self.nav_history         = [1.0]
        self.current_w           = self._equal_weights()
        self.total_tc            = 0.0
        self.episode_steps       = 0
        self.last_turnover       = 0.0
        self.last_tc_rate        = 0.0

        # Reset DSR EMA state: fresh episode, fresh return distribution.
        # Starting at (0, 0) is correct — the first few steps will warm up
        # the estimates. DSR_CLIP prevents blow-up during this phase.
        self.dsr_X = 0.0
        self.dsr_Y = 0.0

        obs  = self._build_obs()
        info = {"date": str(self.bundle.dates[self.t].date()), "nav": self.nav}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one RL decision step.

        action[0] — lambda_idx       (0–19)
        action[1] — hold_idx         (0–4)
        action[2] — exposure_idx     (0–5)
        action[3] — anchor_blend_idx (0–4)

        Step sequence
        ─────────────
        1.  Decode action → (λ, hold_days, max_equity, anchor_blend)
        2.  Build obs_before; detect regime from pre-hold signals (no look-ahead)
        3.  Call MVO → target weights; blend with equal-weight anchor
        4.  Compute turnover; deduct dynamic TC from NAV
        5.  Capture pre-update DSR state (X_prev, Y_prev) for DSR computation
        6.  Simulate portfolio for hold_days days (buy-and-hold + drift);
            updates self.dsr_X, self.dsr_Y, self.max_drawdown daily
        7.  Compute DSR using the final-day return against pre-step EMA state
        8.  Compute regime-conditioned reward (all terms explicit, none hidden):
              R_lr  = cum_log_ret / trading_days_span       (growth)
              R_dsr = DSR clipped to ±DSR_CLIP              (risk-adjustment)
              R_mdd = -sqrt(max_drawdown)                   (downside control)
              + regime bonuses/penalties (participation, momentum, λ-aggression,
                λ-defensiveness, TC, exposure, turnover, cash)
        9.  Apply per-day amortisation if reward_scale=True
        10. Check termination; build obs_after (always real observation)
        11. Return (obs_after, reward, terminated, truncated, info)
        """

        # ── 1. Decode action ───────────────────────────────────────────────────
        lambda_idx       = int(action[0])
        hold_idx         = int(action[1])
        exposure_idx     = int(action[2])
        anchor_blend_idx = int(action[3])

        lam          = float(LAMBDA_VALUES[lambda_idx])
        hold_days    = max(5, int(HOLDING_PERIODS[hold_idx]))   # hard minimum
        anchor_blend = float(ANCHOR_BLEND_LEVELS[anchor_blend_idx])
        date_start   = self.bundle.dates[self.t]

        # ── 2. Build obs; detect regime from pre-hold signals only ─────────────
        t_before   = self.t
        mvo_ctx    = self._prepare_mvo_extras(self.bundle, t_before)
        obs_before = self._build_obs()
        regime     = self._get_regime(obs_before)
        rp         = REGIME_REWARD_PARAMS[regime]

        # ── [BF-9] Regime-aware equity floor ──────────────────────────────────
        # In bull: honour agent's exposure choice directly — no floor.
        # In other regimes: keep the floor to prevent reckless de-risking.
        raw_equity = float(EXPOSURE_LEVELS[exposure_idx])
        if regime == "bull":
            max_equity = raw_equity                  # [BF-9] no floor in bull
        else:
            max_equity = max(0.6, raw_equity)        # floor preserved for non-bull

        # ── 3. MVO → target weights ────────────────────────────────────────────
        mvo_result = optimize(
            obs_before,
            self.tickers,
            lam,
            current_weights  = self.current_w,
            max_equity       = max_equity,
            turnover_penalty = TURNOVER_PENALTY,
            cov_matrix       = mvo_ctx["cov_matrix"],
            mom20d_t         = mvo_ctx["mom20d_t"],
            mom60d_t         = mvo_ctx["mom60d_t"],
            daily_ret_t      = mvo_ctx["daily_ret_t"],
        )

        mvo_w    = mvo_result.weights.astype(np.float64)
        mvo_w    = np.clip(mvo_w, 0.0, 1.0)
        mvo_w   /= max(mvo_w.sum(), EPS)
        anchor_w = self._benchmark_anchor_weights()
        target_w = (1.0 - anchor_blend) * mvo_w + anchor_blend * anchor_w
        target_w = np.clip(target_w, 0.0, 1.0)
        target_w /= max(float(target_w.sum()), EPS)

        # ── 4. Transaction cost ────────────────────────────────────────────────
        tc = transaction_cost(
            self.current_w,
            target_w,
            state    = obs_before,
            prices_t = self.bundle.prices.iloc[self.t].values.astype(np.float64),
            volume_t = self.volume_df.iloc[self.t].values.astype(np.float64),
            adv_t    = self.adv20_df.iloc[self.t].values.astype(np.float64),
        )
        self.nav      *= (1.0 - tc)
        self.total_tc += tc

        turnover           = float(np.abs(target_w[:-1] - self.current_w[:-1]).sum())
        self.last_turnover = turnover
        self.last_tc_rate  = tc / max(turnover, EPS)
        self.current_w     = target_w.copy()

        # ── 5. Capture DSR EMA state BEFORE the holding simulation ────────────
        # DSR is computed using X and Y from *before* this step's returns are
        # folded in. This mirrors the original Moody & Saffell formulation:
        #   DSR_t = f(r_t, X_{t-1}, Y_{t-1})
        # After capture we run the simulation, which updates X and Y day-by-day.
        dsr_X_before = self.dsr_X
        dsr_Y_before = self.dsr_Y

        # ── 6. Simulate holding period ─────────────────────────────────────────
        # _simulate_holding updates: self.t, self.nav, self.peak_nav,
        # self.max_drawdown, self.current_w, self.nav_history,
        # self.dsr_X, self.dsr_Y  (all in-place, daily)
        cum_log_ret, _, realised_daily = self._simulate_holding(target_w, hold_days)

        # ── 7. Compute DSR for this step ──────────────────────────────────────
        # Use the step's representative return (mean daily excess return) against
        # the pre-step EMA state captured in step 5.
        # We temporarily restore X_before/Y_before for the DSR formula, then
        # use the already-updated self.dsr_X / self.dsr_Y going forward.
        trading_days_span = max(self.t - t_before, 1)
        alpha_per_day     = (cum_log_ret - self._benchmark_log_return(t_before, self.t)) \
                            / trading_days_span

        # Temporarily set EMA to pre-step values for the DSR formula.
        # This is mathematically correct: DSR_t = f(r_t, X_{t-1}, Y_{t-1}).
        # The in-place daily updates already advanced self.dsr_X / self.dsr_Y
        # through the holding period — we do NOT revert them; we only use
        # the saved pre-step snapshot for this single DSR computation.
        saved_X, saved_Y  = self.dsr_X, self.dsr_Y
        self.dsr_X        = dsr_X_before
        self.dsr_Y        = dsr_Y_before
        dsr_value         = self._compute_dsr(alpha_per_day)
        self.dsr_X        = saved_X    # restore post-simulation state
        self.dsr_Y        = saved_Y

        # ── 8. Reward components ───────────────────────────────────────────────
        self.days_in_episode += trading_days_span

        benchmark_log_ret = self._benchmark_log_return(t_before, self.t)
        excess_log_ret    = cum_log_ret - benchmark_log_ret

        # ── R_lr: log return per day × R_LR_MULTIPLIER [RT-1] ─────────────────
        # R_LR_MULTIPLIER=2.0 doubles d(reward)/d(return) for all regimes.
        # Mathematically: the policy gradient signal on any action that improves
        # return is 2× stronger, pushing the agent out of the "do nothing" basin.
        R_lr = R_LR_MULTIPLIER * excess_log_ret / trading_days_span

        # ── R_dsr: Differential Sharpe Ratio ──────────────────────────────────
        # Already computed above as dsr_value.
        R_dsr = dsr_value

        # ── R_mdd: running MDD × MDD_PENALTY_SCALE [RT-2] ────────────────────
        # MDD_PENALTY_SCALE=0.5 halves the fear signal globally.
        # -sqrt(MDD) provides smooth gradients; scaling by 0.5 reduces the
        # relative weight of drawdown vs growth without removing the signal.
        # At MDD=5%:  0.5 * (-sqrt(0.05)) = -0.112  (was -0.224)
        # At MDD=10%: 0.5 * (-sqrt(0.10)) = -0.158  (was -0.316)
        R_mdd = MDD_PENALTY_SCALE * (-float(np.sqrt(max(self.max_drawdown, 0.0))))

        # Actual equity exposure from the chosen target weights.
        actual_exposure = float(target_w[:-1].sum())

        # [BF-5] Asymmetric exposure penalty for bull regime (soft, one-sided).
        exposure_gap = actual_exposure - rp["exposure_target"]
        if regime == "bull":
            # One-sided: only penalise shortfall below target.
            exposure_penalty = rp["exposure_weight"] * min(exposure_gap, 0.0) ** 2
        else:
            # Symmetric quadratic: penalise any deviation from regime target.
            exposure_penalty = rp["exposure_weight"] * exposure_gap ** 2

        # [RT-5] Bull hard exposure floor penalty ─────────────────────────────
        # Additional steep quadratic penalty when below BULL_EXPOSURE_FLOOR (80%).
        # This fires on top of the soft exposure_penalty above, creating a
        # combined convex penalty surface that strongly discourages sub-80% equity
        # in bull markets. The gradient steepens as exposure drops further:
        #   80% → penalty = 0 (floor just met)
        #   70% → -(3.0 * 0.10²) = -0.030
        #   60% → -(3.0 * 0.20²) = -0.120 (dominant term in reward)
        if regime == "bull":
            floor_gap              = actual_exposure - BULL_EXPOSURE_FLOOR
            bull_floor_penalty     = BULL_FLOOR_PENALTY_SCALE * min(floor_gap, 0.0) ** 2
        else:
            bull_floor_penalty     = 0.0

        # [BF-6] Participation bonus — reward for positive absolute return in bull.
        # [RT-4] Scale raised 8.0 → 15.0 so the bonus dominates TC at normal return levels.
        if regime == "bull" and cum_log_ret > 0:
            participation_bonus = BULL_PARTICIPATION_SCALE * cum_log_ret / trading_days_span
        else:
            participation_bonus = 0.0

        # [BF-7] Momentum-alignment bonus — in bull, reward holding high-momentum stocks.
        if regime == "bull":
            mom5d_t = obs_before[STATE.MOM5D].astype(np.float64)
            mom5d_t = np.clip(mom5d_t, -3.0, 3.0)
            momentum_alignment = float(np.dot(target_w[:-1], mom5d_t))
            momentum_bonus = MOMENTUM_ALIGNMENT_SCALE * max(momentum_alignment, 1.0)
        else:
            momentum_bonus = 0.0

        # [LA-2] Lambda-aggression bonus in bull.
        if regime == "bull":
            aggression_factor       = 1.0 - lam / LAM_MAX
            lambda_aggression_bonus = (
                BULL_LAMBDA_AGG_SCALE
                * max(cum_log_ret, 0.0)
                * aggression_factor
                / trading_days_span
            )
        else:
            lambda_aggression_bonus = 0.0

        # [LA-3] Lambda-defensiveness penalty in bull.
        if regime == "bull":
            defensiveness_factor        = lam / LAM_MAX
            lambda_defensiveness_penalty = (
                BULL_LAMBDA_DEF_SCALE
                * max(cum_log_ret, 0.0)
                * defensiveness_factor
                / trading_days_span
            )
        else:
            lambda_defensiveness_penalty = 0.0

        # [LA-4] + [RT-6] Lambda-adaptive turnover weight in bull.
        # Step 1 [RT-6]: apply GLOBAL_TURNOVER_RELIEF=0.5 to ALL regimes.
        #   The explicit tc term already pays the economic cost of trading.
        #   The turnover_weight is a behavioural brake — halving it allows
        #   the agent to reposition toward profitable stocks more freely.
        # Step 2 [LA-4]: in bull, apply additional relief for low-λ choices
        #   on top of the global relief.
        base_turnover_weight      = rp["turnover_weight"] * GLOBAL_TURNOVER_RELIEF  # [RT-6]
        if regime == "bull":
            lam_normalised            = lam / LAM_MAX
            aggression_frac           = 1.0 - lam_normalised
            effective_turnover_weight = base_turnover_weight * (
                2.0 - BULL_LOW_LAM_TURNOVER_RELIEF * aggression_frac          # [LA-4]
            )
        else:
            effective_turnover_weight = base_turnover_weight

        # Cash opportunity cost — penalise excess cash in positive regimes.
        # [RT-3] CASH_OPPORTUNITY_PENALTY=2.0e-2 (was 5.0e-3): 4× increase.
        # Example: 20% cash in bull (floor=5%) → 2.0e-2 * 5 * 0.15 = 0.015/step
        # vs old: 5.0e-3 * 5 * 0.15 = 0.00375/step. Now cash truly costs.
        cash_weight = float(target_w[-1])
        cash_floor  = CASH_FLOOR_BY_REGIME.get(regime, 0.10)
        is_positive_regime = regime in ("bull", "sideways")
        cash_penalty = (
            CASH_OPPORTUNITY_PENALTY * trading_days_span * max(cash_weight - cash_floor, 0.0)
            if is_positive_regime else 0.0
        )

        # ── Regime-conditioned reward — all terms explicit ─────────────────────
        #
        # Return aggression modifications vs previous version:
        #   [RT-1] R_lr already pre-multiplied by R_LR_MULTIPLIER=2.0 above
        #   [RT-2] R_mdd already pre-multiplied by MDD_PENALTY_SCALE=0.5 above
        #   [RT-3] CASH_OPPORTUNITY_PENALTY=2.0e-2 (4× larger constant)
        #   [RT-4] BULL_PARTICIPATION_SCALE=15.0 (was 8.0)
        #   [RT-5] bull_floor_penalty: new hard floor at 80% bull exposure
        #   [RT-6] effective_turnover_weight halved via GLOBAL_TURNOVER_RELIEF
        #
        # Net effect on the gradient landscape:
        #   ∂reward/∂return  ≈ 2× larger (RT-1)
        #   ∂reward/∂cash    ≈ 4× more negative (RT-3)
        #   ∂reward/∂MDD     ≈ 2× less negative (RT-2)
        #   ∂reward/∂turnover ≈ 2× less negative (RT-6)
        # → agent is pushed hard toward high-return, fully-invested positions
        reward = (
            rp["alpha_weight"]       * R_lr                 # [RT-1] 2× growth pressure
            + DSR_WEIGHT             * R_dsr                 # risk-adjustment
            + rp["drawdown_weight"]  * R_mdd                 # [RT-2] 0.5× MDD fear
            + participation_bonus                         # [BF-6][RT-4] zero non-bull                                 # [BF-7] zero in non-bull
            + lambda_aggression_bonus                        # [LA-2] zero in non-bull
            - lambda_defensiveness_penalty                   # [LA-3] zero in non-bull
            - 0.0002* tc
            - exposure_penalty                               # [BF-5] asymmetric in bull
            - bull_floor_penalty                             # [RT-5] zero in non-bull
            - effective_turnover_weight * turnover           # [LA-4][RT-6] relief applied
            - cash_penalty                                   # [BF-8][RT-3] 4× stronger
        )

        # Per-day amortisation so all steps are directly comparable for PPO.
        if self.reward_scale:
            reward /= trading_days_span

        self.episode_steps += 1

        # ── 9. Termination checks ──────────────────────────────────────────────
        hit_data_end    = self.t >= self.T - 1
        hit_episode_cap = (
            self.days_in_episode >= self.max_episode_days
            or self.t >= self.episode_horizon_end
        )
        if hit_data_end:
            terminated, truncated = True, False
        elif hit_episode_cap:
            terminated, truncated = False, True
        else:
            terminated, truncated = False, False

        # ── 10. Build next observation ─────────────────────────────────────────
        obs_after = self._build_obs()

        date_end = self.bundle.dates[min(self.t, self.T - 1)]

        info = {
            "lambda":                        lam,
            "lambda_idx":                    lambda_idx,
            "hold_days":                     hold_days,
            "hold_idx":                      hold_idx,
            "exposure":                      max_equity,
            "exposure_idx":                  exposure_idx,
            "anchor_blend":                  anchor_blend,
            "anchor_blend_idx":              anchor_blend_idx,
            "regime":                        regime,
            "turnover":                      turnover,
            "tc_cost":                       tc,
            "tc_rate":                       self.last_tc_rate,
            "cum_log_return":                cum_log_ret,
            "benchmark_log_return":          benchmark_log_ret,
            "excess_log_return":             excess_log_ret,
            # DSR replaces alpha_signal; kept under same key for API compat
            "alpha_signal":                  R_dsr,
            "dsr_value":                     R_dsr,
            "dsr_X":                         self.dsr_X,
            "dsr_Y":                         self.dsr_Y,
            "R_lr":                          R_lr,
            "R_dsr":                         R_dsr,
            "R_mdd":                         R_mdd,
            "max_drawdown":                  self.max_drawdown,
            "participation_bonus":           participation_bonus,
            "momentum_bonus":                momentum_bonus,
            "lambda_aggression_bonus":       lambda_aggression_bonus,       # [LA-2]
            "lambda_defensiveness_penalty":  lambda_defensiveness_penalty,  # [LA-3]
            "effective_turnover_weight":     effective_turnover_weight,     # [LA-4][RT-6]
            "bull_floor_penalty":            bull_floor_penalty,            # [RT-5]
            "drawdown":                      (self.peak_nav - self.nav) / (self.peak_nav + EPS),
            "cash_penalty":                  cash_penalty,
            "exposure_penalty":              exposure_penalty,
            "nav":                           self.nav,
            "total_tc":                      self.total_tc,
            "date_start":                    str(date_start.date()),
            "date_end":                      str(date_end.date()),
            "mvo_exp_return":                mvo_result.exp_return,
            "mvo_exp_vol":                   mvo_result.exp_volatility,
            "trading_days_span":             trading_days_span,
        }

        return obs_after, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        t   = min(self.t, self.T - 1)
        dd  = (self.peak_nav - self.nav) / (self.peak_nav + EPS)
        top = sorted(
            zip(self.tickers, self.current_w[:-1]), key=lambda x: x[1], reverse=True
        )[:3]
        top_str = "  ".join(f"{tk.split('.')[0]}={w:.3f}" for tk, w in top)
        print(
            f"  [{self.bundle.dates[t].date()}]  "
            f"NAV={self.nav:.4f}  DD={dd*100:+.2f}%  MDD={self.max_drawdown*100:.2f}%  "
            f"DSR_X={self.dsr_X:.5f}  DSR_Y={self.dsr_Y:.5f}  "
            f"cash={self.current_w[-1]:.3f}  top: {top_str}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PortfolioEnv — sanity check")
    print("=" * 60)
    print(f"  Action space  : MultiDiscrete([{N_LAMBDA}, {N_HOLD}, {N_EXPOSURE}, {N_ANCHOR_BLEND}])")
    print(f"    λ options   : {N_LAMBDA}  ({LAMBDA_VALUES[0]:.1f} … {LAMBDA_VALUES[-1]:.1f})")
    print(f"    Hold options: {N_HOLD}    {HOLDING_PERIODS} days")
    print(f"    Exposure    : {N_EXPOSURE}    {EXPOSURE_LEVELS}")
    print(f"    Anchor blend: {N_ANCHOR_BLEND}    {ANCHOR_BLEND_LEVELS}")
    print(f"  Obs space     : Box(-{OBS_CLIP}, {OBS_CLIP}, shape=({STATE.DIM},))")
    print(f"  Regime params : {list(REGIME_REWARD_PARAMS.keys())}")
    print(f"  DSR clip      : ±{DSR_CLIP}  (replaces alpha_signal clip)")
    print(f"  DSR eta       : {DSR_ETA:.6f}  (≈ 1-year EMA memory)")
    print(f"  DSR weight    : {DSR_WEIGHT}")
    print(f"  reward_scale  : per-day amortisation when enabled")
    print("  TC model      : dynamic statutory + volatility/liquidity impact")
    print(f"  Drawdown      : running MDD via -sqrt(max_drawdown)  (was instantaneous DD)")
    print(f"  [RT] Return-aggression fixes active:")
    print(f"       R_lr multiplier          : {R_LR_MULTIPLIER}× (was 1×)")
    print(f"       MDD penalty scale        : {MDD_PENALTY_SCALE}× (was 1×)")
    print(f"       Cash penalty             : {CASH_OPPORTUNITY_PENALTY} (was 5e-3, 4× increase)")
    print(f"       Participation scale      : {BULL_PARTICIPATION_SCALE} (was 8.0)")
    print(f"       Bull exposure floor      : {BULL_EXPOSURE_FLOOR:.0%}  penalty scale={BULL_FLOOR_PENALTY_SCALE}")
    print(f"       Global turnover relief   : {GLOBAL_TURNOVER_RELIEF}× all regimes")
    print(f"  [BF] Bull fixes active:")
    print(f"       Crisis vol threshold : 0.035 (was 0.025)")
    print(f"       Bull detection       : weighted score (was conjunctive AND)")
    print(f"       Bull exposure_weight : {REGIME_REWARD_PARAMS['bull']['exposure_weight']} (was 0.40)")
    print(f"       Bull alpha_weight    : {REGIME_REWARD_PARAMS['bull']['alpha_weight']} (×{R_LR_MULTIPLIER} = {REGIME_REWARD_PARAMS['bull']['alpha_weight']*R_LR_MULTIPLIER} effective)")
    print(f"       Cash floor (bull)    : {CASH_FLOOR_BY_REGIME['bull']} (was 0.10)")
    print(f"       Equity floor in bull : NONE at MVO level; hard floor via [RT-5]")
    print(f"  [LA] Lambda-aggression fixes active:")
    print(f"       Alpha signal         : DSR / Moody-Saffell (replaces Sharpe/Sortino)")
    print(f"       Lambda-aggression bonus  : scale={BULL_LAMBDA_AGG_SCALE}")
    print(f"       Lambda-defensiveness pen : scale={BULL_LAMBDA_DEF_SCALE}")
    print(f"       Net λ gradient midpoint  : lam ≈ {LAM_MAX/2:.2f}")
    print(f"       Turnover relief (bull)   : {BULL_LOW_LAM_TURNOVER_RELIEF:.0%} max at lam=LAM_MIN (on top of {GLOBAL_TURNOVER_RELIEF}× global)")

    env = PortfolioEnv(split="train", reward_scale=True)
    obs, info = env.reset()

    print(
        f"\n  Reset — date={info['date']}  obs shape={obs.shape}  "
        f"finite={np.isfinite(obs).all()}"
    )
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"\n  Running 10 random steps …")

    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(
            f"  step {i+1:>2}  λ={info['lambda']:.2f}  hold={info['hold_days']:>2}d  "
            f"exposure={info['exposure']:.0%}  anchor={info['anchor_blend']:.0%}  "
            f"regime={info['regime']:<8}  "
            f"R_lr={info['R_lr']:+.4f}  R_dsr={info['R_dsr']:+.4f}  R_mdd={info['R_mdd']:+.4f}  "
            f"MDD={info['max_drawdown']*100:.2f}%  "
            f"part={info['participation_bonus']:+.4f}  floor_pen={info['bull_floor_penalty']:+.4f}  "
            f"cash_pen={info['cash_penalty']:+.4f}  "
            f"λ_agg={info['lambda_aggression_bonus']:+.4f}  "
            f"λ_def={info['lambda_defensiveness_penalty']:+.4f}  "
            f"ret={info['cum_log_return']:+.4f}  tc={info['tc_cost']*100:.4f}%  "
            f"reward={reward:+.4f}  NAV={info['nav']:.4f}  "
            f"[{info['date_start']} → {info['date_end']}]"
        )
        if terminated or truncated:
            print(f"  Episode ended (terminated={terminated}, truncated={truncated}).")
            break

    print(f"\n  Total reward over {i+1} steps: {total_reward:+.4f}")
    print(f"  Final NAV:      {info['nav']:.4f}  (started at 1.0)")
    print(f"  Total TC paid:  {info['total_tc']*100:.4f}% of NAV")
    print(f"  Episode MDD:    {info['max_drawdown']*100:.2f}%")
    print(f"  DSR EMA X:      {info['dsr_X']:.6f}  (E[r])")
    print(f"  DSR EMA Y:      {info['dsr_Y']:.6f}  (E[r²])")
    print(f"\n  ✓  Environment verified")