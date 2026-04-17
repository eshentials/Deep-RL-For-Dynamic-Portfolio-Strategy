"""
env.py — PortfolioEnv: Gymnasium environment for Deep-RL portfolio management.

The RL agent jointly decides two things at every step:
  1. Risk value   — which λ to pass to the Mean-Variance Optimizer (MVO).
                    λ ∈ LAMBDA_VALUES = linspace(0.1, 2.0, 20).
                    Low λ → return-seeking; high λ → minimum-variance.
  2. Holding period — how many trading days to hold the MVO portfolio before
                      the next decision.
                      HOLDING_PERIODS = [1, 5, 10, 21, 63] days.

At each step the environment:
  (a) Passes (state, λ) to efficient_frontier.optimize() → target weights.
  (b) Deducts a STATIC transaction cost of 0.1% × turnover from NAV.
      TRANSACTION_COST_RATE is a hyperparameter fixed in state_spec.py;
      the environment never modifies it.
  (c) Simulates the portfolio forward for `hold_days` trading days,
      updating weights by daily price drift (buy-and-hold within period).
  (d) Returns reward = cumulative log return − TC.

Action space:  MultiDiscrete([20, 5])
  action[0] ∈ {0…19}  → lambda_idx → LAMBDA_VALUES[lambda_idx]
  action[1] ∈ {0…4}   → hold_idx   → HOLDING_PERIODS[hold_idx]

Observation space: Box(-5, 5, shape=(49,), dtype=float32)
  The 49-dim state vector defined in state_spec.py (clipped for NN stability).

Transaction cost (STATIC):
  cost = TRANSACTION_COST_RATE × Σ|Δwᵢ|   (stock positions only)
  TRANSACTION_COST_RATE = 0.001  (0.1%) — defined once in state_spec.py.
  This is a fixed hyperparameter and is NOT learnable or configurable
  from inside the environment.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from features import load_bundle, FeatureBundle
from state_spec import (
    STATE,
    build_state_vector,
    transaction_cost,
    TRANSACTION_COST_RATE,
    MACRO_PATH,
    EPS,
)
from efficient_frontier import optimize, LAMBDA_VALUES

# ── Action-space constants ─────────────────────────────────────────────────────
HOLDING_PERIODS: list[int] = [1, 5, 10, 21, 63]   # trading days
N_LAMBDA: int = len(LAMBDA_VALUES)                 # 20
N_HOLD:   int = len(HOLDING_PERIODS)               # 5

# ── Observation clipping (prevents extreme values blowing up the policy net) ──
OBS_CLIP: float = 5.0

# ── Reward shaping ────────────────────────────────────────────────────────────
DRAWDOWN_PENALTY_SCALE: float = 0.5   # multiplier on current drawdown in reward


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

    The MVO (efficient_frontier.optimize) computes the portfolio weights;
    the RL agent only selects the risk level and time horizon.

    Parameters
    ──────────
    split        : data split to use  ("train" | "test" | "val")
    reward_scale : divide reward by holding_days to get per-day equivalent
                   (True = comparable across holding lengths; False = raw return)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, split: str = "train", reward_scale: bool = False):
        super().__init__()
        self.split        = split
        self.reward_scale = reward_scale

        # ── Load data ──────────────────────────────────────────────────────────
        self.bundle: FeatureBundle = load_bundle(split)
        self.macro_df: pd.DataFrame = pd.read_csv(
            MACRO_PATH, index_col=0, parse_dates=True
        )
        self.tickers: list[str] = self.bundle.tickers
        self.N: int             = self.bundle.n_assets
        self.T: int             = len(self.bundle.dates)
        self.warm_up: int       = self.bundle.warm_up_rows()   # 90

        # ── Spaces ─────────────────────────────────────────────────────────────
        self.action_space = spaces.MultiDiscrete([N_LAMBDA, N_HOLD])
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP,
            shape=(STATE.DIM,), dtype=np.float32,
        )

        # ── Episode state (initialised in reset) ──────────────────────────────
        self.t:              int          = self.warm_up
        self.nav:            float        = 1.0
        self.nav_history:    list[float]  = []
        self.current_w:      np.ndarray   = np.ones(self.N + 1) / (self.N + 1)
        self.peak_nav:       float        = 1.0
        self.total_tc:       float        = 0.0
        self.episode_steps:  int          = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _equal_weights(self) -> np.ndarray:
        return np.ones(self.N + 1, dtype=np.float64) / (self.N + 1)

    def _build_obs(self) -> np.ndarray:
        """Construct the 49-dim state vector at current timestep t."""
        t = self.t
        state = build_state_vector(
            daily_ret_t = self.bundle.daily_ret.iloc[t].values.astype(np.float32),
            momentum_t  = self.bundle.momentum.iloc[t].values.astype(np.float32),
            vol20d_t    = self.bundle.realized_vol.iloc[t].values.astype(np.float32),
            ma_spread_t = self.bundle.ma_spread.iloc[t].values.astype(np.float32),
            cov_t       = self.bundle.cov[t],
            weights     = self.current_w[:-1].astype(np.float32),
            tickers     = self.tickers,
            macro_df    = self.macro_df,
            t_date      = self.bundle.dates[t],
            nav_series  = np.array(self.nav_history if self.nav_history else [self.nav],
                                   dtype=np.float64),
            log_ret_t   = self.bundle.log_ret.iloc[t].values.astype(np.float32),
        )
        return np.clip(state, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _simulate_holding(self, target_w: np.ndarray,
                          hold_days: int) -> tuple[float, np.ndarray]:
        """
        Simulate portfolio forward for `hold_days` days with buy-and-hold.

        Starting from self.t, holds `target_w` and drifts with daily prices.
        Updates self.t, self.nav, self.current_w, self.nav_history in-place.

        Returns
        ───────
        (cumulative_log_return, final_weights)
        """
        t_end      = min(self.t + hold_days, self.T - 1)
        cum_log_ret = 0.0
        w          = target_w.copy()

        for t_step in range(self.t, t_end):
            stock_rets  = self.bundle.daily_ret.iloc[t_step].values.astype(np.float64)
            daily_ret   = float((stock_rets * w[:-1]).sum())
            self.nav   *= (1.0 + daily_ret)
            self.peak_nav = max(self.peak_nav, self.nav)
            cum_log_ret += float(np.log1p(daily_ret))
            w           = _drift_weights(w, stock_rets)
            self.nav_history.append(self.nav)

        self.t         = t_end
        self.current_w = w
        return cum_log_ret, w

    # ── Gymnasium API ──────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.t             = self.warm_up
        self.nav           = 1.0
        self.peak_nav      = 1.0
        self.nav_history   = [1.0]
        self.current_w     = self._equal_weights()
        self.total_tc      = 0.0
        self.episode_steps = 0

        obs  = self._build_obs()
        info = {"date": str(self.bundle.dates[self.t].date()), "nav": self.nav}
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one RL decision step.

        action[0] — lambda_idx  (0–19)
        action[1] — hold_idx    (0–4)

        Step sequence
        ─────────────
        1. Decode action → (λ, hold_days)
        2. Build state at current t (already available from _build_obs)
        3. Call MVO: optimize(state, tickers, λ) → target_weights
        4. Compute turnover, deduct static 0.1% TC from NAV
        5. Simulate portfolio for hold_days days (buy-and-hold + drift)
        6. Compute reward = cumulative_log_return − TC
                           + drawdown_shaping
        7. Advance t, check termination
        8. Return (new_obs, reward, terminated, truncated, info)
        """
        lambda_idx = int(action[0])
        hold_idx   = int(action[1])
        lam        = float(LAMBDA_VALUES[lambda_idx])
        hold_days  = int(HOLDING_PERIODS[hold_idx])

        date_start = self.bundle.dates[self.t]
        nav_before = self.nav

        # ── 1. Build state and run MVO ─────────────────────────────────────────
        obs_before = self._build_obs()
        mvo_result = optimize(obs_before, self.tickers, lam)
        target_w   = mvo_result.weights.astype(np.float64)

        # ── 2. Transaction cost (STATIC hyperparameter, never modified) ────────
        tc       = transaction_cost(self.current_w, target_w)
        self.nav *= (1.0 - tc)
        self.total_tc += tc

        turnover = float(np.abs(target_w[:-1] - self.current_w[:-1]).sum())

        # Set weights to MVO output BEFORE simulating
        self.current_w = target_w.copy()

        # ── 3. Simulate holding period ─────────────────────────────────────────
        cum_log_ret, _ = self._simulate_holding(target_w, hold_days)

        # ── 4. Reward ──────────────────────────────────────────────────────────
        # Base: cumulative log return over the holding period, minus TC paid
        reward = cum_log_ret - tc

        # Drawdown penalty: discourage large peak-to-trough losses
        drawdown = (self.nav - self.peak_nav) / (self.peak_nav + EPS)
        if drawdown < 0:
            reward += DRAWDOWN_PENALTY_SCALE * drawdown   # adds negative value

        # Optional: normalise to per-day equivalent for cross-period comparability
        actual_hold = max(self.t - self.warm_up - self.episode_steps, 1)
        if self.reward_scale and hold_days > 1:
            reward /= hold_days

        self.episode_steps += 1

        # ── 5. Termination ─────────────────────────────────────────────────────
        terminated = (self.t >= self.T - 1)
        truncated  = False

        # ── 6. New observation ─────────────────────────────────────────────────
        if not terminated:
            obs_after = self._build_obs()
        else:
            obs_after = np.zeros(STATE.DIM, dtype=np.float32)

        date_end = self.bundle.dates[min(self.t, self.T - 1)]

        info = {
            "lambda":          lam,
            "lambda_idx":      lambda_idx,
            "hold_days":       hold_days,
            "hold_idx":        hold_idx,
            "turnover":        turnover,
            "tc_cost":         tc,
            "tc_rate":         TRANSACTION_COST_RATE,   # confirm it's static
            "cum_log_return":  cum_log_ret,
            "drawdown":        drawdown,
            "nav":             self.nav,
            "total_tc":        self.total_tc,
            "date_start":      str(date_start.date()),
            "date_end":        str(date_end.date()),
            "mvo_exp_return":  mvo_result.exp_return,
            "mvo_exp_vol":     mvo_result.exp_volatility,
            "mvo_sharpe":      mvo_result.sharpe,
        }

        return obs_after, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> None:
        t   = min(self.t, self.T - 1)
        dd  = (self.nav - self.peak_nav) / (self.peak_nav + EPS)
        top = sorted(zip(self.tickers, self.current_w[:-1]),
                     key=lambda x: x[1], reverse=True)[:3]
        top_str = "  ".join(f"{tk.split('.')[0]}={w:.3f}" for tk, w in top)
        print(f"  [{self.bundle.dates[t].date()}]  "
              f"NAV={self.nav:.4f}  DD={dd*100:+.2f}%  "
              f"cash={self.current_w[-1]:.3f}  top: {top_str}")


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  PortfolioEnv — sanity check")
    print("=" * 60)
    print(f"  Action space  : MultiDiscrete([{N_LAMBDA}, {N_HOLD}])")
    print(f"    λ options   : {N_LAMBDA}  ({LAMBDA_VALUES[0]:.1f} … {LAMBDA_VALUES[-1]:.1f})")
    print(f"    Hold options: {N_HOLD}    {HOLDING_PERIODS} days")
    print(f"  Obs space     : Box(-{OBS_CLIP}, {OBS_CLIP}, shape=({STATE.DIM},))")
    print(f"  TC rate       : {TRANSACTION_COST_RATE*100:.1f}%  (static hyperparameter)")

    env = PortfolioEnv(split="train")
    obs, info = env.reset()

    print(f"\n  Reset — date={info['date']}  obs shape={obs.shape}  finite={np.isfinite(obs).all()}")
    print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    print(f"\n  Running 10 random steps …")
    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  step {i+1:>2}  λ={info['lambda']:.2f}  hold={info['hold_days']:>2}d  "
              f"ret={info['cum_log_return']:+.4f}  tc={info['tc_cost']*100:.4f}%  "
              f"reward={reward:+.4f}  NAV={info['nav']:.4f}  "
              f"[{info['date_start']} → {info['date_end']}]")
        if terminated:
            print("  Episode terminated.")
            break

    print(f"\n  Total reward over {i+1} steps: {total_reward:+.4f}")
    print(f"  Final NAV: {info['nav']:.4f}  (started at 1.0)")
    print(f"  Total TC paid: {info['total_tc']*100:.4f}% of NAV")
    print(f"\n  ✓  Environment verified")
