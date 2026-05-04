from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from features import load_bundle
from state_spec import (
    STATE,
    build_state_vector,
    transaction_cost,
    TRANSACTION_COST_RATE,
    MACRO_PATH,
    EPS,
)
from core.smart_tangency import (
    SmartOptimizer,
    extract_smart_inputs,
    UTILITY_MV,
    UTILITIES,
    N_LAMBDA,
)

HOLDING_PERIODS_SMART = list(range(5, 65, 5))
N_HOLD_SMART = len(HOLDING_PERIODS_SMART)
OBS_CLIP = 5.0

RETURNS_LOOKBACK_MV = 60
RETURNS_LOOKBACK_TAIL = 252


def _drift_weights(w, stock_returns):
    w_stocks = w[:-1] * (1.0 + stock_returns)
    w_cash = w[-1]
    total = w_stocks.sum() + w_cash
    if total < EPS:
        N = len(stock_returns)
        return np.ones(N + 1) / (N + 1)
    return np.append(w_stocks / total, w_cash / total)


class SmartPortfolioEnv(gym.Env):
    def __init__(self, split="train", utility=UTILITY_MV, reward_scale=False):
        super().__init__()

        self.split = split
        self.utility = utility
        self.reward_scale = reward_scale
        self.optimizer = SmartOptimizer(utility)

        self.bundle = load_bundle(split)
        self.macro_df = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)

        self.tickers = self.bundle.tickers
        self.N = self.bundle.n_assets
        self.T = len(self.bundle.dates)
        self.warm_up = self.bundle.warm_up_rows()

        self.action_space = spaces.MultiDiscrete([N_LAMBDA, N_HOLD_SMART])
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP,
            shape=(STATE.DIM,), dtype=np.float32
        )

    def _equal_weights(self):
        return np.ones(self.N + 1) / (self.N + 1)

    def _build_obs(self):
        t = self.t
        state = build_state_vector(
            daily_ret_t=self.bundle.daily_ret.iloc[t].values.astype(np.float32),
            momentum_t=self.bundle.momentum.iloc[t].values.astype(np.float32),
            vol20d_t=self.bundle.realized_vol.iloc[t].values.astype(np.float32),
            ma_spread_t=self.bundle.ma_spread.iloc[t].values.astype(np.float32),
            cov_t=self.bundle.cov[t],
            weights=self.current_w[:-1].astype(np.float32),
            tickers=self.tickers,
            macro_df=self.macro_df,
            t_date=self.bundle.dates[t],
            nav_series=np.array(self.nav_history if self.nav_history else [self.nav]),
            log_ret_t=self.bundle.log_ret.iloc[t].values.astype(np.float32),
        )
        return np.clip(state, -OBS_CLIP, OBS_CLIP).astype(np.float32)

    def _get_optimizer_inputs(self, state):
        lookback = RETURNS_LOOKBACK_MV if self.utility == UTILITY_MV else RETURNS_LOOKBACK_TAIL
        return extract_smart_inputs(
            state=state,
            bundle=self.bundle,
            t=self.t,
            lookback=lookback,
        )

    def _benchmark_log_return(self, t_start, t_end):
        log_ret = 0.0
        for t in range(t_start, t_end):
            r = self.bundle.daily_ret.iloc[t].values
            log_ret += np.log1p(np.mean(r))
        return log_ret

    def _simulate_holding(self, target_w, hold_days):
        t_end = min(self.t + hold_days, self.T - 1)
        cum_log_ret = 0.0
        w = target_w.copy()

        for t_step in range(self.t, t_end):
            r = self.bundle.daily_ret.iloc[t_step].values
            daily_ret = (r * w[:-1]).sum()

            self.nav *= (1.0 + daily_ret)
            self.peak_nav = max(self.peak_nav, self.nav)

            cum_log_ret += np.log1p(daily_ret)
            w = _drift_weights(w, r)

            self.nav_history.append(self.nav)

        self.t = t_end
        self.current_w = w
        return cum_log_ret, w

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = self.warm_up
        self.nav = 1.0
        self.peak_nav = 1.0
        self.prev_drawdown = 0.0
        self.nav_history = [1.0]
        self.current_w = self._equal_weights()
        self.total_tc = 0.0
        self.episode_steps = 0

        return self._build_obs(), {}

    def step(self, action):
        lam_idx = int(action[0]) + 1
        hold_days = HOLDING_PERIODS_SMART[int(action[1])]

        obs_before = self._build_obs()
        inputs = self._get_optimizer_inputs(obs_before)

        # FIXED: clean μ extraction
        mu = inputs["mu"]

        try:
            result = self.optimizer.optimize(
                mu=mu,
                cov=inputs["cov"],
                lam_idx=lam_idx,
                tickers=self.tickers,
                returns_hist=inputs["returns_hist"],
            )
            target_w = result.weights
        except Exception:
            target_w = self.current_w.copy()

        # transaction cost
        tc = transaction_cost(self.current_w, target_w)
        self.nav *= (1.0 - tc)
        self.total_tc += tc
        self.current_w = target_w.copy()

        cum_log_ret, _ = self._simulate_holding(target_w, hold_days)

        t_after = self.t
        t_before = max(t_after - hold_days, self.warm_up)

        benchmark = self._benchmark_log_return(t_before, t_after)
        excess_log = cum_log_ret - benchmark
        excess_return = excess_log / max(t_after - t_before, 1)

        # ============================
        # ✅ FIXED REWARD (CRITICAL)
        # ============================

        recent_nav = np.array(self.nav_history[-21:], dtype=np.float64)

        if len(recent_nav) > 1:
            returns = np.diff(np.log(recent_nav))
            rolling_vol = np.std(returns) + 1e-6
        else:
            rolling_vol = 1e-3

        reward = excess_return / rolling_vol

        # penalties
        reward -= 0.05 * tc
        reward -= 0.005 * (1.0 / hold_days)

        # clipping for stability
        reward = float(np.clip(reward, -5.0, 5.0))

        if self.reward_scale:
            reward /= hold_days

        terminated = (self.t >= self.T - 1)
        obs = self._build_obs() if not terminated else np.zeros(STATE.DIM)

        info = {
    "nav": self.nav,
    "lambda_idx": lam_idx,
    "hold_days": hold_days,
    "tc_cost": tc,
    "total_tc": self.total_tc,
    }      
        return obs, reward, terminated, False, info