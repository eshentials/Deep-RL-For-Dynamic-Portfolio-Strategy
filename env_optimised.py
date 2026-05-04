import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from efficient_frontier import optimize, LAMBDA_VALUES
from features import load_bundle
from state_spec import build_state_vector, MACRO_PATH

HOLDING_PERIODS = [1, 5, 10, 21, 63]

class PortfolioEnv(gym.Env):
    def __init__(self, split="train"):
        super().__init__()

        self.bundle = load_bundle(split)
        self.macro = pd.read_csv(MACRO_PATH, index_col=0, parse_dates=True)

        self.tickers = self.bundle.tickers
        self.N = len(self.tickers)
        self.T = len(self.bundle.dates)

        self.warm_up = 90

        self.action_space = spaces.MultiDiscrete([len(LAMBDA_VALUES), len(HOLDING_PERIODS)])
        self.observation_space = spaces.Box(low=-5, high=5, shape=(49,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # RANDOM START (prevents overfitting)
        self.t = np.random.randint(self.warm_up, self.T - 100)

        self.nav = 1.0
        self.peak_nav = 1.0
        self.prev_weights = np.ones(self.N) / self.N

        return self._get_obs(), {}

    def _get_obs(self):
        state = build_state_vector(
            self.bundle.daily_ret.iloc[self.t].values,
            self.bundle.momentum.iloc[self.t].values,
            self.bundle.realized_vol.iloc[self.t].values,
            self.bundle.ma_spread.iloc[self.t].values,
            self.bundle.cov[self.t],
            self.prev_weights,
            self.tickers,
            self.macro,
            self.bundle.dates[self.t],
            np.array([self.nav]),
            self.bundle.log_ret.iloc[self.t].values
        )
        return np.clip(state, -5, 5).astype(np.float32)

    def step(self, action):
        lam = LAMBDA_VALUES[action[0]]
        hold_days = HOLDING_PERIODS[action[1]]

        # MVO allocation
        weights = optimize(self._get_obs(), self.tickers, lam).weights

        # --- TRANSACTION COST ---
        turnover = np.sum(np.abs(weights - self.prev_weights))
        tc_cost = 0.0025 * turnover

        self.nav *= (1 - tc_cost)

        cumulative_return = 0

        for _ in range(hold_days):
            if self.t >= self.T - 1:
                break

            returns = self.bundle.daily_ret.iloc[self.t].values
            port_ret = np.dot(weights, returns)

            self.nav *= (1 + port_ret)
            cumulative_return += np.log(1 + port_ret)

            self.t += 1

        # --- RISK PENALTY ---
        vol_penalty = 0.1 * (cumulative_return ** 2)

        # --- DRAWDOWN ---
        self.peak_nav = max(self.peak_nav, self.nav)
        drawdown = (self.nav - self.peak_nav) / (self.peak_nav + 1e-8)
        drawdown_penalty = 0.5 * abs(drawdown)

        # --- FINAL REWARD ---
        reward = cumulative_return \
                 - tc_cost \
                 - vol_penalty \
                 - drawdown_penalty

        self.prev_weights = weights

        done = self.t >= self.T - 1

        return self._get_obs(), reward, done, False, {}