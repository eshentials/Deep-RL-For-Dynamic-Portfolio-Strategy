"""Gymnasium wrapper: MultiDiscrete([4, 2]) -> (exposure, horizon) on env.engine.Sim.

The observation builder here is the only one; evaluation policies call it too, so
training and evaluation see identical states.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env.engine import MarketData, Sim

# Fixed scalings for the path features (they depend on the policy, so they are not
# z-scored from data): exposure in [0, 1]; drawdown x10; annualised vol x5.
PATH_SCALE = np.array([1.0, 10.0, 5.0])


def build_obs(data: MarketData, sim: Sim, mu: np.ndarray, sd: np.ndarray, clip: float) -> np.ndarray:
    m = (data.feats[sim.t] - mu) / sd
    m = np.clip(np.nan_to_num(m, nan=0.0), -clip, clip)
    p = np.clip(sim.path_features() * PATH_SCALE, -clip, clip)
    return np.concatenate([m, p]).astype(np.float32)


def interval_reward(rets: np.ndarray, rf: np.ndarray, dd0: float, dd1: float,
                    kappa: float, scale: float) -> float:
    """Sum of daily excess log returns minus kappa * increase in drawdown magnitude."""
    excess = np.log1p(rets) - np.log1p(rf)
    return float((excess.sum() - kappa * max(0.0, dd1 - dd0)) * scale)


class RiskBudgetEnv(gym.Env):
    metadata: dict = {}

    def __init__(self, data: MarketData, lo: int, hi: int, mu: np.ndarray, sd: np.ndarray, *,
                 exposures, horizons, kappa: float, scale: float, episode_sessions: int,
                 cost_bps: float, clip: float):
        super().__init__()
        self.data, self.lo, self.hi = data, lo, hi
        self.mu, self.sd, self.clip = mu, sd, clip
        self.E, self.H = list(exposures), list(horizons)
        self.kappa, self.scale, self.L = kappa, scale, episode_sessions
        self.sim = Sim(data, cost_bps=cost_bps)
        self.action_space = spaces.MultiDiscrete([len(self.E), len(self.H)])
        dim = data.feats.shape[1] + len(PATH_SCALE)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

    def _dd(self) -> float:
        return 1.0 - self.sim.nav_pre / self.sim.peak  # drawdown magnitude >= 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        top = max(self.lo, self.hi - self.L)
        s = int(self.np_random.integers(self.lo, top + 1))
        self.sim.reset(s, min(s + self.L, self.hi))
        return build_obs(self.data, self.sim, self.mu, self.sd, self.clip), {}

    def step(self, action):
        e, hz = self.E[int(action[0])], self.H[int(action[1])]
        t0, dd0 = self.sim.t, self._dd()
        rets, done = self.sim.act(e, hz)
        rf = self.data.rf[t0 + 1:self.sim.t + 1]
        reward = interval_reward(rets, rf, dd0, self._dd(), self.kappa, self.scale)
        obs = build_obs(self.data, self.sim, self.mu, self.sd, self.clip)
        return obs, reward, False, done, {}
