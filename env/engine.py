"""Portfolio engine shared by the PPO environment and every baseline.

Conventions (protocol Section 3):
- A decision at the close of session t trades at that close; the new book earns
  returns from t+1. With lag=1 the same decision trades one session later.
- Target book = e * weights[t] in stocks, (1 - e) in cash earning the risk-free rate.
- Cost = cost_rate * sum|target - current| (in NAV units), charged at the trade.
- Holdings drift between decisions. At the close of a stock's last member session
  (or its last traded day) the holding is sold, cost charged, proceeds to cash.
- Daily return r[t] = NAV_pre[t] / NAV_pre[t-1] - 1, where NAV_pre is the value at a
  close before any trade at that close; costs paid at close t-1 therefore land in r[t].
- With force_year_end, a decision is forced at the last session of each calendar
  year, so every return dated in year Y comes from a decision for year Y.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANN = np.sqrt(252.0)


@dataclass
class MarketData:
    dates: pd.DatetimeIndex
    R: np.ndarray            # (T, N) simple returns close t-1 -> t; NaN where no price
    W: np.ndarray            # (T, N) allocator weights at close t (rows sum to 1, or 0 if none)
    exit_mask: np.ndarray    # (T, N) holdings must be sold at close t
    delist_mask: np.ndarray  # (T, N) subset of exits where the stock stopped trading as a member
    rf: np.ndarray           # (T,) cash return close t-1 -> t
    feats: np.ndarray        # (T, K) raw market features at close t
    rp_ret: np.ndarray       # (T,) unlevered risk-parity daily return
    symbols: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.R0 = np.nan_to_num(self.R)  # no price -> holding value unchanged that day
        years = self.dates.year.to_numpy()
        T = len(self.dates)
        self.year_end = np.zeros(T, bool)
        self.year_end[:-1] = years[1:] != years[:-1]
        self.year_end[-1] = True
        nxt = np.full(T, T - 1)
        last = T - 1
        for t in range(T - 1, -1, -1):
            nxt[t] = last
            if self.year_end[t]:
                last = t
        self.next_year_end = nxt  # first year-end strictly after t
        self.exits = [np.flatnonzero(self.exit_mask[t]) for t in range(T)]

    def index_of(self, date) -> int:
        return int(self.dates.get_indexer([pd.Timestamp(date)])[0])

    def last_session_before(self, year: int) -> int:
        """Index of the last session of year-1 (the decision close that starts `year`)."""
        return int(np.flatnonzero(self.dates.year < year)[-1])

    def last_session_of(self, year: int) -> int:
        return int(np.flatnonzero(self.dates.year <= year)[-1])


class Sim:
    """One book moving through MarketData; drives both evaluation and the Gym env."""

    def __init__(self, data: MarketData, weights: np.ndarray | None = None, cost_bps: float = 10.0,
                 lag: int = 0, haircut: float = 0.0, force_year_end: bool = True):
        self.d = data
        self.Wt = data.W if weights is None else weights
        self.c = cost_bps / 1e4
        self.lag = lag
        self.haircut = haircut
        self.force_year_end = force_year_end

    # ----- state -------------------------------------------------------------
    def reset(self, start: int, end: int) -> None:
        """All cash at the close of `start`; the first decision is made there."""
        self.start, self.end, self.t = start, end, start
        N = self.d.R.shape[1]
        self.h = np.zeros(N)
        self.cash = 1.0
        self.nav_pre = 1.0
        self.peak = 1.0
        self.rets: list[float] = []
        self.costs = 0.0
        self.turnover = 0.0
        self.n_trades = 0

    @property
    def nav(self) -> float:
        return float(self.h.sum() + self.cash)

    def path_features(self) -> np.ndarray:
        nav = self.nav
        exposure = float(self.h.sum() / nav) if nav > 0 else 0.0
        own_dd = self.nav_pre / self.peak - 1.0
        hist = self.rets[-20:]
        if len(hist) < 20:  # pre-start days count as cash
            pre = self.d.rf[max(0, self.start - (20 - len(hist)) + 1):self.start + 1]
            hist = list(pre) + hist
        own_vol = float(np.std(hist, ddof=1) * ANN) if len(hist) > 1 else 0.0
        return np.array([exposure, own_dd, own_vol])

    # ----- trading -----------------------------------------------------------
    def _trade(self, e: float, w: np.ndarray, exclude: np.ndarray | None = None) -> None:
        w = w.copy()
        if exclude is not None and exclude.size:
            w[exclude] = 0.0  # stocks being sold at this close are not bought back
        nav = self.nav
        target = e * w * nav
        turn = float(np.abs(target - self.h).sum())
        cost = self.c * turn
        nav_after = nav - cost
        self.h = e * w * nav_after
        self.cash = nav_after - float(self.h.sum())
        self.costs += cost
        self.turnover += turn / nav if nav > 0 else 0.0
        self.n_trades += 1

    def _forced_exits(self, t: int) -> None:
        for i in self.d.exits[t]:
            v = self.h[i]
            if v <= 0:
                continue
            if self.d.delist_mask[t, i] and self.haircut:
                v *= 1.0 + self.haircut
            cost = self.c * v
            self.cash += v - cost
            self.costs += cost
            self.turnover += self.h[i] / self.nav_pre if self.nav_pre > 0 else 0.0
            self.h[i] = 0.0

    def next_decision(self, hz: int) -> int:
        nd = self.t + hz
        if self.force_year_end:
            nd = min(nd, int(self.d.next_year_end[self.t]))
        return min(nd, self.end)

    def act(self, e: float, hz: int) -> tuple[np.ndarray, bool]:
        """Decide at the current close, then run to the next decision close.

        Returns the daily returns of the sessions in the interval and whether the
        segment has ended.
        """
        t0 = self.t
        w0 = self.Wt[t0]
        exclude = self.d.exits[t0]
        if self.lag == 0:
            self._trade(e, w0, exclude)
            pending = None
        else:
            pending = (e, w0)
        nd = self.next_decision(hz)
        out = np.empty(nd - t0)
        R0, rf = self.d.R0, self.d.rf
        for k, s in enumerate(range(t0 + 1, nd + 1)):
            self.h *= 1.0 + R0[s]
            self.cash *= 1.0 + rf[s]
            nav = self.nav
            r = nav / self.nav_pre - 1.0
            out[k] = r
            self.rets.append(r)
            self.nav_pre = nav
            if nav > self.peak:
                self.peak = nav
            self._forced_exits(s)
            if pending is not None and s == t0 + self.lag:
                self._trade(pending[0], pending[1], self.d.exits[s])
                pending = None
        self.t = nd
        return out, nd >= self.end


@dataclass
class Result:
    dates: pd.DatetimeIndex     # sessions start+1..end
    returns: np.ndarray         # daily strategy returns
    rf: np.ndarray              # matching cash returns
    decisions: pd.DataFrame     # one row per decision
    costs: float
    turnover: float


def run(data: MarketData, policy, start: int, end: int, **sim_kw) -> Result:
    """Evaluate `policy(sim, t) -> (exposure, horizon)` from close `start` to close `end`."""
    sim = Sim(data, **sim_kw)
    sim.reset(start, end)
    rets, log = [], []
    done = start >= end
    while not done:
        e, hz = policy(sim, sim.t)
        log.append({"date": data.dates[sim.t], "exposure": float(e), "horizon": int(hz),
                    "invested_before": float(sim.h.sum() / sim.nav)})
        r, done = sim.act(float(e), int(hz))
        rets.append(r)
    ret = np.concatenate(rets) if rets else np.empty(0)
    return Result(dates=data.dates[start + 1:end + 1], returns=ret, rf=data.rf[start + 1:end + 1],
                  decisions=pd.DataFrame(log), costs=sim.costs, turnover=sim.turnover)
