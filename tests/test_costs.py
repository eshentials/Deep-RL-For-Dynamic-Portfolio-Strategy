"""Hand-computed trades on a 3-stock book (10 bp cost, zero cash rate)."""
import numpy as np
import pandas as pd
import pytest

from env.engine import MarketData, Sim

W0 = np.array([0.5, 0.3, 0.2])


def tiny_world():
    T, N = 8, 3
    R = np.zeros((T, N))
    R[0] = np.nan
    R[1] = [0.10, 0.0, 0.0]      # stock 0 +10% on day 1
    R[4] = [0.0, 0.0, 0.50]      # stock 2 +50% on day 4, after it has left
    exit_mask = np.zeros((T, N), bool)
    exit_mask[3, 2] = True       # stock 2's last member session is 3
    W = np.tile(W0, (T, 1))
    W[3:, 2] = 0.0               # as env/build.py does: not eligible at its last member close
    W[3:] /= W[3:].sum(axis=1, keepdims=True)
    return MarketData(dates=pd.bdate_range("2020-01-01", periods=T), R=R, W=W,
                      exit_mask=exit_mask, delist_mask=np.zeros((T, N), bool),
                      rf=np.zeros(T), feats=np.zeros((T, 1)), rp_ret=np.zeros(T))


def test_entry_from_cash_costs_seven_bp_at_70pct():
    sim = Sim(tiny_world(), cost_bps=10, force_year_end=False)
    sim.reset(0, 7)
    rets, _ = sim.act(0.7, 1)
    # turnover 0.7 of NAV -> cost 0.001 * 0.7 = 0.0007
    assert sim.costs == pytest.approx(0.0007, abs=1e-15)
    # day 1: (1 - 0.0007) * (1 + 0.7 * 0.5 * 0.10) - 1
    assert rets[0] == pytest.approx(0.9993 * 1.035 - 1, abs=1e-15)


def test_drifted_rebalance_cost():
    sim = Sim(tiny_world(), cost_bps=10, force_year_end=False)
    sim.reset(0, 7)
    sim.act(0.7, 1)
    h, nav = sim.h.copy(), sim.nav
    c0 = sim.costs
    sim.act(0.7, 1)
    expected = 0.001 * np.abs(0.7 * W0 * nav - h).sum()
    assert sim.costs - c0 == pytest.approx(expected, abs=1e-15)
    assert expected > 0


def test_zero_change_rebalance_is_free():
    sim = Sim(tiny_world(), cost_bps=10, force_year_end=False)
    sim.reset(4, 7)
    sim.act(0.85, 1)     # trade at close 4; day 5 has zero returns, no exits
    c0 = sim.costs
    sim.act(0.85, 1)     # same target at close 5
    assert sim.costs - c0 == pytest.approx(0.0, abs=1e-15)


def test_forced_exit_sells_at_last_member_close_and_is_not_rebought():
    sim = Sim(tiny_world(), cost_bps=10, force_year_end=False)
    sim.reset(2, 7)
    sim.act(1.0, 1)                       # buy at close 2, hold to close 3
    h2 = sim.h[2]
    assert h2 == 0.0                      # sold at close 3 (exit), inside act()
    nav3, c3 = sim.nav, sim.costs
    sim.act(1.0, 1)                       # decide at close 3: stock 2 excluded
    assert sim.h[2] == 0.0
    # stock 2's +50% on day 4 must not reach NAV: stocks 0 and 1 earn 0 on day 4
    assert sim.nav == pytest.approx(nav3 - (sim.costs - c3), rel=1e-12)


def test_exit_cost_charged():
    sim = Sim(tiny_world(), cost_bps=10, force_year_end=False)
    sim.reset(2, 7)
    sim.act(1.0, 1)
    nav_after_entry = 1.0 - 0.001 * 1.0
    v = nav_after_entry * W0[2]           # value of stock 2 at close 3 (zero returns on day 3)
    assert sim.costs == pytest.approx(0.001 * 1.0 + 0.001 * v, abs=1e-15)
    assert sim.cash == pytest.approx(v * (1 - 0.001), abs=1e-15)
