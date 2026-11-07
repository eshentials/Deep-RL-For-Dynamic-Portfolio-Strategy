import numpy as np
import pandas as pd
import pytest


def synthetic_world(T: int = 700, N: int = 8, seed: int = 0):
    """i.i.d. one-factor returns with roster changes, a delisting and a late joiner."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-01", periods=T)
    mkt = rng.normal(0.0004, 0.01, T)
    R = mkt[:, None] * rng.uniform(0.6, 1.4, N) + rng.normal(0, 0.012, (T, N))
    R[0] = np.nan
    member = np.ones((T, N), bool)
    member[:300, N - 1] = False          # joins at session 300
    R[:200, N - 1] = np.nan              # listed at 200
    member[450:, 0] = False              # leaves the index after session 449
    R[501:, 1] = np.nan                  # stops trading after 500 while still a member
    idx = 1000 * np.exp(np.cumsum(np.nan_to_num(R).mean(axis=1)))
    market = pd.DataFrame({
        "nsei": idx,
        "vix": 15 + 5 * np.abs(rng.normal(size=T)),
        "usdinr": 70 * np.exp(np.cumsum(rng.normal(0, 0.003, T))),
    }, index=dates)
    rf = np.full(T, 0.065 / 252)
    return dates, market, R, member, rf


@pytest.fixture(scope="session")
def world():
    from env.build import assemble
    dates, market, R, member, rf = synthetic_world()
    return assemble(dates, market, R, member, rf, window=60)
