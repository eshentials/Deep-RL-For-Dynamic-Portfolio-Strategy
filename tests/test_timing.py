"""Positive control: the engine must credit returns to the interval after a decision.

An oracle that peeks at the next interval's return must look spectacular; the same
rule using only the interval that has already happened must not.
"""
import numpy as np

from env.build import assemble
from env.engine import run
from tests.conftest import synthetic_world


def sharpe(res):
    ex = res.returns - res.rf
    return ex.mean() / ex.std(ddof=1) * np.sqrt(252)


def test_oracle_positive_control():
    dates, market, R, member, rf = synthetic_world(T=3000, seed=5)
    data = assemble(dates, market, R, member, rf, window=60)

    def oracle(sim, t):  # deliberate look-ahead
        return (1.0 if data.rp_ret[t + 1:t + 6].sum() > 0 else 0.5), 5

    def lagged(sim, t):  # same rule on information that exists at t
        return (1.0 if data.rp_ret[t - 4:t + 1].sum() > 0 else 0.5), 5

    def const(sim, t):
        return 1.0, 5

    kw = dict(start=130, end=len(dates) - 1)
    s_or, s_lag, s_c = (sharpe(run(data, p, **kw)) for p in (oracle, lagged, const))
    assert s_or - s_c > 1.5, (s_or, s_c)
    assert abs(s_lag - s_c) < 0.6, (s_lag, s_c)


def test_execution_lag_changes_the_path():
    dates, market, R, member, rf = synthetic_world(T=900, seed=6)
    data = assemble(dates, market, R, member, rf, window=60)

    def alt(sim, t):
        return (0.5 if (t // 5) % 2 else 1.0), 5

    r0 = run(data, alt, 130, 800)
    r1 = run(data, alt, 130, 800, lag=1)
    assert len(r0.returns) == len(r1.returns)
    assert not np.allclose(r0.returns, r1.returns)
