"""The ensemble must be exactly mean-exposure / median-horizon of its members, on its own state."""
import numpy as np

from env.build import assemble
from env.engine import run
from tests.conftest import synthetic_world
from train.common import Ensemble, FoldSwitch


def _world():
    dates, market, R, member, rf = synthetic_world(T=700, seed=8)
    return assemble(dates, market, R, member, rf, window=60)


def state_policy(k):
    def pol(sim, t):
        p = sim.path_features()
        e = [0.5, 0.7, 0.85, 1.0][(t + k) % 4]
        return e, (5 if (p[1] < -0.01) ^ (k % 2) else 21)
    return pol


def test_ensemble_of_identical_policies_reproduces_the_policy():
    data = _world()
    single = run(data, state_policy(1), 130, 650)
    ens = run(data, Ensemble([state_policy(1)] * 5), 130, 650)
    assert np.array_equal(single.returns, ens.returns)
    assert single.decisions.equals(ens.decisions)


def test_ensemble_aggregates_members_on_its_own_state():
    data = _world()
    members = [state_policy(k) for k in range(5)]
    seen = []

    class Spy(Ensemble):
        def __call__(self, sim, t):
            acts = [m(sim, t) for m in self.members]
            out = super().__call__(sim, t)
            seen.append((acts, out))
            return out

    run(data, Spy(members), 130, 400)
    assert len(seen) > 10
    for acts, (e, h) in seen:
        assert e == np.mean([a[0] for a in acts])
        assert h == int(np.median([a[1] for a in acts]))


def test_fold_switch_uses_the_year_of_the_next_session():
    data = _world()
    years = sorted(set(data.dates.year))
    calls = []

    def tagged(y):
        def pol(sim, t):
            calls.append((y, int(data.dates[t + 1].year)))
            return 0.7, 5
        return pol

    run(data, FoldSwitch(data, {y: tagged(y) for y in years}), 130, len(data.dates) - 2)
    assert calls and all(policy_year == governed for policy_year, governed in calls)
