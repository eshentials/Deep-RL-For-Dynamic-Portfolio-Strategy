"""Future-perturbation invariance: nothing up to t* may change when data after t* does."""
import re
from pathlib import Path

import numpy as np

from env.build import assemble
from env.engine import run
from env.features import zscore_stats
from tests.conftest import synthetic_world

ROOT = Path(__file__).resolve().parents[1]
T_STAR = 400


def state_policy(data):
    """Deterministic policy that uses every kind of state the agent sees."""
    def pol(sim, t):
        f = data.feats[t]
        p = sim.path_features()
        score = np.nan_to_num(f).sum() + 3 * p.sum() + 50 * data.rp_ret[t]
        e = [0.5, 0.7, 0.85, 1.0][int(abs(score * 1000)) % 4]
        return e, (5 if score > 0 else 21)
    return pol


def perturbed(seed=11):
    dates, market, R, member, rf = synthetic_world()
    rng = np.random.default_rng(seed)
    R2, m2, rf2, mem2 = R.copy(), market.copy(), rf.copy(), member.copy()
    fut = slice(T_STAR + 1, None)
    R2[fut] = R2[fut] * rng.uniform(0.3, 1.7, R2[fut].shape) + rng.normal(0, 0.01, R2[fut].shape)
    m2.iloc[T_STAR + 1:] = m2.iloc[T_STAR + 1:].to_numpy() * rng.uniform(0.8, 1.2, (len(dates) - T_STAR - 1, 3))
    rf2[fut] *= 3.0
    mem2[T_STAR + 2:, 3] = ~mem2[T_STAR + 2:, 3]   # roster change announced after t*+1
    a = assemble(dates, market, R, member, rf, window=60)
    b = assemble(dates, m2, R2, mem2, rf2, window=60)
    return a, b


def test_features_and_weights_invariant():
    a, b = perturbed()
    k = T_STAR + 1
    assert np.array_equal(a.feats[:k], b.feats[:k], equal_nan=True)
    assert np.array_equal(a.W[:k], b.W[:k])
    assert np.array_equal(a.exit_mask[:k], b.exit_mask[:k])
    assert np.array_equal(a.rp_ret[:k], b.rp_ret[:k])
    mu_a, sd_a = zscore_stats(_df(a), 100, T_STAR)
    mu_b, sd_b = zscore_stats(_df(b), 100, T_STAR)
    assert np.array_equal(mu_a, mu_b) and np.array_equal(sd_a, sd_b)
    # and the perturbation really changed the future
    assert not np.array_equal(a.feats[k + 30:], b.feats[k + 30:], equal_nan=True)


def test_decisions_and_nav_invariant():
    a, b = perturbed()
    ra = run(a, state_policy(a), 130, len(a.dates) - 1)
    rb = run(b, state_policy(b), 130, len(b.dates) - 1)
    upto = a.dates[T_STAR]
    da = ra.decisions[ra.decisions["date"] <= upto].reset_index(drop=True)
    db = rb.decisions[rb.decisions["date"] <= upto].reset_index(drop=True)
    assert len(da) > 10
    assert da.equals(db)
    n = int((ra.dates <= upto).sum())
    assert np.array_equal(ra.returns[:n], rb.returns[:n])
    assert not np.array_equal(ra.returns[n + 30:], rb.returns[n + 30:])


def test_no_backward_fill_in_pipeline_code():
    pat = re.compile(r"bfill|backfill|method=['\"]bfill|shift\(\s*-")
    for folder in ("data", "env", "baselines", "train"):
        for p in (ROOT / folder).glob("*.py"):
            assert not pat.search(p.read_text()), f"backward-looking fill in {p}"


def _df(data):
    import pandas as pd
    return pd.DataFrame(data.feats, index=data.dates)
