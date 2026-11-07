"""The post-hoc VIX threshold rule reads only the VIX known at the decision close."""
import copy

import numpy as np

from env.engine import run
from eval.vix_rule import HIGH, LOW, VIX_COL, vix_threshold


def test_vix_rule_maps_levels_and_ignores_the_future(world):
    th = float(np.nanmedian(world.feats[:, VIX_COL]))
    pol = vix_threshold(world, th, 5)
    for t in (250, 300, 350):
        assert pol(None, t)[0] == (HIGH if world.feats[t, VIX_COL] >= th else LOW)

    base = run(world, pol, 200, 600)
    w2 = copy.copy(world)
    w2.feats = world.feats.copy()
    w2.feats[401:, VIX_COL] = 10.0                     # perturb everything after session 400
    pert = run(w2, vix_threshold(w2, th, 5), 200, 600)
    cut = world.dates[400]
    a = base.decisions[base.decisions["date"] <= cut]["exposure"].to_numpy()
    b = pert.decisions[pert.decisions["date"] <= cut]["exposure"].to_numpy()
    assert np.array_equal(a, b)
    assert np.array_equal(base.returns[:200], pert.returns[:200])   # sessions 201..400
