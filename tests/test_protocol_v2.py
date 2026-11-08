"""Protocol v2 (Nifty Next 50): the frozen-pipeline claim, study isolation, Holm, the VIX rule."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from config import ROOT
from env.build import assemble
from tests.conftest import synthetic_world

# Every key protocol v2 may change relative to v1. Anything else differing breaks the
# claim that v2 runs the frozen v1 pipeline on another universe.
ALLOWED = {
    ("version",), ("study",),
    ("data", "universe"), ("data", "calendar"), ("data", "index"), ("data", "benchmark"),
    ("data", "benchmark_key"),
    ("folds", "previously_seen"),
    ("baselines", "vix_rule"),
    ("evaluation", "primary"), ("evaluation", "multiplicity"), ("evaluation", "secondary", "against"),
}


def _diff(a, b, path=()):
    if isinstance(a, dict) and isinstance(b, dict):
        out = set()
        for k in set(a) | set(b):
            out |= _diff(a.get(k), b.get(k), path + (k,))
        return out
    return set() if a == b else {path}


def test_v2_changes_only_the_registered_keys():
    v1 = yaml.safe_load((ROOT / "config" / "protocol.yaml").read_text())
    v2 = yaml.safe_load((ROOT / "config" / "protocol_v2.yaml").read_text())
    changed = _diff(v1, v2)
    assert changed <= ALLOWED, f"unregistered changes: {sorted(changed - ALLOWED)}"
    for k in ("portfolio", "reward", "state", "training", "ensemble"):
        assert v1[k] == v2[k]
    assert v1["folds"]["test_years"] == v2["folds"]["test_years"]
    assert v1["baselines"]["vol_target"] == v2["baselines"]["vol_target"]
    assert v1["evaluation"]["bootstrap"] == v2["evaluation"]["bootstrap"]
    assert v1["data"]["freeze"] == v2["data"]["freeze"]


def _ask(code, env_value=None):
    env = {k: v for k, v in os.environ.items() if k != "RB_PROTOCOL"}
    if env_value:
        env["RB_PROTOCOL"] = env_value
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                          text=True, check=True).stdout.split()


def test_each_protocol_has_its_own_study_root_and_tag():
    code = "import config; print(config.study_root().relative_to(config.ROOT), config.protocol_tag())"
    assert _ask(code) == [".", "protocol-v1"]
    assert _ask(code, "protocol_v2.yaml") == ["studies/next50", "protocol-v2"]


def test_v1_only_scripts_refuse_under_v2():
    env = {**os.environ, "RB_PROTOCOL": "protocol_v2.yaml"}
    r = subprocess.run([sys.executable, "-m", "eval.vix_rule"], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "belongs to protocol v1" in r.stderr


def test_holm_step_down():
    from eval.report import holm
    d = holm({"H1": 0.01, "H2": 0.04}, 0.05)
    assert d["H1"]["rejected"] and d["H2"]["rejected"] and d["H1"]["holm_threshold"] == 0.025
    d = holm({"H1": 0.03, "H2": 0.04}, 0.05)          # smallest p misses alpha/2: nothing rejected
    assert not d["H1"]["rejected"] and not d["H2"]["rejected"]
    d = holm({"H1": 0.001, "H2": 0.30}, 0.05)
    assert d["H1"]["rejected"] and not d["H2"]["rejected"]


def test_index_column_drives_index_features():
    """A study with its own index ('index' column) must take index features from it, not from nsei."""
    dates, market, R, member, rf = synthetic_world()
    a = assemble(dates, market, R, member, rf, window=60)
    m2 = market.copy()
    m2["index"] = market["nsei"] * np.exp(np.random.default_rng(1).normal(0, 0.01, len(market))).cumprod()
    b = assemble(dates, m2, R, member, rf, window=60)
    c = assemble(dates, m2.drop(columns="nsei").rename(columns={"index": "nsei"}), R, member, rf, window=60)
    assert not np.allclose(np.nan_to_num(a.feats[:, :6]), np.nan_to_num(b.feats[:, :6]))
    np.testing.assert_array_equal(b.feats, c.feats)


def test_post_hoc_and_registered_vix_rule_are_the_same_function(world):
    from baselines.rules import vix_threshold as registered
    from eval.vix_rule import vix_threshold as post_hoc
    v = world.feats[:, 7]
    th = float(np.nanmedian(v))
    for t in range(200, 650, 7):
        assert post_hoc(world, th, 21)(None, t) == registered(world, th, 21)(None, t)


def test_v2_yaml_vix_grid_is_the_post_hoc_grid():
    from eval.vix_rule import HIGH, HORIZONS, LOW, THRESHOLDS
    vr = yaml.safe_load((ROOT / "config" / "protocol_v2.yaml").read_text())["baselines"]["vix_rule"]
    assert vr["thresholds"] == THRESHOLDS and vr["horizons"] == HORIZONS
    assert vr["low"] == LOW and vr["high"] == HIGH
    assert isinstance(pd.Series(vr["thresholds"]).iloc[0], float)


# ---- built Next 50 roster (skipped until `python -m data.roster_next50 build` has run)
import pytest  # noqa: E402

N50 = ROOT / "studies" / "next50" / "data" / "roster"
needs_roster = pytest.mark.skipif(not (N50 / "membership_daily.parquet").exists(), reason="v2 roster not built")


@needs_roster
def test_next50_has_fifty_members_every_session_and_never_overlaps_nifty50():
    d = pd.read_parquet(N50 / "membership_daily.parquet")
    assert (d.sum(axis=1) == 50).all()
    n50 = pd.read_parquet(ROOT / "data" / "roster" / "membership_daily.parquet").reindex(d.index).fillna(False)
    al = pd.read_csv(N50 / "symbol_aliases.csv")
    canon = dict(zip(al["alias"], al["symbol"]))
    n50.columns = [canon.get(c, c) for c in n50.columns]
    both = [c for c in d.columns if c in n50.columns]
    assert not (d[both] & n50[both].astype(bool)).to_numpy().any()


@needs_roster
def test_every_change_rests_on_an_nse_notice_and_every_reading_matches():
    ch = pd.read_csv(N50 / "changes.csv")
    assert ch["source"].str.startswith("https://www.niftyindices.com/Press_Release/").all()
    assert (ch["action"] == "add").sum() == (ch["action"] == "remove").sum()
    val = pd.read_csv(N50 / "validation.csv")
    assert len(val) >= 30 and val["match"].all()
