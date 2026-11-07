import numpy as np
import pytest

from eval.bootstrap import block_indices, paired_sharpe_diff
from eval.dsr import dsr_from_stats, psr


def test_dsr_reproduces_bailey_lopez_de_prado_example():
    # Bailey and Lopez de Prado (2014): annualised SR 2.5 over T=1250 daily obs,
    # skew -3, kurtosis 10, N=100 trials with annualised V[SR]=0.5 -> DSR ~ 0.9004.
    out = dsr_from_stats(sr=2.5 / np.sqrt(250), T=1250, sk=-3.0, ku=10.0,
                         n_trials=100, var_sr=0.5 / 250)
    assert out["dsr"] == pytest.approx(0.9004, abs=5e-4)
    assert out["sr0"] * np.sqrt(250) == pytest.approx(1.789, abs=5e-3)


def test_single_trial_dsr_equals_psr():
    out = dsr_from_stats(sr=0.05, T=1000, sk=0.0, ku=3.0, n_trials=1)
    assert out["dsr"] == out["psr"] == pytest.approx(psr(0.05, 1000, 0.0, 3.0))


def test_identical_series_give_zero_difference():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 800)
    rf = np.full(800, 0.0002)
    out = paired_sharpe_diff(r, r, rf, block=20, draws=500, seed=1)
    assert out["observed"] == 0.0
    assert np.all(out["draws"] == 0.0)


def test_block_indices_are_contiguous_circular_blocks():
    rng = np.random.default_rng(2)
    idx = block_indices(100, 20, 3, rng)
    assert idx.shape == (3, 100)
    for row in idx:
        for b in range(5):
            blk = row[b * 20:(b + 1) * 20]
            assert np.all(np.diff(blk) % 100 == 1)


def test_bootstrap_detects_a_real_edge():
    rng = np.random.default_rng(3)
    n = 2000
    common = rng.normal(0.0003, 0.01, n)
    a = common + rng.normal(0.0006, 0.002, n)   # same risk, higher mean
    rf = np.zeros(n)
    out = paired_sharpe_diff(a, common, rf, draws=2000, seed=4)
    assert out["observed"] > 0 and out["p05"] > 0
