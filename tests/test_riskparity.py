import numpy as np

from env.riskparity import erc_weights, risk_contributions


def test_diagonal_covariance_gives_inverse_vol():
    sig = np.array([0.1, 0.2, 0.4, 0.25])
    w = erc_weights(np.diag(sig**2))
    expected = (1 / sig) / (1 / sig).sum()
    assert np.allclose(w, expected, atol=1e-12)


def test_equal_risk_contributions_random_covariance():
    rng = np.random.default_rng(3)
    for n in (2, 10, 50):
        A = rng.normal(size=(n, n))
        cov = A @ A.T / n + 0.05 * np.eye(n)
        w = erc_weights(cov)
        rc = risk_contributions(w, cov)
        assert (w > 0).all()
        assert abs(w.sum() - 1) < 1e-12
        assert np.max(np.abs(rc - 1 / n)) < 1e-4 / n


def test_precomputed_rows(world):
    W, elig = world.W, world.eligible
    sums = W.sum(axis=1)
    has_any = elig.any(axis=1)
    has_any[:59] = False  # no weights before the first full window
    assert np.allclose(sums[has_any], 1.0)
    assert (W[~elig] == 0).all()
    assert (W >= 0).all()
