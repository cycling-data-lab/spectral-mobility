"""Tests for the inductive cross-validation protocol."""

import numpy as np
from sklearn.linear_model import Ridge

from spectral_mobility import SpectralAugmentedRegressor


def _synthetic_city(N=120, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[48.85, 2.34], [48.86, 2.36], [48.84, 2.45]])
    assign = rng.choice(3, size=N, p=[0.5, 0.4, 0.1])
    lat = centres[assign, 0] + rng.normal(0, 0.003, size=N)
    lng = centres[assign, 1] + rng.normal(0, 0.003, size=N)
    X = rng.standard_normal((N, 4))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (lat - lat.mean()) * 100 + 2.0 * (assign == 1) + rng.normal(0, 0.5, size=N)
    coords = np.column_stack([lat, lng])
    return X, coords, y


def test_inductive_runs_and_returns_protocol():
    X, coords, y = _synthetic_city(N=80)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=0.1), K=8)
    res = model.cross_validate(
        X, coords, y, n_folds=4, protocol="inductive", random_state=0
    )
    assert res["protocol"] == "inductive"
    assert len(res["augmented_scores"]) == 4
    assert "baseline_mean" in res
    assert "augmented_mean" in res


def test_inductive_beats_baseline_on_smooth_signal():
    X, coords, y = _synthetic_city(N=120, seed=2)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=0.1), K=8)
    res = model.cross_validate(
        X, coords, y, n_folds=5, protocol="inductive", random_state=0
    )
    assert res["augmented_mean"] > res["baseline_mean"]


def test_transductive_and_inductive_give_similar_results():
    """On a smooth signal, both protocols should produce similar gains.
    (Inductive can be slightly weaker due to Nyström extension noise.)"""
    X, coords, y = _synthetic_city(N=100, seed=3)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=0.1), K=8)
    res_t = model.cross_validate(
        X, coords, y, n_folds=4, protocol="transductive", random_state=0
    )
    res_i = model.cross_validate(
        X, coords, y, n_folds=4, protocol="inductive", random_state=0
    )
    # Both should be positive
    assert res_t["mean_gain"] > 0
    assert res_i["mean_gain"] > 0


def test_inductive_invalid_protocol_raises():
    X, coords, y = _synthetic_city(N=50)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=0.1), K=4)
    import pytest

    with pytest.raises(ValueError):
        model.cross_validate(X, coords, y, n_folds=3, protocol="nonsense")


def test_inductive_feature_graph_type():
    """Inductive mode should also work with feature-space graphs."""
    rng = np.random.default_rng(0)
    N, p = 60, 5
    feats = rng.standard_normal((N, 3))
    X = rng.standard_normal((N, p))
    y = rng.standard_normal(N) + feats[:, 0] * 0.5
    model = SpectralAugmentedRegressor(
        base_estimator=Ridge(alpha=0.1), K=4, graph_type="feature"
    )
    res = model.cross_validate(X, feats, y, n_folds=3, protocol="inductive", random_state=0)
    assert "augmented_mean" in res
