"""Tests for spectral_mobility.predictor.SpectralAugmentedRegressor."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from spectral_mobility import SpectralAugmentedRegressor


def _synthetic_city(N=120, seed=0):
    """A synthetic city where demand is smooth in space — augmentation
    should help substantially."""
    rng = np.random.default_rng(seed)
    centres = np.array([[48.85, 2.34], [48.86, 2.36], [48.84, 2.45]])
    assign = rng.choice(3, size=N, p=[0.5, 0.4, 0.1])
    lat = centres[assign, 0] + rng.normal(0, 0.003, size=N)
    lng = centres[assign, 1] + rng.normal(0, 0.003, size=N)
    X = rng.standard_normal((N, 4))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Demand: smooth gradient + cluster effect + noise
    y = (lat - lat.mean()) * 100 + 2.0 * (assign == 1) + rng.normal(0, 0.5, size=N)
    coords = np.column_stack([lat, lng])
    return X, coords, y


def test_fit_predict_basic():
    X, coords, y = _synthetic_city()
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=1.0), K=8)
    model.fit(X, coords, y)
    y_hat = model.predict()
    assert y_hat.shape == (X.shape[0],)
    # Should be reasonable correlation
    from scipy.stats import pearsonr

    r, _ = pearsonr(y, y_hat)
    assert r > 0.5


def test_attributes_after_fit():
    X, coords, y = _synthetic_city(N=50)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=1.0), K=4)
    model.fit(X, coords, y)
    assert model.eigvecs_.shape == (50, 50)
    assert model.eigvals_.shape == (50,)
    assert model.X_aug_.shape == (50, 4 + 4)
    assert model.n_imd_features_ == 4
    assert model.sigma_ > 0


def test_train_mask_respected():
    X, coords, y = _synthetic_city()
    train_mask = np.ones(len(X), dtype=bool)
    train_mask[:20] = False  # hold out first 20
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=1.0), K=4)
    model.fit(X, coords, y, train_mask=train_mask)
    y_held = model.predict(test_mask=~train_mask)
    assert y_held.shape == (20,)


def test_ceiling_matches_spectral_bound():
    X, coords, y = _synthetic_city(N=50)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=1.0), K=8)
    model.fit(X, coords, y)
    res = model.ceiling()
    assert 0.0 <= res.r2_imd <= 1.0
    assert 0.0 <= res.r2_augmented <= 1.0
    assert res.r2_augmented >= res.r2_imd - 1e-9
    assert res.K == 8


def test_cross_validate_augmented_beats_baseline_on_smooth_signal():
    X, coords, y = _synthetic_city(N=80, seed=1)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=0.1), K=8)
    res = model.cross_validate(X, coords, y, n_folds=4, random_state=0)
    assert "baseline_scores" in res and "augmented_scores" in res
    assert res["baseline_mean"] < res["augmented_mean"], (
        f"On a smooth synthetic city, augmented should beat baseline; "
        f"got baseline={res['baseline_mean']:.3f}, "
        f"augmented={res['augmented_mean']:.3f}"
    )
    assert res["ceiling"]["delta_r2"] > 0


def test_cross_validate_return_predictions():
    X, coords, y = _synthetic_city(N=60)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(alpha=1.0), K=4)
    res = model.cross_validate(X, coords, y, n_folds=3, return_predictions=True)
    preds = res["predictions"]
    assert preds["baseline"].shape == (60,)
    assert preds["augmented"].shape == (60,)
    # All slots filled by some fold
    assert not np.isnan(preds["baseline"]).any()
    assert not np.isnan(preds["augmented"]).any()


def test_feature_graph_type():
    rng = np.random.default_rng(0)
    N, p = 40, 5
    coords = rng.standard_normal((N, 3))  # 3-D feature space
    X = rng.standard_normal((N, p))
    y = rng.standard_normal(N)
    model = SpectralAugmentedRegressor(
        base_estimator=Ridge(alpha=1.0), K=4, graph_type="feature"
    )
    model.fit(X, coords, y)
    y_hat = model.predict()
    assert y_hat.shape == (N,)


def test_predict_before_fit_raises():
    model = SpectralAugmentedRegressor()
    with pytest.raises(RuntimeError):
        model.predict()


def test_shape_mismatch_raises():
    X, coords, y = _synthetic_city(N=30)
    model = SpectralAugmentedRegressor(base_estimator=Ridge(), K=4)
    with pytest.raises(ValueError):
        model.fit(X, coords, y[:20])  # y too short
