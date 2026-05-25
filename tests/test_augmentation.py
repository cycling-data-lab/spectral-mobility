"""Tests for spectral_mobility.augmentation."""

import numpy as np
import pytest

from spectral_mobility import (
    augment_features,
    build_geographic_knn,
    select_K,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)


def _setup(N=40, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(48.8, 48.9, size=N)
    lng = rng.uniform(2.3, 2.4, size=N)
    W, _ = build_geographic_knn(lat, lng, k=6)
    L = symmetric_normalised_laplacian(W, dense=True)
    eigvals, eigvecs = spectral_decomposition(L)
    return eigvecs, rng, N


def test_augment_features_shape():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N, 4))
    X_aug = augment_features(X, eigvecs, K=10)
    assert X_aug.shape == (N, 4 + 10)


def test_augment_preserves_original_columns():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N, 3))
    X_aug = augment_features(X, eigvecs, K=5, standardise=False)
    np.testing.assert_allclose(X_aug[:, :3], X)


def test_augment_standardise_appended_columns():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N, 2))
    X_aug = augment_features(X, eigvecs, K=4, standardise=True)
    appended = X_aug[:, 2:]
    # After standardisation, each appended column should have std ≈ 1
    np.testing.assert_allclose(appended.std(axis=0), 1.0, atol=1e-6)


def test_augment_no_standardise_uses_raw_eigenvectors():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N, 1))
    X_aug = augment_features(X, eigvecs, K=3, standardise=False)
    np.testing.assert_allclose(X_aug[:, 1:], eigvecs[:, :3])


def test_augment_K_larger_than_eigvec_count():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N, 2))
    # Request more eigenvectors than available — clamps to available
    X_aug = augment_features(X, eigvecs, K=eigvecs.shape[1] + 100)
    assert X_aug.shape[1] == 2 + eigvecs.shape[1]


def test_augment_rejects_row_mismatch():
    eigvecs, rng, N = _setup()
    X = rng.standard_normal((N + 1, 2))
    with pytest.raises(ValueError):
        augment_features(X, eigvecs, K=4)


def test_select_K_fixed():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    K, info = select_K(eigvecs, y, method="fixed", fixed_K=12)
    assert K == 12
    assert info["method"] == "fixed"


def test_select_K_elbow_returns_valid_K():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    X = rng.standard_normal((N, 3))
    K, info = select_K(eigvecs, y, X, method="elbow", K_max=32)
    assert K > 0
    assert info["method"] == "elbow"
    assert len(info["K_grid"]) == len(info["r2_curve"])
    # r2 curve should be non-decreasing
    r2c = info["r2_curve"]
    for a, b in zip(r2c, r2c[1:]):
        assert b >= a - 1e-9


def test_select_K_ratio_target():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    K, info = select_K(eigvecs, y, method="ratio", K_max=32, ratio_target=0.5)
    assert info["method"] == "ratio"
    assert 1 <= K <= 32
