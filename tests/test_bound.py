"""Tests for spectral_mobility.bound."""

import numpy as np
import pytest

from spectral_mobility import (
    build_geographic_knn,
    r2_spec_subspace,
    spectral_bound,
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


def test_r2_spec_zero_columns():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20)
    S = np.zeros((20, 0))
    assert r2_spec_subspace(S, y) == 0.0


def test_r2_spec_full_subspace():
    # If S has N orthonormal columns, R²_spec must be 1
    rng = np.random.default_rng(0)
    N = 20
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    y = rng.standard_normal(N)
    r2 = r2_spec_subspace(Q, y - y.mean())
    np.testing.assert_allclose(r2, 1.0, atol=1e-8)


def test_r2_spec_self_projection():
    # If S spans exactly y, R²_spec must be 1
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    y_centred = y - y.mean()
    r2 = r2_spec_subspace(y_centred.reshape(-1, 1), y_centred)
    np.testing.assert_allclose(r2, 1.0, atol=1e-10)


def test_r2_spec_orthogonal_subspace():
    # If S is orthogonal to y, R²_spec must be 0
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    y_centred = y - y.mean()
    # Pick a vector orthogonal to y
    z = rng.standard_normal(30)
    z -= (z @ y_centred) / (y_centred @ y_centred) * y_centred
    r2 = r2_spec_subspace(z.reshape(-1, 1), y_centred)
    assert r2 < 1e-8


def test_spectral_bound_augmentation_monotonic():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    X = rng.standard_normal((N, 3))
    for K in [1, 4, 8, 16]:
        result = spectral_bound(eigvecs, y, encoder_features=X, K=K)
        # Augmented ceiling must be at least as high as the IMD-only ceiling
        assert result.r2_augmented >= result.r2_imd - 1e-9
        assert result.delta_r2 >= -1e-9
        assert 0.0 <= result.r2_imd <= 1.0
        assert 0.0 <= result.r2_augmented <= 1.0


def test_spectral_bound_no_imd_features():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    result = spectral_bound(eigvecs, y, encoder_features=None, K=8)
    assert result.n_imd_features == 0
    assert result.r2_imd == 0.0
    # No-features bound equals spectral-only bound
    np.testing.assert_allclose(result.r2_augmented, result.r2_spectral_only, atol=1e-10)


def test_spectral_bound_monotonic_in_K():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N)
    X = rng.standard_normal((N, 2))
    r2_prev = -np.inf
    for K in [1, 2, 4, 8, 16]:
        if K > eigvecs.shape[1]: break
        result = spectral_bound(eigvecs, y, encoder_features=X, K=K)
        # Augmented R² is monotonically nondecreasing in K
        assert result.r2_augmented >= r2_prev - 1e-9
        r2_prev = result.r2_augmented


def test_spectral_bound_shape_mismatch():
    eigvecs, rng, N = _setup()
    y = rng.standard_normal(N + 1)
    with pytest.raises(ValueError):
        spectral_bound(eigvecs, y, encoder_features=None, K=4)
