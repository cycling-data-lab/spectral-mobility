"""Tests for spectral_mobility.graph."""

import numpy as np
import pytest
from scipy.sparse import issparse

from spectral_mobility import (
    build_feature_knn,
    build_geographic_knn,
    haversine_distance_matrix,
    symmetric_normalised_laplacian,
)


def test_haversine_self_zero():
    lat = np.array([48.85, 51.51, 40.71])
    lng = np.array([2.35, -0.13, -74.00])
    D = haversine_distance_matrix(lat, lng)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-6)


def test_haversine_known_distance_paris_london():
    # Paris -> London great-circle distance ~ 343 km
    lat = np.array([48.8566, 51.5074])
    lng = np.array([2.3522, -0.1278])
    D = haversine_distance_matrix(lat, lng)
    km = D[0, 1] / 1000
    assert 330 < km < 360, f"Paris-London should be ~ 343 km, got {km:.0f} km"


def test_haversine_symmetric():
    rng = np.random.default_rng(0)
    lat = rng.uniform(40, 50, size=20)
    lng = rng.uniform(-5, 5, size=20)
    D = haversine_distance_matrix(lat, lng)
    np.testing.assert_allclose(D, D.T, atol=1e-9)


def test_build_geographic_knn_basic_shape():
    rng = np.random.default_rng(0)
    N = 30
    lat = rng.uniform(48.8, 48.9, size=N)
    lng = rng.uniform(2.3, 2.4, size=N)
    W, sigma = build_geographic_knn(lat, lng, k=5)
    assert W.shape == (N, N)
    assert sigma > 0
    # symmetry
    diff = (W - W.T).toarray() if issparse(W) else W - W.T
    np.testing.assert_allclose(diff, 0, atol=1e-12)


def test_build_geographic_knn_sigma_auto():
    rng = np.random.default_rng(0)
    lat = rng.uniform(48.8, 48.9, size=50)
    lng = rng.uniform(2.3, 2.4, size=50)
    _, sigma_auto = build_geographic_knn(lat, lng, k=10)
    _, sigma_fixed = build_geographic_knn(lat, lng, k=10, sigma=1000.0)
    assert sigma_auto > 0
    assert abs(sigma_fixed - 1000.0) < 1e-9


def test_build_geographic_knn_rejects_invalid_input():
    with pytest.raises(ValueError):
        build_geographic_knn(np.array([1.0]), np.array([1.0]), k=2)


def test_build_feature_knn_basic():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5))
    W, sigma = build_feature_knn(X, k=8)
    assert W.shape == (40, 40)
    assert sigma > 0


def test_symmetric_normalised_laplacian_psd():
    rng = np.random.default_rng(0)
    lat = rng.uniform(48.8, 48.9, size=20)
    lng = rng.uniform(2.3, 2.4, size=20)
    W, _ = build_geographic_knn(lat, lng, k=5)
    L = symmetric_normalised_laplacian(W, dense=True)
    eigvals = np.linalg.eigvalsh(L)
    # Smallest eigenvalue should be ≈ 0 (constant vector); all ≥ 0
    assert eigvals[0] > -1e-8
    # Largest should be ≤ 2 for normalised Laplacian
    assert eigvals[-1] < 2.0 + 1e-6


def test_symmetric_normalised_laplacian_sparse_dense_agreement():
    rng = np.random.default_rng(0)
    lat = rng.uniform(48.8, 48.9, size=15)
    lng = rng.uniform(2.3, 2.4, size=15)
    W, _ = build_geographic_knn(lat, lng, k=4)
    L_dense = symmetric_normalised_laplacian(W, dense=True)
    L_sparse = symmetric_normalised_laplacian(W, dense=False).toarray()
    np.testing.assert_allclose(L_dense, L_sparse, atol=1e-10)
