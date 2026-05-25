"""Tests for spectral_mobility.spectral."""

import numpy as np
import pytest

from spectral_mobility import (
    build_geographic_knn,
    inverse_participation_ratio,
    level_spacing_ratios,
    participation_ratio,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)


def _toy_laplacian(N=30, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(48.8, 48.9, size=N)
    lng = rng.uniform(2.3, 2.4, size=N)
    W, _ = build_geographic_knn(lat, lng, k=5)
    return symmetric_normalised_laplacian(W, dense=True), N


def test_dense_full_spectrum():
    L, N = _toy_laplacian()
    eigvals, eigvecs = spectral_decomposition(L)
    assert eigvals.shape == (N,)
    assert eigvecs.shape == (N, N)
    # ascending
    assert np.all(np.diff(eigvals) >= -1e-9)
    # orthonormal
    np.testing.assert_allclose(eigvecs.T @ eigvecs, np.eye(N), atol=1e-8)


def test_dense_partial_spectrum():
    L, N = _toy_laplacian()
    eigvals, eigvecs = spectral_decomposition(L, k=5, which="smallest")
    assert eigvals.shape == (5,)
    assert eigvecs.shape == (N, 5)
    # These should be the LOWEST 5
    full_vals, _ = spectral_decomposition(L)
    np.testing.assert_allclose(eigvals, full_vals[:5], atol=1e-10)


def test_ipr_extended_state():
    # A perfectly uniform vector has IPR = 1/N exactly
    N = 50
    psi = np.ones((N, 1)) / np.sqrt(N)
    ipr = inverse_participation_ratio(psi)
    assert ipr.shape == (1,)
    np.testing.assert_allclose(ipr[0], 1.0 / N, atol=1e-12)


def test_ipr_localized_state():
    # Delta function on a single node has IPR = 1
    N = 50
    psi = np.zeros((N, 1))
    psi[3, 0] = 1.0
    ipr = inverse_participation_ratio(psi)
    np.testing.assert_allclose(ipr[0], 1.0, atol=1e-12)


def test_participation_ratio_bounds():
    L, N = _toy_laplacian()
    eigvals, eigvecs = spectral_decomposition(L)
    pr = participation_ratio(eigvecs)
    assert pr.shape == (N,)
    assert (pr > 0).all()
    assert (pr <= 1 + 1e-6).all()


def test_level_spacing_extended_regime():
    # GOE-like eigenvalues (random matrix); expect <r> ≈ 0.5295
    rng = np.random.default_rng(0)
    A = rng.standard_normal((300, 300))
    A = (A + A.T) / 2  # symmetrise
    eigvals = np.linalg.eigvalsh(A)
    r = level_spacing_ratios(eigvals)
    mean_r = r.mean()
    # GOE expected 0.5295; with finite N noise tolerate ± 0.04
    assert 0.48 < mean_r < 0.58, f"GOE benchmark check: mean r = {mean_r}"


def test_level_spacing_localized_regime():
    # Independent Poisson-distributed eigenvalues; expect <r> ≈ 0.3863
    rng = np.random.default_rng(0)
    eigvals = np.cumsum(rng.exponential(size=2000))
    r = level_spacing_ratios(eigvals)
    mean_r = r.mean()
    assert 0.35 < mean_r < 0.42, f"Poisson benchmark check: mean r = {mean_r}"
