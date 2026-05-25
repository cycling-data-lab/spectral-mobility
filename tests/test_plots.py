"""Smoke tests for spectral_mobility.plots."""

import numpy as np
import pytest

# Importable only if matplotlib is installed
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spectral_mobility import (
    build_geographic_knn,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)
from spectral_mobility.plots import (
    plot_bottleneck_map,
    plot_ceiling_curve,
    plot_cv_comparison,
    plot_spectrum,
)


def _toy(N=40, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(48.8, 48.9, size=N)
    lng = rng.uniform(2.3, 2.4, size=N)
    W, _ = build_geographic_knn(lat, lng, k=5)
    L = symmetric_normalised_laplacian(W)
    eigvals, eigvecs = spectral_decomposition(L)
    coords = np.column_stack([lat, lng])
    return eigvals, eigvecs, coords


def test_plot_ceiling_curve_runs():
    ax = plot_ceiling_curve([1, 2, 4, 8, 16], [0.1, 0.15, 0.25, 0.35, 0.42],
                            r2_baseline=0.08)
    assert ax.get_xlabel() == "K (low-frequency eigenvectors augmented)"
    plt.close("all")


def test_plot_bottleneck_map_runs():
    eigvals, eigvecs, coords = _toy()
    # Pick the most-localized mode
    from spectral_mobility import bottleneck_modes
    idx = bottleneck_modes(eigvecs, n_top=1)[0]
    ax = plot_bottleneck_map(coords, eigvecs[:, idx])
    assert ax is not None
    plt.close("all")


def test_plot_spectrum_runs():
    eigvals, eigvecs, coords = _toy()
    from spectral_mobility import inverse_participation_ratio
    ipr = inverse_participation_ratio(eigvecs)
    ax = plot_spectrum(eigvals, ipr, extended_threshold=5.0 / len(eigvals))
    assert ax.get_xlabel() == r"$\lambda$  (Laplacian eigenvalue)"
    plt.close("all")


def test_plot_cv_comparison_runs():
    fake_cv = {
        "baseline_scores": [0.05, 0.10, 0.07, 0.12, 0.08],
        "augmented_scores": [0.30, 0.40, 0.35, 0.42, 0.38],
        "baseline_mean": 0.084,
        "augmented_mean": 0.37,
        "mean_gain": 0.286,
    }
    ax = plot_cv_comparison(fake_cv)
    assert "augmented" in ax.get_title()
    plt.close("all")


def test_plot_bottleneck_map_shape_mismatch():
    eigvals, eigvecs, coords = _toy()
    with pytest.raises(ValueError):
        plot_bottleneck_map(coords[:20], eigvecs[:, 0])
    plt.close("all")
