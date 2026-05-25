"""01_quickstart.py — minimal end-to-end demo on a synthetic city.

Builds a 100-station synthetic city graph, computes the spectral
applicability bound on a baseline IMD-like encoder, then augments
with the top-16 low-frequency Laplacian eigenvectors and reports the
ceiling lift.

Run with:
    python examples/01_quickstart.py
"""
from __future__ import annotations

import numpy as np

from spectral_mobility import (
    augment_features,
    build_geographic_knn,
    select_K,
    spectral_bound,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)


def main() -> None:
    rng = np.random.default_rng(42)
    N = 100

    # ── Fake "city": stations clustered into 2 city-centres + 1 isolated district
    centres = np.array([[48.85, 2.34], [48.86, 2.36], [48.84, 2.45]])
    weights = np.array([0.5, 0.4, 0.1])
    assignments = rng.choice(3, size=N, p=weights)
    lat = centres[assignments, 0] + rng.normal(0, 0.003, size=N)
    lng = centres[assignments, 1] + rng.normal(0, 0.003, size=N)

    # ── Fake IMD-like encoder features (4 columns)
    X = rng.standard_normal((N, 4))
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # ── Synthetic demand: smooth gradient + cluster effect + noise
    y = (lat - lat.mean()) * 100 + 2.0 * (assignments == 1) + rng.normal(0, 0.5, size=N)

    # ── Build graph and Laplacian
    W, sigma = build_geographic_knn(lat, lng, k=6)
    L = symmetric_normalised_laplacian(W)
    print(f"Graph:  N = {N}  k = 6  σ_auto = {sigma:.0f} m")

    # ── Full eigendecomposition (small N → dense is fine)
    eigvals, eigvecs = spectral_decomposition(L)
    print(f"Spectrum: λ ∈ [{eigvals.min():.4f}, {eigvals.max():.4f}]")

    # ── Spectral bound at K = 16
    result = spectral_bound(eigvecs, y, encoder_features=X, K=16)
    print(f"\nSpectral bound at K = {result.K}")
    print(f"  R²_IMD only       = {result.r2_imd:.4f}")
    print(f"  R²_spectral only  = {result.r2_spectral_only:.4f}")
    print(f"  R²_augmented      = {result.r2_augmented:.4f}")
    print(f"  ΔR²               = +{result.delta_r2:.4f}")

    # ── Automatic K selection by elbow heuristic
    K_auto, info = select_K(eigvecs, y, X, method="elbow", K_max=64)
    print(f"\nAuto-selected K = {K_auto} (elbow heuristic)")
    print(f"  R² curve: {[f'{r:.3f}' for r in info['r2_curve']]}")
    print(f"  K grid:   {info['K_grid']}")

    # ── Build augmented feature matrix for downstream ML
    X_aug = augment_features(X, eigvecs, K=K_auto)
    print(f"\nAugmented feature matrix:  X had {X.shape[1]} columns, "
          f"X_aug has {X_aug.shape[1]} (= {X.shape[1]} + {K_auto})")
    print("Feed X_aug into your favourite gradient-boosting / sklearn / "
          "neural-net model and enjoy the lifted ceiling.")


if __name__ == "__main__":
    main()
