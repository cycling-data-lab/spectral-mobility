"""Feature augmentation for downstream ML pipelines."""

from __future__ import annotations

from typing import Literal

import numpy as np

from spectral_mobility.bound import r2_spec_subspace


def augment_features(
    X: np.ndarray,
    eigvecs: np.ndarray,
    K: int = 16,
    *,
    standardise: bool = True,
) -> np.ndarray:
    """Augment a feature matrix with the top-K low-frequency
    Laplacian eigenvectors.

    Parameters
    ----------
    X : (N, p) np.ndarray
        Bare encoder feature matrix.
    eigvecs : (N, M) np.ndarray
        Laplacian eigenvectors, columns ordered by ascending
        eigenvalue.  Pass the output of
        :func:`spectral_decomposition`.
    K : int, default 16
        Number of low-frequency eigenvectors to append.
    standardise : bool, default True
        If True, z-score the appended eigenvector columns to unit
        variance so they match the typical scale of feature columns.
        (Each raw eigenvector has unit ℓ₂-norm, which means standard
        deviation ≈ 1/√N — too small for tree-based models to pick
        up by default.)

    Returns
    -------
    X_aug : (N, p + K) np.ndarray
        Augmented feature matrix, ready to feed to LightGBM /
        XGBoost / sklearn / any model that consumes a 2-D matrix.
    """
    X = np.asarray(X, dtype=float)
    eigvecs = np.asarray(eigvecs, dtype=float)
    if X.shape[0] != eigvecs.shape[0]:
        raise ValueError("X and eigvecs must have the same number of rows")
    K_eff = min(int(K), eigvecs.shape[1])
    Uk = eigvecs[:, :K_eff].copy()
    if standardise:
        std = Uk.std(axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        Uk = Uk / std[None, :]
    return np.column_stack([X, Uk])


def select_K(
    eigvecs: np.ndarray,
    y: np.ndarray,
    X: np.ndarray | None = None,
    *,
    method: Literal["elbow", "ratio", "fixed"] = "elbow",
    K_max: int = 64,
    ratio_target: float = 0.95,
    fixed_K: int = 16,
    elbow_tol: float = 0.01,
) -> tuple[int, dict]:
    """Heuristically select a value of K for augmentation.

    Three modes:

    - ``"elbow"`` (default): scan ``K = 1, 2, 4, 8, ..., K_max`` and
      return the smallest K such that the marginal gain in
      ``R²_spec`` over the previous step falls below ``elbow_tol``.
    - ``"ratio"``: return the smallest K that achieves
      ``R²_spec(K) ≥ ratio_target * R²_spec(K_max)``.
    - ``"fixed"``: simply return ``fixed_K``.

    Parameters
    ----------
    eigvecs : (N, M) np.ndarray
    y : (N,) np.ndarray
    X : (N, p) np.ndarray, optional
        Bare encoder features.  If supplied, gains are measured
        relative to ``R²_spec(X)``; otherwise relative to 0.
    method : str
    K_max : int, default 64
    ratio_target : float, default 0.95
    fixed_K : int, default 16
    elbow_tol : float, default 0.01

    Returns
    -------
    K : int
        Selected number of eigenvectors.
    info : dict
        Diagnostic data:
        ``{"K_grid": [...], "r2_curve": [...], "method": ...}``.
    """
    eigvecs = np.asarray(eigvecs, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    y_c = y - y.mean()
    if X is None:
        X_arr = np.zeros((y.shape[0], 0))
    else:
        X_arr = np.asarray(X, dtype=float)
    K_max_eff = min(int(K_max), eigvecs.shape[1])

    if method == "fixed":
        return int(min(fixed_K, K_max_eff)), {"method": "fixed", "K_grid": [fixed_K]}

    # geometric K grid 1, 2, 4, 8, ..., K_max
    K_grid: list[int] = []
    k = 1
    while k <= K_max_eff:
        K_grid.append(k)
        k *= 2
    if K_grid[-1] != K_max_eff:
        K_grid.append(K_max_eff)

    r2_curve = []
    for k in K_grid:
        Uk = eigvecs[:, :k]
        S_aug = np.column_stack([X_arr, Uk]) if X_arr.size else Uk
        r2_curve.append(r2_spec_subspace(S_aug, y_c))

    if method == "ratio":
        target = ratio_target * r2_curve[-1]
        for ki, r2 in zip(K_grid, r2_curve):
            if r2 >= target:
                return ki, {
                    "method": "ratio",
                    "K_grid": K_grid,
                    "r2_curve": r2_curve,
                    "target": target,
                }
        return K_grid[-1], {
            "method": "ratio",
            "K_grid": K_grid,
            "r2_curve": r2_curve,
            "target": target,
        }

    if method == "elbow":
        # Detect elbow only AFTER we have captured at least half of the
        # eventual achievable gain — otherwise an initial plateau
        # would be mistaken for the elbow.
        r2_max = r2_curve[-1]
        progress_threshold = 0.5 * r2_max
        for i in range(1, len(K_grid)):
            if r2_curve[i - 1] < progress_threshold:
                continue
            gain = r2_curve[i] - r2_curve[i - 1]
            if gain < elbow_tol:
                return K_grid[i - 1], {
                    "method": "elbow",
                    "K_grid": K_grid,
                    "r2_curve": r2_curve,
                    "elbow_tol": elbow_tol,
                    "progress_threshold": progress_threshold,
                }
        return K_grid[-1], {
            "method": "elbow",
            "K_grid": K_grid,
            "r2_curve": r2_curve,
            "elbow_tol": elbow_tol,
            "progress_threshold": progress_threshold,
        }

    raise ValueError(f"unknown method: {method}")
