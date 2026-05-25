"""Spectral applicability bound: ``R²_spec`` on encoder subspaces.

The structural applicability bound of Fossé & Pallares (2026) states
that the expected leave-node-out loss of any predictor in an
encoder class with column-span ``S`` is at least
``(1 - R²_spec(S, y)) · Var(y)``, where

    R²_spec(S, y) = ||P_S y||² / ||y||²

and ``P_S`` is the orthogonal projector onto ``S``.

This module exposes:

- :func:`r2_spec_subspace` — compute R²_spec for an arbitrary encoder
  matrix.
- :func:`spectral_bound` — compute the bound with and without
  augmentation by the top-K low-frequency Laplacian eigenvectors,
  returning a :class:`SpectralBoundResult` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralBoundResult:
    """Result of an augmented-spectral-bound computation.

    Attributes
    ----------
    r2_imd : float
        R²_spec ceiling on the bare encoder features (``X``) only.
    r2_spectral_only : float
        R²_spec ceiling on the top-K eigenvectors alone (no IMD).
    r2_augmented : float
        R²_spec ceiling on the augmented subspace ``[X | U_K]``.
    delta_r2 : float
        ``r2_augmented - r2_imd`` — the ceiling gain from augmentation.
    K : int
        Number of low-frequency eigenvectors used.
    n_imd_features : int
        Number of columns in the bare encoder ``X``.
    """

    r2_imd: float
    r2_spectral_only: float
    r2_augmented: float
    delta_r2: float
    K: int
    n_imd_features: int


def r2_spec_subspace(S: np.ndarray, y: np.ndarray) -> float:
    """Compute R²_spec = ||P_S y||² / ||y||² for an encoder ``S``.

    Implementation note: uses thin QR decomposition for numerical
    stability when ``S`` has near-collinear columns.

    Parameters
    ----------
    S : (N, p) np.ndarray
        Encoder matrix.  Columns need not be orthonormal.
    y : (N,) np.ndarray
        Target vector.  Should be centred (mean-subtracted) for
        ``R²_spec`` to be interpretable as a fraction of variance.

    Returns
    -------
    r2 : float
        ``R²_spec`` ∈ [0, 1].
    """
    S = np.asarray(S, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if S.shape[0] != y.shape[0]:
        raise ValueError(f"S has {S.shape[0]} rows but y has {y.shape[0]}")
    if S.size == 0 or S.shape[1] == 0:
        return 0.0
    # Drop zero columns (degenerate features)
    col_norms = np.linalg.norm(S, axis=0)
    keep = col_norms > 1e-12
    if not keep.any():
        return 0.0
    S = S[:, keep]
    Q, _ = np.linalg.qr(S)
    proj = Q @ (Q.T @ y)
    denom = float(y @ y)
    if denom <= 0:
        return 0.0
    return float(proj @ proj) / denom


def spectral_bound(
    eigvecs: np.ndarray,
    y: np.ndarray,
    *,
    encoder_features: np.ndarray | None = None,
    K: int = 16,
) -> SpectralBoundResult:
    """Compute the spectral bound with and without augmentation.

    Parameters
    ----------
    eigvecs : (N, M) np.ndarray
        Laplacian eigenvectors, columns ordered by ascending
        eigenvalue (low-frequency first).  Typically the output of
        :func:`spectral_decomposition`.
    y : (N,) np.ndarray
        Target signal.  Will be centred internally.
    encoder_features : (N, p) np.ndarray, optional
        Bare encoder feature matrix (e.g.\\ IMD-4).  If omitted,
        ``r2_imd`` is reported as 0 and only the spectral-only
        ceiling is computed.
    K : int, default 16
        Number of low-frequency eigenvectors to augment with.

    Returns
    -------
    result : SpectralBoundResult
    """
    eigvecs = np.asarray(eigvecs, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if eigvecs.shape[0] != y.shape[0]:
        raise ValueError(
            f"eigvecs has {eigvecs.shape[0]} rows but y has {y.shape[0]}"
        )
    K_eff = min(int(K), eigvecs.shape[1])

    y_centred = y - y.mean()

    if encoder_features is None:
        X = np.zeros((y.shape[0], 0))
        n_imd = 0
    else:
        X = np.asarray(encoder_features, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"encoder_features has {X.shape[0]} rows but y has {y.shape[0]}"
            )
        n_imd = X.shape[1]

    r2_imd = r2_spec_subspace(X, y_centred)
    Uk = eigvecs[:, :K_eff]
    r2_spec_only = r2_spec_subspace(Uk, y_centred)
    if X.shape[1] == 0:
        r2_aug = r2_spec_only
    else:
        S_aug = np.column_stack([X, Uk])
        r2_aug = r2_spec_subspace(S_aug, y_centred)

    return SpectralBoundResult(
        r2_imd=float(r2_imd),
        r2_spectral_only=float(r2_spec_only),
        r2_augmented=float(r2_aug),
        delta_r2=float(r2_aug - r2_imd),
        K=int(K_eff),
        n_imd_features=int(n_imd),
    )
