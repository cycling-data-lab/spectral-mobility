"""Spectral analysis primitives: eigendecomposition, IPR, level statistics."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh


def spectral_decomposition(
    L: np.ndarray | "object",
    k: int | None = None,
    *,
    which: Literal["smallest", "largest", "both"] = "smallest",
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of a real symmetric matrix.

    For dense input, uses :func:`numpy.linalg.eigh` (full spectrum).
    For sparse input with ``k`` specified, uses ARPACK
    :func:`scipy.sparse.linalg.eigsh` (partial spectrum).

    Parameters
    ----------
    L : (N, N) array-like or scipy.sparse matrix
        A symmetric matrix — typically the symmetric-normalised
        Laplacian from :func:`build_geographic_knn` →
        :func:`symmetric_normalised_laplacian`.
    k : int, optional
        Number of eigenpairs to compute.  If None, compute the full
        spectrum (only available for dense input).
    which : {"smallest", "largest", "both"}, default "smallest"
        Which end of the spectrum to compute when k is set.
        "smallest" returns the lowest-frequency eigenpairs (the
        relevant ones for the structural bound).

    Returns
    -------
    eigvals : (k,) np.ndarray
        Eigenvalues, ascending.
    eigvecs : (N, k) np.ndarray
        Corresponding eigenvectors (columns).
    """
    is_sparse = issparse(L)
    N = L.shape[0]
    if k is None:
        if is_sparse:
            raise ValueError(
                "k must be specified for sparse input; "
                "for the full spectrum, convert to dense first."
            )
        eigvals, eigvecs = np.linalg.eigh(np.asarray(L, dtype=float))
        return eigvals, eigvecs

    k = int(k)
    if k <= 0 or k > N - 1:
        raise ValueError(f"k must be in [1, N-1] = [1, {N-1}]; got {k}")

    if is_sparse:
        sigma_eigsh = "SA" if which == "smallest" else ("LA" if which == "largest" else None)
        if which == "both":
            ev_lo, vc_lo = eigsh(L, k=k, which="SA")
            ev_hi, vc_hi = eigsh(L, k=k, which="LA")
            order_lo = np.argsort(ev_lo)
            order_hi = np.argsort(ev_hi)
            return (
                np.concatenate([ev_lo[order_lo], ev_hi[order_hi]]),
                np.column_stack([vc_lo[:, order_lo], vc_hi[:, order_hi]]),
            )
        eigvals, eigvecs = eigsh(L, k=k, which=sigma_eigsh)
        order = np.argsort(eigvals)
        return eigvals[order], eigvecs[:, order]

    # dense: compute full, then slice
    full_vals, full_vecs = np.linalg.eigh(np.asarray(L, dtype=float))
    if which == "smallest":
        return full_vals[:k], full_vecs[:, :k]
    elif which == "largest":
        return full_vals[-k:], full_vecs[:, -k:]
    else:
        return (
            np.concatenate([full_vals[:k], full_vals[-k:]]),
            np.column_stack([full_vecs[:, :k], full_vecs[:, -k:]]),
        )


def inverse_participation_ratio(eigvecs: np.ndarray) -> np.ndarray:
    """Inverse Participation Ratio per eigenmode.

    For each column ψ of ``eigvecs``,
    ``IPR(ψ) = Σ_i ψ_i^4 / (Σ_i ψ_i^2)^2``.

    Interpretation
    --------------
    - ``IPR ≈ 1/N``: ψ is uniformly extended over all N nodes (the
      "metallic" / GOE regime).
    - ``IPR ≈ O(1)``: ψ is concentrated on O(1) nodes (the "insulating" /
      Poisson regime).
    - ``IPR = 1`` means ψ is entirely on a single node.

    Parameters
    ----------
    eigvecs : (N, K) np.ndarray

    Returns
    -------
    ipr : (K,) np.ndarray
    """
    psi2 = np.asarray(eigvecs, dtype=float) ** 2
    norm = psi2.sum(axis=0)
    norm = np.where(norm > 0, norm, 1.0)
    psi2 = psi2 / norm[None, :]
    return (psi2 ** 2).sum(axis=0)


def participation_ratio(eigvecs: np.ndarray) -> np.ndarray:
    """Participation ratio = 1 / (N * IPR), in (0, 1].

    PR ≈ 1 → fully extended, PR ≈ 1/N → fully localized.
    """
    N = eigvecs.shape[0]
    ipr = inverse_participation_ratio(eigvecs)
    return 1.0 / (N * ipr + 1e-30)


def level_spacing_ratios(eigvals: np.ndarray) -> np.ndarray:
    """Adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

    Diagnostic for the level statistics:
    - ⟨r⟩ ≈ 0.5295 → Gaussian Orthogonal Ensemble (extended regime)
    - ⟨r⟩ ≈ 0.3863 → Poisson (localized regime)

    Returns an array of length N-2.
    """
    eigvals = np.sort(np.asarray(eigvals, dtype=float))
    s = np.diff(eigvals)
    r = np.minimum(s[:-1], s[1:]) / (np.maximum(s[:-1], s[1:]) + 1e-30)
    return r
