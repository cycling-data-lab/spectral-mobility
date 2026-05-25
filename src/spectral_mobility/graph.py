"""Graph construction utilities.

Two graph builders are provided:

- :func:`build_geographic_knn` — k-nearest-neighbour graph from
  geographic coordinates using the haversine distance.  This is the
  graph type used in the topological-localization-mobility paper at
  station scale and at commune scale.

- :func:`build_feature_knn` — k-nearest-neighbour graph from a generic
  feature matrix using Euclidean distance.  This is the graph type
  used in the materials-applicability-bound paper.

Both functions return a sparse symmetric weight matrix in
``scipy.sparse.csr_matrix`` form, with Gaussian-RBF edge weights at a
bandwidth ``sigma`` that is either user-supplied or auto-selected as
the median k-th-nearest-neighbour distance.

The :func:`symmetric_normalised_laplacian` function then builds
``L_sym = I - D^{-1/2} W D^{-1/2}`` from the weight matrix.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix, eye as sp_eye, issparse
from sklearn.neighbors import BallTree, NearestNeighbors

EARTH_RADIUS_METRES = 6_371_000.0


def haversine_distance_matrix(
    lat: np.ndarray, lng: np.ndarray
) -> np.ndarray:
    """Dense haversine distance matrix in metres.

    Use only for small N (< ~5,000); for larger panels use
    :func:`build_geographic_knn` which avoids materialising the full
    pairwise distance matrix.

    Parameters
    ----------
    lat, lng : np.ndarray of shape (N,)
        Latitude and longitude in decimal degrees.

    Returns
    -------
    D : np.ndarray of shape (N, N)
        Pairwise haversine distances in metres; ``D[i, i] = 0``.
    """
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    lat_r = np.deg2rad(lat)
    lng_r = np.deg2rad(lng)
    dphi = lat_r[:, None] - lat_r[None, :]
    dlam = lng_r[:, None] - lng_r[None, :]
    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlam / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METRES * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build_geographic_knn(
    lat: np.ndarray,
    lng: np.ndarray,
    k: int = 10,
    sigma: float | Literal["auto"] | None = "auto",
) -> tuple[csr_matrix, float]:
    """Build a k-NN graph from geographic coordinates with Gaussian-RBF
    edge weights based on haversine distance.

    Parameters
    ----------
    lat, lng : np.ndarray of shape (N,)
        Latitude and longitude in decimal degrees.
    k : int, default 10
        Number of nearest neighbours per node (excluding self).
    sigma : float, ``"auto"``, or None, default ``"auto"``
        Bandwidth of the Gaussian-RBF edge weights, in metres.
        If ``"auto"`` (or None), σ is set to the median k-th nearest
        neighbour distance, matching the recipe of d03 of the
        mobility-applicability-bound paper.

    Returns
    -------
    W : scipy.sparse.csr_matrix of shape (N, N)
        Symmetrised k-NN weight matrix.
    sigma_used : float
        The σ value used (in metres), useful when ``sigma="auto"``.
    """
    lat = np.asarray(lat, dtype=float)
    lng = np.asarray(lng, dtype=float)
    if lat.shape != lng.shape or lat.ndim != 1:
        raise ValueError("lat and lng must be 1-D arrays of the same length")
    N = lat.size
    if N < 2:
        raise ValueError("need at least 2 points to build a k-NN graph")
    k_eff = min(int(k), N - 1)

    coords_rad = np.deg2rad(np.column_stack([lat, lng]))
    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, idx = tree.query(coords_rad, k=k_eff + 1)
    dist_m = dist_rad[:, 1:] * EARTH_RADIUS_METRES
    idx = idx[:, 1:]

    if sigma in (None, "auto"):
        sigma_used = float(np.median(dist_m[:, -1]))
    else:
        sigma_used = float(sigma)
    if sigma_used <= 0:
        raise ValueError("sigma must be positive")

    rows = np.repeat(np.arange(N), k_eff)
    cols = idx.ravel()
    data = np.exp(-(dist_m.ravel() ** 2) / (2.0 * sigma_used ** 2))
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    return (0.5 * (W + W.T)).tocsr(), sigma_used


def build_feature_knn(
    features: np.ndarray,
    k: int = 10,
    sigma: float | Literal["auto"] | None = "auto",
) -> tuple[csr_matrix, float]:
    """Build a k-NN graph from a feature matrix with Gaussian-RBF
    edge weights based on Euclidean distance.

    Parameters
    ----------
    features : np.ndarray of shape (N, d)
        Standardised feature matrix.  Standardise per-column before
        calling this function for best results.
    k : int, default 10
        Number of nearest neighbours per node (excluding self).
    sigma : float, ``"auto"``, or None, default ``"auto"``
        Bandwidth.  If ``"auto"``, set to the median k-th-NN distance.

    Returns
    -------
    W : scipy.sparse.csr_matrix
    sigma_used : float
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be a 2-D array")
    N = X.shape[0]
    if N < 2:
        raise ValueError("need at least 2 points to build a k-NN graph")
    k_eff = min(int(k), N - 1)

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(X)
    dist, idx = nn.kneighbors(X)
    dist, idx = dist[:, 1:], idx[:, 1:]
    if sigma in (None, "auto"):
        sigma_used = float(np.median(dist[:, -1]))
    else:
        sigma_used = float(sigma)
    if sigma_used <= 0:
        raise ValueError("sigma must be positive")

    rows = np.repeat(np.arange(N), k_eff)
    cols = idx.ravel()
    data = np.exp(-(dist.ravel() ** 2) / (2.0 * sigma_used ** 2))
    W = csr_matrix((data, (rows, cols)), shape=(N, N))
    return (0.5 * (W + W.T)).tocsr(), sigma_used


def unnormalised_laplacian(
    W: csr_matrix | np.ndarray, *, dense: bool | None = None
) -> np.ndarray | csr_matrix:
    """Unnormalised graph Laplacian ``L = D − W``.

    Use this when computing the Dirichlet energy of a node-valued
    *volume* (counts, intensities not divided by anything), because
    ``y^T L y = Σ_{(i,j) ∈ E} w_ij (y_i − y_j)²`` is then the natural
    discrete gradient norm of ``y``.  For per-degree intensities or
    when comparing graphs of widely varying density, prefer
    :func:`symmetric_normalised_laplacian` (its Dirichlet form
    compares ``y_i / √d_i`` instead).

    Parameters
    ----------
    W : (N, N) array-like or scipy.sparse matrix
        Edge weight matrix; should be symmetric.
    dense : bool or None, default None
        Same convention as :func:`symmetric_normalised_laplacian`.

    Returns
    -------
    L : np.ndarray or scipy.sparse.csr_matrix
    """
    if issparse(W):
        N = W.shape[0]
        deg = np.asarray(W.sum(axis=1)).ravel()
    else:
        W = np.asarray(W, dtype=float)
        N = W.shape[0]
        deg = W.sum(axis=1)
    if dense is None:
        dense = N <= 5000
    if dense:
        W_d = W.toarray() if issparse(W) else W
        L = np.diag(deg) - W_d
        return 0.5 * (L + L.T)
    from scipy.sparse import diags
    D = diags(deg)
    return (D - W).tocsr()


def symmetric_normalised_laplacian(
    W: csr_matrix | np.ndarray, *, dense: bool | None = None
) -> np.ndarray | csr_matrix:
    """Symmetric-normalised Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``.

    Parameters
    ----------
    W : array-like or scipy.sparse matrix of shape (N, N)
        Edge weight matrix; should be symmetric.
    dense : bool or None, default None
        If True, return a dense ``np.ndarray``.  If False, return a
        sparse ``csr_matrix``.  If None, return dense when N ≤ 5000
        (eigendecomposition will need a dense matrix anyway) and
        sparse otherwise.

    Returns
    -------
    L_sym : np.ndarray or scipy.sparse.csr_matrix
    """
    if issparse(W):
        N = W.shape[0]
        deg = np.asarray(W.sum(axis=1)).ravel()
    else:
        W = np.asarray(W, dtype=float)
        N = W.shape[0]
        deg = W.sum(axis=1)
    deg_safe = np.maximum(deg, 1e-12)
    Dinv2 = 1.0 / np.sqrt(deg_safe)

    if dense is None:
        dense = N <= 5000

    if dense:
        if issparse(W):
            W_d = W.toarray()
        else:
            W_d = W
        L = np.eye(N) - (W_d * Dinv2[:, None]) * Dinv2[None, :]
        return 0.5 * (L + L.T)  # symmetrise for numerical stability
    else:
        from scipy.sparse import diags

        Dinv2_diag = diags(Dinv2)
        L = sp_eye(N, format="csr") - Dinv2_diag @ W @ Dinv2_diag
        return ((L + L.T) * 0.5).tocsr()
