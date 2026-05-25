"""Diagnostics: identify localized modes and their geographic footprint.

These utilities answer the operational question:
*where are the structural bottlenecks of a given mobility graph?*

A bottleneck is a region of the graph where a low-eigenvalue
eigenvector has concentrated mass.  In the topological-localization
framework, these are the zones that cap the predictability of the
whole network.  Identifying them is a prerequisite for any
infrastructure intervention aimed at lifting the bound.
"""

from __future__ import annotations

import numpy as np

from spectral_mobility.spectral import inverse_participation_ratio


def bottleneck_modes(
    eigvecs: np.ndarray,
    *,
    n_top: int = 5,
    require_low_frequency: int | None = None,
) -> np.ndarray:
    """Indices of the most-localized eigenmodes.

    Parameters
    ----------
    eigvecs : (N, K) np.ndarray
        Laplacian eigenvectors, columns ordered ascending in
        eigenvalue.
    n_top : int, default 5
        Number of localized modes to return (by descending IPR).
    require_low_frequency : int or None
        If set, restricts the search to the first
        ``require_low_frequency`` eigenmodes.  This isolates
        bottlenecks that affect the bound (which depends on
        low-frequency modes) from generic high-frequency
        localization which always exists.

    Returns
    -------
    indices : (n_top,) np.ndarray
        Column indices into ``eigvecs`` of the most-localized modes.
    """
    eigvecs = np.asarray(eigvecs, dtype=float)
    if require_low_frequency is not None:
        upper = min(int(require_low_frequency), eigvecs.shape[1])
        ipr = inverse_participation_ratio(eigvecs[:, :upper])
    else:
        ipr = inverse_participation_ratio(eigvecs)
    order = np.argsort(-ipr)  # descending IPR
    return order[: int(n_top)]


def locate_bottleneck_nodes(
    eigvec: np.ndarray,
    *,
    mass_threshold: float = 0.05,
    min_mass: float = 1e-3,
) -> np.ndarray:
    """Identify the nodes carrying the localized mass of a single
    eigenvector.

    Parameters
    ----------
    eigvec : (N,) np.ndarray
        A single (column) eigenvector.
    mass_threshold : float in (0, 1), default 0.05
        Fraction of the eigenvector's ℓ²-mass to accumulate before
        stopping.  Default 5%: the function returns the top nodes
        until they collectively carry 95% of the |ψ|² mass.

        Wait — actually the convention is the OTHER way: we collect
        nodes from the largest |ψ_i|² downward until the cumulative
        mass crosses ``1 - mass_threshold``.  In other words,
        ``mass_threshold = 0.05`` means "return the nodes that
        account for 95% of the localized mass".
    min_mass : float, default 1e-3
        Per-node minimum |ψ_i|² to include; prevents very tiny tails
        from leaking into the bottleneck list.

    Returns
    -------
    indices : np.ndarray
        Node indices, sorted by |ψ_i|² descending.
    """
    psi = np.asarray(eigvec, dtype=float).ravel()
    mass = psi ** 2
    total = mass.sum()
    if total <= 0:
        return np.array([], dtype=int)
    mass = mass / total
    order = np.argsort(-mass)
    cumulative = np.cumsum(mass[order])
    cap = 1.0 - float(mass_threshold)
    n_keep = int(np.searchsorted(cumulative, cap) + 1)
    keep = order[:n_keep]
    keep = keep[mass[keep] >= float(min_mass)]
    return keep


def extended_subspace_fraction(
    eigvecs: np.ndarray, *, ipr_threshold_factor: float = 5.0
) -> float:
    """Fraction of eigenmodes that are "extended" (low-IPR).

    Definition: a mode is extended iff ``IPR < ipr_threshold_factor / N``.
    For uniformly extended states ``IPR = 1/N``, so the default
    ``factor = 5`` admits modes that are extended up to a 5x mass
    concentration.

    Returns the fraction in [0, 1].
    """
    eigvecs = np.asarray(eigvecs, dtype=float)
    N = eigvecs.shape[0]
    ipr = inverse_participation_ratio(eigvecs)
    threshold = float(ipr_threshold_factor) / N
    return float((ipr < threshold).mean())
