"""Pairwise and multi-city comparison of spectral profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

if TYPE_CHECKING:
    from spectral_mobility.profile import CitySpectralProfile


@dataclass
class CityComparisonResult:
    """Pairwise comparison of two :class:`CitySpectralProfile` objects.

    Attributes
    ----------
    name_a, name_b : str
    n_a, n_b : int
    wasserstein_eigvals : float
        Wasserstein-1 distance between the two normalised eigenvalue
        distributions (eigenvalues mapped to [0, 1]).  0 = identical.
    wasserstein_ipr : float
        Wasserstein-1 distance between the IPR distributions, in log
        space (since IPR spans several orders of magnitude).  0 =
        identical.
    ks_eigvals : float
        Kolmogorov-Smirnov statistic on eigenvalue CDFs.
    ks_ipr : float
        Kolmogorov-Smirnov statistic on IPR CDFs.
    spectral_similarity : float
        ``exp(-α · combined_wasserstein)`` ∈ (0, 1].  1 = identical.
    """

    name_a: str
    name_b: str
    n_a: int
    n_b: int
    wasserstein_eigvals: float
    wasserstein_ipr: float
    ks_eigvals: float
    ks_ipr: float
    spectral_similarity: float

    def __repr__(self) -> str:
        return (
            f"CityComparisonResult({self.name_a!r} vs {self.name_b!r}: "
            f"similarity={self.spectral_similarity:.3f})"
        )


def _normalise_eigvals(eigvals: np.ndarray) -> np.ndarray:
    """Min-max rescale eigenvalues to [0, 1] so different-sized cities
    can be compared on the same axis."""
    eigvals = np.asarray(eigvals, dtype=float)
    lo, hi = float(eigvals.min()), float(eigvals.max())
    if hi - lo < 1e-12:
        return np.zeros_like(eigvals)
    return (eigvals - lo) / (hi - lo)


def compare_cities(
    profile_a: "CitySpectralProfile",
    profile_b: "CitySpectralProfile",
    *,
    alpha: float = 5.0,
) -> CityComparisonResult:
    """Compare two cities' spectral profiles.

    The result is symmetric in the two arguments.

    Parameters
    ----------
    profile_a, profile_b : CitySpectralProfile
    alpha : float, default 5.0
        Scale used in the ``spectral_similarity = exp(-α · combined_wd)``
        score.  Higher α penalises differences more steeply.

    Returns
    -------
    result : CityComparisonResult
    """
    ea = _normalise_eigvals(profile_a.eigvals)
    eb = _normalise_eigvals(profile_b.eigvals)
    wd_eig = float(wasserstein_distance(ea, eb))

    log_ipr_a = np.log10(profile_a.ipr + 1e-12)
    log_ipr_b = np.log10(profile_b.ipr + 1e-12)
    wd_ipr = float(wasserstein_distance(log_ipr_a, log_ipr_b))
    # Normalise log-IPR distance by its natural range (~3 decades)
    wd_ipr_norm = wd_ipr / 3.0

    ks_eig = float(ks_2samp(ea, eb).statistic)
    ks_ipr = float(ks_2samp(log_ipr_a, log_ipr_b).statistic)

    combined = 0.5 * wd_eig + 0.5 * wd_ipr_norm
    similarity = float(np.exp(-alpha * combined))

    return CityComparisonResult(
        name_a=profile_a.name,
        name_b=profile_b.name,
        n_a=profile_a.N,
        n_b=profile_b.N,
        wasserstein_eigvals=wd_eig,
        wasserstein_ipr=wd_ipr,
        ks_eigvals=ks_eig,
        ks_ipr=ks_ipr,
        spectral_similarity=similarity,
    )


def cross_city_similarity_matrix(
    profiles: Iterable["CitySpectralProfile"],
    *,
    alpha: float = 5.0,
) -> tuple[np.ndarray, list[str]]:
    """Pairwise spectral-similarity matrix over a list of profiles.

    Parameters
    ----------
    profiles : iterable of CitySpectralProfile
    alpha : float, default 5.0

    Returns
    -------
    similarity : (N, N) np.ndarray
        Symmetric matrix; diagonal = 1.  ``similarity[i, j]`` is the
        spectral similarity score between city i and city j.
    names : list of str
        Names of the cities, in the same order as the matrix rows.
    """
    profiles = list(profiles)
    n = len(profiles)
    if n < 2:
        raise ValueError("need at least 2 profiles to compute a similarity matrix")
    names = [p.name for p in profiles]
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = compare_cities(profiles[i], profiles[j], alpha=alpha)
            M[i, j] = M[j, i] = r.spectral_similarity
    return M, names
