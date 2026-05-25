"""Multi-scale spectral similarity: per-frequency-band comparisons.

A single Wasserstein-1 number on the full eigenvalue distribution
collapses three distinct kinds of urban similarity:

  • LOW frequency (the bottom 10-20 % of the spectrum) carries
    global geometric structure — the city's overall shape, the
    coarse split into clusters.
  • MID frequency (20 % - 50 %) carries regional / neighbourhood
    structure.
  • HIGH frequency (the top half of the spectrum) is noise-like
    local connectivity.

Two cities can be similar in low frequency (both compact and
roughly the same size) yet very different in mid frequency (one
mono-centric, one poly-centric).  This module decomposes the
similarity by band.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from spectral_mobility.profile import CitySpectralProfile


DEFAULT_BANDS = [(0.0, 0.10), (0.10, 0.50), (0.50, 1.00)]
DEFAULT_BAND_NAMES = ["low", "mid", "high"]


def _band_eigvals(eigvals: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Return the eigenvalues whose RANK falls in the given quantile band."""
    n = len(eigvals)
    lo_idx = int(np.floor(band[0] * n))
    hi_idx = int(np.ceil(band[1] * n))
    hi_idx = max(hi_idx, lo_idx + 1)  # guarantee non-empty
    return np.sort(eigvals)[lo_idx:hi_idx]


def _band_ipr(ipr: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Same logic on IPR, in spectral order."""
    n = len(ipr)
    lo_idx = int(np.floor(band[0] * n))
    hi_idx = int(np.ceil(band[1] * n))
    hi_idx = max(hi_idx, lo_idx + 1)
    return ipr[lo_idx:hi_idx]


def multiscale_similarity(
    profile_a: "CitySpectralProfile",
    profile_b: "CitySpectralProfile",
    *,
    bands: list[tuple[float, float]] | None = None,
    band_names: list[str] | None = None,
    alpha: float = 5.0,
) -> dict[str, float]:
    """Per-band Wasserstein-based spectral similarity.

    For each band ``(q_lo, q_hi)`` in ``bands``, the eigenvalues of
    each profile restricted to that quantile range are min-max
    rescaled to ``[0, 1]`` and compared via Wasserstein-1.  The
    similarity in the band is ``exp(-alpha · wasserstein)``.

    Parameters
    ----------
    profile_a, profile_b : CitySpectralProfile
    bands : list of (float, float), default ``[(0, 0.1), (0.1, 0.5), (0.5, 1)]``
        Quantile boundaries of the bands.
    band_names : list of str, default ``["low", "mid", "high"]``
    alpha : float, default 5.0

    Returns
    -------
    out : dict
        Maps band name → similarity score in ``(0, 1]``.  Also
        returns the raw Wasserstein distances under
        ``"wasserstein_<name>"``.
    """
    bands = bands or DEFAULT_BANDS
    band_names = band_names or DEFAULT_BAND_NAMES
    if len(bands) != len(band_names):
        raise ValueError("bands and band_names must have the same length")

    out: dict[str, float] = {}
    for name, band in zip(band_names, bands):
        ea = _band_eigvals(profile_a.eigvals, band)
        eb = _band_eigvals(profile_b.eigvals, band)
        # Rescale each band to [0, 1] independently so wavelengths align
        def _rescale(x):
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-12:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        ea_n = _rescale(ea)
        eb_n = _rescale(eb)
        wd = float(wasserstein_distance(ea_n, eb_n))
        out[f"similarity_{name}"] = float(np.exp(-alpha * wd))
        out[f"wasserstein_{name}"] = wd
    return out


def multiscale_similarity_matrix(
    profiles: list["CitySpectralProfile"],
    *,
    bands: list[tuple[float, float]] | None = None,
    band_names: list[str] | None = None,
    alpha: float = 5.0,
) -> dict[str, np.ndarray]:
    """Pairwise multi-band similarity matrix.

    Returns one (n, n) matrix per band, plus the list of city names.
    Diagonal is 1.
    """
    bands = bands or DEFAULT_BANDS
    band_names = band_names or DEFAULT_BAND_NAMES
    n = len(profiles)
    if n < 2:
        raise ValueError("need at least 2 profiles")
    names = [p.name for p in profiles]
    matrices = {name: np.eye(n) for name in band_names}
    for i in range(n):
        for j in range(i + 1, n):
            r = multiscale_similarity(
                profiles[i], profiles[j], bands=bands,
                band_names=band_names, alpha=alpha,
            )
            for name in band_names:
                s = r[f"similarity_{name}"]
                matrices[name][i, j] = matrices[name][j, i] = s
    return {"matrices": matrices, "names": names, "band_names": band_names}
