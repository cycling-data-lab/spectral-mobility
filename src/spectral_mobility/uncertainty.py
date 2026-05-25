"""Uncertainty quantification: bootstrap CIs and permutation tests.

Two utilities for putting confidence intervals and p-values around
spectral similarity claims:

- :func:`bootstrap_similarity_matrix` — repeatedly subsample stations
  within each city and recompute the pairwise similarity matrix.
  Returns (median, lower, upper) bootstrap estimates per pair.

- :func:`permutation_test_block_contrast` — test whether the
  difference in mean within-block vs between-block similarity is
  significantly larger than would be obtained by permuting block
  labels at random.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from spectral_mobility.compare import (
    CityComparisonResult,
    compare_cities,
    cross_city_similarity_matrix,
)

if TYPE_CHECKING:
    from spectral_mobility.profile import CitySpectralProfile


def _subsample_profile(
    profile: "CitySpectralProfile",
    fraction: float,
    rng: np.random.Generator,
) -> "CitySpectralProfile":
    """Return a CitySpectralProfile rebuilt on a random subsample of nodes."""
    from spectral_mobility.profile import CitySpectralProfile

    N = profile.N
    n_keep = max(30, int(round(fraction * N)))
    n_keep = min(n_keep, N)
    if n_keep == N:
        return profile
    idx = rng.choice(N, size=n_keep, replace=False)
    coords_sub = profile.coords[idx]
    features_sub = profile.features[idx] if profile.features is not None else None
    target_sub = profile.target[idx] if profile.target is not None else None
    return CitySpectralProfile.from_coords(
        name=profile.name,
        coords=coords_sub,
        features=features_sub,
        target=target_sub,
        k_nn=profile.k_nn,
        sigma="auto",
        graph_type=profile.graph_type,
    )


def bootstrap_similarity_matrix(
    profiles: list["CitySpectralProfile"],
    *,
    n_boot: int = 200,
    subsample_fraction: float = 0.7,
    alpha: float = 5.0,
    ci_level: float = 0.95,
    seed: int = 42,
    progress: bool = False,
) -> dict:
    """Bootstrap a pairwise spectral-similarity matrix.

    For each bootstrap replicate, every profile in ``profiles`` is
    rebuilt on a random subsample of its nodes; the pairwise
    similarity matrix is then recomputed.  Across replicates we
    report the median and a 1-``alpha`` central confidence interval
    per pair.

    Parameters
    ----------
    profiles : list of CitySpectralProfile
    n_boot : int, default 200
    subsample_fraction : float in (0, 1], default 0.7
        Fraction of each city's stations kept per replicate.
    alpha : float, default 5.0
        Decay rate in the similarity score.  See
        :func:`compare_cities`.
    ci_level : float in (0, 1), default 0.95
    seed : int, default 42
    progress : bool, default False
        If True, print one dot per 20 replicates.

    Returns
    -------
    result : dict
        ``{"median": (n, n) np.ndarray, "lower": ..., "upper": ...,
            "names": list[str], "raw": (n_boot, n, n) np.ndarray,
            "n_boot": int, "ci_level": float}``.
    """
    rng = np.random.default_rng(seed)
    names = [p.name for p in profiles]
    n_cities = len(profiles)
    if n_cities < 2:
        raise ValueError("need at least 2 profiles for the bootstrap")

    raw = np.empty((n_boot, n_cities, n_cities), dtype=float)
    for b in range(n_boot):
        boot_profiles = [_subsample_profile(p, subsample_fraction, rng)
                         for p in profiles]
        M_b, _ = cross_city_similarity_matrix(boot_profiles, alpha=alpha)
        raw[b] = M_b
        if progress and (b + 1) % 20 == 0:
            print(".", end="", flush=True)
    if progress:
        print()

    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    median = np.median(raw, axis=0)
    lower = np.quantile(raw, lo_q, axis=0)
    upper = np.quantile(raw, hi_q, axis=0)

    return {
        "median": median,
        "lower": lower,
        "upper": upper,
        "names": names,
        "raw": raw,
        "n_boot": int(n_boot),
        "subsample_fraction": float(subsample_fraction),
        "ci_level": float(ci_level),
    }


def _block_contrast(
    M: np.ndarray, group_a: list[int], group_b: list[int]
) -> float:
    """Mean within-block similarity minus mean between-block similarity."""
    if len(group_a) < 2 or len(group_b) < 2:
        return float("nan")
    within_a = [M[i, j] for i in group_a for j in group_a if i < j]
    within_b = [M[i, j] for i in group_b for j in group_b if i < j]
    between = [M[i, j] for i in group_a for j in group_b]
    if not within_a or not within_b or not between:
        return float("nan")
    return 0.5 * (np.mean(within_a) + np.mean(within_b)) - float(np.mean(between))


def permutation_test_block_contrast(
    M: np.ndarray,
    names: list[str],
    group_a: list[str],
    group_b: list[str],
    *,
    n_perm: int = 10_000,
    seed: int = 42,
) -> dict:
    """Permutation test of the block-contrast statistic.

    For a pre-specified partition of city names into two groups,
    compute the observed contrast::

        contrast = (within-A + within-B) / 2  −  between-AB

    and test the null hypothesis that the city labels are
    exchangeable by repeatedly shuffling the assignment of which
    cities belong to which group.

    Parameters
    ----------
    M : (n, n) np.ndarray
        Pairwise similarity matrix (symmetric, diagonal = 1).
    names : list of str, length ``n``
        City names in row/column order of ``M``.
    group_a, group_b : list of str
        Pre-specified groups.  Must contain names that exist in
        ``names``; unknown names are ignored.
    n_perm : int, default 10_000
    seed : int, default 42

    Returns
    -------
    result : dict
        Contains ``observed_contrast``, ``null_mean``, ``null_std``,
        ``p_value`` (one-tailed, ``contrast >= observed``),
        ``null_distribution`` (length ``n_perm``).
    """
    rng = np.random.default_rng(seed)
    M = np.asarray(M)
    n = M.shape[0]

    set_a = set(group_a)
    set_b = set(group_b)
    idx_a = [i for i, nm in enumerate(names) if nm in set_a]
    idx_b = [i for i, nm in enumerate(names) if nm in set_b]
    if len(idx_a) < 2 or len(idx_b) < 2:
        raise ValueError("each group must contain at least 2 known names")
    pool = idx_a + idx_b
    n_a = len(idx_a)

    observed = _block_contrast(M, idx_a, idx_b)

    null = np.empty(n_perm)
    for p in range(n_perm):
        shuffled = rng.permutation(pool)
        a_perm = list(shuffled[:n_a])
        b_perm = list(shuffled[n_a:])
        null[p] = _block_contrast(M, a_perm, b_perm)

    null_mean = float(np.mean(null))
    null_std = float(np.std(null))
    # one-tailed (we hypothesised contrast > null)
    p_value = float((null >= observed).mean())
    return {
        "observed_contrast": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": p_value,
        "n_perm": int(n_perm),
        "null_distribution": null,
        "n_in_group_a": n_a,
        "n_in_group_b": len(idx_b),
    }


def ci_summary_table(boot_result: dict) -> pd.DataFrame:
    """Return a tidy DataFrame of pairwise similarities with CIs."""
    names = boot_result["names"]
    med = boot_result["median"]
    lo = boot_result["lower"]
    hi = boot_result["upper"]
    rows = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "city_a": names[i],
                    "city_b": names[j],
                    "median": float(med[i, j]),
                    "lower": float(lo[i, j]),
                    "upper": float(hi[i, j]),
                    "ci_width": float(hi[i, j] - lo[i, j]),
                }
            )
    return pd.DataFrame(rows).sort_values("median", ascending=False).reset_index(drop=True)
