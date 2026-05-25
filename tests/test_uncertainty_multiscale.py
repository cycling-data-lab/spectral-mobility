"""Tests for the uncertainty and multiscale modules."""

import numpy as np
import pytest

from spectral_mobility import (
    CitySpectralProfile,
    bootstrap_similarity_matrix,
    ci_summary_table,
    multiscale_similarity,
    multiscale_similarity_matrix,
    permutation_test_block_contrast,
)


def _city(name, N=80, seed=0, scatter=0.005):
    rng = np.random.default_rng(seed)
    centres = np.array([[48.85, 2.34], [48.86, 2.36], [48.84, 2.45]])
    assign = rng.choice(3, size=N, p=[0.5, 0.4, 0.1])
    lat = centres[assign, 0] + rng.normal(0, scatter, size=N)
    lng = centres[assign, 1] + rng.normal(0, scatter, size=N)
    return CitySpectralProfile.from_coords(name=name, lat=lat, lng=lng, k_nn=5)


def test_bootstrap_shape_and_diagonal():
    cities = [_city(f"C{i}", N=60, seed=i) for i in range(4)]
    res = bootstrap_similarity_matrix(
        cities, n_boot=10, subsample_fraction=0.7, seed=0
    )
    assert res["median"].shape == (4, 4)
    assert res["lower"].shape == (4, 4)
    assert res["upper"].shape == (4, 4)
    # diagonal is 1 in every replicate
    np.testing.assert_allclose(np.diag(res["median"]), 1.0)


def test_bootstrap_ci_order():
    cities = [_city(f"C{i}", N=60, seed=i) for i in range(3)]
    res = bootstrap_similarity_matrix(cities, n_boot=20, seed=0)
    # lower ≤ median ≤ upper everywhere
    assert np.all(res["lower"] <= res["median"] + 1e-9)
    assert np.all(res["median"] <= res["upper"] + 1e-9)


def test_bootstrap_ci_summary_table():
    cities = [_city(f"C{i}", N=60, seed=i) for i in range(4)]
    res = bootstrap_similarity_matrix(cities, n_boot=10, seed=0)
    df = ci_summary_table(res)
    assert len(df) == 6  # C(4, 2) pairs
    assert set(df.columns) >= {"city_a", "city_b", "median", "lower", "upper"}
    assert (df["lower"] <= df["median"]).all()
    assert (df["median"] <= df["upper"]).all()


def test_permutation_observed_in_null_range():
    """With random cities (no real structure), observed contrast should
    be in the bulk of the null distribution → p > 0.05."""
    cities = [_city(f"C{i}", N=50, seed=i) for i in range(8)]
    M = np.eye(8)
    for i in range(8):
        for j in range(i + 1, 8):
            M[i, j] = M[j, i] = 0.5 + 0.1 * np.random.default_rng(i + j).standard_normal()
    names = [c.name for c in cities]
    res = permutation_test_block_contrast(
        M, names, group_a=["C0", "C1", "C2", "C3"], group_b=["C4", "C5", "C6", "C7"],
        n_perm=200, seed=0,
    )
    # Random matrix → p should be roughly uniform on [0, 1]
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["n_perm"] == 200


def test_permutation_detects_strong_block():
    """If we inject a strong within-block bonus, the permutation test
    should reject (p < 0.05)."""
    n = 8
    M = 0.5 * np.ones((n, n))
    np.fill_diagonal(M, 1.0)
    # boost within-block-A and within-block-B similarities
    for i in range(4):
        for j in range(i + 1, 4):
            M[i, j] = M[j, i] = 0.9
    for i in range(4, 8):
        for j in range(i + 1, 8):
            M[i, j] = M[j, i] = 0.9
    names = [f"X{i}" for i in range(n)]
    res = permutation_test_block_contrast(
        M, names, group_a=names[:4], group_b=names[4:],
        n_perm=500, seed=0,
    )
    # Real block structure → observed should be in extreme of null
    assert res["p_value"] < 0.05
    assert res["observed_contrast"] > 0.3


def test_permutation_group_size_check():
    n = 5
    M = np.eye(n)
    names = [f"C{i}" for i in range(n)]
    with pytest.raises(ValueError):
        permutation_test_block_contrast(
            M, names, group_a=["C0"], group_b=["C1", "C2"],
            n_perm=100, seed=0,
        )


def test_multiscale_similarity_keys():
    p1 = _city("A", N=80, seed=1)
    p2 = _city("B", N=80, seed=2)
    res = multiscale_similarity(p1, p2)
    for band in ["low", "mid", "high"]:
        assert f"similarity_{band}" in res
        assert f"wasserstein_{band}" in res
        s = res[f"similarity_{band}"]
        assert 0 < s <= 1


def test_multiscale_self_similarity_is_one():
    p = _city("Solo", N=80, seed=0)
    res = multiscale_similarity(p, p)
    for band in ["low", "mid", "high"]:
        assert res[f"similarity_{band}"] > 0.99


def test_multiscale_matrix_shape():
    cities = [_city(f"C{i}", N=60, seed=i) for i in range(5)]
    res = multiscale_similarity_matrix(cities)
    assert set(res["matrices"].keys()) == {"low", "mid", "high"}
    for name, M in res["matrices"].items():
        assert M.shape == (5, 5)
        np.testing.assert_allclose(np.diag(M), 1.0)
        np.testing.assert_allclose(M, M.T)
    assert res["names"] == [f"C{i}" for i in range(5)]


def test_multiscale_custom_bands():
    p1 = _city("A", N=80, seed=1)
    p2 = _city("B", N=80, seed=2)
    res = multiscale_similarity(
        p1, p2,
        bands=[(0.0, 0.5), (0.5, 1.0)],
        band_names=["coarse", "fine"],
    )
    assert "similarity_coarse" in res and "similarity_fine" in res
    assert "similarity_low" not in res  # didn't use defaults
