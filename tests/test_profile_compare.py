"""Tests for CitySpectralProfile + comparison."""

import numpy as np
import pytest

from spectral_mobility import (
    CityComparisonResult,
    CitySpectralProfile,
    compare_cities,
    cross_city_similarity_matrix,
)


def _city(name, N=50, seed=0, scatter=0.005):
    """Helper: build a synthetic city profile."""
    rng = np.random.default_rng(seed)
    centres = np.array([[48.85, 2.34], [48.86, 2.36], [48.84, 2.45]])
    assign = rng.choice(3, size=N, p=[0.5, 0.4, 0.1])
    lat = centres[assign, 0] + rng.normal(0, scatter, size=N)
    lng = centres[assign, 1] + rng.normal(0, scatter, size=N)
    X = rng.standard_normal((N, 4))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (lat - lat.mean()) * 100 + 2.0 * (assign == 1) + rng.normal(0, 0.5, size=N)
    return CitySpectralProfile.from_coords(
        name=name, lat=lat, lng=lng, features=X, target=y, k_nn=5
    )


def test_profile_basic_construction():
    p = _city("Paris", N=50)
    assert p.name == "Paris"
    assert p.N == 50
    assert p.eigvals.shape == (50,)
    assert p.eigvecs.shape == (50, 50)
    assert p.ipr.shape == (50,)
    assert p.pr.shape == (50,)
    assert 0 <= p.mean_ipr <= 1
    assert 0 <= p.extended_fraction <= 1


def test_profile_constructor_coords_only():
    rng = np.random.default_rng(0)
    coords = rng.uniform(48.8, 48.9, size=(30, 2))
    p = CitySpectralProfile.from_coords(name="Test", coords=coords, k_nn=4)
    assert p.N == 30


def test_profile_constructor_requires_lat_lng_or_coords():
    with pytest.raises(ValueError):
        CitySpectralProfile.from_coords(name="X")


def test_profile_predictability_ceiling():
    p = _city("Paris", N=60)
    res = p.predictability_ceiling(K=8)
    assert 0 <= res.r2_imd <= 1
    assert 0 <= res.r2_augmented <= 1
    assert res.r2_augmented >= res.r2_imd - 1e-9


def test_profile_ceiling_requires_target():
    rng = np.random.default_rng(0)
    coords = rng.uniform(48.8, 48.9, size=(20, 2))
    p = CitySpectralProfile.from_coords(name="No target", coords=coords, k_nn=4)
    with pytest.raises(ValueError):
        p.predictability_ceiling(K=4)


def test_profile_bottleneck_zones():
    p = _city("Paris", N=80)
    zones = p.bottleneck_zones(n=3, mass_threshold=0.1)
    assert len(zones) == 3
    for z in zones:
        assert "mode_idx" in z
        assert "eigval" in z
        assert "ipr" in z
        assert "nodes" in z
        assert "coords" in z
        assert "mass" in z
        # nodes should be ordered by |ψ|² descending
        assert (z["mass"][:-1] >= z["mass"][1:]).all()


def test_profile_summary():
    p = _city("Lyon", N=40)
    s = p.summary()
    expected_keys = {"name", "N", "sigma", "mean_ipr", "extended_fraction",
                     "mean_level_spacing_r", "R2_imd", "R2_augmented_K16"}
    assert expected_keys.issubset(s.keys())
    assert s["name"] == "Lyon"
    assert s["N"] == 40


def test_compare_cities_symmetric():
    p1 = _city("A", N=50, seed=1)
    p2 = _city("B", N=50, seed=2)
    r12 = compare_cities(p1, p2)
    r21 = compare_cities(p2, p1)
    # Symmetric metrics
    assert abs(r12.wasserstein_eigvals - r21.wasserstein_eigvals) < 1e-9
    assert abs(r12.spectral_similarity - r21.spectral_similarity) < 1e-9


def test_compare_cities_self_similarity():
    p = _city("Solo", N=50, seed=0)
    r = compare_cities(p, p)
    # Identical → maximum similarity
    assert r.wasserstein_eigvals < 1e-9
    assert r.spectral_similarity > 0.99


def test_compare_cities_returns_result_dataclass():
    p1 = _city("A", N=40, seed=10)
    p2 = _city("B", N=40, seed=20)
    r = compare_cities(p1, p2)
    assert isinstance(r, CityComparisonResult)
    assert r.name_a == "A" and r.name_b == "B"
    assert r.n_a == 40 and r.n_b == 40
    assert 0 < r.spectral_similarity <= 1


def test_cross_city_similarity_matrix():
    cities = [_city(f"C{i}", N=40, seed=i) for i in range(4)]
    M, names = cross_city_similarity_matrix(cities)
    assert M.shape == (4, 4)
    assert names == ["C0", "C1", "C2", "C3"]
    # Symmetric
    np.testing.assert_allclose(M, M.T, atol=1e-9)
    # Diagonal = 1
    np.testing.assert_allclose(np.diag(M), 1.0)
    # Entries in [0, 1]
    assert M.min() >= 0 and M.max() <= 1


def test_cross_city_requires_at_least_two():
    p = _city("Solo", N=30)
    with pytest.raises(ValueError):
        cross_city_similarity_matrix([p])
