"""Tests for spectral_mobility.diagnostics."""

import numpy as np

from spectral_mobility import (
    bottleneck_modes,
    build_geographic_knn,
    extended_subspace_fraction,
    locate_bottleneck_nodes,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)


def _setup(N=30, seed=0):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(48.8, 48.9, size=N)
    lng = rng.uniform(2.3, 2.4, size=N)
    W, _ = build_geographic_knn(lat, lng, k=4)
    L = symmetric_normalised_laplacian(W, dense=True)
    eigvals, eigvecs = spectral_decomposition(L)
    return eigvecs, rng, N


def test_bottleneck_modes_returns_n_top():
    eigvecs, rng, N = _setup()
    idx = bottleneck_modes(eigvecs, n_top=5)
    assert idx.shape == (5,)
    # All distinct
    assert len(np.unique(idx)) == 5


def test_bottleneck_modes_orders_by_descending_ipr():
    from spectral_mobility import inverse_participation_ratio

    eigvecs, _, _ = _setup()
    idx = bottleneck_modes(eigvecs, n_top=10)
    ipr = inverse_participation_ratio(eigvecs)
    # IPR should be non-increasing along idx
    sorted_ipr = ipr[idx]
    for a, b in zip(sorted_ipr, sorted_ipr[1:]):
        assert a >= b - 1e-12


def test_locate_bottleneck_nodes_localized_state():
    # Eigenvector concentrated on a single node should return that node
    N = 30
    psi = np.zeros(N)
    psi[7] = 1.0
    nodes = locate_bottleneck_nodes(psi, mass_threshold=0.05)
    assert nodes.shape == (1,)
    assert nodes[0] == 7


def test_locate_bottleneck_nodes_uniform_state():
    # Uniform eigenvector should require nearly all nodes to reach 95%
    N = 20
    psi = np.ones(N) / np.sqrt(N)
    nodes = locate_bottleneck_nodes(psi, mass_threshold=0.05)
    assert len(nodes) >= int(0.9 * N)


def test_locate_bottleneck_nodes_zero_input():
    psi = np.zeros(10)
    nodes = locate_bottleneck_nodes(psi)
    assert nodes.shape == (0,)


def test_extended_subspace_fraction_uniform():
    # If all eigenvectors were uniform, every IPR = 1/N < 5/N → all extended
    N = 20
    eigvecs = np.tile((np.ones(N) / np.sqrt(N))[:, None], (1, 10))
    frac = extended_subspace_fraction(eigvecs)
    assert frac == 1.0


def test_extended_subspace_fraction_localized():
    # All eigenvectors are deltas → IPR = 1 → none extended
    N = 20
    eigvecs = np.zeros((N, 10))
    for k in range(10):
        eigvecs[k, k] = 1.0
    frac = extended_subspace_fraction(eigvecs)
    assert frac == 0.0


def test_extended_subspace_fraction_real_graph():
    eigvecs, _, _ = _setup(N=50)
    frac = extended_subspace_fraction(eigvecs)
    # On a random spatial k-NN graph, should be in (0, 1)
    assert 0.0 < frac < 1.0
