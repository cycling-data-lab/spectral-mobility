"""spectral-mobility — spectral feature augmentation and
applicability-domain bounds for urban mobility prediction.

Quick start
-----------

    >>> import numpy as np
    >>> from spectral_mobility import (
    ...     build_geographic_knn, symmetric_normalised_laplacian,
    ...     spectral_decomposition, spectral_bound, augment_features,
    ... )
    >>> # 50 stations on a small spatial grid
    >>> rng = np.random.default_rng(0)
    >>> coords = rng.uniform(0, 0.01, size=(50, 2))  # ~1 km x 1 km
    >>> y = rng.standard_normal(50)
    >>> W, sigma = build_geographic_knn(coords[:, 0], coords[:, 1], k=6)
    >>> L = symmetric_normalised_laplacian(W)
    >>> eigvals, eigvecs = spectral_decomposition(L)
    >>> # Features could be anything — here, two random demographic-like ones
    >>> X = rng.standard_normal((50, 2))
    >>> result = spectral_bound(eigvecs, y, encoder_features=X, K=8)
    >>> result.r2_imd, result.r2_augmented, result.delta_r2  # doctest: +SKIP
    (0.04, 0.18, 0.14)

See README.md for the full reference and the theoretical background.
"""

from spectral_mobility.augmentation import augment_features, select_K
from spectral_mobility.bound import (
    SpectralBoundResult,
    r2_spec_subspace,
    spectral_bound,
)
from spectral_mobility.diagnostics import (
    bottleneck_modes,
    extended_subspace_fraction,
    locate_bottleneck_nodes,
)
from spectral_mobility.graph import (
    build_feature_knn,
    build_geographic_knn,
    haversine_distance_matrix,
    symmetric_normalised_laplacian,
)
from spectral_mobility.predictor import SpectralAugmentedRegressor
from spectral_mobility.spectral import (
    inverse_participation_ratio,
    level_spacing_ratios,
    participation_ratio,
    spectral_decomposition,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    # graph
    "build_geographic_knn",
    "build_feature_knn",
    "haversine_distance_matrix",
    "symmetric_normalised_laplacian",
    # spectral
    "spectral_decomposition",
    "inverse_participation_ratio",
    "participation_ratio",
    "level_spacing_ratios",
    # bound
    "r2_spec_subspace",
    "spectral_bound",
    "SpectralBoundResult",
    # augmentation
    "augment_features",
    "select_K",
    # diagnostics
    "bottleneck_modes",
    "locate_bottleneck_nodes",
    "extended_subspace_fraction",
    # predictor
    "SpectralAugmentedRegressor",
]
