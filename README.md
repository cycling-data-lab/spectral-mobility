# spectral-mobility

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-72%20passing-brightgreen.svg)](./tests/)
[![Version](https://img.shields.io/badge/version-0.4.0-orange.svg)](./pyproject.toml)

**Spectral feature augmentation and applicability-domain bounds for urban mobility prediction.**

`spectral-mobility` is a small, focused Python package that implements
the **structural applicability bound** of
[Fossé & Pallares (2026)](https://github.com/cycling-data-lab/structural-bounds-framework)
and turns its diagnostic into a practical feature-engineering tool.
For any prediction task on a graph (bike-share demand, modal share,
station-level usage, …) the package gives you:

1. **The theoretical ceiling** of any model — `R²_spec(features, y)`,
   computed in closed form, no training required.
2. **A feature-augmentation recipe** that empirically lifts this
   ceiling by adding the top-K low-frequency Laplacian eigenvectors
   to your existing feature matrix.
3. **A diagnostic toolkit** to identify the geographic bottlenecks
   that cap predictability of the underlying network.

Validated on 76 independent tests (24 bike-share networks + 13
French metropolitan regions × 4 commute modes); all 76 show
positive ΔR² from augmentation, with median gain of +0.22 R² at
station scale.

## Installation

```bash
pip install spectral-mobility            # from PyPI (when released)
pip install -e ".[dev]"                  # from a local clone
```

Requires Python ≥ 3.10, numpy, scipy, scikit-learn, pandas.  Plotting
helpers and notebook examples require the optional `[plotting]` and
`[examples]` extras.

## Quick start — high-level prediction API

```python
from spectral_mobility import SpectralAugmentedRegressor

# X = (N, p) feature matrix; coords = (N, 2) [lat, lng]; y = (N,) target
model = SpectralAugmentedRegressor(K=16, k_nn=6, sigma=300.0)
model.fit(X, coords, y)
y_hat = model.predict()                  # transductive prediction

# Closed-form ceiling diagnostic
print(model.ceiling())

# Side-by-side comparison: augmented vs baseline (no augmentation), 5-fold LSO
result = model.cross_validate(X, coords, y, n_folds=5)
print(f"baseline R²:  {result['baseline_mean']:+.3f}")
print(f"augmented R²: {result['augmented_mean']:+.3f}")
print(f"gain:         {result['mean_gain']:+.3f}")
```

On Boston Bluebikes (493 stations, 4 IMD features, 5-fold LSO):

| Protocol | Baseline R² | Augmented R² | Gain |
|---|---|---|---|
| Transductive | +0.05 | +0.40 | **+0.35** |
| Inductive (strict, no leakage) | +0.05 | **+0.46** | **+0.41** |

See [`examples/02_boston_bikeshare.py`](./examples/02_boston_bikeshare.py).

The wrapper defaults to LightGBM if available, falling back to
`sklearn.ensemble.GradientBoostingRegressor` otherwise.  Any
sklearn-compatible regressor can be passed as ``base_estimator``.

### Inductive vs transductive

```python
# Strict "deploy to new stations" evaluation — rebuilds eigenbasis per
# fold using only training coordinates, projects test points via
# Nyström-style k-NN extension.  No coordinate leakage.
result = model.cross_validate(X, coords, y, n_folds=5, protocol="inductive")
```

### Visualisation helpers

```python
from spectral_mobility.plots import (
    plot_ceiling_curve, plot_bottleneck_map,
    plot_spectrum, plot_cv_comparison,
)

# Identify and visualise the worst structural bottleneck of the network
from spectral_mobility import bottleneck_modes
worst_mode = bottleneck_modes(model.eigvecs_, n_top=1)[0]
plot_bottleneck_map(coords, model.eigvecs_[:, worst_mode])
```

### City profiles and cross-city comparison

```python
from spectral_mobility import (
    CitySpectralProfile, compare_cities, cross_city_similarity_matrix,
)

paris = CitySpectralProfile.from_coords(
    name="Vélib Paris", lat=lat, lng=lng,
    features=X_imd, target=y_demand, k_nn=6, sigma=300,
)
lyon = CitySpectralProfile.from_coords(...)

paris.summary()                       # dict with all key stats
paris.bottleneck_zones(n=5)           # the 5 worst structural bottlenecks
paris.predictability_ceiling(K=16)    # closed-form R²_spec
paris.plot_overview()                 # 4-panel diagnostic figure

cmp = compare_cities(paris, lyon)
print(cmp.spectral_similarity)        # 0.80 — high

# Multi-city similarity matrix (heatmap + clustering)
profiles = [paris, lyon, marseille, ...]
M, names = cross_city_similarity_matrix(profiles)
```

On a 9-city panel (Boston, DC, Chicago, SF, London, Montréal, Paris,
Lyon, Toulouse) the package recovers an unsupervised
**US-vs-European structural split** without being told anything
about continents or city sizes:

- Top similar pairs: Boston↔Chicago (0.93), DC↔Chicago (0.92),
  Boston↔DC (0.89), Lyon↔Toulouse (0.87)
- Most dissimilar: US-east ↔ Paris/London (~0.48-0.51)

See [`examples/03_paris_vs_lyon.py`](./examples/03_paris_vs_lyon.py).

## Lower-level API

```python
import numpy as np
from spectral_mobility import (
    build_geographic_knn, symmetric_normalised_laplacian,
    spectral_decomposition, spectral_bound, augment_features,
)

# 1. Build the graph from station coordinates
W, sigma = build_geographic_knn(lat, lng, k=6)          # σ auto = median k-th NN

# 2. Get the spectrum
L = symmetric_normalised_laplacian(W)
eigvals, eigvecs = spectral_decomposition(L)

# 3. Diagnose the predictability ceiling
result = spectral_bound(eigvecs, y, encoder_features=X_imd, K=16)
print(f"IMD ceiling : {result.r2_imd:.3f}")
print(f"+ spectral  : {result.r2_augmented:.3f}")
print(f"ΔR²         : +{result.delta_r2:.3f}")

# 4. Augment your feature matrix and drop into any ML pipeline
X_aug = augment_features(X_imd, eigvecs, K=16)
# X_aug is just a numpy array; feed it to LightGBM, XGBoost, sklearn,
# PyTorch — anything that consumes (N, p) feature matrices.
```

See [`examples/01_quickstart.py`](./examples/01_quickstart.py) for a
synthetic end-to-end demo using the low-level API.

## API at a glance

| Function | Purpose |
|---|---|
| `build_geographic_knn(lat, lng, k, sigma)` | Haversine k-NN graph |
| `build_feature_knn(X, k, sigma)` | Euclidean k-NN graph |
| `symmetric_normalised_laplacian(W)` | `L_sym = I − D^{−½} W D^{−½}` |
| `spectral_decomposition(L, k)` | Eigendecomposition (dense or ARPACK) |
| `inverse_participation_ratio(eigvecs)` | IPR per eigenmode |
| `participation_ratio(eigvecs)` | PR per eigenmode |
| `level_spacing_ratios(eigvals)` | Adjacent gap ratio statistic |
| `r2_spec_subspace(S, y)` | The applicability bound on a subspace |
| `spectral_bound(eigvecs, y, X, K)` | Bound with/without augmentation |
| `augment_features(X, eigvecs, K)` | Append top-K eigenvectors as columns |
| `select_K(eigvecs, y, X, method)` | Heuristic K selection (elbow / ratio / fixed) |
| `bottleneck_modes(eigvecs, n_top)` | Indices of most-localized eigenmodes |
| `locate_bottleneck_nodes(psi, mass_threshold)` | Geographic footprint of a localized mode |
| `extended_subspace_fraction(eigvecs)` | What fraction of modes are extended |
| `SpectralAugmentedRegressor(...)` | sklearn-style wrapper: `.fit(X, coords, y)`, `.predict()`, `.ceiling()`, `.cross_validate(...)` |
| `CitySpectralProfile.from_coords(...)` | Self-contained spectral profile of one network |
| `compare_cities(profile_a, profile_b)` | Pairwise Wasserstein / KS / similarity |
| `cross_city_similarity_matrix(profiles)` | Pairwise similarity matrix over N cities |

## Theory in two sentences

The structural lower bound on out-of-distribution loss is
`(1 − R²_spec(S, y)) · Var(y)`, where `R²_spec(S, y)` is the squared
projection of the target on the column-span of an encoder matrix `S`.
Because `R²_spec(S ∪ T, y) ≥ R²_spec(S, y)` for any subspace
extension, appending the top-K low-frequency Laplacian eigenvectors
to `S` can only raise the ceiling — and the gain is measurable in
closed form before training any model.

See the
[topological-localization-mobility paper](https://github.com/cycling-data-lab/topological-localization-mobility)
for the empirical validation and the
[structural-bounds-framework paper](https://github.com/cycling-data-lab/structural-bounds-framework)
for the proof of the bound.

## Cycling Data Lab integration

This package replaces the ad-hoc spectral-graph code currently
copy-pasted across several
[`cycling-data-lab`](https://github.com/cycling-data-lab) repositories
(graph construction, Laplacian, eigendecomposition, IPR, R²_spec, …)
with a single, tested, documented interface.

Existing analyses that re-implement these primitives — including
`d24`, `d28`, `d40b`, `d40c`, `d51`, `d51c`, `d52`, `d53` in the
sibling repositories — can be rewritten in a handful of lines using
this package.

## Status

**v0.1 — alpha.**  Core API + 41 unit tests, all passing.  No PyPI
release yet; install from source.  Breaking API changes possible
before v1.0.

## How to cite

A machine-readable citation is provided in
[`CITATION.cff`](./CITATION.cff).

```bibtex
@software{spectralMobility2026,
  author       = {Foss\'e, Rohan and Pallares, Ga\"el},
  title        = {spectral-mobility: Spectral feature augmentation and
                  applicability-domain bounds for urban mobility prediction},
  year         = {2026},
  url          = {https://github.com/cycling-data-lab/spectral-mobility},
  version      = {0.1.0}
}
```

## License

[MIT](./LICENSE).  Affiliated with [CESI LINEACT (EA 7527)](https://lineact.cesi.fr), Montpellier, France.
