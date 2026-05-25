# spectral-mobility — reproducibility Makefile
# Per Gemini round 5: 'make smoke-test' for fast sanity, 'make reproduce-all'
# for the full pipeline.

.PHONY: help install dev-install test smoke-test reproduce-all reproduce-fast clean \
        lint format docs check-version

PYTHON ?= python
VENV   ?= .venv

help:
	@echo "spectral-mobility — common targets"
	@echo ""
	@echo "  install         Install the package in editable mode"
	@echo "  dev-install     Install with dev dependencies (pytest, ruff)"
	@echo "  test            Run the test suite (82 tests, < 10s)"
	@echo "  smoke-test      Fast end-to-end sanity check (~5 min, 30 cities)"
	@echo "  reproduce-fast  Mid-sized run for the headline results (~30 min)"
	@echo "  reproduce-all   Full pipeline (~6-8 h, 124 cities × 200 perms)"
	@echo "  lint            Run ruff"
	@echo "  format          Apply ruff format"
	@echo "  check-version   Verify pyproject/__init__/CITATION are aligned"
	@echo "  clean           Remove caches and build artifacts"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]" 2>/dev/null || pip install -e . pytest ruff

test:
	pytest -q tests/

# Fast smoke test: validates the *pipeline mechanics* on a 5-city subset,
# 5 bootstrap reps, 100 perm-test draws.  Should finish in < 5 min on a
# modern laptop.  No claim about the science — purely a "does it run".
smoke-test:
	@mkdir -p examples/output
	$(PYTHON) -c "import spectral_mobility; print(f'spectral_mobility {spectral_mobility.__version__} loaded')"
	$(PYTHON) -c "from spectral_mobility import (\
		CitySpectralProfile, bootstrap_similarity_matrix,\
		multiscale_similarity_matrix, dirichlet_energy,\
		unnormalised_laplacian, build_geographic_knn,\
		symmetric_normalised_laplacian, spectral_decomposition,\
	); \
	import numpy as np; \
	rng = np.random.default_rng(0); \
	cities = [CitySpectralProfile.from_coords(name=f'C{i}', lat=rng.uniform(0,1,80), lng=rng.uniform(0,1,80), k_nn=5) for i in range(5)]; \
	r = bootstrap_similarity_matrix(cities, n_boot=5, seed=0); \
	assert r['median'].shape == (5,5); \
	m = multiscale_similarity_matrix(cities); \
	assert set(m['matrices']) == {'low','mid','high'}; \
	W, _ = build_geographic_knn(rng.uniform(0,1,50), rng.uniform(0,1,50), k=5); \
	L = symmetric_normalised_laplacian(W); \
	Lu = unnormalised_laplacian(W); \
	d = dirichlet_energy(L, rng.standard_normal(50)); \
	assert d >= 0; \
	print(f'✓ smoke-test passed: 5-city bootstrap, multi-scale, Dirichlet d={d:.3f}')"

# Mid-sized run: the headline 30-city panel from script 05.  ~30 min.
# Reproduces the bootstrap CI, US-vs-non-US permutation test, multi-scale
# fingerprint, all on the curated panel.
reproduce-fast:
	$(PYTHON) examples/05_paper_ready_validation.py

# Full pipeline.  ~6-8 h.  Re-runs every numbered script from 05 to 20.
# WARNING: produces ~50 MB of intermediate artifacts in examples/output/.
reproduce-all:
	@mkdir -p examples/output
	$(PYTHON) examples/05_paper_ready_validation.py
	$(PYTHON) examples/06_discover_structure.py
	$(PYTHON) examples/07_taxonomy.py
	$(PYTHON) examples/08_atlas_taxonomy.py
	$(PYTHON) examples/09_size_corrected_taxonomy.py
	$(PYTHON) examples/11_kde_null_and_gmm.py
	$(PYTHON) examples/12_math_validation.py
	$(PYTHON) examples/13_saturation_smooth_y.py
	$(PYTHON) examples/14_demand_validation.py
	$(PYTHON) examples/15_goldilocks_test.py
	$(PYTHON) examples/16_goldilocks_atlas_proxy.py
	$(PYTHON) examples/17_dirichlet_and_aug_gain.py
	$(PYTHON) examples/18_pre_reg_v2_check.py
	$(PYTHON) examples/19_kde_bandwidth_sensitivity.py
	$(PYTHON) examples/20_block_spatial_cv.py
	@echo "✓ All artifacts in examples/output/"

lint:
	ruff check src/ tests/ examples/

format:
	ruff format src/ tests/ examples/

check-version:
	@grep -E '^version' pyproject.toml
	@grep -E '__version__' src/spectral_mobility/__init__.py
	@grep -E '^version' CITATION.cff

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__ \
	       .pytest_cache .ruff_cache .coverage htmlcov \
	       build dist src/*.egg-info
