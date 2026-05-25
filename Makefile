# spectral-mobility — package Makefile

.PHONY: help install dev-install test smoke-test build clean lint format check-version

PYTHON ?= python

help:
	@echo "spectral-mobility — package targets"
	@echo ""
	@echo "  install         pip install -e ."
	@echo "  dev-install     pip install -e .[dev] (pytest, ruff)"
	@echo "  test            Run the test suite (82 tests, < 10s)"
	@echo "  smoke-test      Fast end-to-end sanity check on a synthetic city"
	@echo "  build           Build sdist + wheel for PyPI"
	@echo "  lint            Run ruff"
	@echo "  format          Apply ruff format"
	@echo "  check-version   Verify pyproject/__init__/CITATION are aligned"
	@echo "  clean           Remove caches and build artifacts"
	@echo ""
	@echo "Research scripts (05-22) are in the companion paper repo:"
	@echo "  https://github.com/cycling-data-lab/paper-spectral-cv-illusion"

install:
	pip install -e .

dev-install:
	pip install -e ".[dev]" 2>/dev/null || pip install -e . pytest ruff

test:
	pytest -q tests/

smoke-test:
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

build:
	pip install --upgrade build
	$(PYTHON) -m build

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
