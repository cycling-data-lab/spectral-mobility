"""04_us_eu_split_validation.py — Validate the US-vs-European spectral
split against three confounds.

Background.  In examples/03_paris_vs_lyon.py we found that the
pairwise spectral-similarity matrix over a 9-city bike-share panel
spontaneously splits into a US cluster (Boston, DC, Chicago, SF) and
a European cluster (Paris, London + Lyon, Toulouse).  A reviewer can
legitimately object that the split reflects nuisance variation
rather than a structural urban property:

  C1.  SIZE.  US networks (493-1345 stations) are systematically
       larger than European ones (448-1361, but mostly < 800).  Are
       we just clustering by N?

  C2.  DENSITY.  With sigma="auto" (median k-th-NN distance), σ
       adapts to local density.  Larger / denser networks get
       smaller σ, which changes the spectrum.

  C3.  k-NN HYPERPARAMETER.  Are we sensitive to k = 6 specifically?
       Does the split survive k = 10, k = 20?

This script runs the three controls and reports a quantitative
verdict.

VERDICT METRIC.  We compute the pairwise similarity matrix M(control)
under each control, and compare it to the baseline matrix M(baseline)
using the Mantel-style correlation between the two flattened off-
diagonal entries (Spearman).  A high correlation (> 0.7) means the
split is structural; a low correlation (< 0.3) means the original
split was an artefact of the specific (N, σ, k) choices.

Output:
  output/04_split_validation.csv
  output/04_split_validation.json
  output/04_split_validation.{pdf,png}     5-panel figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from spectral_mobility import (
    CitySpectralProfile,
    cross_city_similarity_matrix,
)
from spectral_mobility.plots import plot_similarity_matrix


SIBLING = Path.home() / "cesi-research" / "bikeshare-demand-forecasting"
IMD_DIR = SIBLING / "data_collection" / "imd_international"
OUT_DIR = SIBLING / "experiments" / "outputs"

CITIES = [
    ("boston_bluebikes",    "boston_bluebikes",         "Boston"),
    ("dc_capitalbikeshare", "dc_capitalbikeshare",      "DC"),
    ("chicago_divvy",       "chicago_divvy",            "Chicago"),
    ("sf_baywheels",        "sf_baywheels",             "San Francisco"),
    ("london_tfl",          "london_tfl",               "London"),
    ("montreal_bixi",       "world_ca_bixi_montr_al",   "Montréal"),
    ("tier2_paris",         "world_fr_v_lib_metropole", "Paris"),
    ("tier2_lyon",          "world_fr_v_lo_v",          "Lyon"),
    ("tier2_toulouse",      "world_fr_v_l_toulouse",    "Toulouse"),
]

TIER2_MAP = {
    "tier2_paris": "Paris",
    "tier2_lyon": "lyon",
    "tier2_toulouse": "toulouse",
}


def _demand_path(slug: str) -> Path | None:
    if slug in ("boston_bluebikes", "dc_capitalbikeshare", "chicago_divvy", "sf_baywheels"):
        return OUT_DIR / f"d3_{slug}_predictions.parquet"
    if slug == "london_tfl":
        return OUT_DIR / "d16_london_tfl_predictions.parquet"
    if slug == "montreal_bixi":
        return OUT_DIR / "d14_montreal_bixi_predictions.parquet"
    if slug.startswith("tier2_"):
        return OUT_DIR / f"d10_{TIER2_MAP[slug]}_predictions.parquet"
    return None


def load_station_table(slug: str, stem: str) -> pd.DataFrame | None:
    imd_path = IMD_DIR / f"{stem}.parquet"
    if not imd_path.exists(): return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    if slug == "london_tfl":
        imd["station_id"] = imd["station_id"].str.zfill(6)
    dpath = _demand_path(slug)
    if dpath is None or not dpath.exists(): return None
    df = pd.read_parquet(dpath)
    df["station_id"] = df["station_id"].astype(str)
    if slug == "london_tfl":
        df["station_id"] = df["station_id"].str.zfill(6)
    df["y"] = np.expm1(df["y_true_log"])
    y_map = df.groupby("station_id")["y"].mean().to_dict()
    imd["y"] = imd["station_id"].map(y_map)
    feats = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]
    avail = [f for f in feats if f in imd.columns]
    return imd.dropna(subset=["lat", "lng", "y"] + avail).reset_index(drop=True)


def make_profile(sub: pd.DataFrame, name: str,
                 k_nn: int = 6, sigma="auto") -> CitySpectralProfile:
    feats = ["gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
             "elevation_m", "topography_roughness_index"]
    avail = [f for f in feats if f in sub.columns]
    X = sub[avail].astype(float).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    coords = sub[["lat", "lng"]].astype(float).values
    y = sub["y"].astype(float).values
    return CitySpectralProfile.from_coords(
        name=name, coords=coords, features=X, target=y, k_nn=k_nn, sigma=sigma
    )


def off_diagonal(M: np.ndarray) -> np.ndarray:
    """Return the off-diagonal entries (upper triangle) as a 1D array."""
    n = M.shape[0]
    return M[np.triu_indices(n, k=1)]


def mantel_corr(M_a: np.ndarray, M_b: np.ndarray) -> tuple[float, float]:
    """Spearman correlation between flattened off-diagonal entries
    of two similarity matrices."""
    a = off_diagonal(M_a)
    b = off_diagonal(M_b)
    return spearmanr(a, b)


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # ── Load station tables once ─────────────────────────────────────
    print("Loading station tables...")
    tables: dict[str, pd.DataFrame] = {}
    for slug, stem, pretty in CITIES:
        sub = load_station_table(slug, stem)
        if sub is None:
            print(f"  ✗ {pretty}: missing data"); continue
        tables[pretty] = sub
        print(f"  ✓ {pretty}: N = {len(sub)}")

    names = list(tables.keys())
    n_cities = len(names)

    # ── BASELINE: original recipe k=6, σ=auto ─────────────────────
    print("\n=== BASELINE (k=6, σ=auto) ===")
    profiles_baseline = {n: make_profile(tables[n], n, k_nn=6, sigma="auto")
                         for n in names}
    M_baseline, _ = cross_city_similarity_matrix(profiles_baseline.values())

    # ── CONTROL A: matched-size subsample to N = 400 ─────────────
    print("\n=== CONTROL A — matched-size subsample to N=400 ===")
    rng = np.random.default_rng(42)
    profiles_A = {}
    for n in names:
        N_orig = len(tables[n])
        if N_orig < 400:
            sub = tables[n]  # use all if smaller
            print(f"  {n}: N={N_orig} < 400, using all")
        else:
            idx = rng.choice(N_orig, size=400, replace=False)
            sub = tables[n].iloc[idx].reset_index(drop=True)
            print(f"  {n}: subsampled to 400 from {N_orig}")
        profiles_A[n] = make_profile(sub, n, k_nn=6, sigma="auto")
    M_A, _ = cross_city_similarity_matrix(profiles_A.values())

    # ── CONTROL B: fixed σ = 500 m (not auto) ─────────────────────
    print("\n=== CONTROL B — fixed σ = 500 m (no auto adaptation) ===")
    profiles_B = {n: make_profile(tables[n], n, k_nn=6, sigma=500.0) for n in names}
    M_B, _ = cross_city_similarity_matrix(profiles_B.values())

    # ── CONTROL C: k = 20 (vs baseline k = 6) ─────────────────────
    print("\n=== CONTROL C — k = 20 (vs baseline k = 6) ===")
    profiles_C = {n: make_profile(tables[n], n, k_nn=20, sigma="auto")
                  for n in names}
    M_C, _ = cross_city_similarity_matrix(profiles_C.values())

    # ── CONTROL D: k = 15 + σ = 500 (combined) ────────────────────
    print("\n=== CONTROL D — k = 15 + σ = 500 m (combined) ===")
    profiles_D = {n: make_profile(tables[n], n, k_nn=15, sigma=500.0)
                  for n in names}
    M_D, _ = cross_city_similarity_matrix(profiles_D.values())

    # ── Compute Mantel-style correlations vs baseline ──────────────
    print("\n=== Mantel-style correlations (off-diagonal Spearman vs baseline) ===")
    controls = {
        "A: matched-size N=400": M_A,
        "B: fixed σ=500m": M_B,
        "C: k=20": M_C,
        "D: k=15 + σ=500m": M_D,
    }
    results = []
    for label, M_ctrl in controls.items():
        rho, p = mantel_corr(M_baseline, M_ctrl)
        results.append(dict(control=label, rho=float(rho), p=float(p)))
        verdict = (
            "🟢 ROBUST" if rho > 0.7 else
            ("🟡 PARTIAL" if rho > 0.3 else "🔴 ARTEFACT")
        )
        print(f"  {label:25s}  ρ = {rho:+.3f}  (p = {p:.2e})  {verdict}")

    # ── Verdict on US-EU specifically ──────────────────────────────
    us_block = ["Boston", "DC", "Chicago", "San Francisco"]
    eu_block = ["Paris", "London", "Lyon", "Toulouse"]
    def block_similarity(M, group_a, group_b):
        idx_a = [names.index(g) for g in group_a if g in names]
        idx_b = [names.index(g) for g in group_b if g in names]
        return float(np.mean([M[i, j] for i in idx_a for j in idx_b]))

    print("\n=== US-vs-European block contrast ===")
    print(f"{'Recipe':25s} {'within-US':>10s} {'within-EU':>10s} {'US-vs-EU':>10s} {'contrast':>10s}")
    block_contrasts = []
    for label, M in [("BASELINE", M_baseline)] + list(controls.items()):
        within_us = block_similarity(M, us_block, us_block)
        within_eu = block_similarity(M, eu_block, eu_block)
        between = block_similarity(M, us_block, eu_block)
        # contrast = mean within - between (positive = split holds)
        contrast = 0.5 * (within_us + within_eu) - between
        block_contrasts.append(dict(
            recipe=label, within_us=within_us, within_eu=within_eu,
            between=between, contrast=contrast,
        ))
        print(f"  {label:25s} {within_us:10.3f} {within_eu:10.3f} {between:10.3f}  "
              f"{contrast:+.3f}")

    # ── Figure: 5-panel similarity matrices ────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [
        ("Baseline (k=6, σ=auto)", M_baseline),
        ("A: matched-size N=400", M_A),
        ("B: fixed σ=500m", M_B),
        ("C: k=20", M_C),
        ("D: k=15, σ=500m", M_D),
    ]
    for ax, (title, M) in zip(axes.ravel(), panels):
        plot_similarity_matrix(M, names, ax=ax, annotate=True)
        ax.set_title(title, fontsize=10)
    axes.ravel()[5].set_axis_off()
    fig.suptitle("US-vs-EU spectral split: validation against 4 controls",
                 fontsize=12)
    fig.savefig(out_dir / "04_split_validation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "04_split_validation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\n✓ {out_dir / '04_split_validation.pdf'}")

    # ── Final verdict ──────────────────────────────────────────────
    baseline_contrast = block_contrasts[0]["contrast"]
    surviving = sum(1 for bc in block_contrasts[1:] if bc["contrast"] > 0.05)
    print("\n=== FINAL VERDICT ===")
    print(f"  Baseline US-vs-EU contrast : +{baseline_contrast:.3f}")
    print(f"  Controls preserving contrast > 0.05: {surviving}/4")
    if surviving == 4:
        verdict = (
            "🟢 SPLIT IS ROBUST.  The US-vs-European structural distinction "
            "survives all four controls.  Publishable."
        )
    elif surviving >= 2:
        verdict = (
            "🟡 SPLIT IS PARTIALLY ROBUST.  Survives some controls but not "
            "all.  Caveats required before publication."
        )
    else:
        verdict = (
            "🔴 SPLIT IS AN ARTEFACT.  Does not survive controls.  Do NOT "
            "publish the unsupervised cluster discovery claim."
        )
    print(f"  {verdict}")

    # Save artifacts
    pd.DataFrame(results).to_csv(out_dir / "04_mantel_correlations.csv", index=False)
    pd.DataFrame(block_contrasts).to_csv(out_dir / "04_block_contrasts.csv", index=False)
    with open(out_dir / "04_split_validation.json", "w") as f:
        json.dump({
            "mantel_correlations": results,
            "block_contrasts": block_contrasts,
            "baseline_contrast": baseline_contrast,
            "controls_surviving": surviving,
            "verdict": verdict,
        }, f, indent=2)
    print(f"  ✓ wrote {out_dir / '04_split_validation.json'}")


if __name__ == "__main__":
    main()
