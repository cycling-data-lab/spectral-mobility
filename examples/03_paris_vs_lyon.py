"""03_paris_vs_lyon.py — compare two real bike-share networks via
their spectral profiles, and produce a multi-city similarity matrix.

Loads IMD + Tier-1/Tier-2 demand caches from the sibling
``bikeshare-demand-forecasting`` repository, builds a
``CitySpectralProfile`` for each of nine networks (the panel used in
the topological-localization paper), then:

  1. compares Paris ↔ Lyon as a worked example (3-panel figure
     written to ``examples/output/03_paris_vs_lyon.pdf``);
  2. computes the pairwise spectral-similarity matrix on the 9-city
     panel and saves the heatmap to
     ``examples/output/03_similarity_matrix.pdf``.

Run with::

    python examples/03_paris_vs_lyon.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectral_mobility import (
    CitySpectralProfile,
    compare_cities,
    cross_city_similarity_matrix,
)
from spectral_mobility.plots import plot_city_comparison, plot_similarity_matrix


SIBLING = Path.home() / "cesi-research" / "bikeshare-demand-forecasting"
IMD_DIR = SIBLING / "data_collection" / "imd_international"
OUT_DIR = SIBLING / "experiments" / "outputs"

# Same panel as the topological-localization v0.1 paper
CITIES = [
    ("boston_bluebikes",       "boston_bluebikes",         "Boston"),
    ("dc_capitalbikeshare",    "dc_capitalbikeshare",      "DC"),
    ("chicago_divvy",          "chicago_divvy",            "Chicago"),
    ("sf_baywheels",           "sf_baywheels",             "San Francisco"),
    ("london_tfl",             "london_tfl",               "London"),
    ("montreal_bixi",          "world_ca_bixi_montr_al",   "Montréal"),
    ("tier2_paris",            "world_fr_v_lib_metropole", "Paris"),
    ("tier2_lyon",             "world_fr_v_lo_v",          "Lyon"),
    ("tier2_toulouse",         "world_fr_v_l_toulouse",    "Toulouse"),
]

FEATS_IMD = [
    "gtfs_heavy_stops_300m",
    "infra_cyclable_features_300m",
    "elevation_m",
    "topography_roughness_index",
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
    if slug.startswith("tier2_") and slug in TIER2_MAP:
        return OUT_DIR / f"d10_{TIER2_MAP[slug]}_predictions.parquet"
    return None


def load_profile(slug: str, stem: str, pretty: str) -> CitySpectralProfile | None:
    imd_path = IMD_DIR / f"{stem}.parquet"
    if not imd_path.exists():
        print(f"  ✗ {pretty}: missing IMD parquet"); return None
    imd = pd.read_parquet(imd_path)
    imd["station_id"] = imd["station_id"].astype(str)
    if slug == "london_tfl":
        imd["station_id"] = imd["station_id"].str.zfill(6)

    dpath = _demand_path(slug)
    if dpath is None or not dpath.exists():
        print(f"  ✗ {pretty}: no demand parquet"); return None
    df = pd.read_parquet(dpath)
    df["station_id"] = df["station_id"].astype(str)
    if slug == "london_tfl":
        df["station_id"] = df["station_id"].str.zfill(6)
    df["y"] = np.expm1(df["y_true_log"])
    y_map = df.groupby("station_id")["y"].mean().to_dict()
    imd["y"] = imd["station_id"].map(y_map)

    avail = [f for f in FEATS_IMD if f in imd.columns]
    sub = imd.dropna(subset=["lat", "lng", "y"] + avail).reset_index(drop=True)
    if len(sub) < 30: return None

    X = sub[avail].astype(float).values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    coords = sub[["lat", "lng"]].astype(float).values
    y = sub["y"].astype(float).values

    return CitySpectralProfile.from_coords(
        name=pretty, coords=coords, features=X, target=y, k_nn=6, sigma=300.0
    )


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # ── 1. Build profiles ──────────────────────────────────────────────
    print("Building spectral profiles for 9 bike-share networks...")
    profiles: dict[str, CitySpectralProfile] = {}
    for slug, stem, pretty in CITIES:
        p = load_profile(slug, stem, pretty)
        if p is None: continue
        profiles[pretty] = p
        s = p.summary()
        print(
            f"  ✓ {pretty:14s} N={s['N']:5d}  ⟨IPR⟩={s['mean_ipr']:.3f}  "
            f"ext={s['extended_fraction']:.2f}  ⟨r⟩={s['mean_level_spacing_r']:.3f}  "
            f"R²_IMD={s.get('R2_imd', float('nan')):.3f}  "
            f"ΔR²={s.get('delta_R2_K16', float('nan')):+.3f}"
        )

    if "Paris" not in profiles or "Lyon" not in profiles:
        print("Need both Paris and Lyon for the worked example.  Aborting.")
        return

    # ── 2. Paris vs Lyon worked example ────────────────────────────────
    paris = profiles["Paris"]
    lyon = profiles["Lyon"]
    cmp = compare_cities(paris, lyon)

    print("\n=== Paris ↔ Lyon ===")
    print(f"  Wasserstein on eigenvalues : {cmp.wasserstein_eigvals:.4f}")
    print(f"  Wasserstein on log-IPR     : {cmp.wasserstein_ipr:.4f}")
    print(f"  KS on eigenvalues          : {cmp.ks_eigvals:.4f}")
    print(f"  KS on log-IPR              : {cmp.ks_ipr:.4f}")
    print(f"  Spectral similarity        : {cmp.spectral_similarity:.4f}")

    fig = plot_city_comparison(paris, lyon)
    fig.savefig(out_dir / "03_paris_vs_lyon.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "03_paris_vs_lyon.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ wrote {out_dir / '03_paris_vs_lyon.pdf'}")

    # Individual Paris overview
    fig = paris.plot_overview()
    fig.savefig(out_dir / "03_paris_overview.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "03_paris_overview.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ wrote {out_dir / '03_paris_overview.pdf'}")

    # ── 3. 9-city similarity matrix ────────────────────────────────────
    print("\n=== Cross-city similarity matrix ===")
    M, names = cross_city_similarity_matrix(list(profiles.values()))
    df_M = pd.DataFrame(M, index=names, columns=names)
    print(df_M.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    plot_similarity_matrix(M, names, ax=ax)
    fig.savefig(out_dir / "03_similarity_matrix.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "03_similarity_matrix.png", bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"  ✓ wrote {out_dir / '03_similarity_matrix.pdf'}")

    # ── 4. Cluster-by-similarity summary ───────────────────────────────
    print("\n=== Most similar pairs (top 5) ===")
    n = len(names)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((names[i], names[j], float(M[i, j])))
    pairs.sort(key=lambda t: -t[2])
    for a, b, s in pairs[:5]:
        print(f"  {a:14s} ↔ {b:14s}  : {s:.3f}")
    print("=== Most dissimilar pairs (bottom 5) ===")
    for a, b, s in pairs[-5:]:
        print(f"  {a:14s} ↔ {b:14s}  : {s:.3f}")


if __name__ == "__main__":
    main()
