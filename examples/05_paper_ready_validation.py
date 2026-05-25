"""05_paper_ready_validation.py — full paper-ready spectral analysis
on a curated 30-city international panel, with all three
methodological upgrades:

  (i)   bootstrap confidence intervals on the pairwise similarity
        matrix (200 resamples, 70% subsample per city);
  (ii)  permutation test of the US-vs-non-US block contrast
        (10 000 permutations);
  (iii) multi-scale spectral fingerprint (low / mid / high
        frequency bands), each with its own similarity matrix.

The panel covers North America, Europe, Latin America and Asia,
deliberately mixing capital and secondary cities at different
sizes to break the size confound flagged by external review.

Output (all in ``examples/output/``):
  05_panel.csv                       one row per city
  05_baseline_matrix.csv             pairwise similarity (point estimate)
  05_ci_summary.csv                  bootstrap CIs per pair
  05_permutation.json                US-vs-non-US block contrast test
  05_multiscale.csv                  per-band similarity per pair
  05_paper_validation.{pdf,png}      composite paper-figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

from spectral_mobility import (
    CitySpectralProfile,
    bootstrap_similarity_matrix,
    ci_summary_table,
    cross_city_similarity_matrix,
    multiscale_similarity_matrix,
    permutation_test_block_contrast,
)
from spectral_mobility.plots import plot_similarity_matrix


# ── Curated 30-city international panel ───────────────────────────────
# Each entry: (imd parquet stem, pretty name, ISO country, region label)
PANEL = [
    # North America (USA + Canada)
    ("nyc_citibike",                          "Citi Bike NYC",          "US", "north_america"),
    ("world_us_divvy",                        "Divvy Chicago",          "US", "north_america"),
    ("dc_capitalbikeshare",                   "Capital DC",             "US", "north_america"),
    ("sf_baywheels",                          "Bay Wheels SF",          "US", "north_america"),
    ("boston_bluebikes",                      "Bluebikes Boston",       "US", "north_america"),
    ("world_us_biketown",                     "Biketown Portland",      "US", "north_america"),
    ("world_ca_bixi_montr_al",                "BIXI Montréal",          "CA", "north_america"),
    ("world_ca_bike_share_toronto",           "Bike Share Toronto",     "CA", "north_america"),
    # France
    ("world_fr_v_lib_metropole",              "Vélib Paris",            "FR", "europe"),
    ("world_fr_v_lo_v",                       "Vélo'v Lyon",            "FR", "europe"),
    ("world_fr_v_l_toulouse",                 "VéLÔ Toulouse",          "FR", "europe"),
    ("world_fr_le_v_lo_par_tbm",              "Le VélO Bordeaux",       "FR", "europe"),
    ("world_fr_lev_lo_marseille",             "Le Vélo Marseille",      "FR", "europe"),
    # UK
    ("london_tfl",                            "Santander London",       "GB", "europe"),
    ("world_gb_beryl_bcp",                    "Beryl Bournemouth",      "GB", "europe"),
    ("world_gb_beryl_greater_manchester",     "Beryl Manchester",       "GB", "europe"),
    # Germany / Central Europe
    ("world_de_nextbike_berlin",              "Nextbike Berlin",        "DE", "europe"),
    ("world_de_swabi_augsburg",               "Swabi Augsburg",         "DE", "europe"),
    ("world_cz_nextbike_praha",               "Nextbike Praha",         "CZ", "europe"),
    # Iberia
    ("world_es_bicimad",                      "BiciMAD Madrid",         "ES", "europe"),
    ("world_es_bicing",                       "Bicing Barcelona",       "ES", "europe"),
    ("world_es_valenbisi",                    "Valenbisi València",     "ES", "europe"),
    # Italy + Benelux + Poland
    ("world_it_milan_bikemi",                 "BikeMi Milan",           "IT", "europe"),
    ("world_be_villo",                        "Villo Bruxelles",        "BE", "europe"),
    ("world_be_velo_antwerpen",               "Velo Antwerpen",         "BE", "europe"),
    ("world_pl_metrorower",                   "Metrorower",             "PL", "europe"),
    ("world_pl_mevo",                         "Mevo Gdańsk",            "PL", "europe"),
    # Switzerland
    ("world_ch_velospot",                     "Velospot CH",            "CH", "europe"),
    # Latin America
    ("world_mx_ecobici",                      "Ecobici Ciudad de Mexico", "MX", "latam"),
    ("world_ar_ecobici",                      "Ecobici Buenos Aires",   "AR", "latam"),
    ("world_br_bike_ita_rio",                 "Bike Itaú Rio",          "BR", "latam"),
]

SIBLING = Path.home() / "cesi-research" / "bikeshare-demand-forecasting"
IMD_DIR = SIBLING / "data_collection" / "imd_international"


def load_profile(stem: str, name: str, k_nn: int = 6) -> CitySpectralProfile | None:
    path = IMD_DIR / f"{stem}.parquet"
    if not path.exists():
        print(f"  ✗ {name}: missing parquet ({stem})"); return None
    df = pd.read_parquet(path)
    df = df.dropna(subset=["lat", "lng"]).reset_index(drop=True)
    if len(df) < 50:
        print(f"  ✗ {name}: N = {len(df)} < 50"); return None
    coords = df[["lat", "lng"]].astype(float).values
    return CitySpectralProfile.from_coords(
        name=name, coords=coords, k_nn=k_nn, sigma="auto",
    )


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # ── Load profiles ────────────────────────────────────────────────
    print(f"Loading {len(PANEL)} curated networks...")
    profiles = []
    panel_meta = []
    for stem, name, iso, region in PANEL:
        prof = load_profile(stem, name)
        if prof is None: continue
        profiles.append(prof)
        panel_meta.append({"name": name, "iso": iso, "region": region,
                           "N": prof.N, "sigma": prof.sigma,
                           "mean_ipr": prof.mean_ipr,
                           "extended_fraction": prof.extended_fraction})
    pd.DataFrame(panel_meta).to_csv(out_dir / "05_panel.csv", index=False)
    n_cities = len(profiles)
    names = [p.name for p in profiles]
    iso_codes = [m["iso"] for m in panel_meta]
    region_codes = [m["region"] for m in panel_meta]
    print(f"\n  → {n_cities} networks loaded")

    # ── 1. Baseline similarity matrix (single estimate) ──────────────
    print("\n=== 1. Baseline similarity matrix ===")
    M_base, _ = cross_city_similarity_matrix(profiles)
    pd.DataFrame(M_base, index=names, columns=names).to_csv(
        out_dir / "05_baseline_matrix.csv"
    )

    # ── 2. Bootstrap CIs (the big one — 200 reps × 30 cities) ─────────
    print("\n=== 2. Bootstrap 95% CIs on similarity (200 reps × 30 cities) ===")
    print("    (this takes ~5-10 minutes)")
    boot = bootstrap_similarity_matrix(
        profiles, n_boot=200, subsample_fraction=0.7, seed=42, progress=True
    )
    df_ci = ci_summary_table(boot)
    df_ci.to_csv(out_dir / "05_ci_summary.csv", index=False)
    print(f"    median CI width: {df_ci['ci_width'].median():.3f}")
    print(f"    Top 5 most-similar pairs (95% CIs):")
    for _, r in df_ci.head(5).iterrows():
        print(f"      {r['city_a']:24s} ↔ {r['city_b']:24s}  "
              f"{r['median']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]")
    print(f"    Top 5 most-dissimilar pairs:")
    for _, r in df_ci.tail(5).iloc[::-1].iterrows():
        print(f"      {r['city_a']:24s} ↔ {r['city_b']:24s}  "
              f"{r['median']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]")

    # ── 3. Permutation test of US-vs-non-US block contrast ───────────
    print("\n=== 3. Permutation test of US-vs-non-US block contrast ===")
    us_cities = [m["name"] for m in panel_meta if m["iso"] == "US"]
    non_us_cities = [m["name"] for m in panel_meta if m["iso"] != "US"]
    print(f"    US block: {len(us_cities)} networks")
    print(f"    non-US block: {len(non_us_cities)} networks")
    perm = permutation_test_block_contrast(
        boot["median"], names, us_cities, non_us_cities,
        n_perm=10_000, seed=42,
    )
    print(f"    Observed contrast: +{perm['observed_contrast']:.4f}")
    print(f"    Null distribution: mean={perm['null_mean']:+.4f}  "
          f"std={perm['null_std']:.4f}")
    print(f"    One-tailed p-value: {perm['p_value']:.4f}")
    if perm["p_value"] < 0.001:
        print("    🟢 HIGHLY SIGNIFICANT — US split is structural, not artefactual")
    elif perm["p_value"] < 0.05:
        print("    🟡 SIGNIFICANT — moderate support for US split")
    else:
        print("    🔴 NOT SIGNIFICANT — US split is not distinguishable from chance")

    # Save without the heavy null distribution
    perm_to_save = {k: v for k, v in perm.items() if k != "null_distribution"}
    with open(out_dir / "05_permutation.json", "w") as f:
        json.dump(perm_to_save, f, indent=2)

    # ── 4. Permutation test on Europe-vs-non-Europe (cross check) ────
    print("\n=== 4. Cross-check: Europe-vs-rest block contrast ===")
    eu_cities = [m["name"] for m in panel_meta if m["region"] == "europe"]
    rest_cities = [m["name"] for m in panel_meta if m["region"] != "europe"]
    perm_eu = permutation_test_block_contrast(
        boot["median"], names, eu_cities, rest_cities,
        n_perm=10_000, seed=43,
    )
    print(f"    Observed contrast: +{perm_eu['observed_contrast']:.4f}")
    print(f"    p-value: {perm_eu['p_value']:.4f}")

    # ── 5. Multi-scale spectral fingerprint ──────────────────────────
    print("\n=== 5. Multi-scale spectral fingerprint (low / mid / high) ===")
    ms = multiscale_similarity_matrix(profiles)
    rows = []
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            rows.append({
                "city_a": names[i], "city_b": names[j],
                "similarity_low": float(ms["matrices"]["low"][i, j]),
                "similarity_mid": float(ms["matrices"]["mid"][i, j]),
                "similarity_high": float(ms["matrices"]["high"][i, j]),
                "iso_a": iso_codes[i], "iso_b": iso_codes[j],
            })
    df_ms = pd.DataFrame(rows)
    df_ms.to_csv(out_dir / "05_multiscale.csv", index=False)
    # mean within-block vs between-block per band
    print(f"    {'band':6s}  {'mean within-US':>15s}  {'mean US↔non-US':>15s}  {'contrast':>10s}")
    for band in ["low", "mid", "high"]:
        col = f"similarity_{band}"
        within_us = df_ms[(df_ms.iso_a == "US") & (df_ms.iso_b == "US")][col].mean()
        between = df_ms[((df_ms.iso_a == "US") ^ (df_ms.iso_b == "US"))][col].mean()
        contrast = within_us - between
        print(f"    {band:6s}  {within_us:15.4f}  {between:15.4f}  {contrast:+.4f}")

    # ── 6. Generate composite figure ─────────────────────────────────
    print("\n=== 6. Generating composite paper figure ===")
    fig = plt.figure(figsize=(18, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    # Panel A: Baseline matrix (with bootstrap CI annotations)
    ax_A = fig.add_subplot(gs[0, 0])
    plot_similarity_matrix(boot["median"], names, ax=ax_A, annotate=False)
    ax_A.set_title("(a) Median similarity matrix\n(bootstrap 200 reps × 70% subsample)",
                    fontsize=10)

    # Panel B: CI width
    ax_B = fig.add_subplot(gs[0, 1])
    ci_w = boot["upper"] - boot["lower"]
    np.fill_diagonal(ci_w, 0)
    im = ax_B.imshow(ci_w, cmap="viridis", aspect="auto")
    ax_B.set_xticks(range(n_cities))
    ax_B.set_yticks(range(n_cities))
    ax_B.set_xticklabels(names, rotation=45, ha="right", fontsize=6)
    ax_B.set_yticklabels(names, fontsize=6)
    plt.colorbar(im, ax=ax_B, label="95% CI width", shrink=0.7)
    ax_B.set_title(f"(b) CI widths (median = {ci_w[np.triu_indices(n_cities, 1)].mean():.3f})",
                    fontsize=10)

    # Panel C: Permutation null distribution
    ax_C = fig.add_subplot(gs[0, 2])
    ax_C.hist(perm["null_distribution"], bins=40, color="grey",
              alpha=0.7, density=True, label="permutation null")
    ax_C.axvline(perm["observed_contrast"], color="red", lw=2.5,
                 label=f"observed = {perm['observed_contrast']:+.3f}")
    ax_C.set_xlabel("US-vs-non-US block contrast")
    ax_C.set_ylabel("density")
    ax_C.set_title(f"(c) Permutation test ({perm['n_perm']:,} reps)\n"
                   f"p-value = {perm['p_value']:.4f}", fontsize=10)
    ax_C.legend(fontsize=8, loc="upper left")
    ax_C.grid(True, alpha=0.3)

    # Panels D-F: Multi-scale matrices
    for ax_idx, band in enumerate(["low", "mid", "high"]):
        ax = fig.add_subplot(gs[1, ax_idx])
        plot_similarity_matrix(ms["matrices"][band], names, ax=ax, annotate=False)
        ax.set_title(f"({chr(ord('d') + ax_idx)}) Spectral similarity — "
                     f"{band}-frequency band",
                     fontsize=10)

    # Panel G: Dendrogram of clustering
    ax_G = fig.add_subplot(gs[2, 0:2])
    D = 1 - boot["median"]
    np.fill_diagonal(D, 0)
    D = (D + D.T) / 2
    from scipy.spatial.distance import squareform
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")
    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=7,
               ax=ax_G, color_threshold=0.4)
    ax_G.set_ylabel("1 − spectral similarity (distance)")
    ax_G.set_title("(g) Hierarchical clustering of 30 international networks",
                   fontsize=10)
    ax_G.grid(axis="y", alpha=0.3)

    # Panel H: Per-band contrasts
    ax_H = fig.add_subplot(gs[2, 2])
    band_contrasts = []
    for band in ["low", "mid", "high"]:
        col = f"similarity_{band}"
        within_us = df_ms[(df_ms.iso_a == "US") & (df_ms.iso_b == "US")][col].mean()
        between = df_ms[((df_ms.iso_a == "US") ^ (df_ms.iso_b == "US"))][col].mean()
        band_contrasts.append(within_us - between)
    ax_H.bar(["low", "mid", "high"], band_contrasts, color=["C0", "C2", "C3"], alpha=0.8)
    ax_H.axhline(0, color="black", lw=0.5)
    ax_H.set_ylabel("US-vs-non-US contrast")
    ax_H.set_title("(h) Multi-scale contrast", fontsize=10)
    ax_H.grid(axis="y", alpha=0.3)
    for i, c in enumerate(band_contrasts):
        ax_H.text(i, c, f"{c:+.3f}", ha="center", va="bottom" if c > 0 else "top",
                  fontsize=9)

    fig.suptitle(f"Paper-ready spectral validation on {n_cities} international bike-share networks",
                 fontsize=13)
    fig.savefig(out_dir / "05_paper_validation.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "05_paper_validation.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"\n✓ {out_dir / '05_paper_validation.pdf'}")
    print("\n=== ALL DONE ===")
    print(f"  artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
