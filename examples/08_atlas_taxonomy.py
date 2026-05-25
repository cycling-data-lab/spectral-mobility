"""08_atlas_taxonomy.py — extend the 30-city taxonomy to the full
international bike-share atlas (~190 networks).

For every Atlas parquet we compute the per-city spectral features
(mean IPR, extended fraction) — a per-city operation, so the cost
scales linearly with the number of cities, not quadratically.  Then
we project every city onto the taxonomy plane defined in 07 and
classify it by nearest cluster centroid (k=3, using cluster centroids
from the curated panel as anchors).

Output (in examples/output/):
  08_atlas_features.csv     per-city (N, sigma, mean_ipr, ext_frac)
  08_atlas_taxonomy.csv     per-city + predicted type
  08_atlas_summary.csv      per-type aggregates over the atlas
  08_atlas_taxonomy.{pdf,png}  scatter (IPR, ext_frac) of all atlas
                              cities coloured by type, with the
                              curated panel overlaid as larger markers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spectral_mobility import CitySpectralProfile


ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
OUT = Path(__file__).parent / "output"

MIN_STATIONS = 50    # skip tiny networks
MAX_STATIONS = 6000  # cap for compute time (NYC ~3000, biggest few are 5-6k)
K_NN = 5


def safe_load(parquet_path: Path) -> CitySpectralProfile | None:
    try:
        df = pd.read_parquet(parquet_path)
        if "lat" not in df.columns or "lng" not in df.columns:
            return None
        df = df.dropna(subset=["lat", "lng"]).drop_duplicates(["lat", "lng"])
        if len(df) < MIN_STATIONS:
            return None
        if len(df) > MAX_STATIONS:
            df = df.sample(n=MAX_STATIONS, random_state=0)
        name = parquet_path.stem
        return CitySpectralProfile.from_coords(
            name=name,
            lat=df["lat"].values,
            lng=df["lng"].values,
            k_nn=K_NN,
        )
    except Exception as e:
        print(f"  ✗ {parquet_path.name}: {e}")
        return None


def main():
    parquets = sorted(ATLAS_DIR.glob("*.parquet"))
    print(f"Found {len(parquets)} Atlas parquets")

    rows = []
    for i, p in enumerate(parquets, 1):
        prof = safe_load(p)
        if prof is None:
            continue
        rows.append({
            "name": prof.name,
            "N": prof.N,
            "sigma": prof.sigma,
            "mean_ipr": prof.mean_ipr,
            "extended_fraction": prof.extended_fraction,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(parquets)}] {prof.name:40s}  N={prof.N:5d}  "
                  f"IPR={prof.mean_ipr:.3f}  ext={prof.extended_fraction:.2f}")

    feat = pd.DataFrame(rows)
    feat.to_csv(OUT / "08_atlas_features.csv", index=False)
    print(f"\n  ✓ {len(feat)} / {len(parquets)} cities profiled")

    # ── Load curated-panel cluster centroids as anchors ──────────────
    panel = pd.read_csv(OUT / "05_panel.csv")
    clusters = pd.read_csv(OUT / "06_discovered_clusters.csv")
    panel = panel.merge(clusters[["name", "cluster_k3", "cluster_k4"]],
                        on="name")

    centroids = panel.groupby("cluster_k3")[["mean_ipr", "extended_fraction"]] \
                     .mean().sort_index()
    print("\n=== Curated-panel cluster centroids (k=3) ===")
    print(centroids)

    # Standardise both axes by panel std for clean Euclidean distance
    sd = panel[["mean_ipr", "extended_fraction"]].std()
    cen_z = centroids / sd
    feat_z = feat[["mean_ipr", "extended_fraction"]] / sd

    # Nearest-centroid classification
    dists = np.zeros((len(feat), len(cen_z)))
    for i, (_, c) in enumerate(cen_z.iterrows()):
        dists[:, i] = np.sqrt(
            (feat_z["mean_ipr"] - c["mean_ipr"]) ** 2
            + (feat_z["extended_fraction"] - c["extended_fraction"]) ** 2
        )
    cluster_labels = cen_z.index.values
    feat["predicted_type"] = cluster_labels[np.argmin(dists, axis=1)]
    feat["dist_to_centroid"] = dists.min(axis=1)

    # Human-readable type labels (re-deriving from centroid order)
    cen_sorted = centroids.reset_index().sort_values("mean_ipr", ascending=False)
    type_label = {}
    labels = ["A: compact / hub-centric",
              "B: distributed / poly-centric",
              "C: mid-sized sprawl",
              "D: outlier-like"][: len(cen_sorted)]
    # We expect: highest IPR=A (compact), lowest IPR=B (distributed), middle=C
    cen_ordered = centroids.sort_values("mean_ipr", ascending=False)
    label_map = {cid: labels[i] for i, cid in enumerate(cen_ordered.index)}
    feat["predicted_label"] = feat["predicted_type"].map(label_map)
    feat.to_csv(OUT / "08_atlas_taxonomy.csv", index=False)

    summary = (feat.groupby("predicted_label")
               .agg(n=("name", "size"),
                    ipr_mean=("mean_ipr", "mean"),
                    ipr_std=("mean_ipr", "std"),
                    ext_mean=("extended_fraction", "mean"),
                    ext_std=("extended_fraction", "std"),
                    N_median=("N", "median"))
               .round(3))
    summary.to_csv(OUT / "08_atlas_summary.csv")
    print("\n=== Atlas-wide taxonomy ===")
    print(summary)

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    palette = {labels[0]: "C0", labels[1]: "C3", labels[2]: "C2"}
    if len(labels) > 3:
        palette[labels[3]] = "C4"

    ax = axes[0]
    for lab in summary.index:
        sub = feat[feat["predicted_label"] == lab]
        ax.scatter(sub["mean_ipr"], sub["extended_fraction"],
                   s=20, alpha=0.5, color=palette[lab],
                   label=f"{lab} (n={len(sub)})")
    # overlay curated-panel anchors
    for _, row in panel.iterrows():
        ax.scatter(row["mean_ipr"], row["extended_fraction"],
                   s=80, marker="*", edgecolor="black",
                   facecolor=palette[label_map[row["cluster_k3"]]],
                   lw=0.7, zorder=5)
    # centroids
    for cid, c in centroids.iterrows():
        ax.scatter(c["mean_ipr"], c["extended_fraction"],
                   s=300, marker="X", edgecolor="black",
                   facecolor=palette[label_map[cid]], lw=1.5, zorder=6)
    ax.set_xlabel("mean IPR  (mode localisation strength)")
    ax.set_ylabel("extended fraction")
    ax.set_title(f"(a) Atlas-wide spectral taxonomy\n"
                 f"{len(feat)} bike-share networks projected onto the panel-discovered plane")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # right: histogram of type counts + IPR by type
    ax = axes[1]
    ax.violinplot(
        [feat.loc[feat["predicted_label"] == lab, "mean_ipr"].values
         for lab in summary.index],
        showmeans=True, showmedians=True,
    )
    ax.set_xticks(range(1, len(summary) + 1))
    ax.set_xticklabels([lab.split(":")[0] for lab in summary.index])
    ax.set_ylabel("mean IPR")
    ax.set_title("(b) Mean IPR distribution per discovered type")
    ax.grid(alpha=0.3)

    fig.suptitle("Atlas-wide spectral taxonomy of bike-share mobility networks",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "08_atlas_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "08_atlas_taxonomy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '08_atlas_taxonomy.pdf'}")


if __name__ == "__main__":
    main()
