"""07_taxonomy.py — characterise the spectral clusters discovered in 06.

Goal: turn the hierarchical clusters into a *physical taxonomy* of
urban bike-share networks. For each cluster we report:
  - prototypical mean IPR + extended fraction (the dominant axes
    from the Mantel analysis)
  - typical size / spatial scale
  - geographic spread (do clusters cross continents?)
  - a 1-line physical label (mono-centric compact vs sprawl, etc.)

Output:
  07_taxonomy.csv          one row per (k, cluster) with summary stats
  07_taxonomy_members.csv  long-format city → cluster (all k)
  07_taxonomy.pdf/png      scatter (IPR vs extended_fraction) coloured
                           by cluster, + dendrogram with cluster bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

OUT = Path(__file__).parent / "output"


def main():
    panel = pd.read_csv(OUT / "05_panel.csv")
    clusters = pd.read_csv(OUT / "06_discovered_clusters.csv")
    M = pd.read_csv(OUT / "05_baseline_matrix.csv", index_col=0).values

    df = panel.merge(clusters[["name", "cluster_k3", "cluster_k4", "cluster_k5"]],
                     on="name")

    # ── 1. Per-cluster summary at k=3 (the most stable level) ────────
    rows = []
    for k in (3, 4, 5):
        col = f"cluster_k{k}"
        for cid in sorted(df[col].unique()):
            sub = df[df[col] == cid]
            rows.append({
                "k": k,
                "cluster": int(cid),
                "n_cities": len(sub),
                "ipr_mean": float(sub["mean_ipr"].mean()),
                "ipr_std": float(sub["mean_ipr"].std()),
                "ext_frac_mean": float(sub["extended_fraction"].mean()),
                "ext_frac_std": float(sub["extended_fraction"].std()),
                "N_median": int(sub["N"].median()),
                "sigma_median": float(sub["sigma"].median()),
                "regions": ", ".join(sorted(sub["region"].unique())),
                "countries": ", ".join(sorted(sub["iso"].unique())),
                "members": ", ".join(sub["name"].tolist()),
            })
    tax = pd.DataFrame(rows)
    tax.to_csv(OUT / "07_taxonomy.csv", index=False)

    print("=== k=3 taxonomy ===")
    for _, r in tax[tax["k"] == 3].iterrows():
        print(f"\nCluster {r['cluster']}  (n={r['n_cities']})")
        print(f"  mean IPR        : {r['ipr_mean']:.3f} ± {r['ipr_std']:.3f}")
        print(f"  extended frac.  : {r['ext_frac_mean']:.3f} ± {r['ext_frac_std']:.3f}")
        print(f"  median N        : {r['N_median']}")
        print(f"  median σ (m)    : {r['sigma_median']:.0f}")
        print(f"  regions         : {r['regions']}")
        print(f"  members         : {r['members'][:120]}{'...' if len(r['members'])>120 else ''}")

    print("\n=== k=4 taxonomy ===")
    for _, r in tax[tax["k"] == 4].iterrows():
        print(f"\nCluster {r['cluster']}  (n={r['n_cities']})")
        print(f"  mean IPR        : {r['ipr_mean']:.3f} ± {r['ipr_std']:.3f}")
        print(f"  extended frac.  : {r['ext_frac_mean']:.3f} ± {r['ext_frac_std']:.3f}")
        print(f"  median N        : {r['N_median']}")
        print(f"  regions         : {r['regions']}")
        print(f"  members         : {r['members'][:120]}{'...' if len(r['members'])>120 else ''}")

    # ── 2. Figure: scatter on the discovered (IPR, ext_frac) plane ───
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    palette = plt.cm.tab10.colors
    for cid in sorted(df["cluster_k3"].unique()):
        sub = df[df["cluster_k3"] == cid]
        ax.scatter(sub["mean_ipr"], sub["extended_fraction"],
                   s=80, color=palette[cid - 1], edgecolor="black", lw=0.5,
                   label=f"cluster {cid} (n={len(sub)})")
        for _, r in sub.iterrows():
            ax.annotate(r["name"].split()[0][:8], (r["mean_ipr"], r["extended_fraction"]),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean IPR (mode localisation strength)")
    ax.set_ylabel("extended fraction (above-bulk eigenvectors)")
    ax.set_title("(a) Spectral taxonomy on the (IPR, extended) plane\n"
                 "30 cities, k=3 hierarchical clusters")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # (b) dendrogram with cluster colours
    ax = axes[1]
    D = 1.0 - M
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = df["name"].tolist()
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7,
               color_threshold=0.7 * D.max(), ax=ax)
    ax.set_title("(b) Hierarchical clustering of the similarity matrix")
    ax.set_ylabel("1 − similarity")

    fig.suptitle("A spectral taxonomy of urban bike-share networks",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "07_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "07_taxonomy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '07_taxonomy.pdf'}")


if __name__ == "__main__":
    main()
