"""06_discover_structure.py — let the data speak.

The pre-registered US-vs-non-US split was falsified on the 30-city panel
(p=0.41). This script does the honest follow-up:

  1. Cluster the baseline similarity matrix hierarchically and label
     the cities by *discovered* clusters.
  2. For each candidate explanation of similarity (continent, country,
     network size, spatial scale σ, mean IPR, extended fraction),
     build a per-pair distance matrix and compute the Mantel
     correlation with the spectral similarity matrix.
  3. Take the top-correlated metadata, binarise it at the median, and
     run a permutation test on the resulting block contrast.

Output (in examples/output/):
  06_metadata_mantel.csv      Mantel correlations + permutation p-values
  06_discovered_clusters.csv  city → discovered-cluster assignment
  06_discovery.json           summary of strongest signal + p-value
  06_discovery.pdf/png        dendrogram + Mantel bar + best-partition test
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from spectral_mobility import permutation_test_block_contrast


OUT = Path(__file__).parent / "output"


def mantel_corr(D1: np.ndarray, D2: np.ndarray, n_perm: int = 5000, seed: int = 0):
    """Mantel correlation between two symmetric distance matrices.

    Returns (r_observed, p_value_two_tailed)."""
    rng = np.random.default_rng(seed)
    n = D1.shape[0]
    iu = np.triu_indices(n, k=1)
    v1 = D1[iu]
    v2 = D2[iu]
    r_obs = float(np.corrcoef(v1, v2)[0, 1])
    # permute one matrix's row/col indices
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        null[k] = np.corrcoef(v1, D2p[iu])[0, 1]
    p = float((np.abs(null) >= abs(r_obs)).mean())
    return r_obs, p


def dist_from_scalar(values: np.ndarray) -> np.ndarray:
    """|x_i - x_j| distance matrix from a 1-D feature."""
    v = np.asarray(values, dtype=float)
    return np.abs(v[:, None] - v[None, :])


def dist_from_categorical(labels: list[str]) -> np.ndarray:
    """0 if same category, 1 otherwise."""
    a = np.asarray(labels)
    return (a[:, None] != a[None, :]).astype(float)


def main():
    panel = pd.read_csv(OUT / "05_panel.csv")
    M = pd.read_csv(OUT / "05_baseline_matrix.csv", index_col=0).values
    names = panel["name"].tolist()
    n = len(names)
    assert M.shape == (n, n)

    # similarity → distance for clustering/Mantel
    D_spec = 1.0 - M
    np.fill_diagonal(D_spec, 0.0)
    # symmetrize numerically
    D_spec = 0.5 * (D_spec + D_spec.T)

    # ── 1. Hierarchical clustering ─────────────────────────────────────
    Z = linkage(squareform(D_spec, checks=False), method="average")
    for k_clusters in (3, 4, 5):
        labels_k = fcluster(Z, t=k_clusters, criterion="maxclust")
        panel[f"cluster_k{k_clusters}"] = labels_k

    # ── 2. Mantel correlations vs candidate metadata ───────────────────
    candidates = {
        "continent (region)":      dist_from_categorical(panel["region"].tolist()),
        "country (ISO)":           dist_from_categorical(panel["iso"].tolist()),
        "log network size N":      dist_from_scalar(np.log(panel["N"].values)),
        "spatial scale σ":         dist_from_scalar(panel["sigma"].values),
        "mean IPR":                dist_from_scalar(panel["mean_ipr"].values),
        "extended fraction":       dist_from_scalar(panel["extended_fraction"].values),
    }
    rows = []
    for label, D_meta in candidates.items():
        r, p = mantel_corr(D_spec, D_meta, n_perm=5000, seed=0)
        rows.append({"feature": label, "mantel_r": r, "p_value": p})
        print(f"  {label:25s}  r={r:+.3f}  p={p:.4f}")
    mantel_df = pd.DataFrame(rows).sort_values("mantel_r", key=lambda s: -s.abs())
    mantel_df.to_csv(OUT / "06_metadata_mantel.csv", index=False)

    # ── 3. Best continuous feature → binarise at median → perm test ────
    cont_features = mantel_df[mantel_df["feature"].isin(
        ["log network size N", "spatial scale σ", "mean IPR", "extended fraction"]
    )].copy()
    cont_features["abs_r"] = cont_features["mantel_r"].abs()
    cont_features = cont_features.sort_values("abs_r", ascending=False)
    top = cont_features.iloc[0]
    feat_to_col = {
        "log network size N":      ("N", lambda x: np.log(x)),
        "spatial scale σ":         ("sigma", lambda x: x),
        "mean IPR":                ("mean_ipr", lambda x: x),
        "extended fraction":       ("extended_fraction", lambda x: x),
    }
    col, transform = feat_to_col[top["feature"]]
    vals = transform(panel[col].values)
    median = float(np.median(vals))
    high_idx = panel.loc[vals >= median, "name"].tolist()
    low_idx = panel.loc[vals < median, "name"].tolist()

    perm_res = permutation_test_block_contrast(
        M, names, group_a=high_idx, group_b=low_idx,
        n_perm=10_000, seed=0,
    )
    perm_res["partition_feature"] = top["feature"]
    perm_res["median_split_value"] = median
    perm_res["mantel_r"] = float(top["mantel_r"])
    print()
    print(f"=== Best continuous split: {top['feature']} ===")
    print(f"  Mantel r:        {top['mantel_r']:+.3f}")
    print(f"  High-group size: {len(high_idx)}")
    print(f"  Low-group size:  {len(low_idx)}")
    print(f"  Observed contrast: {perm_res['observed_contrast']:+.4f}")
    print(f"  Null mean ± std:   {perm_res['null_mean']:+.4f} ± {perm_res['null_std']:.4f}")
    print(f"  p-value (1-tail):  {perm_res['p_value']:.4f}")

    # drop the null distribution for json serialisation
    perm_save = {k: v for k, v in perm_res.items() if k != "null_distribution"}
    with open(OUT / "06_discovery.json", "w") as f:
        json.dump(perm_save, f, indent=2, default=float)

    panel[["name", "iso", "region", "cluster_k3", "cluster_k4", "cluster_k5"]].to_csv(
        OUT / "06_discovered_clusters.csv", index=False
    )

    # ── 4. Composite figure ────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])

    # (a) dendrogram
    ax1 = fig.add_subplot(gs[0, :])
    dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * D_spec.max(), ax=ax1)
    ax1.set_title("(a) Hierarchical clustering of the 30-city spectral similarity matrix")
    ax1.set_ylabel("1 − similarity")

    # (b) Mantel bar
    ax2 = fig.add_subplot(gs[1, 0])
    mdf = mantel_df.copy().sort_values("mantel_r", key=lambda s: s.abs())
    colours = ["C3" if p < 0.05 else "C7" for p in mdf["p_value"]]
    ax2.barh(mdf["feature"], mdf["mantel_r"], color=colours)
    ax2.axvline(0, color="black", lw=0.5)
    ax2.set_xlabel("Mantel correlation with spectral similarity")
    ax2.set_title("(b) Which metadata explains spectral similarity?")
    for i, (r, p) in enumerate(zip(mdf["mantel_r"], mdf["p_value"])):
        ax2.text(r + (0.005 if r >= 0 else -0.005), i,
                 f"p={p:.3f}", va="center",
                 ha="left" if r >= 0 else "right", fontsize=8)

    # (c) null distribution for best feature
    ax3 = fig.add_subplot(gs[1, 1])
    null = perm_res["null_distribution"]
    ax3.hist(null, bins=40, color="C7", alpha=0.7)
    ax3.axvline(perm_res["observed_contrast"], color="C3", lw=2,
                label=f"observed = {perm_res['observed_contrast']:+.3f}")
    ax3.set_xlabel("block contrast (high vs low)")
    ax3.set_ylabel("count")
    ax3.set_title(f"(c) Permutation test — split by {top['feature']}\n"
                  f"p = {perm_res['p_value']:.4f}")
    ax3.legend(fontsize=9)

    fig.suptitle("Structure discovery on the 30-city spectral panel",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "06_discovery.pdf", bbox_inches="tight")
    fig.savefig(OUT / "06_discovery.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '06_discovery.pdf'}")
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
