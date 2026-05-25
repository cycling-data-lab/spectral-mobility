"""09_size_corrected_taxonomy.py — rigorous size-corrected version of
the Atlas-wide spectral taxonomy.

The naive (IPR, extended_fraction) plane is confounded with network
size: small networks have higher IPR by combinatorial accident. To
remove this, we:

  (a) Compute a random-spatial-null IPR for each city (Monte-Carlo
      over uniform random points in the city's bounding box, same N,
      same k_NN, 10 replicates).
  (b) Use the *localisation excess* z = (IPR_obs − IPR_null) /
      σ_null as a size-independent axis.
  (c) Re-classify into types using this excess.
  (d) Stratify by size: split N into (<200, 200-800, >800) and show
      the type distribution in each bin.
  (e) Robust subset: restrict to N ≥ 300 and confirm taxonomy holds.

Output:
  09_atlas_excess.csv          per-city excess + size-corrected coords
  09_size_stratified.csv       n_cities per (size bin × type)
  09_size_corrected.{pdf,png}  4-panel diagnostic figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

from spectral_mobility import CitySpectralProfile


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
N_NULL = 10
MIN_STATIONS = 50
MAX_STATIONS = 6000
K_NN = 5


def random_null_ipr(N: int, lat_range: tuple[float, float],
                    lng_range: tuple[float, float],
                    *, n_replicates: int = N_NULL,
                    seed: int = 0) -> tuple[float, float]:
    """Return (mean, std) of mean_IPR over n_replicates uniform-random
    spatial nulls with N points in the same bounding box."""
    rng = np.random.default_rng(seed)
    iprs = []
    for r in range(n_replicates):
        lat = rng.uniform(lat_range[0], lat_range[1], size=N)
        lng = rng.uniform(lng_range[0], lng_range[1], size=N)
        try:
            null_prof = CitySpectralProfile.from_coords(
                name=f"null_{r}", lat=lat, lng=lng, k_nn=K_NN,
            )
            iprs.append(null_prof.mean_ipr)
        except Exception:
            continue
    if len(iprs) < 3:
        return float("nan"), float("nan")
    return float(np.mean(iprs)), float(np.std(iprs))


def main():
    parquets = sorted(ATLAS_DIR.glob("*.parquet"))
    rows = []
    for i, p in enumerate(parquets, 1):
        try:
            df = pd.read_parquet(p)
            if "lat" not in df or "lng" not in df:
                continue
            df = df.dropna(subset=["lat", "lng"]).drop_duplicates(["lat", "lng"])
            if len(df) < MIN_STATIONS:
                continue
            if len(df) > MAX_STATIONS:
                df = df.sample(n=MAX_STATIONS, random_state=0)
            prof = CitySpectralProfile.from_coords(
                name=p.stem, lat=df["lat"].values, lng=df["lng"].values, k_nn=K_NN,
            )
            null_mu, null_sd = random_null_ipr(
                N=len(df),
                lat_range=(df["lat"].min(), df["lat"].max()),
                lng_range=(df["lng"].min(), df["lng"].max()),
                seed=i,
            )
            rows.append({
                "name": prof.name,
                "N": prof.N,
                "sigma": prof.sigma,
                "mean_ipr": prof.mean_ipr,
                "ext_frac": prof.extended_fraction,
                "ipr_null_mean": null_mu,
                "ipr_null_std": null_sd,
                "ipr_excess_z": (prof.mean_ipr - null_mu) / (null_sd if null_sd > 1e-9 else float("nan")),
                "ipr_ratio": prof.mean_ipr / null_mu if null_mu > 1e-9 else float("nan"),
            })
            if i % 20 == 0:
                z = rows[-1]["ipr_excess_z"]
                print(f"  [{i}/{len(parquets)}] {prof.name:42s}  N={prof.N:5d}  "
                      f"z={z:+.2f}  ratio={rows[-1]['ipr_ratio']:.2f}")
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
            continue

    feat = pd.DataFrame(rows).dropna(subset=["ipr_excess_z"])
    print(f"\n  ✓ {len(feat)} cities with valid null baseline")

    # ── Size strata ──────────────────────────────────────────────────
    bins = [0, 200, 800, 100000]
    labels = ["small (<200)", "medium (200-800)", "large (>800)"]
    feat["size_bin"] = pd.cut(feat["N"], bins=bins, labels=labels)

    # ── Cluster on the size-corrected plane (z, ext_frac) ────────────
    X = feat[["ipr_excess_z", "ext_frac"]].values
    # standardise so both axes weigh equally
    Xz = (X - X.mean(0)) / X.std(0)
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(Xz)
    feat["type_corrected"] = km.labels_
    # name types by mean excess z
    by_z = (feat.groupby("type_corrected")["ipr_excess_z"]
            .mean().sort_values(ascending=False))
    type_label = {}
    type_names = ["A: hyper-localised  (z high)",
                  "B: typical  (z near null)",
                  "C: hyper-delocalised  (z low)"]
    for i, cid in enumerate(by_z.index):
        type_label[cid] = type_names[i]
    feat["type_label"] = feat["type_corrected"].map(type_label)
    feat.to_csv(OUT / "09_atlas_excess.csv", index=False)

    print("\n=== Size-corrected types ===")
    print(feat.groupby("type_label").agg(
        n=("name", "size"),
        z_mean=("ipr_excess_z", "mean"),
        ratio_mean=("ipr_ratio", "mean"),
        ext_mean=("ext_frac", "mean"),
        N_median=("N", "median"),
    ).round(3))

    # cross-tab size × type
    tab = pd.crosstab(feat["size_bin"], feat["type_label"])
    tab.to_csv(OUT / "09_size_stratified.csv")
    print("\n=== Type distribution by size bin ===")
    print(tab)
    print("\n=== Row-normalised (fraction in each type) ===")
    print((tab.div(tab.sum(axis=1), axis=0) * 100).round(1))

    # ── Robust subset N >= 300 ───────────────────────────────────────
    feat_large = feat[feat["N"] >= 300].copy()
    print(f"\n=== Robust subset (N>=300): {len(feat_large)} cities ===")
    print(feat_large.groupby("type_label").agg(
        n=("name", "size"),
        z_mean=("ipr_excess_z", "mean"),
        N_median=("N", "median"),
    ).round(3))

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    palette = {type_names[0]: "C3", type_names[1]: "C0", type_names[2]: "C2"}

    # (a) raw IPR vs N (the confound)
    ax = axes[0, 0]
    for lab in palette:
        sub = feat[feat["type_label"] == lab]
        ax.scatter(sub["N"], sub["mean_ipr"], s=20, alpha=0.6,
                   color=palette[lab], label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("N (number of stations)")
    ax.set_ylabel("mean IPR (raw)")
    ax.set_title("(a) Raw IPR is heavily size-confounded\n"
                 "(smaller networks → mechanically higher IPR)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    # (b) excess z vs N (the corrected axis)
    ax = axes[0, 1]
    for lab in palette:
        sub = feat[feat["type_label"] == lab]
        ax.scatter(sub["N"], sub["ipr_excess_z"], s=20, alpha=0.6,
                   color=palette[lab], label=lab)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("N (number of stations)")
    ax.set_ylabel("IPR excess (z-score vs random spatial null)")
    ax.set_title("(b) IPR excess removes the size confound\n"
                 "the three types now span all sizes")
    ax.grid(alpha=0.3)

    # (c) corrected plane
    ax = axes[1, 0]
    for lab in palette:
        sub = feat[feat["type_label"] == lab]
        ax.scatter(sub["ipr_excess_z"], sub["ext_frac"], s=20, alpha=0.6,
                   color=palette[lab], label=f"{lab} (n={len(sub)})")
    ax.set_xlabel("IPR excess (z-score)")
    ax.set_ylabel("extended fraction")
    ax.set_title("(c) Size-corrected taxonomy plane")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) size × type heatmap
    ax = axes[1, 1]
    tab_pct = (tab.div(tab.sum(axis=1), axis=0) * 100).round(1)
    im = ax.imshow(tab_pct.values, aspect="auto", cmap="Blues")
    for i in range(tab_pct.shape[0]):
        for j in range(tab_pct.shape[1]):
            ax.text(j, i, f"{tab_pct.values[i, j]:.0f}%\n(n={tab.values[i, j]})",
                    ha="center", va="center", fontsize=10,
                    color="white" if tab_pct.values[i, j] > 50 else "black")
    ax.set_xticks(range(tab_pct.shape[1]))
    ax.set_xticklabels([c.split(":")[0] for c in tab_pct.columns],
                        rotation=0)
    ax.set_yticks(range(tab_pct.shape[0]))
    ax.set_yticklabels(tab_pct.index)
    ax.set_xlabel("size-corrected type")
    ax.set_ylabel("size bin")
    ax.set_title("(d) Type composition by size bin\n"
                 "if types reflect real geometry, columns should\nhave non-zero entries in every row")
    plt.colorbar(im, ax=ax, label="% within size bin")

    fig.suptitle(
        "Atlas-wide spectral taxonomy — size-corrected via random spatial null",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(OUT / "09_size_corrected.pdf", bbox_inches="tight")
    fig.savefig(OUT / "09_size_corrected.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '09_size_corrected.pdf'}")


if __name__ == "__main__":
    main()
