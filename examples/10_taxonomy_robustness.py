"""10_taxonomy_robustness.py — two robustness tests of the
size-corrected Atlas taxonomy.

(1) **Per-city bootstrap stability** — for each Atlas city we
    resample 70% of its stations B times, recompute (IPR excess z,
    extended fraction), re-classify by nearest centroid (the
    centroids are *fixed* from the full-data k-means run in 09).  The
    stability is the fraction of replicates that land in the same
    type as the full-data assignment.

(2) **k_NN sensitivity** — repeat the *whole* pipeline (per-city
    spectrum + random null + k-means classification) at k=3 and k=7
    and report the adjusted Rand index (ARI) of the type assignments
    against the k=5 reference (which is what 09 produced).

Output:
  10_stability_per_city.csv   per-city bootstrap stability
  10_knn_ari.csv              ARI(k_ref=5, k=k') for k' in {3, 7}
  10_robustness.{pdf,png}     stability histogram + per-type stability
                              + ARI bar
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from spectral_mobility import CitySpectralProfile


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
N_BOOT = 30           # per-city bootstrap replicates (smaller than 200; this is per-city)
N_NULL_BOOT = 5       # nulls per bootstrap (smaller for compute)
N_NULL_FULL = 10
MIN_STATIONS = 50
MAX_STATIONS = 6000
SUBSAMPLE_FRACTION = 0.7


def per_city_features(lat, lng, k_nn):
    prof = CitySpectralProfile.from_coords(
        name="x", lat=lat, lng=lng, k_nn=k_nn,
    )
    return prof.mean_ipr, prof.extended_fraction, prof.N


def random_null_ipr(N, lat_range, lng_range, k_nn, n_rep, seed):
    rng = np.random.default_rng(seed)
    iprs = []
    for r in range(n_rep):
        la = rng.uniform(lat_range[0], lat_range[1], size=N)
        ln = rng.uniform(lng_range[0], lng_range[1], size=N)
        try:
            p = CitySpectralProfile.from_coords(name=f"n{r}", lat=la, lng=ln, k_nn=k_nn)
            iprs.append(p.mean_ipr)
        except Exception:
            continue
    if len(iprs) < 3:
        return float("nan"), float("nan")
    return float(np.mean(iprs)), float(np.std(iprs))


def excess_z(ipr_obs, ipr_null_mean, ipr_null_std):
    if ipr_null_std is None or ipr_null_std < 1e-9:
        return float("nan")
    return (ipr_obs - ipr_null_mean) / ipr_null_std


def run_full_pipeline_for_k(k_nn, seed=0):
    """Replicate of script 09 at a different k_nn. Returns DataFrame
    with columns: name, ipr_excess_z, ext_frac, type_corrected (0/1/2)."""
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
            ipr, ext, N = per_city_features(df["lat"].values, df["lng"].values, k_nn)
            null_mu, null_sd = random_null_ipr(
                N, (df["lat"].min(), df["lat"].max()),
                (df["lng"].min(), df["lng"].max()),
                k_nn, N_NULL_FULL, seed + i,
            )
            z = excess_z(ipr, null_mu, null_sd)
            if not np.isfinite(z):
                continue
            rows.append({"name": p.stem, "ipr_excess_z": z, "ext_frac": ext})
        except Exception:
            continue
    feat = pd.DataFrame(rows)
    X = feat[["ipr_excess_z", "ext_frac"]].values
    Xz = (X - X.mean(0)) / X.std(0)
    km = KMeans(n_clusters=3, random_state=0, n_init=10).fit(Xz)
    feat["type"] = km.labels_
    return feat, km, X.mean(0), X.std(0)


def main():
    print("=== (1) Per-city bootstrap stability (k=5) ===\n")
    # Load the reference centroids from 09
    ref = pd.read_csv(OUT / "09_atlas_excess.csv")
    X_ref = ref[["ipr_excess_z", "ext_frac"]].values
    mu = X_ref.mean(0); sd = X_ref.std(0)
    Xz_ref = (X_ref - mu) / sd
    km_ref = KMeans(n_clusters=3, random_state=0, n_init=10).fit(Xz_ref)
    ref["type_full"] = km_ref.labels_
    centroids_z = km_ref.cluster_centers_

    # iterate cities (limit to those present in 09_atlas_excess.csv)
    parquets = {p.stem: p for p in ATLAS_DIR.glob("*.parquet")}
    stab_rows = []
    for i, (_, row) in enumerate(ref.iterrows(), 1):
        name = row["name"]
        if name not in parquets:
            continue
        try:
            df = pd.read_parquet(parquets[name])
            df = df.dropna(subset=["lat", "lng"]).drop_duplicates(["lat", "lng"])
            if len(df) > MAX_STATIONS:
                df = df.sample(n=MAX_STATIONS, random_state=0)
            N_full = len(df)
        except Exception:
            continue

        rng = np.random.default_rng(hash(name) % (2**32))
        same = 0
        valid = 0
        for b in range(N_BOOT):
            n_keep = max(MIN_STATIONS, int(round(SUBSAMPLE_FRACTION * N_full)))
            n_keep = min(n_keep, N_full)
            idx = rng.choice(N_full, size=n_keep, replace=False)
            sub = df.iloc[idx]
            try:
                ipr, ext, _ = per_city_features(sub["lat"].values, sub["lng"].values, 5)
                null_mu, null_sd = random_null_ipr(
                    n_keep,
                    (sub["lat"].min(), sub["lat"].max()),
                    (sub["lng"].min(), sub["lng"].max()),
                    5, N_NULL_BOOT, hash((name, b)) % (2**32),
                )
                z = excess_z(ipr, null_mu, null_sd)
                if not np.isfinite(z):
                    continue
                pt = np.array([(z - mu[0]) / sd[0], (ext - mu[1]) / sd[1]])
                d = np.linalg.norm(centroids_z - pt, axis=1)
                pred = int(np.argmin(d))
                if pred == row["type_full"]:
                    same += 1
                valid += 1
            except Exception:
                continue
        if valid == 0:
            continue
        stab_rows.append({
            "name": name,
            "type_full": int(row["type_full"]),
            "type_label": row["type_label"],
            "N": int(row["N"]),
            "stability": same / valid,
            "n_valid_boot": valid,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(ref)}] {name[:40]:40s}  "
                  f"stab={same/valid:.2f} ({valid} reps)")

    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT / "10_stability_per_city.csv", index=False)
    print(f"\n  ✓ {len(stab)} cities with stability scores")
    print(f"\n  median stability:  {stab['stability'].median():.3f}")
    print(f"  fraction >= 0.80:  {(stab['stability'] >= 0.80).mean():.2%}")
    print("\n  per-type stability:")
    print(stab.groupby("type_label").agg(
        n=("name", "size"),
        median_stab=("stability", "median"),
        frac_above_80=("stability", lambda s: (s >= 0.80).mean()),
    ).round(3))

    # ── (2) k_NN sensitivity ─────────────────────────────────────────
    print("\n\n=== (2) k_NN sensitivity ===")
    print("Running full pipeline at k=3 ...")
    feat_k3, _, _, _ = run_full_pipeline_for_k(3, seed=100)
    print(f"  {len(feat_k3)} cities profiled")
    print("Running full pipeline at k=7 ...")
    feat_k7, _, _, _ = run_full_pipeline_for_k(7, seed=200)
    print(f"  {len(feat_k7)} cities profiled")

    # Align on common cities with the k=5 reference
    ref_lookup = ref[["name", "type_full"]].rename(columns={"type_full": "type_k5"})
    merged_k3 = ref_lookup.merge(
        feat_k3[["name", "type"]].rename(columns={"type": "type_k3"}),
        on="name",
    )
    merged_k7 = ref_lookup.merge(
        feat_k7[["name", "type"]].rename(columns={"type": "type_k7"}),
        on="name",
    )
    ari_k3 = adjusted_rand_score(merged_k3["type_k5"], merged_k3["type_k3"])
    ari_k7 = adjusted_rand_score(merged_k7["type_k5"], merged_k7["type_k7"])
    print(f"\n  ARI(k=3 vs k=5) = {ari_k3:.3f}  (n={len(merged_k3)})")
    print(f"  ARI(k=7 vs k=5) = {ari_k7:.3f}  (n={len(merged_k7)})")

    pd.DataFrame([
        {"k_alt": 3, "n_common": len(merged_k3), "ARI": ari_k3},
        {"k_alt": 7, "n_common": len(merged_k7), "ARI": ari_k7},
    ]).to_csv(OUT / "10_knn_ari.csv", index=False)

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.hist(stab["stability"], bins=20, color="C0", alpha=0.8, edgecolor="black")
    ax.axvline(stab["stability"].median(), color="C3", lw=2,
               label=f"median = {stab['stability'].median():.2f}")
    ax.axvline(0.80, color="black", lw=1, ls=":",
               label=f"{(stab['stability'] >= 0.80).mean():.0%} pass 0.80")
    ax.set_xlabel("bootstrap stability (fraction of reps in same type)")
    ax.set_ylabel("number of cities")
    ax.set_title("(a) Per-city bootstrap stability\n"
                 f"30 reps × {len(stab)} cities, 70% subsample")
    ax.legend()

    ax = axes[1]
    types = stab.groupby("type_label")["stability"].apply(list).to_dict()
    pos = list(range(1, len(types) + 1))
    parts = ax.violinplot(list(types.values()), positions=pos, showmeans=True,
                          showmedians=True)
    ax.set_xticks(pos)
    ax.set_xticklabels([k.split(":")[0] for k in types.keys()], rotation=0)
    ax.set_ylabel("bootstrap stability")
    ax.set_ylim(0, 1.05)
    ax.set_title("(b) Stability by predicted type\n(higher = type assignment more robust)")
    ax.axhline(0.80, color="black", lw=0.7, ls=":")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.bar([3, 5, 7], [ari_k3, 1.0, ari_k7],
           color=["C0", "C7", "C0"], edgecolor="black")
    for k_val, ari_val in [(3, ari_k3), (5, 1.0), (7, ari_k7)]:
        ax.text(k_val, ari_val + 0.02, f"{ari_val:.3f}",
                ha="center", fontsize=10)
    ax.set_xlabel("k for k-NN graph")
    ax.set_ylabel("Adjusted Rand Index vs k=5 reference")
    ax.set_title("(c) k_NN sensitivity\n(higher = taxonomy stable across k)")
    ax.set_xticks([3, 5, 7])
    ax.set_ylim(0, 1.1)
    ax.axhline(0.70, color="black", lw=0.7, ls=":")
    ax.grid(alpha=0.3)

    fig.suptitle("Taxonomy robustness — bootstrap stability + k_NN sensitivity",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "10_robustness.pdf", bbox_inches="tight")
    fig.savefig(OUT / "10_robustness.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '10_robustness.pdf'}")


if __name__ == "__main__":
    main()
