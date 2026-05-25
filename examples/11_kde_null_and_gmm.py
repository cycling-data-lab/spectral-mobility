"""11_kde_null_and_gmm.py — implement Gemini's 4 corrections to the
Atlas-wide taxonomy:

  (a) KDE-based null model.  Sample N points from a Gaussian KDE of
      the city's stations (Scott's rule, scaled up 1.5×).  This
      destroys micro-structure (hubs) while respecting the macro
      shape (no stations in rivers / oceans / outside the urban
      footprint), unlike the bounding-box uniform null in 09.

  (b) GMM with BIC for k ∈ [1, 8] on the size-corrected plane
      (z_excess, extended_fraction).  If BIC selects k=1 the
      "taxonomy" is a continuum and we report tertiles of z instead.

  (c) Finite-size falsification of type C.  Take the strongest type-A
      cities and downsample them progressively to N=200, 100, 75.
      If their z_excess collapses into the type-C zone, type C is a
      finite-size artefact.

  (d) Output the continuous (z, ext_frac) coordinates that the OSF
      pre-registration document will lock in.

Output:
  11_kde_excess.csv             per-city z under KDE null
  11_gmm_bic.csv                BIC for k ∈ [1, 8]
  11_finite_size_test.csv       downsample trajectories of large type-A
  11_kde_taxonomy.{pdf,png}     4-panel: KDE vs bbox null, BIC curve,
                                continuous plane, downsample test
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

from spectral_mobility import CitySpectralProfile


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
N_NULL = 10
MIN_STATIONS = 50
MAX_STATIONS = 6000
K_NN = 5
BW_FACTOR = 1.5  # over-smooth the KDE: destroys micro-structure


def kde_null_ipr(lat, lng, N, k_nn, n_rep, seed):
    """Generate n_rep Monte-Carlo null layouts by sampling from a
    smoothed KDE of the observed stations. Returns (mean, std) of
    mean_IPR over replicates."""
    rng = np.random.default_rng(seed)
    pts = np.vstack([np.asarray(lat), np.asarray(lng)])
    try:
        kde = gaussian_kde(pts)
        kde.set_bandwidth(kde.factor * BW_FACTOR)
    except Exception:
        return float("nan"), float("nan")
    iprs = []
    for r in range(n_rep):
        try:
            sample = kde.resample(N, seed=rng.integers(0, 2**31)).T
            la = sample[:, 0]
            ln = sample[:, 1]
            p = CitySpectralProfile.from_coords(
                name=f"null{r}", lat=la, lng=ln, k_nn=k_nn,
            )
            iprs.append(p.mean_ipr)
        except Exception:
            continue
    if len(iprs) < 3:
        return float("nan"), float("nan")
    return float(np.mean(iprs)), float(np.std(iprs))


def main():
    parquets = sorted(ATLAS_DIR.glob("*.parquet"))
    print(f"=== (a) KDE-null spectral features over {len(parquets)} Atlas parquets ===\n")
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
            null_mu, null_sd = kde_null_ipr(
                df["lat"].values, df["lng"].values,
                N=len(df), k_nn=K_NN, n_rep=N_NULL, seed=i,
            )
            if not np.isfinite(null_sd) or null_sd < 1e-9:
                continue
            z = (prof.mean_ipr - null_mu) / null_sd
            ratio = prof.mean_ipr / null_mu if null_mu > 1e-9 else float("nan")
            rows.append({
                "name": prof.name,
                "N": prof.N,
                "mean_ipr": prof.mean_ipr,
                "ext_frac": prof.extended_fraction,
                "kde_null_mean": null_mu,
                "kde_null_std": null_sd,
                "z_kde": z,
                "ratio_kde": ratio,
            })
            if i % 20 == 0:
                print(f"  [{i}/{len(parquets)}] {prof.name[:42]:42s}  N={prof.N:5d}  "
                      f"z_kde={z:+.2f}  ratio={ratio:.2f}")
        except Exception as e:
            print(f"  ✗ {p.name}: {e}")
            continue

    feat = pd.DataFrame(rows)
    feat.to_csv(OUT / "11_kde_excess.csv", index=False)
    print(f"\n  ✓ {len(feat)} cities with valid KDE null")

    # Merge with the bbox null for comparison
    bbox = pd.read_csv(OUT / "09_atlas_excess.csv")[
        ["name", "ipr_excess_z", "ipr_ratio"]
    ].rename(columns={"ipr_excess_z": "z_bbox", "ipr_ratio": "ratio_bbox"})
    cmp = feat.merge(bbox, on="name", how="inner")
    print(f"\n  Cities in both KDE and bbox: {len(cmp)}")
    print(f"  Pearson corr(z_kde, z_bbox): {cmp[['z_kde','z_bbox']].corr().iloc[0,1]:.3f}")
    print(f"  median |z_kde - z_bbox|:     {(cmp['z_kde'] - cmp['z_bbox']).abs().median():.2f}")
    print(f"  mean   z_kde - z_bbox:       {(cmp['z_kde'] - cmp['z_bbox']).mean():+.2f}")

    # ── (b) GMM with BIC ─────────────────────────────────────────────
    print("\n=== (b) GMM with BIC for k in [1, 8] ===")
    X = feat[["z_kde", "ext_frac"]].values
    Xz = (X - X.mean(0)) / X.std(0)

    bic_rows = []
    aic_rows = []
    for k in range(1, 9):
        gm = GaussianMixture(n_components=k, random_state=0,
                             covariance_type="full", n_init=5).fit(Xz)
        bic_rows.append({"k": k, "BIC": gm.bic(Xz), "AIC": gm.aic(Xz),
                          "log_lik": gm.score(Xz) * len(Xz)})
    bic_df = pd.DataFrame(bic_rows)
    bic_df.to_csv(OUT / "11_gmm_bic.csv", index=False)
    print(bic_df.round(2))
    k_best = int(bic_df.loc[bic_df["BIC"].idxmin(), "k"])
    print(f"\n  BIC-best k = {k_best}")

    # tertiles of z_kde as fallback (continuous descriptor)
    feat["tertile"] = pd.qcut(feat["z_kde"], q=3,
                              labels=["T1: low z (delocalised)",
                                      "T2: medium z",
                                      "T3: high z (localised)"])

    # also fit a GMM at the BIC-best k and at k=3 for comparison
    gm_best = GaussianMixture(n_components=k_best, random_state=0,
                              covariance_type="full", n_init=5).fit(Xz)
    feat["gmm_label_best"] = gm_best.predict(Xz)
    if k_best != 3:
        gm3 = GaussianMixture(n_components=3, random_state=0,
                              covariance_type="full", n_init=5).fit(Xz)
        feat["gmm_label_k3"] = gm3.predict(Xz)
    else:
        feat["gmm_label_k3"] = feat["gmm_label_best"]

    print("\n  tertile summary on z_kde:")
    print(feat.groupby("tertile").agg(
        n=("name", "size"),
        z_mean=("z_kde", "mean"),
        z_min=("z_kde", "min"),
        z_max=("z_kde", "max"),
        N_median=("N", "median"),
    ).round(3))

    feat.to_csv(OUT / "11_kde_excess.csv", index=False)

    # ── (c) Finite-size falsification test ───────────────────────────
    print("\n=== (c) Finite-size falsification: downsample large type-A cities ===")
    # Take the top-5 highest z_kde cities with N >= 800
    big_a = (feat[(feat["N"] >= 800)]
             .sort_values("z_kde", ascending=False)
             .head(8))
    print(f"  large-N hyper-localised cities to downsample:")
    print(big_a[["name", "N", "z_kde", "ratio_kde"]].round(2).to_string(index=False))

    targets = [800, 400, 200, 100]
    test_rows = []
    for _, row in big_a.iterrows():
        # reload original parquet
        try:
            df = pd.read_parquet(ATLAS_DIR / f"{row['name']}.parquet")
            df = df.dropna(subset=["lat", "lng"]).drop_duplicates(["lat", "lng"])
            if len(df) > MAX_STATIONS:
                df = df.sample(n=MAX_STATIONS, random_state=0)
            for n_target in targets:
                if n_target > len(df):
                    continue
                rng = np.random.default_rng(hash(row["name"]) % (2**32))
                # 5 subsamples per N to estimate spread
                zs = []
                for r in range(5):
                    idx = rng.choice(len(df), size=n_target, replace=False)
                    sub = df.iloc[idx]
                    try:
                        prof = CitySpectralProfile.from_coords(
                            name=f"{row['name']}_n{n_target}_r{r}",
                            lat=sub["lat"].values, lng=sub["lng"].values, k_nn=K_NN,
                        )
                        null_mu, null_sd = kde_null_ipr(
                            sub["lat"].values, sub["lng"].values,
                            N=n_target, k_nn=K_NN, n_rep=5,
                            seed=hash((row["name"], n_target, r)) % (2**32),
                        )
                        if np.isfinite(null_sd) and null_sd > 1e-9:
                            zs.append((prof.mean_ipr - null_mu) / null_sd)
                    except Exception:
                        continue
                if zs:
                    test_rows.append({
                        "name": row["name"],
                        "N_full": int(row["N"]),
                        "z_full": float(row["z_kde"]),
                        "N_subsample": n_target,
                        "z_subsample_mean": float(np.mean(zs)),
                        "z_subsample_std": float(np.std(zs)),
                        "n_replicates": len(zs),
                    })
                    print(f"  {row['name'][:38]:38s}  "
                          f"N={n_target:4d}  z={np.mean(zs):+.2f} ± {np.std(zs):.2f}")
        except Exception as e:
            print(f"  ✗ {row['name']}: {e}")
            continue
    finite = pd.DataFrame(test_rows)
    finite.to_csv(OUT / "11_finite_size_test.csv", index=False)

    # check: at N=100, did the hyper-localised cities drop into type-C zone?
    if len(finite):
        at100 = finite[finite["N_subsample"] == 100]
        if len(at100):
            tertile_c_max = feat[feat["tertile"] == "T1: low z (delocalised)"]["z_kde"].max()
            print(f"\n  Tertile T1 (delocalised) upper bound: z = {tertile_c_max:+.2f}")
            n_collapse = (at100["z_subsample_mean"] <= tertile_c_max).sum()
            print(f"  Type-A cities downsampled to N=100 that fall into T1: "
                  f"{n_collapse}/{len(at100)}")
            print(f"  → if all {len(at100)} fall in, type C is partly a finite-size artefact.")

    # ── (d) Figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # (a) KDE vs bbox null scatter
    ax = axes[0, 0]
    ax.scatter(cmp["z_bbox"], cmp["z_kde"], s=20, alpha=0.7, color="C0",
               edgecolor="black", lw=0.3)
    lo = min(cmp["z_bbox"].min(), cmp["z_kde"].min()) - 1
    hi = max(cmp["z_bbox"].max(), cmp["z_kde"].max()) + 1
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    ax.set_xlabel("z under bbox-uniform null (script 09)")
    ax.set_ylabel("z under KDE null (script 11)")
    ax.set_title(f"(a) KDE null vs bbox null\n"
                 f"corr = {cmp[['z_kde','z_bbox']].corr().iloc[0,1]:.3f}, "
                 f"shift = {(cmp['z_kde']-cmp['z_bbox']).mean():+.2f}")
    ax.grid(alpha=0.3)

    # (b) BIC curve
    ax = axes[0, 1]
    ax.plot(bic_df["k"], bic_df["BIC"], "o-", color="C0", label="BIC")
    ax.plot(bic_df["k"], bic_df["AIC"], "s--", color="C7", label="AIC", alpha=0.7)
    ax.axvline(k_best, color="C3", lw=2, ls=":",
               label=f"BIC-best k = {k_best}")
    ax.set_xlabel("k (number of Gaussian components)")
    ax.set_ylabel("information criterion")
    ax.set_title(f"(b) GMM model selection on (z_kde, ext_frac)")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) continuous plane, tertile-coloured
    ax = axes[1, 0]
    tert_palette = {"T1: low z (delocalised)": "C2",
                    "T2: medium z": "C0",
                    "T3: high z (localised)": "C3"}
    for lab, col in tert_palette.items():
        sub = feat[feat["tertile"] == lab]
        ax.scatter(sub["z_kde"], sub["ext_frac"], s=25, alpha=0.6,
                   color=col, label=f"{lab} (n={len(sub)})")
    ax.set_xlabel("z under KDE null (corrected, size-independent)")
    ax.set_ylabel("extended fraction")
    ax.set_title("(c) Continuous taxonomy plane\n(tertiles of z_kde)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (d) finite-size test trajectories
    ax = axes[1, 1]
    if len(finite):
        for name, grp in finite.groupby("name"):
            grp = grp.sort_values("N_subsample", ascending=False)
            ax.plot(grp["N_subsample"], grp["z_subsample_mean"],
                    "o-", alpha=0.7, label=name[:18])
        # overlay tertile boundaries
        for lab, col in tert_palette.items():
            sub_max = feat[feat["tertile"] == lab]["z_kde"].max()
            sub_min = feat[feat["tertile"] == lab]["z_kde"].min()
            ax.axhspan(sub_min, sub_max, color=col, alpha=0.1)
        ax.set_xlabel("N (downsampled)")
        ax.set_ylabel("z_kde at this N")
        ax.set_title("(d) Finite-size falsification of type C\n"
                     "hyper-localised cities downsampled — do they fall into T1?")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)

    fig.suptitle("KDE-null taxonomy + GMM-BIC + finite-size falsification",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "11_kde_taxonomy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "11_kde_taxonomy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '11_kde_taxonomy.pdf'}")


if __name__ == "__main__":
    main()
