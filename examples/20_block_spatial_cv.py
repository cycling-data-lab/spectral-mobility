"""20_block_spatial_cv.py — Block Spatial Cross-Validation for the
pre-reg v2 hypotheses.

The random 70/30 holdout in scripts 14–18 leaks spatial autocorrelation:
a held-out station can be 50 m from a training station, so the model
effectively interpolates between very close points.  The strict
inductive Nyström helps but does not fully simulate a "deploy in a
new area" use case.  Per Gemini round 5, this is the most likely
attack vector from a spatial-econometrics reviewer.

Protocol:
  1. K-means cluster the station coordinates into K_BLOCKS = 10
     contiguous spatial blocks.
  2. Leave-one-block-out CV: hold out one entire block at a time,
     train on the other 9.
  3. Compute baseline Ridge and augmented Ridge R² on the held-out
     block, average across blocks.
  4. Refit H1a (per-city OLS) and H1b (LMM gain ~ z_kde, z>−5).

Compare with the 70/30 random-CV results to quantify spatial leakage.

Output:
  20_block_cv_results.csv  per-(city, block) R²_base, R²_aug, gain
  20_block_cv_lmm.json     H1a + H1b under block-CV
  20_block_cv.{pdf,png}    4-panel comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree

from spectral_mobility import (
    build_geographic_knn,
    dirichlet_energy,
    spectral_decomposition,
    symmetric_normalised_laplacian,
    unnormalised_laplacian,
)


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
PRED_DIR = Path("/Users/rfosse/cesi-research/imd-national-catalogue/"
                "paper_demand/experiments/outputs")
K_NN = 5
K_SPEC = 10
K_BLOCKS = 10
MIN_MATCHED = 60
MIN_TRAIN = 30
SEED = 2026


def _to_dense(W):
    return np.asarray(W.toarray()) if hasattr(W, "toarray") else np.asarray(W)


def find_matching_atlas(pred_ids, atlas_files):
    best = None; best_n = 0
    for atlas_path in atlas_files:
        try:
            df = pd.read_parquet(atlas_path, columns=["station_id"])
        except Exception:
            continue
        n = len(set(df["station_id"].astype(str)) & pred_ids)
        if n > best_n:
            best = atlas_path; best_n = n
    return best, best_n


def nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn):
    from spectral_mobility.graph import EARTH_RADIUS_METRES
    train_rad = np.deg2rad(coords_tr); test_rad = np.deg2rad(coords_te)
    tree = BallTree(train_rad, metric="haversine")
    dist_rad, idx = tree.query(test_rad, k=k_nn)
    dist_m = dist_rad * EARTH_RADIUS_METRES
    w = np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, Uk[idx])


def inductive_block_fit(coords, X_bare, y, train_mask, K, k_nn):
    """Ridge baseline + augmented, where train/test split is given by
    a boolean mask (block-CV)."""
    test_mask = ~train_mask
    coords_tr = coords[train_mask]; coords_te = coords[test_mask]
    X_tr = X_bare[train_mask]; X_te = X_bare[test_mask]
    y_tr = y[train_mask]; y_te = y[test_mask]
    if len(y_tr) < MIN_TRAIN or len(y_te) < 5:
        return float("nan"), float("nan")
    r2_base = r2_score(y_te, Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te))
    W_tr, sigma = build_geographic_knn(coords_tr[:, 0], coords_tr[:, 1], k=k_nn)
    L_tr = _to_dense(symmetric_normalised_laplacian(_to_dense(W_tr)))
    _, eigvecs_tr = spectral_decomposition(L_tr)
    Uk = eigvecs_tr[:, :K]
    Uk_te = nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn)
    aug = Ridge(alpha=1.0).fit(np.column_stack([X_tr, Uk]), y_tr)
    r2_aug = r2_score(y_te, aug.predict(np.column_stack([X_te, Uk_te])))
    return float(r2_base), float(r2_aug)


def main():
    atlas_files = sorted(ATLAS_DIR.glob("*.parquet"))
    pred_files = sorted(PRED_DIR.glob("d*_predictions.parquet"))
    kde = pd.read_csv(OUT / "11_kde_excess.csv")
    kde_lookup = dict(zip(kde["name"], kde["z_kde"]))

    rows = []
    for pred_path in pred_files:
        try:
            preds = pd.read_parquet(pred_path, columns=["station_id", "y_true_log"])
            preds["station_id"] = preds["station_id"].astype(str)
            per_station = preds.groupby("station_id")["y_true_log"].mean()
        except Exception:
            continue
        atlas_path, n_ov = find_matching_atlas(set(per_station.index), atlas_files)
        if atlas_path is None or n_ov < MIN_MATCHED:
            continue
        atlas = pd.read_parquet(atlas_path)
        atlas["station_id"] = atlas["station_id"].astype(str)
        bare_cols = [c for c in [
            "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
            "elevation_m", "topography_roughness_index",
            "n_stations_within_500m", "n_stations_within_1km",
            "catchment_density_per_km2",
        ] if c in atlas.columns]
        atlas = atlas.dropna(subset=["lat", "lng"] + bare_cols) \
                     .drop_duplicates("station_id")
        merged = atlas.merge(per_station.rename("y").reset_index(),
                             on="station_id", how="inner")
        if len(merged) < MIN_MATCHED:
            continue
        z_kde = kde_lookup.get(atlas_path.stem, np.nan)
        if not np.isfinite(z_kde):
            continue
        coords = merged[["lat", "lng"]].values
        X_bare = merged[bare_cols].values.astype(float)
        X_bare = (X_bare - X_bare.mean(0)) / (X_bare.std(0) + 1e-9)
        y = merged["y"].values
        n = len(merged)
        city = pred_path.stem.replace("_predictions", "")

        # Dirichlet on L_unnorm (per pre-reg v2)
        W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=K_NN)
        W = _to_dense(W)
        L_un = _to_dense(unnormalised_laplacian(W))
        D_un = float(dirichlet_energy(L_un, y))

        # Block partitioning via k-means in coord space
        k_blocks = min(K_BLOCKS, n // (MIN_TRAIN + 5))
        if k_blocks < 3:
            print(f"  ⚠ {city[:28]:28s} too small for {K_BLOCKS} blocks")
            continue
        km = KMeans(n_clusters=k_blocks, random_state=SEED, n_init=10).fit(coords)
        labels = km.labels_

        # Leave-one-block-out
        for b in range(k_blocks):
            train_mask = labels != b
            r2_base, r2_aug = inductive_block_fit(
                coords, X_bare, y, train_mask, K_SPEC, K_NN,
            )
            if not (np.isfinite(r2_base) and np.isfinite(r2_aug)):
                continue
            rows.append({
                "city": city, "atlas": atlas_path.stem,
                "block": b, "N": n,
                "n_train": int(train_mask.sum()),
                "n_test": int((~train_mask).sum()),
                "z_kde": float(z_kde),
                "dirichlet_unnorm": D_un,
                "r2_base": r2_base,
                "r2_aug": r2_aug,
                "gain": r2_aug - r2_base,
            })

        sub = [r for r in rows if r["city"] == city]
        if sub:
            print(f"  ✓ {city[:28]:28s}  N={n:4d}  z_kde={z_kde:+.2f}  "
                  f"R²_aug={np.mean([r['r2_aug'] for r in sub]):.3f}  "
                  f"gain={np.mean([r['gain'] for r in sub]):+.3f}  "
                  f"({len(sub)} blocks)")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "20_block_cv_results.csv", index=False)
    print(f"\n  ✓ {df['city'].nunique()} cities × ~{K_BLOCKS} blocks = {len(df)} obs")

    summary = {}
    if df.empty:
        print("No results"); return

    # ── H1a — per-city OLS on Dirichlet + log N ──────────────────────
    per_city = df.groupby("city").agg(
        z_kde=("z_kde", "first"),
        D=("dirichlet_unnorm", "first"),
        N=("N", "first"),
        r2_aug=("r2_aug", "mean"),
        r2_base=("r2_base", "mean"),
        gain=("gain", "mean"),
    ).reset_index()
    Xm = sm.add_constant(np.column_stack([
        per_city["D"], np.log(per_city["N"])
    ]))
    ols = sm.OLS(per_city["r2_aug"], Xm).fit()
    print(f"\n=== H1a under block-CV (per-city OLS, n={len(per_city)}) ===")
    print(ols.summary())
    summary["h1a_block"] = {
        "n_cities": int(len(per_city)),
        "dirichlet_beta": float(ols.params[1]),
        "dirichlet_p": float(ols.pvalues[1]),
        "log_N_beta": float(ols.params[2]),
        "log_N_p": float(ols.pvalues[2]),
        "R2_overall": float(ols.rsquared),
        "sign_correct": bool(ols.params[1] < 0),
        "sig_001": bool(ols.pvalues[1] <= 0.01),
    }

    # ── H1b — LMM gain ~ z_kde on z>−5 ───────────────────────────────
    df_h1b = df[df["z_kde"] > -5].copy()
    n_h1b = df_h1b["city"].nunique()
    md = smf.mixedlm("gain ~ z_kde", df_h1b, groups=df_h1b["city"])
    try:
        res = md.fit(method="lbfgs")
        beta = float(res.fe_params["z_kde"])
        p_val = float(res.pvalues["z_kde"])
        print(f"\n=== H1b under block-CV (LMM, n_cities={n_h1b}) ===")
        print(res.summary())
        summary["h1b_block"] = {
            "n_cities": int(n_h1b), "z_kde_beta": beta, "z_kde_p": p_val,
            "sign_correct": bool(beta < 0), "sig_005": bool(p_val <= 0.05),
        }
    except Exception as e:
        print(f"\nH1b block-CV LMM failed: {e}")
        summary["h1b_block"] = {"error": str(e)}

    # Comparison vs random-CV (from 18_*)
    try:
        rand = pd.read_csv(OUT / "18_pre_reg_v2_check.csv")
        rand_per_city = rand.groupby("city").agg(
            z_kde=("z_kde", "first"),
            D=("dirichlet_unnorm", "first"),
            N=("N", "first"),
            r2_aug=("r2_aug", "mean"),
            gain=("gain", "mean"),
        ).reset_index()
        merged = per_city.merge(
            rand_per_city[["city", "r2_aug", "gain"]].rename(
                columns={"r2_aug": "r2_aug_random", "gain": "gain_random"}
            ),
            on="city",
        )
        merged["r2_drop"] = merged["r2_aug_random"] - merged["r2_aug"]
        merged["gain_drop"] = merged["gain_random"] - merged["gain"]
        merged.to_csv(OUT / "20_block_vs_random_compare.csv", index=False)
        print(f"\n=== Block-CV vs Random-CV drop ===")
        print(f"  mean R²_aug drop: {merged['r2_drop'].mean():+.3f}")
        print(f"  mean gain drop:   {merged['gain_drop'].mean():+.3f}")
        summary["block_vs_random"] = {
            "mean_r2_drop": float(merged["r2_drop"].mean()),
            "mean_gain_drop": float(merged["gain_drop"].mean()),
        }
    except Exception as e:
        print(f"comparison failed: {e}")

    with open(OUT / "20_block_cv_lmm.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.scatter(per_city["D"], per_city["r2_aug"], s=80,
               c=per_city["z_kde"], cmap="coolwarm",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["D"], r["r2_aug"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet (L_unnorm)")
    ax.set_ylabel("R²_aug under block-CV")
    info = summary.get("h1a_block", {})
    ax.set_title(f"(a) H1a under block-CV\nβ={info.get('dirichlet_beta', float('nan')):+.4f}  "
                 f"p={info.get('dirichlet_p', float('nan')):.4f}")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(per_city["z_kde"], per_city["gain"], s=80, color="C0",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["z_kde"], r["gain"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(-5, color="black", lw=0.7, ls=":")
    info = summary.get("h1b_block", {})
    ax.set_title(f"(b) H1b under block-CV (z>-5)\nβ={info.get('z_kde_beta', float('nan')):+.5f}  "
                 f"p={info.get('z_kde_p', float('nan')):.4f}")
    ax.set_xlabel("z_kde"); ax.set_ylabel("gain (block-CV)")
    ax.grid(alpha=0.3)

    # comparison plots
    try:
        ax = axes[1, 0]
        ax.scatter(merged["r2_aug_random"], merged["r2_aug"], s=80,
                   edgecolor="black", lw=0.5)
        for _, r in merged.iterrows():
            ax.annotate(r["city"][:12], (r["r2_aug_random"], r["r2_aug"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        lo = min(merged[["r2_aug", "r2_aug_random"]].min().min() - 0.05, 0)
        hi = max(merged[["r2_aug", "r2_aug_random"]].max().max() + 0.05, 0.7)
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_xlabel("R²_aug under random-CV (script 18)")
        ax.set_ylabel("R²_aug under block-CV (script 20)")
        ax.set_title("(c) Spatial leakage gap")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.scatter(merged["gain_random"], merged["gain"], s=80,
                   edgecolor="black", lw=0.5)
        for _, r in merged.iterrows():
            ax.annotate(r["city"][:12], (r["gain_random"], r["gain"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        lo = min(merged[["gain", "gain_random"]].min().min() - 0.02, -0.02)
        hi = max(merged[["gain", "gain_random"]].max().max() + 0.02, 0.15)
        ax.plot([lo, hi], [lo, hi], "k:", lw=1)
        ax.set_xlabel("gain under random-CV")
        ax.set_ylabel("gain under block-CV")
        ax.set_title("(d) Augmentation gain — leakage gap")
        ax.grid(alpha=0.3)
    except Exception:
        pass

    fig.suptitle("Block Spatial Cross-Validation (Gemini round 5 hardening)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "20_block_cv.pdf", bbox_inches="tight")
    fig.savefig(OUT / "20_block_cv.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '20_block_cv.pdf'}")


if __name__ == "__main__":
    main()
