"""17_dirichlet_and_aug_gain.py — test two ideas from Gemini round 3.

(A) Dirichlet energy of demand as missing covariate.
    For each city: D(y) = y^T L y / y^T y, where L is the normalised
    Laplacian and y is log-demand.  D measures spatial smoothness of
    the demand on the graph.  Hypothesis: residual R²_spec variance
    (after z_kde) is explained by 1 - D(y) — predictable cities are
    those where demand is smooth.

(B) Augmentation gain as the *real* target.
    For each city: compute baseline R² (using only k_NN-of-coords
    features) vs augmented R² (adding top-K eigenvectors).  Gain
    = R²_aug - R²_base.  Hypothesis: gain peaks at z_kde near 0
    (type B / typical networks benefit most from spectral priors).

We run both on the 18 cities with real demand (matched via the
script-14 pipeline) using strict inductive Nyström.

Output:
  17_dirichlet_results.csv  per-city Dirichlet energy + R²_spec
  17_aug_gain_results.csv   per-(city, split) baseline + aug R²
  17_lmm_summary.json       both LMM fits
  17_results.{pdf,png}      4-panel diagnostic
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree

from spectral_mobility import (
    build_geographic_knn,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
PRED_DIR = Path("/Users/rfosse/cesi-research/imd-national-catalogue/"
                "paper_demand/experiments/outputs")
K_NN = 5
K_SPEC = 10
HOLDOUT = 0.30
N_SPLITS = 10
SEED = 2026
MIN_MATCHED = 60


def _to_dense(W):
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray())
    return np.asarray(W)


def find_matching_atlas(pred_ids, atlas_files):
    best = None
    best_n = 0
    for atlas_path in atlas_files:
        try:
            df = pd.read_parquet(atlas_path, columns=["station_id"])
        except Exception:
            continue
        n_ov = len(set(df["station_id"].astype(str)) & pred_ids)
        if n_ov > best_n:
            best = atlas_path
            best_n = n_ov
    return best, best_n


def dirichlet_energy(L, y):
    """y^T L y / y^T y on standardised y. Lower = smoother."""
    y = y - y.mean()
    num = float(y @ L @ y)
    den = float(y @ y) + 1e-12
    return num / den


def nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn):
    from spectral_mobility.graph import EARTH_RADIUS_METRES
    train_rad = np.deg2rad(coords_tr)
    test_rad = np.deg2rad(coords_te)
    tree = BallTree(train_rad, metric="haversine")
    dist_rad, idx = tree.query(test_rad, k=k_nn)
    dist_m = dist_rad * EARTH_RADIUS_METRES
    w = np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, Uk[idx])


def inductive_split(coords, X_bare, y, train_idx, test_idx, K, k_nn):
    """Return (R²_base, R²_aug). Baseline = Ridge(X_bare). Aug = Ridge([X_bare, U_K])."""
    coords_tr = coords[train_idx]
    coords_te = coords[test_idx]
    X_tr = X_bare[train_idx]
    X_te = X_bare[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    # Baseline (no spectral features)
    base = Ridge(alpha=1.0).fit(X_tr, y_tr)
    r2_base = r2_score(y_te, base.predict(X_te))

    # Augmented: top-K eigvecs from train graph, Nyström extended
    W_tr, sigma = build_geographic_knn(coords_tr[:, 0], coords_tr[:, 1], k=k_nn)
    L_tr = _to_dense(symmetric_normalised_laplacian(_to_dense(W_tr)))
    _, eigvecs_tr = spectral_decomposition(L_tr)
    Uk = eigvecs_tr[:, :K]
    Uk_te = nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn)
    X_aug_tr = np.column_stack([X_tr, Uk])
    X_aug_te = np.column_stack([X_te, Uk_te])
    aug = Ridge(alpha=1.0).fit(X_aug_tr, y_tr)
    r2_aug = r2_score(y_te, aug.predict(X_aug_te))
    return float(r2_base), float(r2_aug)


def main():
    atlas_files = sorted(ATLAS_DIR.glob("*.parquet"))
    pred_files = sorted(PRED_DIR.glob("d*_predictions.parquet"))
    kde = pd.read_csv(OUT / "11_kde_excess.csv")
    kde_lookup = dict(zip(kde["name"], kde["z_kde"]))

    dir_rows = []
    aug_rows = []

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

        # Bare encoder features available in every atlas parquet
        bare_cols = [
            c for c in [
                "gtfs_heavy_stops_300m", "infra_cyclable_features_300m",
                "elevation_m", "topography_roughness_index",
                "n_stations_within_500m", "n_stations_within_1km",
                "catchment_density_per_km2",
            ]
            if c in atlas.columns
        ]
        atlas = atlas.dropna(subset=["lat", "lng"] + bare_cols) \
                     .drop_duplicates("station_id")
        merged = atlas.merge(per_station.rename("y_log").reset_index(),
                             on="station_id", how="inner")
        if len(merged) < MIN_MATCHED:
            continue
        z_kde = kde_lookup.get(atlas_path.stem, np.nan)
        if not np.isfinite(z_kde):
            continue

        coords = merged[["lat", "lng"]].values
        X_bare = merged[bare_cols].values.astype(float)
        # standardise X_bare columnwise
        X_bare = (X_bare - X_bare.mean(0)) / (X_bare.std(0) + 1e-9)
        y = merged["y_log"].values
        n = len(merged)
        city_name = pred_path.stem.replace("_predictions", "")

        # ── (A) Dirichlet energy on the full graph ───────────────────
        try:
            W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=K_NN)
            L = _to_dense(symmetric_normalised_laplacian(_to_dense(W)))
            D_y = dirichlet_energy(L, y)
        except Exception:
            continue
        dir_rows.append({
            "city": city_name, "atlas": atlas_path.stem,
            "N": int(n), "z_kde": float(z_kde),
            "dirichlet_y": float(D_y),
        })

        # ── (B) Augmentation gain over splits ────────────────────────
        for split_id in range(N_SPLITS):
            split_rng = np.random.default_rng(
                hash((city_name, split_id)) % (2**32)
            )
            idx = split_rng.permutation(n)
            n_test = int(round(HOLDOUT * n))
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            try:
                r2_base, r2_aug = inductive_split(
                    coords, X_bare, y, train_idx, test_idx, K_SPEC, K_NN,
                )
            except Exception:
                continue
            aug_rows.append({
                "city": city_name, "split_id": split_id,
                "N": int(n), "z_kde": float(z_kde),
                "dirichlet_y": float(D_y),
                "r2_base": r2_base,
                "r2_aug": r2_aug,
                "gain": r2_aug - r2_base,
            })

        last = aug_rows[-N_SPLITS:] if len(aug_rows) >= N_SPLITS else aug_rows
        last = [r for r in last if r["city"] == city_name]
        if last:
            g = np.mean([r["gain"] for r in last])
            ba = np.mean([r["r2_base"] for r in last])
            au = np.mean([r["r2_aug"] for r in last])
            print(f"  ✓ {city_name[:28]:28s}  N={n:4d}  z_kde={z_kde:+.2f}  "
                  f"D(y)={D_y:.4f}  R²_base={ba:.3f}  R²_aug={au:.3f}  "
                  f"gain={g:+.3f}")

    dir_df = pd.DataFrame(dir_rows)
    aug_df = pd.DataFrame(aug_rows)
    dir_df.to_csv(OUT / "17_dirichlet_results.csv", index=False)
    aug_df.to_csv(OUT / "17_aug_gain_results.csv", index=False)

    summary = {}

    # ── (A) Dirichlet test: per-city R²_spec correlation ─────────────
    if len(aug_df) > 0:
        per_city = aug_df.groupby("city").agg(
            z_kde=("z_kde", "first"),
            dirichlet=("dirichlet_y", "first"),
            r2_aug_mean=("r2_aug", "mean"),
            r2_base_mean=("r2_base", "mean"),
            gain_mean=("gain", "mean"),
        ).reset_index()
        print(f"\n=== (A) Dirichlet energy correlations ===")
        for tgt in ["r2_aug_mean", "r2_base_mean", "gain_mean"]:
            corr_z = np.corrcoef(per_city["z_kde"], per_city[tgt])[0, 1]
            corr_d = np.corrcoef(per_city["dirichlet"], per_city[tgt])[0, 1]
            print(f"  {tgt:14s}  corr(z_kde)={corr_z:+.3f}  corr(Dirichlet)={corr_d:+.3f}")
        summary["dirichlet_correlations"] = {
            tgt: {
                "z_kde": float(np.corrcoef(per_city["z_kde"], per_city[tgt])[0, 1]),
                "dirichlet": float(np.corrcoef(per_city["dirichlet"], per_city[tgt])[0, 1]),
            }
            for tgt in ["r2_aug_mean", "r2_base_mean", "gain_mean"]
        }

    # ── LMM on R²_spec ~ z_kde + Dirichlet + (1|city) ────────────────
    if len(aug_df) > 0:
        try:
            md = smf.mixedlm("r2_aug ~ z_kde + dirichlet_y", aug_df,
                             groups=aug_df["city"])
            res = md.fit(method="lbfgs")
            print(f"\n=== LMM: R²_aug ~ z_kde + Dirichlet ===")
            print(res.summary())
            summary["lmm_r2_aug"] = {
                "z_kde_beta": float(res.fe_params["z_kde"]),
                "z_kde_p": float(res.pvalues["z_kde"]),
                "dirichlet_beta": float(res.fe_params["dirichlet_y"]),
                "dirichlet_p": float(res.pvalues["dirichlet_y"]),
            }
        except Exception as e:
            print(f"LMM r2_aug failed: {e}")

    # ── (B) Augmentation gain LMM ────────────────────────────────────
    if len(aug_df) > 0:
        try:
            aug_df["z_kde2"] = aug_df["z_kde"] ** 2
            md = smf.mixedlm("gain ~ z_kde + z_kde2", aug_df,
                             groups=aug_df["city"])
            res = md.fit(method="lbfgs")
            print(f"\n=== (B) LMM: gain ~ z_kde + z_kde² ===")
            print(res.summary())
            b1 = float(res.fe_params["z_kde"])
            b2 = float(res.fe_params["z_kde2"])
            z_peak = -b1 / (2 * b2) if abs(b2) > 1e-9 else float("nan")
            summary["lmm_gain"] = {
                "intercept": float(res.fe_params["Intercept"]),
                "z_kde_beta": b1,
                "z_kde_p": float(res.pvalues["z_kde"]),
                "z_kde2_beta": b2,
                "z_kde2_p": float(res.pvalues["z_kde2"]),
                "z_peak": float(z_peak),
                "sign_correct": bool(b2 < 0),
                "sig_005": bool(res.pvalues["z_kde2"] <= 0.05),
                "sig_001": bool(res.pvalues["z_kde2"] <= 0.01),
            }
            print(f"  β₂={b2:+.6f}  p={res.pvalues['z_kde2']:.4f}  "
                  f"sign:{'✓' if b2 < 0 else '✗'}  "
                  f"sig@0.05:{'✓' if res.pvalues['z_kde2'] <= 0.05 else '✗'}")
            print(f"  fitted peak at z = {z_peak:.2f}")
        except Exception as e:
            print(f"LMM gain failed: {e}")

    with open(OUT / "17_lmm_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Figure ───────────────────────────────────────────────────────
    if len(aug_df) > 0:
        per_city = aug_df.groupby("city").agg(
            z_kde=("z_kde", "first"),
            dirichlet=("dirichlet_y", "first"),
            r2_aug=("r2_aug", "mean"),
            r2_aug_std=("r2_aug", "std"),
            r2_base=("r2_base", "mean"),
            gain=("gain", "mean"),
            gain_std=("gain", "std"),
            N=("N", "first"),
        ).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

        # (a) Dirichlet vs R²_aug
        ax = axes[0, 0]
        sc = ax.scatter(per_city["dirichlet"], per_city["r2_aug"], s=70,
                        c=per_city["z_kde"], cmap="coolwarm",
                        edgecolor="black", lw=0.5)
        for _, r in per_city.iterrows():
            ax.annotate(r["city"][:12], (r["dirichlet"], r["r2_aug"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        corr = np.corrcoef(per_city["dirichlet"], per_city["r2_aug"])[0, 1]
        ax.set_xlabel("Dirichlet energy of demand  (y^T L y / y^T y)")
        ax.set_ylabel("R²_aug (Nyström inductive)")
        ax.set_title(f"(a) Dirichlet energy vs R²_aug  corr={corr:+.3f}")
        plt.colorbar(sc, ax=ax, label="z_kde")
        ax.grid(alpha=0.3)

        # (b) z_kde vs gain (the main test)
        ax = axes[0, 1]
        ax.errorbar(per_city["z_kde"], per_city["gain"],
                    yerr=per_city["gain_std"], fmt="o",
                    color="C0", capsize=2, markersize=8)
        for _, r in per_city.iterrows():
            ax.annotate(r["city"][:12], (r["z_kde"], r["gain"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        if "lmm_gain" in summary:
            info = summary["lmm_gain"]
            xs = np.linspace(per_city["z_kde"].min() - 0.5,
                             per_city["z_kde"].max() + 0.5, 200)
            ys = info["intercept"] + info["z_kde_beta"] * xs + info["z_kde2_beta"] * xs ** 2
            ax.plot(xs, ys, "C3-", lw=2,
                    label=f"β₂={info['z_kde2_beta']:+.5f}  p={info['z_kde2_p']:.3f}")
            ax.axhline(0, color="black", lw=0.7, ls=":")
            ax.legend(fontsize=9)
        ax.set_xlabel("z_kde")
        ax.set_ylabel("ΔR² = R²_aug − R²_base  (augmentation gain)")
        ax.set_title("(b) Augmentation gain vs z_kde")
        ax.grid(alpha=0.3)

        # (c) baseline vs augmented R²
        ax = axes[1, 0]
        ax.scatter(per_city["r2_base"], per_city["r2_aug"], s=70,
                   c=per_city["z_kde"], cmap="coolwarm",
                   edgecolor="black", lw=0.5)
        for _, r in per_city.iterrows():
            ax.annotate(r["city"][:12], (r["r2_base"], r["r2_aug"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        lim = max(per_city[["r2_base", "r2_aug"]].max().max() + 0.05, 0.1)
        ax.plot([-0.1, lim], [-0.1, lim], "k:", lw=1)
        ax.set_xlabel("R²_baseline (encoder features only)")
        ax.set_ylabel("R²_augmented (+ top-K eigvecs)")
        ax.set_title("(c) Augmentation helps when above the diagonal")
        ax.grid(alpha=0.3)

        # (d) Dirichlet vs z_kde
        ax = axes[1, 1]
        ax.scatter(per_city["z_kde"], per_city["dirichlet"], s=70,
                   c=per_city["gain"], cmap="RdYlGn",
                   edgecolor="black", lw=0.5,
                   vmin=-0.05, vmax=0.15)
        for _, r in per_city.iterrows():
            ax.annotate(r["city"][:12], (r["z_kde"], r["dirichlet"]),
                        fontsize=7, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("z_kde")
        ax.set_ylabel("Dirichlet energy of demand")
        ax.set_title("(d) Are z_kde and Dirichlet independent?")
        ax.grid(alpha=0.3)

        fig.suptitle("Two new angles from Gemini r3: Dirichlet energy + augmentation gain",
                     fontsize=13, y=1.00)
        fig.tight_layout()
        fig.savefig(OUT / "17_results.pdf", bbox_inches="tight")
        fig.savefig(OUT / "17_results.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"\n✓ {OUT / '17_results.pdf'}")


if __name__ == "__main__":
    main()
