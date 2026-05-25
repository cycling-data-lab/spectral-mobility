"""15_goldilocks_test.py — strict inductive Nyström test of the
Goldilocks (quadratic) hypothesis.

After the falsification of the linear H1 (script 14, r=-0.017), the
revised hypothesis is:

  H1*: For real bike-share demand y_log = log(trips + 1), the inductive
       R²_spec follows an inverted-U in z_kde — maximal near z=0,
       suppressed at both extremes (Goldilocks zone of structural
       priors).  In a linear mixed-effects model
            R²_spec ~ β1·z_kde + β2·z_kde² + (1 | city)
       fit over multiple (city, holdout-split) pairs, the *quadratic
       coefficient* β2 is **negative and significant** at α = 0.01.

The Goldilocks justification (Gemini round 2): hyper-delocalised
graphs have lattice-like Fourier eigenvectors that cannot represent
local demand peaks; hyper-localised graphs have Dirac-spike
eigenvectors that ignore the global tissue; the centre z≈0 lives in
a Goldilocks zone.

Two methodological corrections vs script 14:
  (i)   strict inductive Nyström: train graph + eigenvectors built on
        the in-sample 70% only, then extended to held-out 30% via
        the package's Nyström machinery.  No transductive leakage.
  (ii)  raw log(y+1), no within-city z-scoring (preserves absolute
        signal-to-noise ratio across cities).

We run n_splits = 10 holdouts per city → 18 cities × 10 = 180
observations; LMM with city as random intercept.

Output:
  15_goldilocks_results.csv
  15_goldilocks_lmm.json
  15_goldilocks.{pdf,png}
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


def find_matching_atlas(pred_station_ids, atlas_files):
    best = None
    best_n = 0
    for atlas_path in atlas_files:
        try:
            df = pd.read_parquet(atlas_path, columns=["station_id"])
        except Exception:
            continue
        n_overlap = len(set(df["station_id"].astype(str)) & pred_station_ids)
        if n_overlap > best_n:
            best = atlas_path
            best_n = n_overlap
    return best, best_n


def nystrom_inductive_r2(coords_all, y_all, train_idx, test_idx, K, k_nn):
    """Strict inductive R²_spec.

      1. Build k-NN graph on training coords ONLY.
      2. Compute top-K eigenvectors of the in-sample Laplacian.
      3. Extend eigenvectors to held-out coords via Gaussian-RBF
         weighted Nyström k-NN extension (same as
         SpectralAugmentedRegressor._augment_inductive).
      4. OLS y_train ~ U_K_train + intercept.
      5. Score on test: R² = 1 − SS_res / SS_tot.
    """
    from sklearn.neighbors import BallTree
    from spectral_mobility.graph import EARTH_RADIUS_METRES

    coords_tr = coords_all[train_idx]
    coords_te = coords_all[test_idx]
    y_tr = y_all[train_idx]
    y_te = y_all[test_idx]

    # 1) Training graph and eigendecomposition
    W_tr, sigma = build_geographic_knn(
        coords_tr[:, 0], coords_tr[:, 1], k=k_nn,
    )
    W_tr = _to_dense(W_tr)
    L_tr = _to_dense(symmetric_normalised_laplacian(W_tr))
    eigvals_tr, eigvecs_tr = spectral_decomposition(L_tr)
    Uk = eigvecs_tr[:, :K]

    # 2) Nyström extension: each test point ← weighted mean of its
    #    k_nn nearest training neighbours' eigvec rows.
    train_rad = np.deg2rad(coords_tr)
    test_rad = np.deg2rad(coords_te)
    tree = BallTree(train_rad, metric="haversine")
    dist_rad, idx = tree.query(test_rad, k=k_nn)
    dist_m = dist_rad * EARTH_RADIUS_METRES
    weights = np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    Uk_te = np.einsum("ij,ijk->ik", weights, Uk[idx])

    # 3) OLS train, eval test (with intercept)
    Xtr = np.column_stack([Uk, np.ones(len(Uk))])
    Xte = np.column_stack([Uk_te, np.ones(len(Uk_te))])
    beta, *_ = np.linalg.lstsq(Xtr, y_tr, rcond=None)
    y_pred = Xte @ beta
    ss_res = float(((y_te - y_pred) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum() + 1e-12)
    return max(0.0, 1.0 - ss_res / ss_tot)


def main():
    atlas_files = sorted(ATLAS_DIR.glob("*.parquet"))
    pred_files = sorted(PRED_DIR.glob("d*_predictions.parquet"))
    kde = pd.read_csv(OUT / "11_kde_excess.csv")
    kde_lookup = dict(zip(kde["name"], kde["z_kde"]))

    master_rng = np.random.default_rng(SEED)
    rows = []

    for pred_path in pred_files:
        try:
            preds = pd.read_parquet(pred_path, columns=["station_id", "y_true_log"])
            preds["station_id"] = preds["station_id"].astype(str)
            per_station = preds.groupby("station_id")["y_true_log"].mean()
            station_ids = set(per_station.index)
        except Exception:
            continue

        atlas_path, n_overlap = find_matching_atlas(station_ids, atlas_files)
        if atlas_path is None or n_overlap < MIN_MATCHED:
            continue

        atlas = pd.read_parquet(atlas_path)
        atlas["station_id"] = atlas["station_id"].astype(str)
        atlas = atlas.dropna(subset=["lat", "lng"]).drop_duplicates("station_id")
        merged = atlas.merge(per_station.rename("y_log").reset_index(),
                             on="station_id", how="inner")
        if len(merged) < MIN_MATCHED:
            continue

        z_kde = kde_lookup.get(atlas_path.stem, np.nan)
        if not np.isfinite(z_kde):
            continue

        coords = merged[["lat", "lng"]].values
        # y is already log-transformed from y_true_log; treat it as log(trips+1)
        y = merged["y_log"].values
        n = len(merged)

        # 10 random holdout splits
        city_name = pred_path.stem.replace("_predictions", "")
        for split_id in range(N_SPLITS):
            split_rng = np.random.default_rng(
                hash((city_name, split_id)) % (2**32)
            )
            idx = split_rng.permutation(n)
            n_test = int(round(HOLDOUT * n))
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            try:
                r2 = nystrom_inductive_r2(
                    coords, y, train_idx, test_idx, K_SPEC, K_NN,
                )
            except Exception as e:
                continue
            rows.append({
                "city": city_name,
                "atlas": atlas_path.stem,
                "split_id": split_id,
                "N": int(n),
                "z_kde": float(z_kde),
                "r2_spec_inductive": float(r2),
            })
        if rows and rows[-1]["city"] == city_name:
            r2s = [r["r2_spec_inductive"] for r in rows
                   if r["city"] == city_name]
            print(f"  ✓ {city_name[:30]:30s}  N={n:4d}  "
                  f"z_kde={z_kde:+.2f}  R²={np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "15_goldilocks_results.csv", index=False)

    if df.empty:
        print("No results")
        return

    print(f"\n=== LMM fit on {len(df)} (city, split) observations from "
          f"{df['city'].nunique()} cities ===\n")

    # ── Quadratic LMM with city random intercept ─────────────────────
    df["z_kde2"] = df["z_kde"] ** 2
    md = smf.mixedlm("r2_spec_inductive ~ z_kde + z_kde2", df,
                     groups=df["city"])
    res = md.fit(method="lbfgs")
    print(res.summary())

    summary = {
        "n_obs": int(len(df)),
        "n_cities": int(df["city"].nunique()),
        "fixed_effects": {
            "intercept": float(res.fe_params["Intercept"]),
            "z_kde": float(res.fe_params["z_kde"]),
            "z_kde2": float(res.fe_params["z_kde2"]),
        },
        "z_kde_p": float(res.pvalues["z_kde"]),
        "z_kde2_p": float(res.pvalues["z_kde2"]),
        "z_kde2_quadratic_negative": bool(res.fe_params["z_kde2"] < 0),
        "z_kde2_significant_001": bool(res.pvalues["z_kde2"] <= 0.01),
        "log_likelihood": float(res.llf),
        "preregistered_direction": "negative quadratic coefficient",
    }
    # Also fit linear-only for comparison
    md_lin = smf.mixedlm("r2_spec_inductive ~ z_kde", df, groups=df["city"])
    res_lin = md_lin.fit(method="lbfgs")
    summary["linear_only_z_kde"] = float(res_lin.fe_params["z_kde"])
    summary["linear_only_z_kde_p"] = float(res_lin.pvalues["z_kde"])
    summary["delta_loglik_quadratic_vs_linear"] = float(res.llf - res_lin.llf)

    with open(OUT / "15_goldilocks_lmm.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== Goldilocks verdict ===")
    print(f"  β2 (z_kde²)        = {summary['fixed_effects']['z_kde2']:+.5f}")
    print(f"  p(β2)              = {summary['z_kde2_p']:.4f}")
    print(f"  sign correct?      = {summary['z_kde2_quadratic_negative']}")
    print(f"  sig at α=0.01?     = {summary['z_kde2_significant_001']}")
    print(f"  Δ log-lik vs lin   = {summary['delta_loglik_quadratic_vs_linear']:.3f}")

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) per-city mean R² with error bar vs z_kde + quadratic fit
    ax = axes[0]
    city_stats = df.groupby("city").agg(
        z_kde=("z_kde", "first"),
        r2_mean=("r2_spec_inductive", "mean"),
        r2_std=("r2_spec_inductive", "std"),
        N=("N", "first"),
    ).reset_index()
    sc = ax.errorbar(city_stats["z_kde"], city_stats["r2_mean"],
                     yerr=city_stats["r2_std"], fmt="o",
                     color="C0", capsize=2, markersize=8,
                     ecolor="gray", lw=0.7)
    for _, r in city_stats.iterrows():
        ax.annotate(r["city"][:14], (r["z_kde"], r["r2_mean"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    # quadratic fit curve from LMM
    xs = np.linspace(df["z_kde"].min() - 0.5, df["z_kde"].max() + 0.5, 200)
    ys = (summary["fixed_effects"]["intercept"]
          + summary["fixed_effects"]["z_kde"] * xs
          + summary["fixed_effects"]["z_kde2"] * xs ** 2)
    ax.plot(xs, ys, "C3-", lw=2,
            label=f"LMM quadratic fit  β₂={summary['fixed_effects']['z_kde2']:+.5f}  "
                  f"p={summary['z_kde2_p']:.3f}")
    ax.set_xlabel("z_kde (localisation excess)")
    ax.set_ylabel("R²_spec (inductive Nyström, 10 splits)")
    ax.set_title("(a) Goldilocks test on 18 real-demand cities\n"
                 "strict inductive Nyström, log(y+1)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    # (b) coefficient table
    ax = axes[1]
    ax.axis("off")
    text = [
        "LINEAR MIXED-EFFECTS MODEL",
        "R²_spec ~ z_kde + z_kde² + (1|city)",
        "",
        f"n_obs       = {summary['n_obs']}",
        f"n_cities    = {summary['n_cities']}",
        "",
        f"β₀ (intercept) = {summary['fixed_effects']['intercept']:+.4f}",
        f"β₁ (z_kde)     = {summary['fixed_effects']['z_kde']:+.5f}   p = {summary['z_kde_p']:.4f}",
        f"β₂ (z_kde²)    = {summary['fixed_effects']['z_kde2']:+.5f}   p = {summary['z_kde2_p']:.4f}",
        "",
        f"Quadratic coefficient sign: {'✓ negative' if summary['z_kde2_quadratic_negative'] else '✗ positive'}",
        f"Significant at α=0.01:      {'✓ yes' if summary['z_kde2_significant_001'] else '✗ no'}",
        "",
        f"Δ log-lik vs linear-only:   {summary['delta_loglik_quadratic_vs_linear']:+.3f}",
        "",
        "Pre-registered H1*: β₂ < 0, p ≤ 0.01",
        f"VERDICT: " +
        ("SUPPORTED" if summary["z_kde2_quadratic_negative"]
         and summary["z_kde2_significant_001"]
         else "NOT SUPPORTED")
    ]
    ax.text(0.05, 0.95, "\n".join(text), fontsize=11,
            transform=ax.transAxes, va="top", family="monospace")

    fig.suptitle("Goldilocks-zone test of the applicability bound (revised H1)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "15_goldilocks.pdf", bbox_inches="tight")
    fig.savefig(OUT / "15_goldilocks.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '15_goldilocks.pdf'}")


if __name__ == "__main__":
    main()
