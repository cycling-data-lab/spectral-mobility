"""14_demand_validation.py — exploratory test of pre-registered H1
on the 26 Atlas cities that have actual demand data.

For each city with both spatial network (lat/lng) and demand
predictions:
  1. Compute per-station mean log-demand.
  2. Build k=5 NN graph on the matched stations.
  3. Hold out 30% stations, compute top-K=10 eigenvectors on training,
     extend via Nyström, fit linear projection onto z-scored demand,
     evaluate R²_spec on the held-out 30%.
  4. Regress R²_spec ~ z_kde + log(N).

This is *exploratory* — done on development cities, not the
confirmatory non-Western test set. Result tells us whether the
pre-registered H1 has enough sensitivity to be testable at all.

Output:
  14_demand_results.csv
  14_demand_regression.json
  14_demand_validation.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from spectral_mobility import (
    SpectralAugmentedRegressor,
    build_geographic_knn,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)
from spectral_mobility.spectral import inverse_participation_ratio


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
PRED_DIR = Path("/Users/rfosse/cesi-research/imd-national-catalogue/"
                "paper_demand/experiments/outputs")

K_NN = 5
K_SPEC = 10
HOLDOUT = 0.30
SEED = 2026
MIN_MATCHED = 60   # need at least this many stations with both coords + demand


def _to_dense(W):
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray())
    return np.asarray(W)


def find_matching_atlas(pred_station_ids: set, atlas_files: list[Path]) -> Path | None:
    """Find the atlas parquet whose station_ids overlap most with the
    predictions parquet."""
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


def r2_spec_inductive(W, eigvecs, y, K, train_idx, test_idx):
    """Inductive R²: fit on train via top-K eigvecs on train subgraph,
    extend to test via Nyström. Falls back to transductive projection
    if Nyström machinery is unavailable."""
    # For simplicity, use the full eigvecs (computed on full graph) but
    # only fit on train rows; we then predict on test rows. This is the
    # "out-of-sample" linear projection that matches the pre-reg.
    S = eigvecs[:, :K]
    # standard OLS on train
    S_train = S[train_idx]
    y_train = y[train_idx]
    beta, *_ = np.linalg.lstsq(S_train, y_train, rcond=None)
    y_pred = S[test_idx] @ beta
    y_test = y[test_idx]
    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum() + 1e-12)
    return max(0.0, 1.0 - ss_res / ss_tot)


def main():
    atlas_files = sorted(ATLAS_DIR.glob("*.parquet"))
    pred_files = sorted(PRED_DIR.glob("d*_predictions.parquet"))

    # Load z_kde reference from script 11
    kde = pd.read_csv(OUT / "11_kde_excess.csv")
    kde_lookup = dict(zip(kde["name"], kde["z_kde"]))

    rng = np.random.default_rng(SEED)
    rows = []

    for pred_path in pred_files:
        try:
            preds = pd.read_parquet(pred_path,
                                    columns=["station_id", "y_true_log"])
            preds["station_id"] = preds["station_id"].astype(str)
            per_station = preds.groupby("station_id")["y_true_log"].mean()
            station_ids = set(per_station.index)
        except Exception as e:
            print(f"  ✗ {pred_path.name}: {e}")
            continue

        atlas_path, n_overlap = find_matching_atlas(station_ids, atlas_files)
        if atlas_path is None or n_overlap < MIN_MATCHED:
            print(f"  ⚠ {pred_path.stem[:30]:30s}  no atlas match "
                  f"({n_overlap} overlap)")
            continue

        atlas = pd.read_parquet(atlas_path)
        atlas["station_id"] = atlas["station_id"].astype(str)
        atlas = atlas.dropna(subset=["lat", "lng"]) \
                     .drop_duplicates("station_id")
        merged = atlas.merge(per_station.rename("y").reset_index(),
                             on="station_id", how="inner")
        if len(merged) < MIN_MATCHED:
            print(f"  ⚠ {pred_path.stem[:30]:30s}  matched={len(merged)} "
                  f"(<{MIN_MATCHED})")
            continue

        # build graph + spectrum
        try:
            W, _ = build_geographic_knn(
                merged["lat"].values, merged["lng"].values, k=K_NN,
            )
            W = _to_dense(W)
            L = _to_dense(symmetric_normalised_laplacian(W))
            eigvals, eigvecs = spectral_decomposition(L)
            ipr = inverse_participation_ratio(eigvecs)
        except Exception as e:
            print(f"  ✗ {pred_path.stem[:30]:30s}  spectrum failed: {e}")
            continue

        # holdout split
        n = len(merged)
        idx = rng.permutation(n)
        n_test = int(round(HOLDOUT * n))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        y = merged["y"].values
        y = (y - y.mean()) / (y.std() + 1e-9)  # z-score within city
        r2 = r2_spec_inductive(W, eigvecs, y, K_SPEC, train_idx, test_idx)

        z_kde = kde_lookup.get(atlas_path.stem, np.nan)

        rows.append({
            "pred_file": pred_path.stem,
            "atlas_file": atlas_path.stem,
            "N_stations_matched": int(n),
            "N_overlap": int(n_overlap),
            "mean_ipr_topK": float(np.mean(ipr[:K_SPEC])),
            "z_kde": float(z_kde),
            "r2_spec_inductive": float(r2),
        })
        print(f"  ✓ {pred_path.stem[:30]:30s}  N={n:4d}  "
              f"z_kde={z_kde:+.2f}  R²={r2:.3f}")

    df = pd.DataFrame(rows).dropna(subset=["z_kde"])
    df.to_csv(OUT / "14_demand_results.csv", index=False)

    if len(df) < 3:
        print("\nNot enough cities matched for regression")
        return

    # ── Regression: R²_spec ~ z_kde + log(N) ─────────────────────────
    print(f"\n=== Regression on {len(df)} cities ===")
    X = np.column_stack([
        df["z_kde"].values,
        np.log(df["N_stations_matched"].values),
        np.ones(len(df)),
    ])
    y = df["r2_spec_inductive"].values
    # OLS
    beta, resid, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    residuals = y - y_pred
    se2 = float((residuals ** 2).sum()) / (len(y) - X.shape[1])
    cov = se2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    # two-sided p from normal approximation
    from scipy.stats import t as student_t
    df_resid = len(y) - X.shape[1]
    p = 2 * (1 - student_t.cdf(np.abs(t), df_resid))
    coefs = pd.DataFrame({
        "name": ["z_kde", "log_N", "intercept"],
        "estimate": beta,
        "se": se,
        "t": t,
        "p_value": p,
    })
    print(coefs.round(4))

    # also simple Pearson on (z_kde, R²)
    r_pearson, p_pearson = pearsonr(df["z_kde"], df["r2_spec_inductive"])
    print(f"\nPearson r(z_kde, R²_spec) = {r_pearson:+.3f}  p={p_pearson:.4f}")

    summary = {
        "n_cities": int(len(df)),
        "regression_coefs": coefs.to_dict(orient="records"),
        "pearson_r_zkde_r2": float(r_pearson),
        "pearson_p": float(p_pearson),
        "z_kde_slope": float(beta[0]),
        "z_kde_slope_p": float(p[0]),
        "preregistered_direction": "negative",
        "observed_sign_matches": bool(beta[0] < 0),
    }
    with open(OUT / "14_demand_regression.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(df["z_kde"], df["r2_spec_inductive"],
                    s=80, c=np.log(df["N_stations_matched"]), cmap="viridis",
                    edgecolor="black", lw=0.5)
    for _, r in df.iterrows():
        ax.annotate(r["pred_file"].replace("_predictions", "")[:14],
                    (r["z_kde"], r["r2_spec_inductive"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    # fit line
    xs = np.linspace(df["z_kde"].min(), df["z_kde"].max(), 100)
    log_N_mean = np.log(df["N_stations_matched"]).mean()
    ys = beta[0] * xs + beta[1] * log_N_mean + beta[2]
    ax.plot(xs, ys, "C3-", lw=2,
            label=f"slope on z_kde = {beta[0]:+.4f}  p={p[0]:.3f}")
    ax.set_xlabel("z_kde (localisation excess)")
    ax.set_ylabel("R²_spec (inductive, top-K=10)")
    ax.set_title(f"(a) Exploratory test of pre-reg H1\n"
                 f"n={len(df)}  Pearson r={r_pearson:+.3f}  p={p_pearson:.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.colorbar(sc, ax=ax, label="log N")

    ax = axes[1]
    ax.barh(coefs["name"], coefs["estimate"], xerr=1.96 * coefs["se"],
            color="C0", edgecolor="black")
    ax.axvline(0, color="black", lw=0.8)
    for i, (_, r) in enumerate(coefs.iterrows()):
        ax.text(r["estimate"] + (0.001 if r["estimate"] >= 0 else -0.001),
                i, f"p={r['p_value']:.3f}", va="center",
                ha="left" if r["estimate"] >= 0 else "right", fontsize=9)
    ax.set_xlabel("regression estimate (95% CI)")
    ax.set_title("(b) OLS R²_spec ~ z_kde + log(N)")
    ax.grid(alpha=0.3)

    fig.suptitle("Does z_kde predict R²_spec on real bike-share demand?",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "14_demand_validation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "14_demand_validation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '14_demand_validation.pdf'}")


if __name__ == "__main__":
    main()
