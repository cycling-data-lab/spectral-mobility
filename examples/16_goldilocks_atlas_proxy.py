"""16_goldilocks_atlas_proxy.py — augment the Goldilocks test by
using spatial covariates as demand-proxies on ALL 124 Atlas cities.

Script 15 had n=18 with real demand → β₂<0 in the right direction but
p=0.27.  Here we use two demand-correlated covariates available for
every Atlas parquet:

  • gtfs_heavy_stops_300m  (transit accessibility, exogenous to bike
    network — pure spatial signal of urban activity).
  • catchment_density_per_km2  (station catchment density — endogenous
    but smooth function of urban form).

Goldilocks predicts: at z_kde ≈ 0 the top-K low-frequency eigenvectors
can best represent spatially-smooth covariates; at the two extremes
(hyper-delocalised lattice OR hyper-localised hubs) the spectral
basis fails to capture them.

Same protocol as 15 but on 124 cities × 10 splits = 1240 obs.

Output:
  16_atlas_proxy_results.csv
  16_atlas_proxy_lmm.json
  16_atlas_proxy.{pdf,png}
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

from spectral_mobility import build_geographic_knn, spectral_decomposition, symmetric_normalised_laplacian


OUT = Path(__file__).parent / "output"
ATLAS_DIR = Path("/Users/rfosse/cesi-research/bikeshare-demand-forecasting/"
                 "data_collection/imd_international")
K_NN = 5
K_SPEC = 10
HOLDOUT = 0.30
N_SPLITS = 10
SEED = 2026
MIN_STATIONS = 60
MAX_STATIONS = 6000


def _to_dense(W):
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray())
    return np.asarray(W)


def nystrom_inductive_r2(coords, y, train_idx, test_idx, K, k_nn):
    from sklearn.neighbors import BallTree
    from spectral_mobility.graph import EARTH_RADIUS_METRES

    coords_tr = coords[train_idx]
    coords_te = coords[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]
    W_tr, sigma = build_geographic_knn(coords_tr[:, 0], coords_tr[:, 1], k=k_nn)
    W_tr = _to_dense(W_tr)
    L_tr = _to_dense(symmetric_normalised_laplacian(W_tr))
    eigvals_tr, eigvecs_tr = spectral_decomposition(L_tr)
    Uk = eigvecs_tr[:, :K]
    train_rad = np.deg2rad(coords_tr)
    test_rad = np.deg2rad(coords_te)
    tree = BallTree(train_rad, metric="haversine")
    dist_rad, idx = tree.query(test_rad, k=k_nn)
    dist_m = dist_rad * EARTH_RADIUS_METRES
    w = np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    Uk_te = np.einsum("ij,ijk->ik", w, Uk[idx])
    Xtr = np.column_stack([Uk, np.ones(len(Uk))])
    Xte = np.column_stack([Uk_te, np.ones(len(Uk_te))])
    beta, *_ = np.linalg.lstsq(Xtr, y_tr, rcond=None)
    y_pred = Xte @ beta
    ss_res = float(((y_te - y_pred) ** 2).sum())
    ss_tot = float(((y_te - y_te.mean()) ** 2).sum() + 1e-12)
    return max(0.0, 1.0 - ss_res / ss_tot)


def main():
    kde = pd.read_csv(OUT / "11_kde_excess.csv")
    kde_lookup = dict(zip(kde["name"], kde["z_kde"]))

    parquets = sorted(ATLAS_DIR.glob("*.parquet"))
    rows = []
    target_columns = ["gtfs_heavy_stops_300m", "catchment_density_per_km2"]
    for i, p in enumerate(parquets, 1):
        try:
            atlas = pd.read_parquet(p)
        except Exception:
            continue
        if "lat" not in atlas or "lng" not in atlas:
            continue
        if not all(c in atlas for c in target_columns):
            continue
        atlas = atlas.dropna(subset=["lat", "lng"] + target_columns) \
                     .drop_duplicates(["lat", "lng"])
        if len(atlas) < MIN_STATIONS:
            continue
        if len(atlas) > MAX_STATIONS:
            atlas = atlas.sample(n=MAX_STATIONS, random_state=0)
        n = len(atlas)

        z_kde = kde_lookup.get(p.stem, np.nan)
        if not np.isfinite(z_kde):
            continue

        coords = atlas[["lat", "lng"]].values
        for target_col in target_columns:
            y = atlas[target_col].values.astype(float)
            # log(y+1) to stabilise; both targets are non-negative counts/densities
            y_log = np.log(y + 1.0)
            if y_log.std() < 1e-6:
                continue
            for split_id in range(N_SPLITS):
                split_rng = np.random.default_rng(
                    hash((p.stem, target_col, split_id)) % (2**32)
                )
                idx = split_rng.permutation(n)
                n_test = int(round(HOLDOUT * n))
                test_idx = idx[:n_test]
                train_idx = idx[n_test:]
                try:
                    r2 = nystrom_inductive_r2(
                        coords, y_log, train_idx, test_idx, K_SPEC, K_NN,
                    )
                except Exception:
                    continue
                rows.append({
                    "city": p.stem,
                    "target": target_col,
                    "split_id": split_id,
                    "N": int(n),
                    "z_kde": float(z_kde),
                    "r2_spec_inductive": float(r2),
                })
        if i % 20 == 0 and rows:
            recent = [r for r in rows if r["city"] == p.stem
                      and r["target"] == target_columns[0]]
            if recent:
                print(f"  [{i}/{len(parquets)}] {p.stem[:38]:38s}  "
                      f"N={n:4d}  z_kde={z_kde:+.2f}  "
                      f"R²(gtfs)={np.mean([r['r2_spec_inductive'] for r in recent]):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "16_atlas_proxy_results.csv", index=False)
    print(f"\n  ✓ {df['city'].nunique()} cities, {len(df)} observations")

    # ── LMM fits per target ──────────────────────────────────────────
    out_summary = {}
    for target in target_columns:
        sub = df[df["target"] == target].copy()
        if sub["city"].nunique() < 10:
            continue
        sub["z_kde2"] = sub["z_kde"] ** 2
        md = smf.mixedlm("r2_spec_inductive ~ z_kde + z_kde2", sub,
                         groups=sub["city"])
        res = md.fit(method="lbfgs")
        md_lin = smf.mixedlm("r2_spec_inductive ~ z_kde", sub,
                             groups=sub["city"])
        res_lin = md_lin.fit(method="lbfgs")
        # peak of fitted parabola
        b1 = float(res.fe_params["z_kde"])
        b2 = float(res.fe_params["z_kde2"])
        z_peak = -b1 / (2 * b2) if abs(b2) > 1e-9 else float("nan")
        out_summary[target] = {
            "n_obs": int(len(sub)),
            "n_cities": int(sub["city"].nunique()),
            "intercept": float(res.fe_params["Intercept"]),
            "z_kde_beta": b1,
            "z_kde_p": float(res.pvalues["z_kde"]),
            "z_kde2_beta": b2,
            "z_kde2_p": float(res.pvalues["z_kde2"]),
            "z_kde2_negative": bool(b2 < 0),
            "z_kde2_sig_005": bool(res.pvalues["z_kde2"] <= 0.05),
            "z_kde2_sig_001": bool(res.pvalues["z_kde2"] <= 0.01),
            "z_peak": float(z_peak),
            "loglik_quadratic": float(res.llf),
            "loglik_linear": float(res_lin.llf),
        }
        print(f"\n=== {target} ===")
        print(res.summary())
        print(f"  β₂ = {b2:+.6f}  p = {res.pvalues['z_kde2']:.4f}  "
              f"sign: {'✓' if b2 < 0 else '✗'}  "
              f"sig@0.05: {'✓' if res.pvalues['z_kde2'] <= 0.05 else '✗'}  "
              f"sig@0.01: {'✓' if res.pvalues['z_kde2'] <= 0.01 else '✗'}")
        print(f"  parabola peak at z = {z_peak:.2f}")

    with open(OUT / "16_atlas_proxy_lmm.json", "w") as f:
        json.dump(out_summary, f, indent=2, default=float)

    # ── Figure ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, target in zip(axes, target_columns):
        sub = df[df["target"] == target]
        city_stats = sub.groupby("city").agg(
            z_kde=("z_kde", "first"),
            r2_mean=("r2_spec_inductive", "mean"),
            r2_std=("r2_spec_inductive", "std"),
            N=("N", "first"),
        ).reset_index()
        sc = ax.scatter(city_stats["z_kde"], city_stats["r2_mean"],
                        s=30, alpha=0.6, c=np.log(city_stats["N"]),
                        cmap="viridis", edgecolor="black", lw=0.3)
        info = out_summary.get(target, {})
        if info:
            xs = np.linspace(sub["z_kde"].min() - 1, sub["z_kde"].max() + 1, 200)
            ys = info["intercept"] + info["z_kde_beta"] * xs + info["z_kde2_beta"] * xs ** 2
            ax.plot(xs, ys, "C3-", lw=2,
                    label=f"β₂={info['z_kde2_beta']:+.5f}  p={info['z_kde2_p']:.4f}\n"
                          f"peak z={info['z_peak']:+.2f}")
            ax.legend(fontsize=9)
        ax.set_xlabel("z_kde")
        ax.set_ylabel("R²_spec (inductive Nyström)")
        ax.set_title(f"target: {target}\n"
                     f"n_cities={city_stats.shape[0]}, n_obs={len(sub)}")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label="log N")
    fig.suptitle("Atlas-wide Goldilocks test on demand-proxy covariates",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "16_atlas_proxy.pdf", bbox_inches="tight")
    fig.savefig(OUT / "16_atlas_proxy.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '16_atlas_proxy.pdf'}")


if __name__ == "__main__":
    main()
