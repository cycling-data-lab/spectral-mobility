"""18_pre_reg_v2_check.py — final exploratory checks before sealing
the OSF pre-registration v2 (Gemini round 4 verdict).

Three sanity checks:

  (1) VIF check.  Are the two pre-registered axes z_kde and the
      Dirichlet energy of demand on the unnormalised Laplacian
      orthogonal *enough* to be jointly identifiable?  Threshold:
      VIF < 3.

  (2) Laplacian choice.  Recompute Dirichlet with L_unnorm = D − W
      (the natural discrete-gradient energy for count-like targets,
      per Gemini Q2).  Verify that the strong negative correlation
      with R²_aug survives.

  (3) H1b range restriction.  Per Gemini Q3, the linear effect on
      z_kde should be evaluated on the typical-to-localised regime
      (z_kde > −5) to avoid the underpowered tail.  Re-fit the LMM
      on this subset and check that β₁ is still negative.

Output:
  18_pre_reg_v2_check.csv
  18_vif_and_lmm.json
  18_pre_reg_v2.{pdf,png}
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
    coords_tr = coords[train_idx]; coords_te = coords[test_idx]
    X_tr = X_bare[train_idx]; X_te = X_bare[test_idx]
    y_tr = y[train_idx]; y_te = y[test_idx]
    base = Ridge(alpha=1.0).fit(X_tr, y_tr)
    r2_base = r2_score(y_te, base.predict(X_te))
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
        X_bare = (X_bare - X_bare.mean(0)) / (X_bare.std(0) + 1e-9)
        y = merged["y_log"].values
        n = len(merged)
        city_name = pred_path.stem.replace("_predictions", "")

        # Both Dirichlets: unnormalised (Gemini choice) and symmetric (comparison)
        W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=K_NN)
        L_un = _to_dense(unnormalised_laplacian(_to_dense(W)))
        L_sym = _to_dense(symmetric_normalised_laplacian(_to_dense(W)))
        D_un = dirichlet_energy(L_un, y)
        D_sym = dirichlet_energy(L_sym, y)

        # Splits
        for split_id in range(N_SPLITS):
            split_rng = np.random.default_rng(hash((city_name, split_id)) % (2**32))
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
            rows.append({
                "city": city_name, "split_id": split_id, "N": int(n),
                "z_kde": float(z_kde),
                "dirichlet_unnorm": float(D_un),
                "dirichlet_sym": float(D_sym),
                "r2_base": r2_base, "r2_aug": r2_aug,
                "gain": r2_aug - r2_base,
            })
        last = [r for r in rows if r["city"] == city_name][-N_SPLITS:]
        if last:
            print(f"  ✓ {city_name[:28]:28s}  N={n:4d}  z_kde={z_kde:+.2f}  "
                  f"D_un={D_un:7.2f}  D_sym={D_sym:.3f}  "
                  f"R²_aug={np.mean([r['r2_aug'] for r in last]):.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "18_pre_reg_v2_check.csv", index=False)
    per_city = df.groupby("city").agg(
        z_kde=("z_kde", "first"),
        dirichlet_unnorm=("dirichlet_unnorm", "first"),
        dirichlet_sym=("dirichlet_sym", "first"),
        N=("N", "first"),
        r2_aug=("r2_aug", "mean"),
        r2_base=("r2_base", "mean"),
        gain=("gain", "mean"),
    ).reset_index()

    summary = {}

    # ── (1) VIF check ────────────────────────────────────────────────
    print("\n=== (1) VIF check between z_kde and Dirichlet axes ===")
    for D_col in ["dirichlet_unnorm", "dirichlet_sym"]:
        X = per_city[["z_kde", D_col, "N"]].values
        Xz = (X - X.mean(0)) / X.std(0)
        # VIF per column = 1/(1-R²) of regressing that column on the others
        vif = []
        for k in range(Xz.shape[1]):
            others = np.delete(Xz, k, axis=1)
            beta, *_ = np.linalg.lstsq(others, Xz[:, k], rcond=None)
            pred = others @ beta
            r2 = 1 - ((Xz[:, k] - pred) ** 2).sum() / (((Xz[:, k] - Xz[:, k].mean()) ** 2).sum() + 1e-12)
            vif.append(1.0 / max(1e-9, 1 - r2))
        corr_zd = float(np.corrcoef(per_city["z_kde"], per_city[D_col])[0, 1])
        summary[f"vif_{D_col}"] = {
            "vif_z_kde": float(vif[0]), "vif_dirichlet": float(vif[1]),
            "vif_log_N": float(vif[2]),
            "corr_z_kde_dirichlet": corr_zd,
        }
        print(f"  using {D_col}:  corr(z_kde, D)={corr_zd:+.3f}  "
              f"VIF(z_kde)={vif[0]:.2f}  VIF(D)={vif[1]:.2f}  "
              f"VIF(N)={vif[2]:.2f}  → "
              f"{'✓ safe' if max(vif[:2]) < 3 else '✗ colinear'}")

    # ── (2) Dirichlet sanity: correlation with R²_aug ────────────────
    print("\n=== (2) Dirichlet correlations (unnorm = pre-reg choice) ===")
    for D_col in ["dirichlet_unnorm", "dirichlet_sym"]:
        r_aug = np.corrcoef(per_city[D_col], per_city["r2_aug"])[0, 1]
        r_base = np.corrcoef(per_city[D_col], per_city["r2_base"])[0, 1]
        r_gain = np.corrcoef(per_city[D_col], per_city["gain"])[0, 1]
        print(f"  {D_col:18s}  R²_aug={r_aug:+.3f}  R²_base={r_base:+.3f}  gain={r_gain:+.3f}")
        summary[f"corr_{D_col}"] = {
            "r_aug": float(r_aug), "r_base": float(r_base), "r_gain": float(r_gain)
        }

    # ── (3) LMM with chosen Laplacian: r2_aug ~ Dirichlet_unnorm + log N
    print("\n=== H1a candidate: R²_aug ~ Dirichlet_unnorm + log(N) + (1|city) ===")
    df_lmm = df.copy()
    df_lmm["log_N"] = np.log(df_lmm["N"])
    try:
        md = smf.mixedlm("r2_aug ~ dirichlet_unnorm + log_N", df_lmm,
                         groups=df_lmm["city"])
        res = md.fit(method="lbfgs")
        print(res.summary())
        summary["h1a"] = {
            "dirichlet_beta": float(res.fe_params["dirichlet_unnorm"]),
            "dirichlet_p": float(res.pvalues["dirichlet_unnorm"]),
            "log_N_beta": float(res.fe_params["log_N"]),
            "log_N_p": float(res.pvalues["log_N"]),
            "sign_correct": bool(res.fe_params["dirichlet_unnorm"] < 0),
            "sig_001": bool(res.pvalues["dirichlet_unnorm"] <= 0.01),
        }
        print(f"  β(Dirichlet) = {res.fe_params['dirichlet_unnorm']:+.6f}  "
              f"p = {res.pvalues['dirichlet_unnorm']:.4f}  "
              f"sign:{'✓' if res.fe_params['dirichlet_unnorm'] < 0 else '✗'}  "
              f"sig@0.01:{'✓' if res.pvalues['dirichlet_unnorm'] <= 0.01 else '✗'}")
    except Exception as e:
        print(f"H1a LMM failed: {e}")

    # ── (4) H1b: gain ~ z_kde linear, restricted to z_kde > -5 ───────
    print("\n=== H1b candidate: gain ~ z_kde + (1|city), restricted to z_kde > -5 ===")
    df_h1b = df[df["z_kde"] > -5].copy()
    n_h1b = df_h1b["city"].nunique()
    try:
        md = smf.mixedlm("gain ~ z_kde", df_h1b, groups=df_h1b["city"])
        res = md.fit(method="lbfgs")
        print(res.summary())
        print(f"  n_cities = {n_h1b} (after restriction z_kde > -5)")
        summary["h1b"] = {
            "z_kde_beta": float(res.fe_params["z_kde"]),
            "z_kde_p": float(res.pvalues["z_kde"]),
            "n_cities": int(n_h1b),
            "sign_correct": bool(res.fe_params["z_kde"] < 0),
            "sig_005": bool(res.pvalues["z_kde"] <= 0.05),
        }
        print(f"  β(z_kde) = {res.fe_params['z_kde']:+.5f}  "
              f"p = {res.pvalues['z_kde']:.4f}  "
              f"sign:{'✓' if res.fe_params['z_kde'] < 0 else '✗'}  "
              f"sig@0.05:{'✓' if res.pvalues['z_kde'] <= 0.05 else '✗'}")
    except Exception as e:
        print(f"H1b LMM failed: {e}")

    with open(OUT / "18_vif_and_lmm.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.scatter(per_city["dirichlet_unnorm"], per_city["r2_aug"],
               s=80, c=per_city["z_kde"], cmap="coolwarm",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["dirichlet_unnorm"], r["r2_aug"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    r_corr = np.corrcoef(per_city["dirichlet_unnorm"], per_city["r2_aug"])[0, 1]
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet energy on L = D − W  (unnormalised, log-scale)")
    ax.set_ylabel("R²_aug")
    ax.set_title(f"(a) H1a candidate axis  corr={r_corr:+.3f}")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(per_city["z_kde"], per_city["gain"],
               s=80, c=per_city["z_kde"], cmap="coolwarm",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["z_kde"], r["gain"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(-5, color="black", lw=0.7, ls=":")
    ax.set_xlabel("z_kde  (vertical line = H1b cut-off z = −5)")
    ax.set_ylabel("Augmentation gain  R²_aug − R²_base")
    ax.set_title("(b) H1b candidate axis (gain)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(per_city["z_kde"], per_city["dirichlet_unnorm"],
               s=80, c=per_city["r2_aug"], cmap="viridis",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["z_kde"], r["dirichlet_unnorm"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    r_corr = np.corrcoef(per_city["z_kde"], per_city["dirichlet_unnorm"])[0, 1]
    ax.set_xlabel("z_kde")
    ax.set_yscale("log")
    ax.set_ylabel("Dirichlet (unnorm, log)")
    ax.set_title(f"(c) Orthogonality of the two axes  corr={r_corr:+.3f}")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    txt = ["PRE-REG v2 STATUS",
           "",
           "Axis 1 — Dirichlet on L=D−W (H1a)",
           f"  corr with R²_aug:  {summary.get('corr_dirichlet_unnorm', {}).get('r_aug', float('nan')):+.3f}",
    ]
    if "h1a" in summary:
        txt.extend([
            f"  LMM β:             {summary['h1a']['dirichlet_beta']:+.5f}",
            f"  LMM p:             {summary['h1a']['dirichlet_p']:.4f}",
            f"  Sign correct?      {summary['h1a']['sign_correct']}",
            f"  Significant α=.01? {summary['h1a']['sig_001']}",
        ])
    txt.extend(["", "Axis 2 — z_kde on z_kde > −5 (H1b)"])
    if "h1b" in summary:
        txt.extend([
            f"  n_cities:          {summary['h1b']['n_cities']}",
            f"  LMM β:             {summary['h1b']['z_kde_beta']:+.5f}",
            f"  LMM p:             {summary['h1b']['z_kde_p']:.4f}",
            f"  Sign correct?      {summary['h1b']['sign_correct']}",
            f"  Significant α=.05? {summary['h1b']['sig_005']}",
        ])
    txt.extend(["", "VIF on (z_kde, D_unnorm, log_N)"])
    if "vif_dirichlet_unnorm" in summary:
        v = summary["vif_dirichlet_unnorm"]
        txt.extend([
            f"  VIF(z_kde):        {v['vif_z_kde']:.2f}",
            f"  VIF(Dirichlet):    {v['vif_dirichlet']:.2f}",
            f"  VIF(log_N):        {v['vif_log_N']:.2f}",
            f"  All < 3?           {max(v['vif_z_kde'], v['vif_dirichlet'], v['vif_log_N']) < 3}",
        ])
    ax.text(0.05, 0.95, "\n".join(txt), fontsize=10,
            transform=ax.transAxes, va="top", family="monospace")

    fig.suptitle("Pre-reg v2 sanity check — Dirichlet on L=D−W + H1b on z_kde > −5",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "18_pre_reg_v2.pdf", bbox_inches="tight")
    fig.savefig(OUT / "18_pre_reg_v2.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '18_pre_reg_v2.pdf'}")


if __name__ == "__main__":
    main()
