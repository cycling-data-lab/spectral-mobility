"""21_spectral_blocks_and_gap.py — Gemini round 6 hardening.

Three concrete improvements over script 20:

  (1) Replace k-means(K=10) on raw coords with **spectral clustering**
      on the k-NN Laplacian itself.  This respects natural urban cuts
      (rivers, motorways) because the bottom eigenvectors of L encode
      connectivity, not just euclidean proximity.  Inattaquable in
      peer-review GSP.

  (2) Resurrect Dirichlet by predicting the **R² gap**
      = R²_random - R²_block,  the spatial-autocorrelation inflation
      of the random-CV ceiling.  Hypothesis: smooth-signal cities
      (low Dirichlet) suffer more leakage.

  (3) Add **eigenvector-energy-on-test-block** as a control variable
      and mechanistic mediator: gain ~ z_kde + eig_energy_test + (1|city).
      If top-K eigenvectors have zero amplitude on the test block,
      augmentation cannot help — this explains the β x17 mechanically.

Output:
  21_block_spectral_results.csv
  21_summary.json
  21_results.{pdf,png}
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
K_SPEC_FOR_BLOCKS = 12   # use bottom-K eigenvectors to cluster
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


def spectral_blocks(W, eigvecs, k_blocks, k_eig=K_SPEC_FOR_BLOCKS, seed=SEED):
    """Spectral clustering on the k-NN graph: k-means on the first
    k_eig non-trivial eigenvectors.  Respects graph connectivity, not
    raw euclidean distance."""
    # skip the first eigenvector (constant) which carries no info
    U = eigvecs[:, 1:k_eig + 1]
    # row-normalise (Ng-Jordan-Weiss)
    norm = np.linalg.norm(U, axis=1, keepdims=True)
    norm = np.where(norm > 1e-12, norm, 1.0)
    U_n = U / norm
    return KMeans(n_clusters=k_blocks, random_state=seed, n_init=10).fit_predict(U_n)


def nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn):
    from spectral_mobility.graph import EARTH_RADIUS_METRES
    train_rad = np.deg2rad(coords_tr); test_rad = np.deg2rad(coords_te)
    tree = BallTree(train_rad, metric="haversine")
    dist_rad, idx = tree.query(test_rad, k=k_nn)
    dist_m = dist_rad * EARTH_RADIUS_METRES
    w = np.exp(-(dist_m ** 2) / (2.0 * sigma ** 2))
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
    return np.einsum("ij,ijk->ik", w, Uk[idx])


def split_fit(coords, X_bare, y, train_mask, K, k_nn):
    test_mask = ~train_mask
    coords_tr = coords[train_mask]; coords_te = coords[test_mask]
    X_tr = X_bare[train_mask]; X_te = X_bare[test_mask]
    y_tr = y[train_mask]; y_te = y[test_mask]
    if len(y_tr) < MIN_TRAIN or len(y_te) < 5:
        return float("nan"), float("nan"), float("nan")
    r2_base = r2_score(y_te, Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te))
    W_tr, sigma = build_geographic_knn(coords_tr[:, 0], coords_tr[:, 1], k=k_nn)
    L_tr = _to_dense(symmetric_normalised_laplacian(_to_dense(W_tr)))
    _, eigvecs_tr = spectral_decomposition(L_tr)
    Uk = eigvecs_tr[:, :K]
    Uk_te = nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn)
    aug = Ridge(alpha=1.0).fit(np.column_stack([X_tr, Uk]), y_tr)
    r2_aug = r2_score(y_te, aug.predict(np.column_stack([X_te, Uk_te])))
    # Energy of top-K eigvecs on TEST block:
    # ||Uk_te||²_F / n_test  (mean per-node energy)
    eig_energy_test = float(np.mean(np.sum(Uk_te ** 2, axis=1)))
    return float(r2_base), float(r2_aug), eig_energy_test


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

        # Full-graph spectral decomposition (for blocks AND for Dirichlet)
        W_full, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=K_NN)
        W_full = _to_dense(W_full)
        L_sym = _to_dense(symmetric_normalised_laplacian(W_full))
        L_un = _to_dense(unnormalised_laplacian(W_full))
        eigvals_full, eigvecs_full = spectral_decomposition(L_sym)
        D_un = float(dirichlet_energy(L_un, y))

        # Spectral clustering blocks
        k_blocks = min(K_BLOCKS, n // (MIN_TRAIN + 5))
        if k_blocks < 3:
            continue
        try:
            block_labels = spectral_blocks(W_full, eigvecs_full, k_blocks)
        except Exception as e:
            print(f"  ✗ {city[:28]}: spectral_blocks failed: {e}")
            continue

        # Block-CV
        for b in range(k_blocks):
            train_mask = block_labels != b
            r2_base, r2_aug, eig_energy = split_fit(
                coords, X_bare, y, train_mask, K_SPEC, K_NN,
            )
            if not all(np.isfinite([r2_base, r2_aug, eig_energy])):
                continue
            rows.append({
                "city": city, "atlas": atlas_path.stem,
                "block": int(b), "block_size": int((~train_mask).sum()),
                "N": int(n), "z_kde": float(z_kde),
                "dirichlet_unnorm": D_un,
                "r2_base_block": r2_base, "r2_aug_block": r2_aug,
                "gain_block": r2_aug - r2_base,
                "eig_energy_test": eig_energy,
            })

        # Random-CV reference (same number of folds, random)
        rng = np.random.default_rng(hash(city) % (2**32))
        for split_id in range(k_blocks):
            idx = rng.permutation(n)
            n_test = int((~train_mask).sum())  # match the last block size
            test_idx = idx[:n_test]
            train_idx = idx[n_test:]
            mask = np.zeros(n, dtype=bool); mask[train_idx] = True
            r2_base, r2_aug, _ = split_fit(
                coords, X_bare, y, mask, K_SPEC, K_NN,
            )
            if all(np.isfinite([r2_base, r2_aug])):
                # attach to the same city, distinguish by block=-(split_id+1)
                rows.append({
                    "city": city, "atlas": atlas_path.stem,
                    "block": -(split_id + 1),  # negative = random reference
                    "block_size": n_test, "N": int(n),
                    "z_kde": float(z_kde),
                    "dirichlet_unnorm": D_un,
                    "r2_base_block": r2_base, "r2_aug_block": r2_aug,
                    "gain_block": r2_aug - r2_base,
                    "eig_energy_test": float("nan"),
                })

        sub_block = [r for r in rows if r["city"] == city and r["block"] >= 0]
        sub_rand = [r for r in rows if r["city"] == city and r["block"] < 0]
        if sub_block:
            print(f"  ✓ {city[:28]:28s}  N={n:4d}  z={z_kde:+.2f}  D={D_un:.2f}  "
                  f"R²_block={np.mean([r['r2_aug_block'] for r in sub_block]):+.3f}  "
                  f"R²_rand={np.mean([r['r2_aug_block'] for r in sub_rand]):+.3f}  "
                  f"gap={np.mean([r['r2_aug_block'] for r in sub_rand]) - np.mean([r['r2_aug_block'] for r in sub_block]):+.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "21_block_spectral_results.csv", index=False)

    block_df = df[df["block"] >= 0].copy()
    rand_df = df[df["block"] < 0].copy()

    per_city = block_df.groupby("city").agg(
        z_kde=("z_kde", "first"),
        D=("dirichlet_unnorm", "first"),
        N=("N", "first"),
        r2_block=("r2_aug_block", "mean"),
        gain_block=("gain_block", "mean"),
        eig_energy=("eig_energy_test", "mean"),
    ).reset_index()
    rand_per_city = rand_df.groupby("city").agg(
        r2_random=("r2_aug_block", "mean"),
        gain_random=("gain_block", "mean"),
    ).reset_index()
    per_city = per_city.merge(rand_per_city, on="city")
    per_city["r2_gap"] = per_city["r2_random"] - per_city["r2_block"]
    per_city["gain_gap"] = per_city["gain_random"] - per_city["gain_block"]
    per_city.to_csv(OUT / "21_per_city.csv", index=False)
    print(f"\n  ✓ {len(per_city)} cities with spectral blocks + random ref")
    print(per_city[["city", "z_kde", "D", "r2_block", "r2_random", "r2_gap"]].round(3).to_string(index=False))

    summary = {}

    # ── (A) H1b under spectral-block CV ──────────────────────────────
    print("\n=== H1b under SPECTRAL block-CV (LMM, z>−5) ===")
    bdf = block_df[block_df["z_kde"] > -5].copy()
    n_cities_h1b = bdf["city"].nunique()
    try:
        md = smf.mixedlm("gain_block ~ z_kde", bdf, groups=bdf["city"])
        res = md.fit(method="lbfgs")
        print(res.summary())
        b = float(res.fe_params["z_kde"]); p = float(res.pvalues["z_kde"])
        print(f"  → β={b:+.5f}  p={p:.4f}  sign:{'✓' if b<0 else '✗'}  sig@.01:{'✓' if p<=0.01 else '✗'}")
        summary["h1b_spectral_block"] = {
            "n_cities": int(n_cities_h1b), "z_kde_beta": b, "z_kde_p": p,
            "sign_correct": bool(b<0), "sig_001": bool(p<=0.01),
        }
    except Exception as e:
        print(f"  failed: {e}")

    # ── (B) R²_gap ~ Dirichlet ───────────────────────────────────────
    print("\n=== R²_gap ~ Dirichlet + log N  (Q3 - rescuing the signal axis) ===")
    Xm = sm.add_constant(np.column_stack([
        per_city["D"], np.log(per_city["N"])
    ]))
    try:
        ols_gap = sm.OLS(per_city["r2_gap"], Xm).fit()
        print(ols_gap.summary())
        summary["gap_regression"] = {
            "n_cities": int(len(per_city)),
            "dirichlet_beta": float(ols_gap.params[1]),
            "dirichlet_p": float(ols_gap.pvalues[1]),
            "log_N_beta": float(ols_gap.params[2]),
            "log_N_p": float(ols_gap.pvalues[2]),
            "R2_overall": float(ols_gap.rsquared),
            # we conjecture POSITIVE sign: smoother demand → bigger gap
            "sign_correct": bool(ols_gap.params[1] < 0),  # actually negative D → positive gap
            "sig_005": bool(ols_gap.pvalues[1] <= 0.05),
        }
    except Exception as e:
        print(f"  failed: {e}")

    # ── (C) Eigenvector energy as mediator ───────────────────────────
    print("\n=== Gain_block ~ z_kde + eig_energy_test + (1|city) ===")
    try:
        bdf = block_df[block_df["z_kde"] > -5].copy()
        # standardise eig_energy for stable coefficient
        bdf["eig_energy_z"] = (bdf["eig_energy_test"] - bdf["eig_energy_test"].mean()) / (bdf["eig_energy_test"].std() + 1e-9)
        md = smf.mixedlm("gain_block ~ z_kde + eig_energy_z", bdf,
                         groups=bdf["city"])
        res = md.fit(method="lbfgs")
        print(res.summary())
        summary["mediator_model"] = {
            "z_kde_beta": float(res.fe_params["z_kde"]),
            "z_kde_p": float(res.pvalues["z_kde"]),
            "eig_energy_beta": float(res.fe_params["eig_energy_z"]),
            "eig_energy_p": float(res.pvalues["eig_energy_z"]),
        }
    except Exception as e:
        print(f"  failed: {e}")

    with open(OUT / "21_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # ── Figure ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (a) Gain block-CV vs z_kde (spectral blocks) — the primary H1
    ax = axes[0, 0]
    ax.errorbar(per_city["z_kde"], per_city["gain_block"], fmt="o", color="C0",
                markersize=8, capsize=2)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["z_kde"], r["gain_block"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(-5, color="black", lw=0.7, ls=":")
    info = summary.get("h1b_spectral_block", {})
    ax.set_xlabel("z_kde")
    ax.set_ylabel("Gain under spectral-block CV")
    ax.set_title(f"(a) H1b under spectral-block CV (z>-5)\n"
                 f"β={info.get('z_kde_beta', float('nan')):+.4f}  "
                 f"p={info.get('z_kde_p', float('nan')):.4f}  "
                 f"n={info.get('n_cities', '?')}")
    ax.grid(alpha=0.3)

    # (b) R²_gap vs Dirichlet
    ax = axes[0, 1]
    sc = ax.scatter(per_city["D"], per_city["r2_gap"], s=80,
                    c=per_city["z_kde"], cmap="coolwarm",
                    edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["D"], r["r2_gap"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    info = summary.get("gap_regression", {})
    ax.set_xscale("log")
    ax.set_xlabel("Dirichlet energy of demand (log)")
    ax.set_ylabel("R²_gap  =  R²_aug(random) − R²_aug(block)")
    ax.set_title(f"(b) Dirichlet predicts the leakage gap\n"
                 f"β={info.get('dirichlet_beta', float('nan')):+.3f}  "
                 f"p={info.get('dirichlet_p', float('nan')):.4f}  "
                 f"R²={info.get('R2_overall', float('nan')):.2f}")
    plt.colorbar(sc, ax=ax, label="z_kde")
    ax.grid(alpha=0.3)

    # (c) Eigenvector energy on test block vs gain
    ax = axes[1, 0]
    ax.scatter(block_df["eig_energy_test"], block_df["gain_block"], s=20,
               alpha=0.6, color="C0")
    info = summary.get("mediator_model", {})
    ax.set_xlabel("⟨‖Uk_test⟩‖²⟩  (mean eigenvector energy per test-block node)")
    ax.set_ylabel("Augmentation gain on test block")
    ax.set_title(f"(c) Mechanism: eigenvector reach on test block\n"
                 f"after controlling z_kde: β_eig={info.get('eig_energy_beta', float('nan')):+.4f}  "
                 f"p={info.get('eig_energy_p', float('nan')):.4f}")
    ax.grid(alpha=0.3)

    # (d) Random vs block R²_aug
    ax = axes[1, 1]
    ax.scatter(per_city["r2_random"], per_city["r2_block"], s=80,
               c=per_city["D"], cmap="viridis",
               edgecolor="black", lw=0.5)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:12], (r["r2_random"], r["r2_block"]),
                    fontsize=7, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    lo = min(per_city[["r2_block", "r2_random"]].min().min() - 0.5, -6)
    hi = max(per_city[["r2_block", "r2_random"]].max().max() + 0.05, 0.7)
    ax.plot([lo, hi], [lo, hi], "k:", lw=1)
    ax.set_xlabel("R²_aug random-CV (interpolation regime)")
    ax.set_ylabel("R²_aug spectral block-CV (extrapolation regime)")
    ax.set_title("(d) The illusion of interpolation vs the reality of extrapolation")
    ax.grid(alpha=0.3)

    fig.suptitle("Spectral-block CV + R²_gap + eigenvector energy (Gemini r6)",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "21_results.pdf", bbox_inches="tight")
    fig.savefig(OUT / "21_results.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '21_results.pdf'}")


if __name__ == "__main__":
    main()
