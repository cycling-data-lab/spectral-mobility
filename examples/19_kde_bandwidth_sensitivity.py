"""19_kde_bandwidth_sensitivity.py — KDE-bandwidth sensitivity of
the pre-reg v2 effects.

Per Gemini round 4 the bandwidth `1.5 × Scott's rule` was an
empirical choice.  Reviewer attack: "your z_kde definition depends
on this arbitrary scalar; β₁ in H1b might flip sign for other
choices".  We sweep the bandwidth over [0.5, 0.75, 1.0, 1.5, 2.0,
3.0] × Scott's rule, recompute z_kde on the 18 real-demand cities,
and check that H1b's slope remains negative across the range.

Since the Dirichlet energy (H1a) doesn't depend on KDE bandwidth,
we focus this sensitivity on H1b.

Output:
  19_bandwidth_sweep.csv
  19_bandwidth_sensitivity.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import BallTree

from spectral_mobility import (
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
N_SPLITS = 10
N_NULL = 10
MIN_MATCHED = 60
BW_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


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


def kde_null_ipr_at(lat, lng, N, k_nn, n_rep, bw_factor, seed):
    rng = np.random.default_rng(seed)
    pts = np.vstack([lat, lng])
    try:
        kde = gaussian_kde(pts)
        kde.set_bandwidth(kde.factor * bw_factor)
    except Exception:
        return float("nan"), float("nan")
    iprs = []
    for r in range(n_rep):
        try:
            sample = kde.resample(N, seed=rng.integers(0, 2**31)).T
            la = sample[:, 0]; ln = sample[:, 1]
            W, _ = build_geographic_knn(la, ln, k=k_nn)
            W = _to_dense(W)
            L = _to_dense(symmetric_normalised_laplacian(W))
            _, eigvecs = spectral_decomposition(L)
            ipr = inverse_participation_ratio(eigvecs)
            iprs.append(float(np.mean(ipr)))
        except Exception:
            continue
    if len(iprs) < 3:
        return float("nan"), float("nan")
    return float(np.mean(iprs)), float(np.std(iprs))


def nystrom_extend(coords_tr, coords_te, Uk, sigma, k_nn):
    from spectral_mobility.graph import EARTH_RADIUS_METRES
    train_rad = np.deg2rad(coords_tr); test_rad = np.deg2rad(coords_te)
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

    # First, load splits and compute gain (independent of bandwidth)
    print("=== Stage 1: gather per-city, per-split gains (bandwidth-independent) ===")
    base_rows = []
    cached_obs_ipr = {}    # cache per-city observed mean IPR + (lat, lng, N)
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
        coords = merged[["lat", "lng"]].values
        X_bare = merged[bare_cols].values.astype(float)
        X_bare = (X_bare - X_bare.mean(0)) / (X_bare.std(0) + 1e-9)
        y = merged["y"].values
        n = len(merged)
        city = pred_path.stem.replace("_predictions", "")

        # observed mean IPR
        W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=K_NN)
        W = _to_dense(W)
        L = _to_dense(symmetric_normalised_laplacian(W))
        _, eigvecs = spectral_decomposition(L)
        ipr_obs = float(np.mean(inverse_participation_ratio(eigvecs)))
        cached_obs_ipr[city] = {
            "lat": coords[:, 0], "lng": coords[:, 1],
            "N": n, "atlas": atlas_path.stem, "ipr_obs": ipr_obs,
        }

        for split_id in range(N_SPLITS):
            rng = np.random.default_rng(hash((city, split_id)) % (2**32))
            idx = rng.permutation(n)
            n_test = int(round(HOLDOUT * n))
            test_idx = idx[:n_test]; train_idx = idx[n_test:]
            try:
                r2b, r2a = inductive_split(coords, X_bare, y, train_idx, test_idx,
                                            K_SPEC, K_NN)
            except Exception:
                continue
            base_rows.append({
                "city": city, "split_id": split_id, "N": n,
                "r2_base": r2b, "r2_aug": r2a, "gain": r2a - r2b,
            })
        print(f"  {city[:28]:28s}  N={n:4d}  IPR_obs={ipr_obs:.4f}")

    base_df = pd.DataFrame(base_rows)

    # Stage 2: per bandwidth, recompute z_kde for each city, then fit H1b
    print("\n=== Stage 2: z_kde at each bandwidth + H1b LMM ===")
    sweep_rows = []
    cities = list(cached_obs_ipr.keys())
    for bw in BW_FACTORS:
        z_per_city = {}
        for city in cities:
            cache = cached_obs_ipr[city]
            null_mu, null_sd = kde_null_ipr_at(
                cache["lat"], cache["lng"], cache["N"], K_NN,
                N_NULL, bw, seed=hash((city, bw)) % (2**31),
            )
            if not np.isfinite(null_sd) or null_sd < 1e-9:
                continue
            z = (cache["ipr_obs"] - null_mu) / null_sd
            z_per_city[city] = z

        # Merge z_kde back into base_df
        bw_df = base_df.copy()
        bw_df["z_kde"] = bw_df["city"].map(z_per_city)
        bw_df = bw_df.dropna(subset=["z_kde"])
        bw_df_restricted = bw_df[bw_df["z_kde"] > -5].copy()
        n_cities = bw_df_restricted["city"].nunique()
        try:
            md = smf.mixedlm("gain ~ z_kde", bw_df_restricted, groups=bw_df_restricted["city"])
            res = md.fit(method="lbfgs")
            beta = float(res.fe_params["z_kde"])
            p = float(res.pvalues["z_kde"])
            ci_lo = float(res.conf_int().loc["z_kde", 0])
            ci_hi = float(res.conf_int().loc["z_kde", 1])
        except Exception as e:
            print(f"  bw={bw}: LMM failed {e}")
            beta = float("nan"); p = float("nan")
            ci_lo = float("nan"); ci_hi = float("nan")
        sweep_rows.append({
            "bw_factor": bw,
            "n_cities_above_minus5": n_cities,
            "z_kde_beta": beta,
            "z_kde_p": p,
            "ci_low": ci_lo, "ci_high": ci_hi,
            "z_kde_min": float(bw_df["z_kde"].min()),
            "z_kde_max": float(bw_df["z_kde"].max()),
        })
        print(f"  bw={bw:.2f}  n={n_cities:2d}  β={beta:+.5f}  p={p:.4f}  "
              f"CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"sign:{'✓' if beta < 0 else '✗'}  "
              f"sig@0.05:{'✓' if p <= 0.05 else '✗'}")

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT / "19_bandwidth_sweep.csv", index=False)

    # ── Figure ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.errorbar(sweep["bw_factor"], sweep["z_kde_beta"],
                yerr=[sweep["z_kde_beta"] - sweep["ci_low"],
                       sweep["ci_high"] - sweep["z_kde_beta"]],
                fmt="o-", color="C0", capsize=4, markersize=8)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.fill_between([min(BW_FACTORS), max(BW_FACTORS)], -1, 0,
                     color="green", alpha=0.05, label="conjectured zone (β < 0)")
    for _, r in sweep.iterrows():
        ax.annotate(f"p={r['z_kde_p']:.3f}\n(n={int(r['n_cities_above_minus5'])})",
                     (r["bw_factor"], r["z_kde_beta"]),
                     fontsize=8, ha="center", va="bottom",
                     xytext=(0, 8), textcoords="offset points")
    ax.set_xlabel("KDE bandwidth factor (multiplier on Scott's rule)")
    ax.set_ylabel("β₁ (z_kde slope in H1b LMM)")
    ax.set_title("H1b slope sensitivity to KDE bandwidth\n"
                 f"pre-reg v2 default = 1.5  →  {sweep.loc[sweep['bw_factor']==1.5, 'z_kde_beta'].values[0]:+.5f}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "19_bandwidth_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "19_bandwidth_sensitivity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '19_bandwidth_sensitivity.pdf'}")
    sign_always_neg = (sweep["z_kde_beta"] < 0).all()
    sig_count = (sweep["z_kde_p"] <= 0.05).sum()
    print(f"\nSummary:")
    print(f"  Sign of β stays negative across all bandwidths: {sign_always_neg}")
    print(f"  Significant @ α=0.05: {sig_count}/{len(sweep)} bandwidths")


if __name__ == "__main__":
    main()
