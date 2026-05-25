"""02_boston_bikeshare.py — real-data demo on Boston Bluebikes.

Loads the Bluebikes IMD parquet + Tier-1 demand cache from the
sibling ``bikeshare-demand-forecasting`` repository, then compares
LightGBM with and without spectral augmentation under 5-fold
leave-station-out cross-validation.

Expected output (with default settings):

    Boston Bluebikes — 493 stations, 4 IMD features
    Ceiling (closed form):
      R² IMD only   : 0.106
      R² augmented  : 0.352
      ΔR²           : +0.246

    Realized 5-fold LSO R²:
      baseline      : 0.0XX (LightGBM on IMD-4)
      augmented     : 0.XXX (LightGBM on IMD-4 + 16 eigenvectors)
      mean gain     : +0.XXX

If you don't have a local clone of bikeshare-demand-forecasting,
adapt the file paths below to point at your own (lat, lng, features,
y) data.

Requires:
    pip install spectral-mobility[examples]   # includes lightgbm + pyarrow
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from spectral_mobility import SpectralAugmentedRegressor, spectral_bound


# ── Data source (sibling repo) ───────────────────────────────────────
SIBLING = Path.home() / "cesi-research" / "bikeshare-demand-forecasting"
IMD_PARQUET = SIBLING / "data_collection" / "imd_international" / "boston_bluebikes.parquet"
DEMAND_PARQUET = SIBLING / "experiments" / "outputs" / "d3_boston_bluebikes_predictions.parquet"

FEATS_IMD = [
    "gtfs_heavy_stops_300m",
    "infra_cyclable_features_300m",
    "elevation_m",
    "topography_roughness_index",
]


def load_boston():
    if not IMD_PARQUET.exists() or not DEMAND_PARQUET.exists():
        raise FileNotFoundError(
            "Could not find Boston Bluebikes data.  Adjust SIBLING path "
            "in this script to point at your local copy of the "
            "bikeshare-demand-forecasting repository."
        )
    imd = pd.read_parquet(IMD_PARQUET)
    imd["station_id"] = imd["station_id"].astype(str)

    demand = pd.read_parquet(DEMAND_PARQUET)
    demand["station_id"] = demand["station_id"].astype(str)
    demand["y"] = np.expm1(demand["y_true_log"])
    y_map = demand.groupby("station_id")["y"].mean().to_dict()
    imd["y"] = imd["station_id"].map(y_map)

    avail = [f for f in FEATS_IMD if f in imd.columns]
    sub = imd.dropna(subset=["lat", "lng", "y"] + avail).reset_index(drop=True)

    X = sub[avail].astype(float).values
    coords = sub[["lat", "lng"]].astype(float).values
    y = sub["y"].astype(float).values
    return X, coords, y, avail


def main() -> None:
    X, coords, y, feats = load_boston()
    print(f"Boston Bluebikes — {len(X)} stations, {X.shape[1]} IMD features")
    print(f"  Features: {feats}")
    print(f"  Demand range: {y.min():.2f} – {y.max():.2f}  (mean {y.mean():.2f})")

    # ── 1. Closed-form ceiling ──────────────────────────────────────
    model = SpectralAugmentedRegressor(K=16, k_nn=6, sigma=300.0)
    model.fit(X, coords, y)
    ceiling = model.ceiling()
    print("\nCeiling (closed-form spectral bound):")
    print(f"  R² IMD only          : {ceiling.r2_imd:.4f}")
    print(f"  R² spectral-only     : {ceiling.r2_spectral_only:.4f}")
    print(f"  R² augmented (K=16)  : {ceiling.r2_augmented:.4f}")
    print(f"  ΔR² (ceiling lift)   : +{ceiling.delta_r2:.4f}")

    # ── 2. Realized LSO comparison ──────────────────────────────────
    print("\nRunning 5-fold leave-station-out cross-validation...")
    cv = model.cross_validate(X, coords, y, n_folds=5, random_state=42)
    print(f"  Baseline LightGBM   (no augmentation): mean R² = {cv['baseline_mean']:+.4f}")
    print(f"  Augmented LightGBM  (K=16 eigvecs)    : mean R² = {cv['augmented_mean']:+.4f}")
    print(f"  Mean realized gain                    : {cv['mean_gain']:+.4f}")

    # ── 3. Per-fold breakdown ───────────────────────────────────────
    print("\nPer-fold R²:")
    for i, (b, a) in enumerate(zip(cv["baseline_scores"], cv["augmented_scores"]), 1):
        print(f"  Fold {i}: baseline {b:+.3f}   |   augmented {a:+.3f}   |   Δ {a-b:+.3f}")

    # ── 4. Operational reading ──────────────────────────────────────
    realized_fraction = cv["mean_gain"] / ceiling.delta_r2 if ceiling.delta_r2 > 0 else 0
    print("\nReading:")
    print(f"  The ceiling lift (closed form) is +{ceiling.delta_r2:.3f}.")
    print(f"  The realised lift (5-fold LSO) is +{cv['mean_gain']:+.3f}.")
    print(f"  LightGBM captures {realized_fraction*100:.0f}% of the achievable spectral gain.")


if __name__ == "__main__":
    main()
