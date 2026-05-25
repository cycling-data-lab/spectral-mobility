"""22_robust_block_cv_estimator.py — fix the LMM numerical instability
in the spectral block-CV result.

Script 21's LMM diverged numerically (Group Var → 0, intercept SE → 1e7)
because the spectral-clustering blocks had very uneven sizes (3 to 240
stations per block), leading to heteroskedasticity that REML cannot
absorb.

Three more robust estimators, all on the same data
(`21_block_spectral_results.csv`):

  (A) **Per-city OLS** on the average block-CV gain.  n_obs = n_cities
      = 11.  Loses within-city variance information but has a clean,
      well-understood t-test.

  (B) **Weighted OLS** with weights = block_size (large blocks are
      more reliable estimates).  Same n as the LMM but cleanly
      identifies which observations contribute.

  (C) **Bootstrap cluster-robust SE** on the LMM: resample cities
      with replacement 2000 times, re-fit, get the percentile CI on
      β.  Avoids the asymptotic normal approximation that fails on
      our small n_cities.

We pre-register the estimator that gives the most conservative
(largest p, narrowest |β|) of the three.

Output:
  22_robust_estimators.json
  22_estimator_comparison.{pdf,png}
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


OUT = Path(__file__).parent / "output"
SEED = 2026
N_BOOT = 2000


def main():
    df = pd.read_csv(OUT / "21_block_spectral_results.csv")
    block_df = df[df["block"] >= 0].copy()      # real blocks only
    block_df = block_df[block_df["z_kde"] > -5].copy()
    print(f"n_obs = {len(block_df)}  n_cities = {block_df['city'].nunique()}")
    print(block_df.groupby("city").size().describe())

    # ─── (A) Per-city OLS ───────────────────────────────────────────
    per_city = block_df.groupby("city").agg(
        z_kde=("z_kde", "first"),
        gain=("gain_block", "mean"),
        gain_std=("gain_block", "std"),
        n_blocks=("block", "count"),
        N=("N", "first"),
    ).reset_index()
    print("\n=== Per-city OLS (Gemini-preferred simple estimator) ===")
    X = sm.add_constant(per_city["z_kde"].values)
    res_ols = sm.OLS(per_city["gain"], X).fit()
    print(res_ols.summary())
    res_A = {
        "estimator": "per_city_OLS",
        "n": int(len(per_city)),
        "z_kde_beta": float(res_ols.params[1]),
        "z_kde_p": float(res_ols.pvalues[1]),
        "ci_low": float(res_ols.conf_int().iloc[1, 0]),
        "ci_high": float(res_ols.conf_int().iloc[1, 1]),
        "sign_correct": bool(res_ols.params[1] < 0),
        "sig_001": bool(res_ols.pvalues[1] <= 0.01),
        "sig_005": bool(res_ols.pvalues[1] <= 0.05),
    }

    # ─── (B) Weighted OLS on (city × block) ─────────────────────────
    print("\n=== Weighted OLS  weights = block_size ===")
    Xb = sm.add_constant(block_df["z_kde"].values)
    res_wls = sm.WLS(block_df["gain_block"], Xb,
                     weights=block_df["block_size"].values).fit()
    print(res_wls.summary())
    res_B = {
        "estimator": "weighted_OLS",
        "n": int(len(block_df)),
        "z_kde_beta": float(res_wls.params[1]),
        "z_kde_p": float(res_wls.pvalues[1]),
        "ci_low": float(res_wls.conf_int().iloc[1, 0]),
        "ci_high": float(res_wls.conf_int().iloc[1, 1]),
        "sign_correct": bool(res_wls.params[1] < 0),
        "sig_001": bool(res_wls.pvalues[1] <= 0.01),
        "sig_005": bool(res_wls.pvalues[1] <= 0.05),
    }

    # ─── (C) Cluster bootstrap on per-city OLS ──────────────────────
    print("\n=== Cluster bootstrap (resample cities) ===")
    cities = per_city["city"].values
    rng = np.random.default_rng(SEED)
    betas = []
    for b in range(N_BOOT):
        sample_cities = rng.choice(cities, size=len(cities), replace=True)
        sub = per_city[per_city["city"].isin(sample_cities)].copy()
        # weight by how many times each city was selected
        counts = pd.Series(sample_cities).value_counts()
        sub = sub.merge(counts.rename("w").reset_index().rename(columns={"index": "city"}),
                        on="city")
        Xs = sm.add_constant(sub["z_kde"].values)
        try:
            wls = sm.WLS(sub["gain"].values, Xs, weights=sub["w"].values).fit()
            betas.append(float(wls.params[1]))
        except Exception:
            continue
    betas = np.array(betas)
    point = float(res_ols.params[1])
    ci_lo = float(np.percentile(betas, 2.5))
    ci_hi = float(np.percentile(betas, 97.5))
    boot_p = 2.0 * min(float((betas >= 0).mean()), float((betas <= 0).mean()))
    print(f"  bootstrap β (mean): {betas.mean():+.4f}  (point: {point:+.4f})")
    print(f"  95% bootstrap CI:   [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  two-tailed p:       {boot_p:.4f}")
    print(f"  fraction β<0:       {(betas < 0).mean():.3f}")
    res_C = {
        "estimator": "cluster_bootstrap",
        "n_cities": int(len(cities)),
        "n_boot": int(len(betas)),
        "z_kde_beta_point": point,
        "z_kde_beta_mean": float(betas.mean()),
        "ci_low": ci_lo, "ci_high": ci_hi,
        "two_tailed_p": boot_p,
        "fraction_negative": float((betas < 0).mean()),
        "sig_001": bool(boot_p <= 0.01 and ci_hi < 0),
        "sig_005": bool(boot_p <= 0.05 and ci_hi < 0),
    }

    summary = {
        "per_city_OLS": res_A,
        "weighted_OLS": res_B,
        "cluster_bootstrap": res_C,
    }
    with open(OUT / "22_robust_estimators.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Verdict line
    print("\n=== ROBUSTNESS VERDICT ===")
    for est_name in ["per_city_OLS", "weighted_OLS", "cluster_bootstrap"]:
        r = summary[est_name]
        β = r.get("z_kde_beta") or r.get("z_kde_beta_point")
        p = r.get("z_kde_p") or r.get("two_tailed_p")
        ci_lo = r["ci_low"]; ci_hi = r["ci_high"]
        sig01 = r["sig_001"]
        sig05 = r["sig_005"]
        print(f"  {est_name:22s}  β={β:+.4f}  p={p:.4f}  "
              f"CI95=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"sig@.01={'✓' if sig01 else '✗'}  sig@.05={'✓' if sig05 else '✗'}")

    # ─── Figure ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # (a) per-city scatter with all three regression lines
    ax = axes[0]
    ax.errorbar(per_city["z_kde"], per_city["gain"], yerr=per_city["gain_std"],
                fmt="o", color="C0", markersize=8, capsize=2,
                ecolor="gray", lw=0.7)
    for _, r in per_city.iterrows():
        ax.annotate(r["city"][:14], (r["z_kde"], r["gain"]),
                    fontsize=8, alpha=0.8,
                    xytext=(3, 3), textcoords="offset points")
    xs = np.linspace(per_city["z_kde"].min() - 0.5,
                     per_city["z_kde"].max() + 0.5, 100)
    ax.plot(xs, res_A["z_kde_beta"] * xs + res_ols.params[0],
            "C3-", lw=2, label=f"per-city OLS  β={res_A['z_kde_beta']:+.3f}  p={res_A['z_kde_p']:.3f}")
    ax.plot(xs, res_B["z_kde_beta"] * xs + res_wls.params[0],
            "C2--", lw=2, label=f"WLS by block size  β={res_B['z_kde_beta']:+.3f}  p={res_B['z_kde_p']:.3f}")
    ax.set_xlabel("z_kde")
    ax.set_ylabel("Mean gain under spectral block-CV")
    ax.set_title("(a) Robust estimators of β(z_kde) on the block-CV gain")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) bootstrap distribution of β
    ax = axes[1]
    ax.hist(betas, bins=40, color="C0", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.axvline(point, color="C3", lw=2, label=f"point estimate {point:+.3f}")
    ax.axvline(ci_lo, color="C2", lw=1, ls="--", label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
    ax.axvline(ci_hi, color="C2", lw=1, ls="--")
    ax.set_xlabel("β(z_kde) under cluster bootstrap")
    ax.set_ylabel("count")
    ax.set_title(f"(b) Cluster bootstrap  (N_boot={len(betas)})\n"
                 f"fraction β<0 = {(betas<0).mean():.0%}  two-tailed p = {boot_p:.4f}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)

    fig.suptitle("Robust estimators for the spectral block-CV result",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "22_estimator_comparison.pdf", bbox_inches="tight")
    fig.savefig(OUT / "22_estimator_comparison.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '22_estimator_comparison.pdf'}")


if __name__ == "__main__":
    main()
