"""13_saturation_smooth_y.py — re-test conjecture B with structured
targets y instead of i.i.d. Gaussian.

The original conjecture B (R²_spec ≤ C · ⟨IPR⟩^{-0.5}) was falsified
for random y because of dimension counting: any orthonormal basis of
dim K captures ~K/N of the variance of random y, regardless of basis
localisation. The localisation only bites when y has *graph-aware
structure*.

We try three flavours of structured y:

  (i)   spatial smoothness: for graphs with spatial coords (RGG, Gauss,
        Mix), y = sin(πx) · cos(πy). Captures macroscopic geometry.
  (ii)  heat-diffused smoothness: y = exp(-t · L_norm) · x_0 where x_0
        is Gaussian, t > 0. This is *Laplacian-aware* smoothness: y
        is forced to live in the low-eigenvalue part of L. Defined
        for any graph.
  (iii) low-band signal: y = Σ_{n=1}^{N_band} φ_n · g_n where g_n ~ N(0,1)
        and N_band ≪ N. This y is *exactly* in the bottom-N_band
        subspace.

For each graph + each y flavour, we measure R²_spec(top-K=10, y) and
correlate with ⟨IPR⟩ of the top-K eigenvectors. Saturation predicts
that localised graphs (high IPR) cannot represent smooth y.

Output:
  13_saturation_smooth.csv
  13_saturation_smooth.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from spectral_mobility import (
    build_geographic_knn,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)
from spectral_mobility.spectral import inverse_participation_ratio


OUT = Path(__file__).parent / "output"


def _to_dense(W):
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray())
    return np.asarray(W)


def _largest_cc_with_idx(W, coords=None):
    W = _to_dense(W)
    n_components, labels = connected_components(W > 0, directed=False)
    if n_components == 1:
        return W, coords
    counts = np.bincount(labels)
    keep = np.argmax(counts)
    mask = labels == keep
    W = W[np.ix_(mask, mask)]
    if coords is not None:
        coords = coords[mask]
    return W, coords


def spectrum(W):
    W = _to_dense(W)
    L = _to_dense(symmetric_normalised_laplacian(W))
    eigvals, eigvecs = spectral_decomposition(L)
    ipr = inverse_participation_ratio(eigvecs)
    return eigvals, eigvecs, ipr


def r2_spec_K(eigvecs, K, y):
    S = eigvecs[:, :K]
    y_proj = S @ (S.T @ y)
    ss_res = float(((y - y_proj) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    return max(0.0, 1.0 - ss_res / ss_tot)


# ── Graph generators with coordinates when available ───────────────

def rgg(N, k_nn, rng, lat_lo=0, lat_hi=1, lng_lo=0, lng_hi=1):
    lat = rng.uniform(lat_lo, lat_hi, size=N)
    lng = rng.uniform(lng_lo, lng_hi, size=N)
    coords = np.column_stack([lat, lng])
    W, _ = build_geographic_knn(lat, lng, k=k_nn)
    return _largest_cc_with_idx(W, coords)


def gaussian_graph(N, k_nn, rng):
    coords = rng.normal(0, 1, size=(N, 2))
    W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=k_nn)
    return _largest_cc_with_idx(W, coords)


def mixture_graph(N, k_nn, rng):
    centers = np.array([[-2, 0], [2, 0]])
    assign = rng.choice(2, size=N, p=[0.5, 0.5])
    coords = centers[assign] + rng.normal(0, 0.5, size=(N, 2))
    W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=k_nn)
    return _largest_cc_with_idx(W, coords)


def matern_cluster(N, k_nn, rng, n_centers=6, sigma=0.05):
    centers = rng.uniform(0, 1, size=(n_centers, 2))
    assign = rng.integers(0, n_centers, size=N)
    coords = centers[assign] + rng.normal(0, sigma, size=(N, 2))
    W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=k_nn)
    return _largest_cc_with_idx(W, coords)


def hub_spoke(N, n_hubs, k_nn, rng):
    """Hub-and-spoke: stations cluster around very few centers."""
    hubs = rng.uniform(0, 1, size=(n_hubs, 2))
    sizes = rng.dirichlet(np.ones(n_hubs)) * N
    sizes = np.maximum(sizes.astype(int), 5)
    sizes[-1] = N - sizes[:-1].sum()  # ensure total = N
    coords = []
    for h, s in zip(hubs, sizes):
        coords.append(h + rng.normal(0, 0.01, size=(int(s), 2)))
    coords = np.vstack(coords)[:N]
    W, _ = build_geographic_knn(coords[:, 0], coords[:, 1], k=k_nn)
    return _largest_cc_with_idx(W, coords)


# ── Three smooth-y generators ──────────────────────────────────────

def y_spatial(coords, rng):
    """y = sin(πx) cos(πy) standardised. Requires coords."""
    if coords is None:
        return None
    # rescale to [0, 1]² so sin/cos make sense
    c = coords.copy().astype(float)
    for d in range(2):
        lo, hi = c[:, d].min(), c[:, d].max()
        c[:, d] = (c[:, d] - lo) / (hi - lo + 1e-9)
    y = np.sin(np.pi * c[:, 0]) * np.cos(np.pi * c[:, 1])
    # add a small bit of noise so y has full rank
    y = y + 0.05 * rng.standard_normal(len(y))
    return y - y.mean()


def y_heat(W, rng, t=2.0):
    """y = exp(-t · L_norm) · x_0 with x_0 Gaussian."""
    L = _to_dense(symmetric_normalised_laplacian(_to_dense(W)))
    eigvals, eigvecs = spectral_decomposition(L)
    x0 = rng.standard_normal(W.shape[0])
    coefs = eigvecs.T @ x0
    coefs = coefs * np.exp(-t * eigvals)
    y = eigvecs @ coefs
    return y - y.mean()


def y_band(W, rng, n_band=20):
    """y = Σ_{n=1..n_band} c_n φ_n, c_n ~ N(0, 1/n²) (decaying)."""
    L = _to_dense(symmetric_normalised_laplacian(_to_dense(W)))
    eigvals, eigvecs = spectral_decomposition(L)
    n_eff = min(n_band, eigvecs.shape[1])
    weights = 1.0 / np.arange(1, n_eff + 1)
    coefs = weights * rng.standard_normal(n_eff)
    y = eigvecs[:, :n_eff] @ coefs
    return y - y.mean()


# ── Main experiment ───────────────────────────────────────────────

def main(seed=0):
    rng = np.random.default_rng(seed)
    K = 10
    N_target = 500
    n_targets_per_config = 30

    families = []
    # parametric sweep over IPR via different generators
    for k_nn in [3, 5, 8, 12, 20]:
        families.append((f"RGG k={k_nn}", lambda k=k_nn: rgg(N_target, k, rng)))
    for k_nn in [5, 10]:
        families.append((f"Gauss k={k_nn}", lambda k=k_nn: gaussian_graph(N_target, k, rng)))
    families.append(("Mix k=5", lambda: mixture_graph(N_target, 5, rng)))
    families.append(("Matérn 4c", lambda: matern_cluster(N_target, 5, rng, n_centers=4, sigma=0.04)))
    families.append(("Matérn 10c", lambda: matern_cluster(N_target, 5, rng, n_centers=10, sigma=0.06)))
    families.append(("Hub-2", lambda: hub_spoke(N_target, 2, 5, rng)))
    families.append(("Hub-4", lambda: hub_spoke(N_target, 4, 5, rng)))
    families.append(("Hub-8", lambda: hub_spoke(N_target, 8, 5, rng)))

    rows = []
    for fam_name, gen in families:
        try:
            W, coords = gen()
            n_actual = W.shape[0]
            eigvals, eigvecs, ipr = spectrum(W)
            mean_ipr_topK = float(np.mean(ipr[:K]))
            mean_ipr_global = float(np.mean(ipr))

            for label, ygen in [
                ("spatial",
                 lambda: y_spatial(coords, rng) if coords is not None else None),
                ("heat-t2",
                 lambda: y_heat(W, rng, t=2.0)),
                ("heat-t5",
                 lambda: y_heat(W, rng, t=5.0)),
                ("band-20",
                 lambda: y_band(W, rng, n_band=20)),
            ]:
                r2_samples = []
                for r in range(n_targets_per_config):
                    y = ygen()
                    if y is None or np.allclose(y, 0):
                        continue
                    r2_samples.append(r2_spec_K(eigvecs, K, y))
                if not r2_samples:
                    continue
                rows.append({
                    "family": fam_name,
                    "y_type": label,
                    "N_actual": n_actual,
                    "mean_ipr_topK": mean_ipr_topK,
                    "mean_ipr_global": mean_ipr_global,
                    "r2_mean": float(np.mean(r2_samples)),
                    "r2_std": float(np.std(r2_samples)),
                    "n_targets": len(r2_samples),
                })
            top = rows[-4:]  # last 4 (one per y_type)
            r2s = {r["y_type"]: r["r2_mean"] for r in top}
            print(f"  {fam_name:14s}  N={n_actual:4d}  ⟨IPR⟩_K={mean_ipr_topK:.4f}  "
                  f"R²: spat={r2s.get('spatial', float('nan')):.3f}  "
                  f"heat2={r2s.get('heat-t2', float('nan')):.3f}  "
                  f"heat5={r2s.get('heat-t5', float('nan')):.3f}  "
                  f"band={r2s.get('band-20', float('nan')):.3f}")
        except Exception as e:
            print(f"  ✗ {fam_name}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "13_saturation_smooth.csv", index=False)

    # ── Per-y-type fit ──
    print("\n=== Per y-type fit (log–log R² vs ⟨IPR⟩_K) ===")
    fits = []
    for y_type, sub in df.groupby("y_type"):
        sub = sub[sub["r2_mean"] > 1e-3]
        if len(sub) < 5:
            continue
        x = np.log(sub["mean_ipr_topK"].values)
        y = np.log(sub["r2_mean"].values)
        b_hat, a_hat = np.polyfit(x, y, 1)
        # also compute Pearson r between IPR and R² (signed monotonicity)
        corr = np.corrcoef(sub["mean_ipr_topK"], sub["r2_mean"])[0, 1]
        fits.append({
            "y_type": y_type, "n": len(sub),
            "exponent": float(b_hat),
            "prefactor": float(np.exp(a_hat)),
            "pearson_r": float(corr),
        })
        print(f"  {y_type:10s} n={len(sub):2d}  exponent={b_hat:+.2f}  "
              f"corr(IPR, R²)={corr:+.3f}  → "
              f"{'SAT (R² ↓ when IPR ↑)' if corr < -0.3 else 'NO SAT' if abs(corr) < 0.3 else 'POS (R² ↑ when IPR ↑)'}")
    fits_df = pd.DataFrame(fits)
    fits_df.to_csv(OUT / "13_saturation_fits.csv", index=False)

    # ── Figure ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    y_types = ["spatial", "heat-t2", "heat-t5", "band-20"]
    for ax, yt in zip(axes.flat, y_types):
        sub = df[df["y_type"] == yt]
        ax.errorbar(sub["mean_ipr_topK"], sub["r2_mean"], yerr=sub["r2_std"],
                    fmt="o", alpha=0.7, color="C0", capsize=2)
        for _, r in sub.iterrows():
            ax.annotate(r["family"][:6], (r["mean_ipr_topK"], r["r2_mean"]),
                        fontsize=7, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        fit = fits_df[fits_df["y_type"] == yt]
        if len(fit):
            f = fit.iloc[0]
            xs = np.linspace(sub["mean_ipr_topK"].min(),
                             sub["mean_ipr_topK"].max(), 100)
            ax.plot(xs, f["prefactor"] * xs ** f["exponent"], "C3-",
                    label=f"fit: R² ∝ ⟨IPR⟩^{f['exponent']:+.2f}  "
                          f"(corr={f['pearson_r']:+.2f})")
            ax.legend(fontsize=9, loc="best")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("⟨IPR⟩ of top-K eigenvectors")
        ax.set_ylabel("⟨R²_spec⟩")
        ax.set_title(f"y type: {yt}")
        ax.grid(alpha=0.3)
    fig.suptitle("Does R²_spec saturate with localisation? — by y structure",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "13_saturation_smooth.pdf", bbox_inches="tight")
    fig.savefig(OUT / "13_saturation_smooth.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '13_saturation_smooth.pdf'}")


if __name__ == "__main__":
    main()
