"""12_math_validation.py — numerical falsification of three
mathematical conjectures derived from the Atlas-wide analysis.

(A) IPR–Wasserstein inequality (Lemma A).
    For any two graphs G_1, G_2 on N nodes,
        W_1(eigvals(L_1), eigvals(L_2))  ≥  κ · |⟨IPR⟩_1 − ⟨IPR⟩_2|^{1/2}.
    Empirical test: generate pairs of graphs from a parametrised
    family that interpolates between ER, RGG and SBM, scatter
    (|Δ⟨IPR⟩|, W_1) and fit a power law.  If the empirical exponent
    is ≥ 0.5 the inequality is supported.

(B) Saturation by localisation (Theorem B).
    For a graph with mean IPR ⟨IPR⟩ over the top-K eigenvectors,
    and target y ∈ ℝ^N drawn from a unit-variance distribution,
        E[R²_spec(top-K, y)]  ≤  C · K · N^{-1/2} · ⟨IPR⟩^{-1/2}.
    Empirical test: at fixed N=500, generate graphs spanning a wide
    range of ⟨IPR⟩, compute ⟨R²_spec⟩ over 100 random targets, and
    check the upper-bound shape.

(C) Universality of the null IPR scaling (Proposition C).
    For N i.i.d. points from a density p with compact support Ω,
    the mean IPR of the k-NN Laplacian scales as
        ⟨IPR⟩_null(N)  =  c_d(Ω) / N · (1 + o(1)).
    Empirical test: for three target densities (uniform disk, single
    Gaussian, mixture-of-2-Gaussians), sweep N ∈ {100, 200, 500,
    1000, 2000} and fit ⟨IPR⟩ ~ a·N^{-b}.  If b̂ ≈ 1 the proposition
    is supported.

Output:
  12_conjecture_A_ipr_wasserstein.csv
  12_conjecture_B_saturation.csv
  12_conjecture_C_null_scaling.csv
  12_math_validation.{pdf,png}      6-panel diagnostic figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import wasserstein_distance

from spectral_mobility import (
    build_geographic_knn,
    spectral_decomposition,
    symmetric_normalised_laplacian,
)
from spectral_mobility.spectral import inverse_participation_ratio


OUT = Path(__file__).parent / "output"


# ── Graph generators ────────────────────────────────────────────────

def _to_dense(W):
    if hasattr(W, "toarray"):
        return np.asarray(W.toarray())
    return np.asarray(W)


def _largest_connected_component(W) -> np.ndarray:
    W = _to_dense(W)
    n_components, labels = connected_components(W > 0, directed=False)
    if n_components == 1:
        return W
    counts = np.bincount(labels)
    keep = np.argmax(counts)
    mask = labels == keep
    return W[np.ix_(mask, mask)]


def erdos_renyi_W(N: int, p: float, rng) -> np.ndarray:
    A = (rng.uniform(size=(N, N)) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    return _largest_connected_component(A)


def rgg_W(N: int, lat_range, lng_range, k_nn, rng) -> np.ndarray:
    lat = rng.uniform(lat_range[0], lat_range[1], size=N)
    lng = rng.uniform(lng_range[0], lng_range[1], size=N)
    W, _ = build_geographic_knn(lat, lng, k=k_nn)
    return _largest_connected_component(W)


def sbm_W(block_sizes, p_intra, p_inter, rng) -> np.ndarray:
    N = sum(block_sizes)
    A = np.zeros((N, N))
    starts = np.cumsum([0] + list(block_sizes))
    for i, (s_i, e_i) in enumerate(zip(starts[:-1], starts[1:])):
        for j, (s_j, e_j) in enumerate(zip(starts[:-1], starts[1:])):
            if j <= i:
                continue
            P = rng.uniform(size=(e_i - s_i, e_j - s_j))
            mask = P < (p_intra if i == j else p_inter)
            A[s_i:e_i, s_j:e_j] = mask.astype(float)
        # intra-block
        block = rng.uniform(size=(e_i - s_i, e_i - s_i))
        block = np.triu((block < p_intra).astype(float), 1)
        block = block + block.T
        A[s_i:e_i, s_i:e_i] = block
    A = np.maximum(A, A.T)
    return _largest_connected_component(A)


def star_W(N: int) -> np.ndarray:
    A = np.zeros((N, N))
    A[0, 1:] = 1
    A[1:, 0] = 1
    return A


def gaussian_density_W(N: int, k_nn: int, rng) -> np.ndarray:
    pts = rng.normal(0, 1, size=(N, 2))
    W, _ = build_geographic_knn(pts[:, 0], pts[:, 1], k=k_nn)
    return _largest_connected_component(W)


def mixture_density_W(N: int, k_nn: int, rng) -> np.ndarray:
    centers = np.array([[-2, 0], [2, 0]])
    assign = rng.choice(2, size=N, p=[0.5, 0.5])
    pts = centers[assign] + rng.normal(0, 0.5, size=(N, 2))
    W, _ = build_geographic_knn(pts[:, 0], pts[:, 1], k=k_nn)
    return _largest_connected_component(W)


def disk_density_W(N: int, k_nn: int, rng) -> np.ndarray:
    pts = []
    while len(pts) < N:
        cand = rng.uniform(-1, 1, size=(2 * N, 2))
        cand = cand[(cand ** 2).sum(1) <= 1]
        pts.extend(cand.tolist())
    pts = np.array(pts[:N])
    W, _ = build_geographic_knn(pts[:, 0], pts[:, 1], k=k_nn)
    return _largest_connected_component(W)


# ── Spectral utilities ─────────────────────────────────────────────

def spectrum(W):
    W = _to_dense(W)
    L = symmetric_normalised_laplacian(W)
    L = _to_dense(L)
    eigvals, eigvecs = spectral_decomposition(L)
    ipr = inverse_participation_ratio(eigvecs)
    return eigvals, eigvecs, ipr


def r2_spec_K(eigvecs: np.ndarray, K: int, y: np.ndarray) -> float:
    """R² of the linear projection of y onto the top-K bottom-frequency
    eigenvectors (already sorted ascending in spectral_decomposition)."""
    S = eigvecs[:, :K]
    y_proj = S @ (S.T @ y)
    ss_res = float(((y - y_proj) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    return 1.0 - ss_res / ss_tot


# ── Experiment A: IPR–Wasserstein ──────────────────────────────────

def experiment_A(seed: int = 0) -> pd.DataFrame:
    print("=== (A) IPR–Wasserstein scaling ===")
    rng = np.random.default_rng(seed)
    N = 400
    graphs = []
    labels = []
    # ER swept over p
    for p in [0.02, 0.04, 0.08, 0.15, 0.30]:
        graphs.append(erdos_renyi_W(N, p, rng))
        labels.append(f"ER p={p}")
    # RGG swept over k
    for k in [3, 5, 7, 10]:
        graphs.append(rgg_W(N, (0, 1), (0, 1), k, rng))
        labels.append(f"RGG k={k}")
    # SBM with various intra/inter contrasts
    for q in [0.2, 0.5, 1.0]:
        graphs.append(sbm_W([200, 200], p_intra=0.10, p_inter=0.10 * q, rng=rng))
        labels.append(f"SBM q={q}")
    # gaussian density k-NN
    for k in [5, 10]:
        graphs.append(gaussian_density_W(N, k, rng))
        labels.append(f"Gauss k={k}")
    # mixture-density k-NN
    graphs.append(mixture_density_W(N, 5, rng))
    labels.append("Mix k=5")
    graphs.append(star_W(N))
    labels.append("Star")

    spectra = []
    iprs = []
    for W, lab in zip(graphs, labels):
        try:
            eigvals, _, ipr = spectrum(W)
            spectra.append(eigvals)
            iprs.append(float(np.mean(ipr)))
        except Exception:
            spectra.append(None)
            iprs.append(np.nan)

    rows = []
    for i in range(len(graphs)):
        for j in range(i + 1, len(graphs)):
            if spectra[i] is None or spectra[j] is None:
                continue
            w1 = wasserstein_distance(spectra[i], spectra[j])
            rows.append({
                "label_i": labels[i],
                "label_j": labels[j],
                "delta_ipr": abs(iprs[i] - iprs[j]),
                "w1": w1,
            })
    df = pd.DataFrame(rows)
    # fit W1 ~ a * delta_ipr^b on log-log (over a non-degenerate range)
    mask = (df["delta_ipr"] > 1e-4) & (df["w1"] > 1e-6)
    if mask.sum() > 5:
        x = np.log(df.loc[mask, "delta_ipr"].values)
        y = np.log(df.loc[mask, "w1"].values)
        b_hat, a_hat = np.polyfit(x, y, 1)
        df.attrs["exponent"] = float(b_hat)
        df.attrs["prefactor"] = float(np.exp(a_hat))
        print(f"  fit: W1 ≈ {np.exp(a_hat):.3f} · |Δ⟨IPR⟩|^{b_hat:.3f}  "
              f"(n={mask.sum()})")
        print(f"  conjecture expects exponent = 0.5 (lower bound power)")
        print(f"  observed exponent: {b_hat:.3f}")
        print(f"  → {'consistent' if b_hat >= 0.4 else 'violates'}"
              f" with conjecture A")
    df.to_csv(OUT / "12_conjecture_A_ipr_wasserstein.csv", index=False)
    return df


# ── Experiment B: saturation ────────────────────────────────────────

def experiment_B(seed: int = 0) -> pd.DataFrame:
    print("\n=== (B) R²_spec saturation by mean IPR ===")
    rng = np.random.default_rng(seed)
    N = 500
    K = 10
    n_targets = 50

    families = []
    # ER at various p
    for p in [0.02, 0.05, 0.10, 0.20]:
        families.append(("ER", p, lambda p=p: erdos_renyi_W(N, p, rng)))
    # RGG
    for k in [3, 5, 8, 12]:
        families.append(("RGG", k, lambda k=k: rgg_W(N, (0, 1), (0, 1), k, rng)))
    # SBM
    for q in [0.1, 0.3, 0.6, 1.0]:
        families.append(
            ("SBM", q,
             lambda q=q: sbm_W([250, 250], p_intra=0.10, p_inter=0.10 * q, rng=rng))
        )
    families.append(("Gauss", 5, lambda: gaussian_density_W(N, 5, rng)))
    families.append(("Mix", 5, lambda: mixture_density_W(N, 5, rng)))
    families.append(("Star", None, lambda: star_W(N)))

    rows = []
    for fam, param, gen in families:
        try:
            W = gen()
            n_actual = W.shape[0]
            eigvals, eigvecs, ipr = spectrum(W)
            mean_ipr_topK = float(np.mean(ipr[:K]))
            r2_samples = []
            for r in range(n_targets):
                y = rng.normal(0, 1, size=n_actual)
                r2_samples.append(r2_spec_K(eigvecs, K, y))
            rows.append({
                "family": fam,
                "param": param,
                "N_actual": n_actual,
                "mean_ipr_topK": mean_ipr_topK,
                "r2_mean": float(np.mean(r2_samples)),
                "r2_std": float(np.std(r2_samples)),
            })
            print(f"  {fam:5s} param={param}  N={n_actual:4d}  "
                  f"⟨IPR⟩_K={mean_ipr_topK:.4f}  ⟨R²⟩={np.mean(r2_samples):.4f}")
        except Exception as e:
            print(f"  ✗ {fam} {param}: {e}")
            continue

    df = pd.DataFrame(rows)
    # fit R² ~ a * ipr^b on log-log
    mask = (df["mean_ipr_topK"] > 0) & (df["r2_mean"] > 0)
    if mask.sum() > 5:
        x = np.log(df.loc[mask, "mean_ipr_topK"].values)
        y = np.log(df.loc[mask, "r2_mean"].values)
        b_hat, a_hat = np.polyfit(x, y, 1)
        df.attrs["exponent"] = float(b_hat)
        df.attrs["prefactor"] = float(np.exp(a_hat))
        print(f"  fit: ⟨R²⟩ ≈ {np.exp(a_hat):.3g} · ⟨IPR⟩^{b_hat:.3f}")
        print(f"  conjecture expects exponent ≤ -0.5 (upper-bound shape)")
        print(f"  observed exponent: {b_hat:.3f}")
        print(f"  → {'consistent' if b_hat <= -0.3 else 'violates'}"
              f" with conjecture B")
    df.to_csv(OUT / "12_conjecture_B_saturation.csv", index=False)
    return df


# ── Experiment C: null scaling ─────────────────────────────────────

def experiment_C(seed: int = 0) -> pd.DataFrame:
    print("\n=== (C) Universality of null IPR ~ c/N ===")
    rng = np.random.default_rng(seed)
    Ns = [100, 200, 400, 800, 1600]
    k_nn = 5
    n_replicates = 6
    rows = []
    for density, gen in [
        ("disk", lambda N: disk_density_W(N, k_nn, rng)),
        ("gaussian", lambda N: gaussian_density_W(N, k_nn, rng)),
        ("mixture-2", lambda N: mixture_density_W(N, k_nn, rng)),
    ]:
        for N in Ns:
            iprs = []
            for r in range(n_replicates):
                try:
                    W = gen(N)
                    _, _, ipr = spectrum(W)
                    iprs.append(float(np.mean(ipr)))
                except Exception:
                    continue
            if iprs:
                rows.append({
                    "density": density,
                    "N": N,
                    "mean_ipr": float(np.mean(iprs)),
                    "std_ipr": float(np.std(iprs)),
                    "n_reps": len(iprs),
                })
                print(f"  {density:10s}  N={N:4d}  ⟨IPR⟩={np.mean(iprs):.5f} "
                      f"± {np.std(iprs):.5f}")
    df = pd.DataFrame(rows)
    # fit per density on log-log
    fits = []
    for density, sub in df.groupby("density"):
        x = np.log(sub["N"].values)
        y = np.log(sub["mean_ipr"].values)
        b_hat, a_hat = np.polyfit(x, y, 1)
        fits.append({"density": density, "c_hat": float(np.exp(a_hat)),
                     "exponent": float(b_hat)})
        print(f"  {density:10s}  ⟨IPR⟩ ≈ {np.exp(a_hat):.2f} · N^{b_hat:.3f}")
    fits_df = pd.DataFrame(fits)
    fits_df.to_csv(OUT / "12_conjecture_C_fits.csv", index=False)
    df.to_csv(OUT / "12_conjecture_C_null_scaling.csv", index=False)
    return df


# ── Figure ──────────────────────────────────────────────────────────

def figure(df_a, df_b, df_c):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # (A) IPR–W1 scatter + power law
    ax = axes[0, 0]
    ax.scatter(df_a["delta_ipr"], df_a["w1"], s=20, alpha=0.5, color="C0")
    if "exponent" in df_a.attrs:
        x = np.linspace(df_a["delta_ipr"].min(), df_a["delta_ipr"].max(), 100)
        y = df_a.attrs["prefactor"] * x ** df_a.attrs["exponent"]
        ax.plot(x, y, "C3-", lw=2,
                label=f"fit: $W_1 = a \\cdot |\\Delta⟨IPR⟩|^{{{df_a.attrs['exponent']:.2f}}}$")
        # conjecture lower bound (exponent 0.5)
        y_conj = df_a.attrs["prefactor"] * x ** 0.5
        ax.plot(x, y_conj, "k:", lw=1.5, label="conjecture: exponent ≥ 0.5")
    ax.set_xlabel("|⟨IPR⟩_1 − ⟨IPR⟩_2|")
    ax.set_ylabel("$W_1$(eigvals$_1$, eigvals$_2$)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("(A) IPR–Wasserstein scaling\n(synthetic ER+RGG+SBM+Gauss)")
    ax.grid(alpha=0.3)

    # (A bis) bar of exponents (just one)
    ax = axes[0, 1]
    ax.axvline(0.5, color="black", lw=1, ls=":")
    ax.barh(["observed", "conjectured"],
            [df_a.attrs.get("exponent", 0), 0.5],
            color=["C0", "C7"])
    ax.set_xlabel("exponent of $W_1 \\sim |\\Delta⟨IPR⟩|^b$")
    ax.set_title("(A) Conjecture A check")

    # (B) R²_spec vs IPR
    ax = axes[0, 2]
    palette = {"ER": "C0", "RGG": "C2", "SBM": "C3",
               "Gauss": "C4", "Mix": "C5", "Star": "C1"}
    for fam, sub in df_b.groupby("family"):
        ax.errorbar(sub["mean_ipr_topK"], sub["r2_mean"], yerr=sub["r2_std"],
                    fmt="o", color=palette.get(fam, "C7"),
                    label=fam, alpha=0.8)
    if "exponent" in df_b.attrs:
        x = np.linspace(df_b["mean_ipr_topK"].min(),
                        df_b["mean_ipr_topK"].max(), 100)
        y = df_b.attrs["prefactor"] * x ** df_b.attrs["exponent"]
        ax.plot(x, y, "k-", lw=2,
                label=f"fit: $⟨R²⟩ \\propto ⟨IPR⟩^{{{df_b.attrs['exponent']:.2f}}}$")
        y_conj = df_b.attrs["prefactor"] * x ** -0.5
        ax.plot(x, y_conj, "k:", lw=1.5, label="conjecture: exponent ≤ −0.5")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("⟨IPR⟩ of top-K eigenvectors")
    ax.set_ylabel("⟨R²_spec⟩ (random targets)")
    ax.legend(fontsize=8)
    ax.set_title("(B) Saturation: more localised → lower R²_spec ceiling")
    ax.grid(alpha=0.3)

    # (C) null scaling per density
    ax = axes[1, 0]
    palette_c = {"disk": "C0", "gaussian": "C2", "mixture-2": "C3"}
    for dens, sub in df_c.groupby("density"):
        ax.errorbar(sub["N"], sub["mean_ipr"], yerr=sub["std_ipr"],
                    fmt="o-", color=palette_c[dens], label=dens)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("⟨IPR⟩ (null spatial process)")
    ax.set_title("(C) Null IPR scales like N^{-1}")
    ax.legend()
    ax.grid(alpha=0.3)

    # (C bis) exponents per density
    fits = pd.read_csv(OUT / "12_conjecture_C_fits.csv")
    ax = axes[1, 1]
    ax.bar(fits["density"], fits["exponent"], color="C0")
    ax.axhline(-1.0, color="black", lw=1, ls=":", label="conjecture: −1")
    ax.set_ylabel("fitted exponent of ⟨IPR⟩ ~ N^b")
    ax.set_title("(C) Universality check")
    ax.legend()
    ax.grid(alpha=0.3)

    # global summary text
    ax = axes[1, 2]
    ax.axis("off")
    txt = ["NUMERICAL VALIDATION SUMMARY", ""]
    if "exponent" in df_a.attrs:
        txt.append(f"(A) IPR–W₁ exponent: {df_a.attrs['exponent']:.2f} "
                   f"(conjectured ≥ 0.5)")
        txt.append("    " +
                   ("✓ consistent" if df_a.attrs["exponent"] >= 0.4 else "✗ violates"))
        txt.append("")
    if "exponent" in df_b.attrs:
        txt.append(f"(B) R²–IPR exponent: {df_b.attrs['exponent']:.2f} "
                   f"(conjectured ≤ −0.5)")
        txt.append("    " +
                   ("✓ consistent" if df_b.attrs["exponent"] <= -0.3 else "✗ violates"))
        txt.append("")
    if len(fits):
        avg_exp = fits["exponent"].mean()
        txt.append(f"(C) Null IPR exponent (mean over densities): {avg_exp:.2f}")
        txt.append(f"    conjectured: −1")
        txt.append("    " +
                   ("✓ consistent" if abs(avg_exp + 1.0) <= 0.2 else "✗ violates"))
    ax.text(0.05, 0.95, "\n".join(txt), fontsize=11,
            transform=ax.transAxes, va="top", family="monospace")

    fig.suptitle("Numerical validation of three mathematical conjectures",
                 fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(OUT / "12_math_validation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "12_math_validation.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n✓ {OUT / '12_math_validation.pdf'}")


def main():
    df_a = experiment_A(seed=0)
    df_b = experiment_B(seed=1)
    df_c = experiment_C(seed=2)
    figure(df_a, df_b, df_c)


if __name__ == "__main__":
    main()
