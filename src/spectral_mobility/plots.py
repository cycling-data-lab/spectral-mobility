"""Matplotlib helpers for visualising spectral diagnostics.

These functions are intentionally simple — they return a matplotlib
``Axes`` so the caller can compose them into multi-panel figures or
attach annotations.  They are *optional*: importing this module
requires ``matplotlib`` to be installed (``pip install
spectral-mobility[plotting]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "spectral_mobility.plots requires matplotlib. "
        "Install with `pip install spectral-mobility[plotting]`."
    ) from e

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def plot_ceiling_curve(
    K_grid: Sequence[int],
    r2_curve: Sequence[float],
    *,
    r2_baseline: float | None = None,
    ax: "Axes | None" = None,
) -> "Axes":
    """Plot R²_spec as a function of K (number of augmenting eigenvectors).

    Parameters
    ----------
    K_grid : sequence of int
        K values evaluated.
    r2_curve : sequence of float
        ``R²_spec`` at each K.
    r2_baseline : float, optional
        Reference R²_IMD-only line (drawn as a horizontal red dashed line).
    ax : matplotlib.axes.Axes, optional
        Axis to draw on.  If None, a new figure is created.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    K_arr = np.asarray(K_grid)
    r2_arr = np.asarray(r2_curve)
    ax.plot(K_arr, r2_arr, "-o", color="C0", lw=1.6, markersize=5,
            label="R²_spec(augmented)")
    if r2_baseline is not None:
        ax.axhline(r2_baseline, color="red", ls="--", lw=1.0,
                   label=f"R²_IMD only = {r2_baseline:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("K (low-frequency eigenvectors augmented)")
    ax.set_ylabel(r"$R^2_\mathrm{spec}$")
    ax.set_title("Predictability ceiling vs. augmentation")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_bottleneck_map(
    coords: np.ndarray,
    eigvec: np.ndarray,
    *,
    ax: "Axes | None" = None,
    log_scale: bool = True,
    title: str | None = None,
) -> "Axes":
    """Scatter map of nodes coloured by an eigenvector's |ψ|² mass.

    Nodes carrying high mass form the geographic footprint of a
    localized eigenmode — i.e.\\ a structural bottleneck of the graph.

    Parameters
    ----------
    coords : (N, 2) array
        Node coordinates.  For geographic graphs, columns are [lat, lng];
        the longitude is plotted on x.
    eigvec : (N,) array
        A single eigenvector (column from
        :func:`spectral_decomposition`).
    ax : matplotlib.axes.Axes, optional
    log_scale : bool, default True
        Use logarithmic colour scale for |ψ|² (useful when mass is
        concentrated on a few nodes).
    title : str, optional
        Axis title.  If None, computes from eigenvec IPR.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    coords = np.asarray(coords)
    psi = np.asarray(eigvec).ravel()
    if coords.shape[0] != psi.shape[0]:
        raise ValueError("coords and eigvec must have same first dimension")
    mass = psi ** 2
    mass = mass / (mass.sum() + 1e-30)
    order = np.argsort(mass)  # paint largest mass on top

    if coords.shape[1] == 2:
        x = coords[order, 1]
        y = coords[order, 0]
        xlabel, ylabel = "longitude", "latitude"
    else:
        # Generic 2-D feature embedding
        x = coords[order, 0]
        y = coords[order, 1]
        xlabel, ylabel = "x", "y"

    if log_scale and (mass > 0).any():
        vmin = max(mass[mass > 0].min(), 1e-10)
        norm = LogNorm(vmin=vmin, vmax=mass.max())
    else:
        norm = None

    sc = ax.scatter(x, y, c=mass[order], cmap="inferno", s=20,
                    norm=norm, edgecolors="none")
    plt.colorbar(sc, ax=ax, label=r"$|\psi|^2$ (normalised mass)", shrink=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="datalim")

    if title is None:
        ipr = float((mass ** 2 * eigvec.size).sum() / (mass.sum() + 1e-30) ** 2)
        # Re-compute IPR exactly on the original eigvec
        psi_normed = psi ** 2 / (psi @ psi + 1e-30)
        ipr_exact = float((psi_normed ** 2).sum())
        title = f"Bottleneck footprint (IPR = {ipr_exact:.4f})"
    ax.set_title(title)
    return ax


def plot_spectrum(
    eigvals: np.ndarray,
    ipr: np.ndarray,
    *,
    ax: "Axes | None" = None,
    extended_threshold: float | None = None,
) -> "Axes":
    """Scatter plot of IPR versus eigenvalue (the canonical Anderson-style
    phenomenology figure).

    Parameters
    ----------
    eigvals : (N,) array
    ipr : (N,) array
    ax : matplotlib.axes.Axes, optional
    extended_threshold : float, optional
        Horizontal threshold line drawn in red (typically 5/N).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    eigvals = np.asarray(eigvals).ravel()
    ipr = np.asarray(ipr).ravel()
    if eigvals.shape != ipr.shape:
        raise ValueError("eigvals and ipr must have the same shape")
    ax.semilogy(eigvals, ipr, ".", markersize=4, alpha=0.7, color="C0")
    N = eigvals.size
    ax.axhline(1.0 / N, color="green", ls="--", lw=0.9,
               label=f"1/N = {1/N:.4f}  (fully extended)")
    if extended_threshold is not None:
        ax.axhline(extended_threshold, color="red", ls=":", lw=0.9,
                   label=f"localized above {extended_threshold:.4f}")
    ax.set_xlabel(r"$\lambda$  (Laplacian eigenvalue)")
    ax.set_ylabel("IPR")
    ax.set_title("Spectrum: localization vs. eigenvalue")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)
    return ax


def plot_city_comparison(
    profile_a,
    profile_b,
    *,
    figsize: tuple[float, float] = (14, 5),
):
    """Side-by-side comparison of two CitySpectralProfile objects.

    Draws an overlay of the eigenvalue distributions and IPR
    distributions, plus a small numerical comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # (a) eigenvalue density (KDE-like via histogram)
    ax = axes[0]
    ax.hist(profile_a.eigvals, bins=50, alpha=0.55, density=True,
            label=profile_a.name, color="C0")
    ax.hist(profile_b.eigvals, bins=50, alpha=0.55, density=True,
            label=profile_b.name, color="C3")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("density")
    ax.set_title("Eigenvalue distributions")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # (b) IPR distribution (log)
    ax = axes[1]
    ax.hist(np.log10(profile_a.ipr + 1e-12), bins=50, alpha=0.55,
            density=True, label=profile_a.name, color="C0")
    ax.hist(np.log10(profile_b.ipr + 1e-12), bins=50, alpha=0.55,
            density=True, label=profile_b.name, color="C3")
    ax.set_xlabel(r"$\log_{10}\,\mathrm{IPR}$")
    ax.set_ylabel("density")
    ax.set_title("IPR distributions (log)")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    # (c) comparison table
    from spectral_mobility.compare import compare_cities

    cmp = compare_cities(profile_a, profile_b)
    sa = profile_a.summary()
    sb = profile_b.summary()
    rows = [
        ("N", sa["N"], sb["N"]),
        ("σ", f"{sa['sigma']:.0f}", f"{sb['sigma']:.0f}"),
        ("⟨IPR⟩", f"{sa['mean_ipr']:.3f}", f"{sb['mean_ipr']:.3f}"),
        ("med IPR", f"{sa['median_ipr']:.3f}", f"{sb['median_ipr']:.3f}"),
        ("ext. frac.", f"{sa['extended_fraction']:.2f}",
         f"{sb['extended_fraction']:.2f}"),
        ("⟨r⟩", f"{sa['mean_level_spacing_r']:.3f}",
         f"{sb['mean_level_spacing_r']:.3f}"),
    ]
    if "R2_imd" in sa and "R2_imd" in sb:
        rows.append(("R²_IMD", f"{sa['R2_imd']:.3f}", f"{sb['R2_imd']:.3f}"))
        rows.append(
            ("R²_aug K16", f"{sa['R2_augmented_K16']:.3f}",
             f"{sb['R2_augmented_K16']:.3f}")
        )

    text_lines = [
        f"{profile_a.name}   ↔   {profile_b.name}",
        "─" * 50,
        f"{'metric':<14s}  {profile_a.name[:12]:>13s}  {profile_b.name[:12]:>13s}",
    ]
    for r in rows:
        text_lines.append(f"{r[0]:<14s}  {str(r[1]):>13s}  {str(r[2]):>13s}")
    text_lines += [
        "─" * 50,
        f"Wasserstein (eigvals)  : {cmp.wasserstein_eigvals:.4f}",
        f"Wasserstein (log IPR)  : {cmp.wasserstein_ipr:.4f}",
        f"KS (eigvals)           : {cmp.ks_eigvals:.4f}",
        f"KS (log IPR)           : {cmp.ks_ipr:.4f}",
        "─" * 50,
        f"Spectral similarity    : {cmp.spectral_similarity:.3f}",
    ]
    axes[2].text(0.01, 0.99, "\n".join(text_lines),
                 ha="left", va="top", family="monospace", fontsize=9,
                 transform=axes[2].transAxes)
    axes[2].set_axis_off()
    fig.suptitle(f"Spectral comparison: {profile_a.name} vs {profile_b.name}",
                 fontsize=11)
    return fig


def plot_similarity_matrix(
    matrix: np.ndarray,
    names: list[str],
    *,
    ax: "Axes | None" = None,
    cmap: str = "viridis",
    annotate: bool = True,
):
    """Heatmap of a pairwise spectral-similarity matrix.

    Parameters
    ----------
    matrix : (n, n) np.ndarray
        Symmetric similarity matrix in ``[0, 1]``.
    names : list of str
        City names; length ``n``.
    ax : matplotlib.axes.Axes, optional
    cmap : str, default "viridis"
    annotate : bool, default True
        Print similarity values in each cell.
    """
    matrix = np.asarray(matrix)
    n = matrix.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(0.7 * n + 2, 0.7 * n + 2),
                              constrained_layout=True)
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    plt.colorbar(im, ax=ax, label="spectral similarity", shrink=0.85)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] < 0.5 else "black")
    ax.set_title("Pairwise spectral similarity")
    return ax


def plot_cv_comparison(
    cv_result: dict,
    *,
    ax: "Axes | None" = None,
) -> "Axes":
    """Bar plot comparing baseline vs augmented per fold.

    Parameters
    ----------
    cv_result : dict
        Output of :meth:`SpectralAugmentedRegressor.cross_validate`.
        Must contain ``baseline_scores`` and ``augmented_scores`` keys.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    baseline = np.asarray(cv_result["baseline_scores"])
    augmented = np.asarray(cv_result["augmented_scores"])
    n = len(baseline)
    ind = np.arange(n)
    width = 0.4
    ax.bar(ind - width / 2, baseline, width=width,
           color="grey", alpha=0.85, label="baseline")
    ax.bar(ind + width / 2, augmented, width=width,
           color="C2", alpha=0.85, label="augmented")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(cv_result["baseline_mean"], color="grey", ls=":", lw=0.7)
    ax.axhline(cv_result["augmented_mean"], color="C2", ls=":", lw=0.7)
    ax.set_xticks(ind)
    ax.set_xticklabels([f"fold {i+1}" for i in range(n)])
    ax.set_ylabel(r"$R^2$ (out-of-sample)")
    gain = cv_result.get("mean_gain", augmented.mean() - baseline.mean())
    ax.set_title(
        f"Cross-validated comparison\n"
        f"baseline = {baseline.mean():+.3f}   "
        f"augmented = {augmented.mean():+.3f}   "
        f"gain = {gain:+.3f}"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)
    return ax
