"""High-level city-level spectral analysis: :class:`CitySpectralProfile`.

A ``CitySpectralProfile`` is a self-contained, lazily-computed summary
of a single urban mobility network from a spectral-graph point of
view.  It packages the graph, its Laplacian, the full eigenbasis,
participation statistics, and predictability diagnostics behind a
single object that can be saved, transported, compared with other
profiles, and visualised.

Typical workflow::

    paris = CitySpectralProfile.from_coords(
        name="Vélib Paris",
        lat=stations_paris["lat"], lng=stations_paris["lng"],
        features=X_imd_paris,
        target=demand_paris,
    )
    print(paris.summary())
    paris.plot_overview()           # 4-panel figure
    paris.bottleneck_zones(n=5)     # the 5 worst structural bottlenecks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from spectral_mobility.augmentation import select_K
from spectral_mobility.bound import SpectralBoundResult, spectral_bound
from spectral_mobility.diagnostics import (
    bottleneck_modes,
    extended_subspace_fraction,
    locate_bottleneck_nodes,
)
from spectral_mobility.graph import (
    build_feature_knn,
    build_geographic_knn,
    symmetric_normalised_laplacian,
)
from spectral_mobility.spectral import (
    inverse_participation_ratio,
    level_spacing_ratios,
    participation_ratio,
    spectral_decomposition,
)


@dataclass
class CitySpectralProfile:
    """Self-contained spectral profile of a single urban network.

    Construct via :meth:`from_coords` rather than the default
    dataclass constructor.

    Attributes
    ----------
    name : str
    coords : (N, 2) np.ndarray
    features : (N, p) np.ndarray, optional
    target : (N,) np.ndarray, optional
    eigvals, eigvecs : np.ndarray
    ipr, pr : np.ndarray
    sigma : float
    k_nn : int
    graph_type : str
    """

    name: str
    coords: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    ipr: np.ndarray
    pr: np.ndarray
    sigma: float
    k_nn: int
    graph_type: Literal["geographic", "feature"]
    features: np.ndarray | None = None
    target: np.ndarray | None = None
    _summary_cache: dict | None = field(default=None, repr=False, compare=False)

    # ─────────────────────────────────────────────────────────────────
    # Constructor
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def from_coords(
        cls,
        name: str,
        lat: np.ndarray | None = None,
        lng: np.ndarray | None = None,
        coords: np.ndarray | None = None,
        features: np.ndarray | None = None,
        target: np.ndarray | None = None,
        k_nn: int = 6,
        sigma: float | Literal["auto"] = "auto",
        graph_type: Literal["geographic", "feature"] = "geographic",
    ) -> "CitySpectralProfile":
        """Build a profile from station/commune coordinates.

        You can pass either ``(lat, lng)`` as 1-D arrays, or
        ``coords`` directly as a 2-D array of shape ``(N, 2)``.

        Parameters
        ----------
        name : str
            Human-readable city name.
        lat, lng : (N,) array-like, optional
        coords : (N, 2) array-like, optional
        features : (N, p) array-like, optional
            Encoder features for the predictability ceiling.
        target : (N,) array-like, optional
            Target signal (demand, modal share, …).
        k_nn : int, default 6
        sigma : float or "auto", default "auto"
        graph_type : {"geographic", "feature"}, default "geographic"
        """
        if coords is None:
            if lat is None or lng is None:
                raise ValueError("provide either (lat, lng) or coords")
            coords = np.column_stack([np.asarray(lat, dtype=float),
                                       np.asarray(lng, dtype=float)])
        coords = np.asarray(coords, dtype=float)

        if graph_type == "geographic":
            W, sigma_used = build_geographic_knn(
                coords[:, 0], coords[:, 1], k=k_nn, sigma=sigma
            )
        elif graph_type == "feature":
            W, sigma_used = build_feature_knn(coords, k=k_nn, sigma=sigma)
        else:
            raise ValueError(
                f"graph_type must be 'geographic' or 'feature', got {graph_type!r}"
            )
        L = symmetric_normalised_laplacian(W)
        eigvals, eigvecs = spectral_decomposition(L)
        ipr = inverse_participation_ratio(eigvecs)
        pr = participation_ratio(eigvecs)
        return cls(
            name=name,
            coords=coords,
            eigvals=eigvals,
            eigvecs=eigvecs,
            ipr=ipr,
            pr=pr,
            sigma=sigma_used,
            k_nn=k_nn,
            graph_type=graph_type,
            features=np.asarray(features, dtype=float) if features is not None else None,
            target=np.asarray(target, dtype=float).ravel() if target is not None else None,
        )

    # ─────────────────────────────────────────────────────────────────
    # Convenience properties
    # ─────────────────────────────────────────────────────────────────
    @property
    def N(self) -> int:
        return self.coords.shape[0]

    @property
    def mean_ipr(self) -> float:
        return float(self.ipr.mean())

    @property
    def median_ipr(self) -> float:
        return float(np.median(self.ipr))

    @property
    def extended_fraction(self) -> float:
        """Fraction of eigenmodes with IPR < 5/N (extended)."""
        return extended_subspace_fraction(self.eigvecs, ipr_threshold_factor=5.0)

    @property
    def level_spacing_mean(self) -> float:
        """Mean adjacent gap ratio ⟨r⟩.  ≈ 0.5295 (GOE) → ≈ 0.3863 (Poisson)."""
        return float(level_spacing_ratios(self.eigvals).mean())

    # ─────────────────────────────────────────────────────────────────
    # Predictability ceiling
    # ─────────────────────────────────────────────────────────────────
    def predictability_ceiling(self, K: int = 16) -> SpectralBoundResult:
        """Closed-form ``R²_spec`` ceiling with and without augmentation."""
        if self.target is None:
            raise ValueError(
                "predictability_ceiling() requires a target; "
                "construct the profile with target=..."
            )
        return spectral_bound(
            self.eigvecs, self.target,
            encoder_features=self.features, K=K,
        )

    def select_K(
        self,
        method: Literal["elbow", "ratio", "fixed"] = "elbow",
        **kwargs,
    ) -> tuple[int, dict]:
        """Heuristically select K for augmentation."""
        if self.target is None:
            raise ValueError("select_K requires a target")
        return select_K(self.eigvecs, self.target, self.features,
                        method=method, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # Bottleneck diagnostics
    # ─────────────────────────────────────────────────────────────────
    def bottleneck_modes(self, n: int = 5, require_low_frequency: int | None = None):
        """Indices of the ``n`` most-localized eigenmodes."""
        return bottleneck_modes(
            self.eigvecs, n_top=n, require_low_frequency=require_low_frequency
        )

    def bottleneck_zones(
        self,
        n: int = 5,
        *,
        mass_threshold: float = 0.05,
        require_low_frequency: int | None = None,
    ) -> list[dict]:
        """Return the geographic footprint of each of the ``n`` worst
        bottlenecks.

        Each entry of the returned list is a dict with keys:
        ``mode_idx``, ``eigval``, ``ipr``, ``nodes`` (indices),
        ``coords`` (Nx2), ``mass`` (per-node normalised |ψ|²).
        """
        idx = self.bottleneck_modes(n=n, require_low_frequency=require_low_frequency)
        zones = []
        for k in idx:
            psi = self.eigvecs[:, k]
            nodes = locate_bottleneck_nodes(psi, mass_threshold=mass_threshold)
            mass = (psi[nodes] ** 2) / (psi @ psi + 1e-30)
            zones.append(
                {
                    "mode_idx": int(k),
                    "eigval": float(self.eigvals[k]),
                    "ipr": float(self.ipr[k]),
                    "nodes": nodes,
                    "coords": self.coords[nodes],
                    "mass": mass,
                }
            )
        return zones

    # ─────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────
    def summary(self) -> dict:
        """Compact dict-summary of the profile (cached after first call)."""
        if self._summary_cache is not None:
            return self._summary_cache
        out: dict = {
            "name": self.name,
            "N": self.N,
            "sigma": self.sigma,
            "k_nn": self.k_nn,
            "graph_type": self.graph_type,
            "mean_ipr": self.mean_ipr,
            "median_ipr": self.median_ipr,
            "ipr_max": float(self.ipr.max()),
            "extended_fraction": self.extended_fraction,
            "mean_level_spacing_r": self.level_spacing_mean,
            "lambda_min": float(self.eigvals.min()),
            "lambda_max": float(self.eigvals.max()),
            "has_features": self.features is not None,
            "has_target": self.target is not None,
        }
        if self.target is not None:
            try:
                ceil = self.predictability_ceiling(K=16)
                out.update(
                    {
                        "R2_imd": ceil.r2_imd,
                        "R2_augmented_K16": ceil.r2_augmented,
                        "delta_R2_K16": ceil.delta_r2,
                    }
                )
            except Exception:
                pass
        self._summary_cache = out
        return out

    def __repr__(self) -> str:
        return (
            f"CitySpectralProfile(name={self.name!r}, N={self.N}, "
            f"graph_type={self.graph_type!r}, "
            f"sigma={self.sigma:.2f}, ext_frac={self.extended_fraction:.2f})"
        )

    # ─────────────────────────────────────────────────────────────────
    # Plotting (optional matplotlib)
    # ─────────────────────────────────────────────────────────────────
    def plot_overview(self, figsize: tuple[float, float] = (13, 9)):
        """Multi-panel overview figure: (a) geographic map of the
        most-localized eigenmode, (b) IPR-vs-eigenvalue spectrum,
        (c) ΔR² ceiling curve over K (if target is set), (d) text
        summary.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from spectral_mobility.plots import (
            plot_bottleneck_map,
            plot_ceiling_curve,
            plot_spectrum,
        )
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

        # (a) bottleneck map
        worst = self.bottleneck_modes(n=1)[0]
        plot_bottleneck_map(self.coords, self.eigvecs[:, worst], ax=axes[0, 0])
        axes[0, 0].set_title(f"{self.name} — worst bottleneck (IPR={self.ipr[worst]:.3f})")

        # (b) spectrum
        plot_spectrum(
            self.eigvals, self.ipr, ax=axes[0, 1],
            extended_threshold=5.0 / self.N,
        )

        # (c) ceiling curve
        if self.target is not None:
            K_grid = [1, 2, 4, 8, 16, 32, 64]
            K_grid = [k for k in K_grid if k <= self.eigvecs.shape[1]]
            r2_curve = []
            for k in K_grid:
                res = self.predictability_ceiling(K=k)
                r2_curve.append(res.r2_augmented)
            r2_baseline = self.predictability_ceiling(K=1).r2_imd
            plot_ceiling_curve(K_grid, r2_curve, r2_baseline=r2_baseline,
                               ax=axes[1, 0])
            axes[1, 0].set_title(f"{self.name} — ceiling vs K")
        else:
            axes[1, 0].text(0.5, 0.5, "no target provided",
                            ha="center", va="center",
                            transform=axes[1, 0].transAxes)
            axes[1, 0].set_axis_off()

        # (d) summary text
        s = self.summary()
        text = "\n".join(
            [
                f"{self.name}",
                f"N = {s['N']}  •  graph: {s['graph_type']}",
                f"σ = {s['sigma']:.1f}  •  k_NN = {s['k_nn']}",
                "",
                f"⟨IPR⟩ = {s['mean_ipr']:.3f}",
                f"median IPR = {s['median_ipr']:.3f}",
                f"max IPR = {s['ipr_max']:.3f}",
                f"extended fraction = {s['extended_fraction']:.2f}",
                f"⟨r⟩ (level spacing) = {s['mean_level_spacing_r']:.3f}",
                "",
            ]
        )
        if "R2_imd" in s:
            text += (
                f"R² IMD     = {s['R2_imd']:.3f}\n"
                f"R² aug K16 = {s['R2_augmented_K16']:.3f}\n"
                f"ΔR² K16    = +{s['delta_R2_K16']:.3f}"
            )
        axes[1, 1].text(0.05, 0.95, text, ha="left", va="top",
                        family="monospace", fontsize=10,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_axis_off()

        fig.suptitle(f"Spectral profile — {self.name}", fontsize=12)
        return fig
