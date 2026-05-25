"""High-level prediction wrapper: spectral-augmented regression.

The :class:`SpectralAugmentedRegressor` class is a thin wrapper that
handles the full pipeline:

    coordinates → graph → Laplacian → eigendecomposition →
    feature augmentation → fit base estimator → predict

It is **transductive**: graph and eigendecomposition are built once
on a fixed set of nodes (passed at ``fit`` time).  Predictions on
held-out nodes work by training only on observed targets but sharing
the eigenbasis across train and test.  This matches the standard
leave-node-out (LNO) evaluation protocol used in the structural-bound
literature.

For inter-network or inter-city transfer (different node sets at
train vs deploy time) the eigenbasis must be recomputed; this is not
handled by this class — use the lower-level primitives directly.
"""

from __future__ import annotations

from typing import Callable, Literal, TYPE_CHECKING

import numpy as np

from spectral_mobility.augmentation import augment_features
from spectral_mobility.bound import SpectralBoundResult, spectral_bound
from spectral_mobility.graph import (
    build_feature_knn,
    build_geographic_knn,
    symmetric_normalised_laplacian,
)
from spectral_mobility.spectral import spectral_decomposition

if TYPE_CHECKING:
    from typing import Any  # noqa: F401


def _default_estimator():
    """Return a default sklearn-compatible regressor.

    Tries LightGBM first; falls back to gradient boosting if LightGBM
    is not installed.  Both are deterministic given a seed.
    """
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    except (ImportError, OSError):
        # OSError catches missing OpenMP runtime on macOS without libomp.
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, random_state=42
        )


class SpectralAugmentedRegressor:
    """Sklearn-style transductive regressor with spectral feature
    augmentation.

    Workflow
    --------
    1. ``fit(X, coords, y, train_mask=mask)`` builds the graph from
       *all* ``coords``, computes the eigendecomposition once, augments
       ``X`` with the top-K low-frequency eigenvectors, and fits the
       base estimator on the rows selected by ``train_mask``.
    2. ``predict(test_mask=mask)`` returns predictions for the rows
       selected by ``test_mask`` (or all rows if not specified), using
       the same augmented features prepared at fit time.
    3. ``ceiling()`` returns the closed-form predictability bound.
    4. ``cross_validate(...)`` runs a k-fold LNO comparison between
       the spectral-augmented model and an unmodified baseline.

    Parameters
    ----------
    base_estimator : sklearn-compatible regressor, optional
        Any object with ``fit(X, y)`` and ``predict(X)``.  If omitted,
        LightGBM is used (falling back to sklearn's
        :class:`GradientBoostingRegressor` if LightGBM is not
        installed).
    K : int, default 16
        Number of low-frequency eigenvectors to augment with.
    k_nn : int, default 6
        Number of nearest neighbours in the proximity graph.
    sigma : float or ``"auto"``, default ``"auto"``
        Gaussian-RBF bandwidth for edge weights.  ``"auto"`` sets σ to
        the median k-th-NN distance.
    graph_type : {"geographic", "feature"}, default "geographic"
        ``"geographic"`` interprets ``coords`` as ``(lat, lng)`` in
        decimal degrees and uses haversine distance.  ``"feature"``
        treats ``coords`` as a generic (N, d) feature matrix and uses
        Euclidean distance.
    standardise_eigvecs : bool, default True
        If True, the appended eigenvector columns are z-scored to
        unit variance to match typical feature scales.

    Attributes
    ----------
    eigvecs_ : (N, N) np.ndarray
        Laplacian eigenvectors computed at ``fit`` time, columns
        ordered ascending in eigenvalue.
    eigvals_ : (N,) np.ndarray
    sigma_ : float
        The σ value actually used (useful when ``sigma="auto"``).
    X_aug_ : (N, p + K) np.ndarray
        Augmented feature matrix.
    n_imd_features_ : int
        Number of bare encoder features (i.e.\\ ``X.shape[1]``).
    """

    def __init__(
        self,
        base_estimator: "Any | None" = None,
        K: int = 16,
        k_nn: int = 6,
        sigma: float | Literal["auto"] = "auto",
        graph_type: Literal["geographic", "feature"] = "geographic",
        standardise_eigvecs: bool = True,
    ) -> None:
        if graph_type not in ("geographic", "feature"):
            raise ValueError(
                f"graph_type must be 'geographic' or 'feature', got {graph_type!r}"
            )
        self.base_estimator = base_estimator
        self.K = int(K)
        self.k_nn = int(k_nn)
        self.sigma = sigma
        self.graph_type = graph_type
        self.standardise_eigvecs = bool(standardise_eigvecs)

    # ─────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────
    def _build_graph(self, coords: np.ndarray):
        if self.graph_type == "geographic":
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(
                    "geographic graph expects coords of shape (N, 2) "
                    "containing [lat, lng] columns"
                )
            return build_geographic_knn(
                coords[:, 0], coords[:, 1], k=self.k_nn, sigma=self.sigma
            )
        return build_feature_knn(coords, k=self.k_nn, sigma=self.sigma)

    def _clone_estimator(self):
        """Return a fresh copy of the base estimator (or default)."""
        if self.base_estimator is None:
            return _default_estimator()
        try:
            from sklearn.base import clone

            return clone(self.base_estimator)
        except Exception:
            # Last resort: assume the caller passes a fresh instance per fit
            return self.base_estimator

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────
    def fit(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        y: np.ndarray,
        train_mask: np.ndarray | None = None,
    ) -> "SpectralAugmentedRegressor":
        """Fit the augmented regressor.

        Parameters
        ----------
        X : (N, p) array-like
            Bare encoder features.  All N rows must be present (the
            graph uses *all* of them).
        coords : (N, 2) array-like
            Coordinates for graph construction.  See ``graph_type``.
        y : (N,) array-like
            Targets.  Used only for the rows selected by ``train_mask``.
        train_mask : (N,) bool array, optional
            If supplied, only these rows are used to fit the base
            estimator.  The graph still uses all N rows
            (transductive).  Useful for leave-node-out workflows.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != coords.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} rows but coords has {coords.shape[0]}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if train_mask is None:
            train_mask = np.ones(X.shape[0], dtype=bool)
        train_mask = np.asarray(train_mask, dtype=bool)
        if train_mask.shape[0] != X.shape[0]:
            raise ValueError("train_mask length must match X.shape[0]")

        # Build graph on ALL nodes (transductive)
        W, sigma_used = self._build_graph(coords)
        self.sigma_ = sigma_used
        L = symmetric_normalised_laplacian(W)
        self.eigvals_, self.eigvecs_ = spectral_decomposition(L)

        # Augment features
        self.X_aug_ = augment_features(
            X, self.eigvecs_, K=self.K, standardise=self.standardise_eigvecs
        )
        self.n_imd_features_ = int(X.shape[1])
        self._X = X
        self._coords = coords
        self._y = y
        self._train_mask = train_mask

        # Fit the base estimator on observed rows only
        self.model_ = self._clone_estimator()
        self.model_.fit(self.X_aug_[train_mask], y[train_mask])
        return self

    def predict(self, test_mask: np.ndarray | None = None) -> np.ndarray:
        """Return predictions on the fit-time node set.

        Parameters
        ----------
        test_mask : (N,) bool array, optional
            Rows to return predictions for.  Defaults to all N.

        Returns
        -------
        y_hat : np.ndarray
        """
        if not hasattr(self, "model_"):
            raise RuntimeError("fit() must be called before predict()")
        if test_mask is None:
            X = self.X_aug_
        else:
            test_mask = np.asarray(test_mask, dtype=bool)
            if test_mask.shape[0] != self.X_aug_.shape[0]:
                raise ValueError("test_mask length must match the fit-time N")
            X = self.X_aug_[test_mask]
        return self.model_.predict(X)

    def ceiling(self) -> SpectralBoundResult:
        """Compute the spectral applicability bound on the fit-time
        data.

        Returns the IMD-only ceiling, the augmented ceiling, and the
        gain ΔR² — see :class:`spectral_mobility.SpectralBoundResult`.
        """
        if not hasattr(self, "X_aug_"):
            raise RuntimeError("fit() must be called before ceiling()")
        return spectral_bound(
            self.eigvecs_,
            self._y,
            encoder_features=self._X,
            K=self.K,
        )

    def _augment_inductive(
        self,
        X_train: np.ndarray,
        coords_train: np.ndarray,
        X_test: np.ndarray,
        coords_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inductive feature augmentation.

        Builds the graph + eigenvectors from training nodes ONLY,
        then projects test nodes onto the training eigenbasis using a
        Nyström-style extension: each test node's k_nn nearest
        training neighbours vote (Gaussian-RBF-weighted) for its
        eigenvector values.

        Returns ``(X_train_aug, X_test_aug)``.
        """
        from sklearn.neighbors import BallTree, NearestNeighbors

        # 1) Build training graph + eigenvectors
        W_train, sigma_used = self._build_graph(coords_train)
        L_train = symmetric_normalised_laplacian(W_train)
        eigvals_train, eigvecs_train = spectral_decomposition(L_train)
        Uk = eigvecs_train[:, : self.K]

        # 2) Standardise eigenvecs columnwise (if requested) — keep
        #    scale factors so test extension matches
        if self.standardise_eigvecs:
            std = Uk.std(axis=0)
            std = np.where(std > 1e-12, std, 1.0)
            Uk_std = Uk / std[None, :]
        else:
            std = np.ones(Uk.shape[1])
            Uk_std = Uk

        # 3) Nyström extension: each test point gets a weighted average
        #    of its k_nn nearest TRAINING neighbours' eigenvec values
        if self.graph_type == "geographic":
            train_rad = np.deg2rad(coords_train[:, ::-1])  # [lng, lat] for haversine? no — BallTree expects [lat, lng]
            # Actually let's just match build_geographic_knn ordering:
            train_rad = np.deg2rad(np.column_stack([coords_train[:, 0], coords_train[:, 1]]))
            test_rad = np.deg2rad(np.column_stack([coords_test[:, 0], coords_test[:, 1]]))
            tree = BallTree(train_rad, metric="haversine")
            dist_rad, idx = tree.query(test_rad, k=self.k_nn)
            from spectral_mobility.graph import EARTH_RADIUS_METRES

            dist_m = dist_rad * EARTH_RADIUS_METRES
            weights = np.exp(-(dist_m ** 2) / (2.0 * sigma_used ** 2))
        else:
            nn = NearestNeighbors(n_neighbors=self.k_nn, algorithm="auto").fit(coords_train)
            dist, idx = nn.kneighbors(coords_test)
            weights = np.exp(-(dist ** 2) / (2.0 * sigma_used ** 2))

        # Normalise weights row-wise so the extension is a proper convex combination
        w_norm = weights.sum(axis=1, keepdims=True)
        w_norm = np.where(w_norm > 1e-12, w_norm, 1.0)
        weights = weights / w_norm

        # For each test row i, eigvec_test[i, :] = sum_j weights[i, j] * Uk_std[idx[i, j], :]
        # Equivalent to (weights @ Uk_std[idx])
        Uk_test = np.einsum("ij,ijk->ik", weights, Uk_std[idx])

        X_train_aug = np.column_stack([X_train, Uk_std])
        X_test_aug = np.column_stack([X_test, Uk_test])
        return X_train_aug, X_test_aug

    def cross_validate(
        self,
        X: np.ndarray,
        coords: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42,
        scoring: Callable[[np.ndarray, np.ndarray], float] | None = None,
        return_predictions: bool = False,
        protocol: Literal["transductive", "inductive"] = "transductive",
    ) -> dict:
        """Compare augmented vs baseline (no augmentation) in K-fold LNO.

        Parameters
        ----------
        X, coords, y : as in :meth:`fit`
        n_folds : int, default 5
        random_state : int, default 42
        scoring : callable, optional
            Function ``scoring(y_true, y_pred) -> float``.  Defaults
            to R² (``sklearn.metrics.r2_score``).
        return_predictions : bool, default False
            If True, include per-fold predictions in the output.
        protocol : {"transductive", "inductive"}, default "transductive"
            ``"transductive"`` builds the graph and eigenvectors on
            all N nodes at fit time; only test ``y`` values are
            held out.  This is the LSO protocol of the structural-
            bound papers, with no ``y`` leakage but coord leakage
            into the eigenbasis is allowed.

            ``"inductive"`` rebuilds the graph from training coords
            only at each fold, then projects test points onto the
            training eigenbasis via a Nyström-style k-NN extension.
            This corresponds to a "deploy to new stations" use case
            and is the strictly leak-free protocol.

        Returns
        -------
        result : dict
            Keys ``baseline_scores``, ``augmented_scores``,
            ``baseline_mean``, ``augmented_mean``, ``mean_gain``,
            ``ceiling``, ``protocol`` and optionally ``predictions``.
        """
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold

        if protocol not in ("transductive", "inductive"):
            raise ValueError(
                f"protocol must be 'transductive' or 'inductive', got {protocol!r}"
            )

        if scoring is None:
            scoring = r2_score
        X = np.asarray(X, dtype=float)
        coords = np.asarray(coords, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        baseline_scores, augmented_scores = [], []
        baseline_preds = np.full_like(y, np.nan, dtype=float)
        augmented_preds = np.full_like(y, np.nan, dtype=float)

        for train_idx, test_idx in kf.split(X):
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[train_idx] = True
            test_mask = ~train_mask

            # Augmented model
            if protocol == "transductive":
                self.fit(X, coords, y, train_mask=train_mask)
                y_pred_aug = self.predict(test_mask=test_mask)
            else:
                # inductive: build augmented features for train and test
                # using only training coords for graph + Nyström extension
                X_train_aug, X_test_aug = self._augment_inductive(
                    X[train_mask], coords[train_mask],
                    X[test_mask], coords[test_mask],
                )
                model = self._clone_estimator()
                model.fit(X_train_aug, y[train_mask])
                y_pred_aug = model.predict(X_test_aug)
            augmented_scores.append(float(scoring(y[test_mask], y_pred_aug)))
            augmented_preds[test_mask] = y_pred_aug

            # Baseline: fit the same base estimator on raw X (no augmentation)
            base = self._clone_estimator()
            base.fit(X[train_mask], y[train_mask])
            y_pred_base = base.predict(X[test_mask])
            baseline_scores.append(float(scoring(y[test_mask], y_pred_base)))
            baseline_preds[test_mask] = y_pred_base

        # Recompute ceiling on the full data once at the end (transductive)
        self.fit(X, coords, y)
        ceiling = self.ceiling()

        result = {
            "baseline_scores": baseline_scores,
            "augmented_scores": augmented_scores,
            "baseline_mean": float(np.mean(baseline_scores)),
            "augmented_mean": float(np.mean(augmented_scores)),
            "mean_gain": float(np.mean(augmented_scores) - np.mean(baseline_scores)),
            "ceiling": {
                "r2_imd": ceiling.r2_imd,
                "r2_augmented": ceiling.r2_augmented,
                "delta_r2": ceiling.delta_r2,
                "K": ceiling.K,
            },
            "protocol": protocol,
        }
        if return_predictions:
            result["predictions"] = {
                "baseline": baseline_preds,
                "augmented": augmented_preds,
                "y_true": y,
            }
        return result
