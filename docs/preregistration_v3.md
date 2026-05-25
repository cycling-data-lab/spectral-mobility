# Pre-registration v3: spatial leakage, spectral block-CV, and the
applicability of graph-spectral priors in urban mobility prediction

**Author**  Rohan Fossé (CESI LINEACT, Montpellier)
**Project** `spectral-mobility` — https://github.com/cycling-data-lab/spectral-mobility
**Code commit** to be inserted at OSF sealing time (currently `de1107b`,
tag `v0.5.0-rc.4`)
**Date drafted** 2026-05-25
**Status**  v3 draft.  Supersedes [v1](preregistration_v1.md) (linear,
falsified) and [v2](preregistration_v2.md) (two-axis on random-CV,
revealed to be partly spatial-leakage artefact).  Sealed on OSF:
pending.

This version locks in the final pre-registered hypotheses after the
**random-CV illusion** discovered in script 20 (k-means block-CV),
script 21 (spectral block-CV) and script 22 (robust estimators).
The exploratory results that motivated v3 are in section 5.

---

## 1. The 4-stage narrative

The pre-registration narrative, in its final form, has four
sequential claims that we will test on networks unseen in
development:

1. **Random-CV inflates the applicability ceiling.**  The standard
   70/30 random hold-out, even with strict inductive Nyström,
   exploits spatial autocorrelation; the achieved $R^2_{\text{aug}}$
   is not the true deployment ceiling.

2. **Spectral-block CV reveals the true extrapolation difficulty.**
   When held-out blocks are defined by *spectral clustering on the
   own k-NN Laplacian*, $R^2_{\text{aug}}$ collapses (often
   becomes negative).

3. **The illusion is predictable from signal smoothness.**  The
   gap
   $$R^2_{\text{gap}} = R^2_{\text{aug, random}} - R^2_{\text{aug, block}}$$
   is anti-correlated with the Dirichlet energy of demand on the
   unnormalised Laplacian: smoother demand → more spatial autocorrelation
   → larger random-CV inflation.

4. **The augmentation gain survives extrapolation and is much more
   predictable.**  Under spectral block-CV, the gain
   $\Delta R^2 = R^2_{\text{aug}} - R^2_{\text{base}}$ remains
   positive and is **17–22× more strongly** predicted by the
   localisation excess $z_{\text{kde}}$ than under random-CV.  This
   is where topological priors actually buy you something.

## 2. Background

The structural applicability bound
$$\mathbb{E}[L] \;\geq\; (1 - R^2_{\text{spec}})\cdot \mathrm{Var}(y)$$
(Théorème 1, prior cycling-data-lab work) holds *mechanically* per
city but does not translate into a clean predictor of which
networks benefit most from graph-spectral priors.  v1 hypothesised
a linear $z_{\text{kde}}$ relation that was falsified; v2
hypothesised a two-axis Dirichlet + $z_{\text{kde}}$ relation that
was revealed in script 20 to be partly spatial leakage.  This
version focuses on the residual *true* signal that survives strict
extrapolation.

## 3. Confirmatory hypotheses

### 3.1 Primary outcome (H1)

> **H1 — Augmentation gain under spectral block-CV.**
> On networks unseen in the development set and with $z_{\text{kde}} > -5$,
> the cluster-bootstrap 95% confidence interval of the slope
> $\beta_1$ in the OLS regression
> $$\overline{\Delta R^2_{c}} \;=\; \beta_0 + \beta_1 z_{\text{kde}}(c)\;+\;\epsilon_c$$
> (per-city averages, $\overline{\Delta R^2_c}$ = mean spectral
> block-CV gain over balanced spectral blocks) **lies entirely
> below 0**, equivalently the two-tailed bootstrap $p \leq 0.01$.

### 3.2 Secondary outcome (H2)

> **H2 — Dirichlet predicts the spatial-leakage gap.**
> On the same confirmatory networks, the OLS regression
> $$R^2_{\text{gap}, c} \;=\; \alpha_0 + \alpha_1 E_D(y_c) + \alpha_2 \log N_c + \epsilon_c$$
> has $\alpha_1 < 0$ (smoother demand → larger inflation) with
> two-tailed $p \leq 0.05$.

H1 is the **primary** outcome and the success/failure of the
pre-registration.  H2 is a **secondary** that does not modify the
H1 decision; it provides mechanistic support but is reported
independently.

### 3.3 Exploratory mechanism

> **M1 — Eigenvector energy as mediator.**
> Per-block ⟨‖U_K[test]‖²⟩, when added to H1's regression as a
> covariate, attenuates $\beta_1$ but is itself not necessarily
> significant.  We report the attenuated $\beta_1$ and the
> $\beta_{\text{eig}}$ point estimate as exploratory mechanism
> evidence, with no significance threshold.

## 4. Anchor analyses (exploratory, not pre-registered)

Performed in development, fully disclosed:

- 30-city panel + bootstrap CIs + Mantel ranking
  (`examples/05`, `examples/06`).  Mean IPR dominated cross-city
  similarity ($r = +0.63$).
- 124-city Atlas KDE-null taxonomy + GMM-BIC selecting 3 types +
  finite-size falsification (`examples/11`).
- Random-CV linear H1 v1: $\beta = -0.017$, $p = 0.95$
  (`examples/14`).
- Random-CV quadratic Goldilocks: $\beta_2 = -0.002$, $p = 0.27$
  (`examples/15`).
- Random-CV bivariate Dirichlet + z_kde (v2): r=−0.93, β_D=−0.21
  ($p < 0.001$ per-city OLS) — later revealed mostly leakage.
- KDE bandwidth sweep [0.5, 3.0] × Scott (`examples/19`): sign
  of $\beta_1$ is **always negative** but $p > 0.05$ under random-CV.
- k-means block-CV (`examples/20`): $\beta_1 = -0.175$, $p < 0.001$;
  H1a sign flipped to $+0.97$, NS.
- Spectral block-CV (`examples/21`): $\beta_1 = -0.196$;
  $R^2_{\text{gap}} \sim -1.29 \cdot E_D$, $p = 0.10$ at $n = 14$.
- Robust estimator suite (`examples/22`): per-city OLS
  $\beta = -0.079$ $p = 0.13$; WLS by block size $\beta = -0.106$
  $p = 0.002$; cluster bootstrap 100% of 2000 resamples give
  $\beta < 0$, 95% CI $[-0.31, -0.01]$, $p = 0.001$.

## 5. Exploratory effect sizes (development cohort, n=14 cities)

```
H1 candidate (cluster bootstrap, gain ~ z_kde, z_kde > -5)
  β point         = -0.079
  bootstrap mean  = -0.099
  95% CI          = [-0.308, -0.015]
  fraction β < 0  = 100% (out of 2000)
  two-tailed p    = 0.001

H2 candidate (OLS, R²_gap ~ Dirichlet + log N)
  β(Dirichlet)    = -1.29
  CI95            = [-2.89, +0.30]
  p               = 0.102
  R²              = 0.333

M1 candidate (LMM gain ~ z_kde + eig_energy_z + (1|city))
  β(z_kde)        = -0.093
  β(eig_energy_z) = -0.039  (NS, p = 0.65)
  z_kde signal partially absorbed by eig_energy
```

These are explicitly *exploratory* (chose the spectral-clustering
protocol after seeing k-means block-CV); the confirmatory test
applies the locked spec below.

## 6. Materials and confirmatory protocol

### 6.1 Test set

The confirmatory test will use bike-share networks satisfying ALL of:
1. ISO country code **not present** in the development set listed
   in Appendix A (enumerated at OSF sealing).
2. Per-station demand data either in
   `paper_demand/experiments/outputs/` or as a publicly accessible
   per-station count series of at least 30 consecutive days.
3. At least 50 stations after de-duplication on (lat, lng).
4. Demand column non-constant and ≥ 60 stations match between
   demand and atlas files.

### 6.2 Stop conditions

- If $n < 25$ qualifying networks: confirmatory test reported as
  **underpowered**.  H1 not decided.
- If $25 \leq n \leq 50$: results reported with a-posteriori power
  caveat for H2 (H1 is robust at $n = 14$, so should be even more
  robust at $n \geq 25$).
- If $n > 50$: standard confirmatory decision applies.

### 6.3 Pre-processing protocol (locked)

For each confirmatory network $c$:
1. Drop NA, de-duplicate on (lat, lng), cap at $N = 6000$ by
   uniform random sampling (seed = 0).
2. Build the symmetric normalised Laplacian on $k = 5$-NN spatial
   graph (Haversine + Gaussian RBF, $\sigma$ = auto).
3. Compute the **full** eigendecomposition.  Extract top-$K = 10$
   eigenvectors for the augmentation step, top-12
   non-trivial eigenvectors for the block partitioning.
4. **Block partitioning**:
   k-means with $K_{\text{blocks}} = \max(3, \min(10, N / 35))$
   on row-normalised eigenvectors 1..12 (Ng-Jordan-Weiss spectral
   clustering).  Constraint: each block must have ≥ 30 stations;
   if violated, decrease $K_{\text{blocks}}$ by 1 and re-cluster
   (up to 3 iterations).
5. Compute $z_{\text{kde}}(c)$ via 10 KDE-null replicates
   (bandwidth = 1.5 × Scott's rule).
6. Compute $E_D(y_c) = (y - \bar y)^\top L_{\text{un}} (y - \bar y) / (y - \bar y)^\top (y - \bar y)$
   where $y = \log(\text{trips} + 1)$ averaged per station.
7. **Leave-one-block-out spectral block-CV**:
   for each block $b$, fit Ridge ($\alpha = 1.0$) on stations in
   other blocks with 7 IMD features → $R^2_{\text{base}}^{(b)}$,
   then with IMD features + Nyström-extended top-10 eigenvectors →
   $R^2_{\text{aug}}^{(b)}$.  Gain $\Delta R^2^{(b)} = R^2_{\text{aug}}^{(b)} - R^2_{\text{base}}^{(b)}$.
8. **Random-CV reference**: same number of folds with random 30%
   holdouts (seed deterministic from `hash(city)`) → $R^2_{\text{random}, c}$.
9. Per-city aggregates:
   $\overline{\Delta R^2_c}$ = mean over blocks,
   $R^2_{\text{gap}, c}$ = $R^2_{\text{aug, random}} - R^2_{\text{aug, block}}$.

### 6.4 Estimation

- **H1**: cluster bootstrap with 5000 city-level resamples on the
  per-city OLS slope.  Implementation: identical to
  `examples/22_robust_block_cv_estimator.py` section (C).
  Random seed = 2026.
- **H2**: per-city OLS via `statsmodels.OLS`, classical SE.
- **M1**: `statsmodels.MixedLM` on (city × block) observations
  with city random intercept and `eig_energy_z = (eig_energy_test
  − mean) / std`.

The python code that produces these estimators is in
`examples/22_robust_block_cv_estimator.py` (functions
`per_city_OLS` and `cluster_bootstrap`) and section M1 in
`examples/21_spectral_blocks_and_gap.py`.  No edits to these
locked functions after sealing.

## 7. Confirmatory decision rules

| H1 outcome | H2 outcome | Decision |
|---|---|---|
| 95% CI fully below 0, $p \leq 0.01$ | $\alpha_1 < 0$, $p \leq 0.05$ | **Theory fully supported** |
| 95% CI fully below 0, $p \leq 0.01$ | $\alpha_1 \geq 0$ or $p > 0.05$ | **Primary supported, mechanism unclear** |
| 95% CI contains 0 or $p > 0.01$ | $\alpha_1 < 0$, $p \leq 0.05$ | **H1 not replicated; manuscript reports failure as primary outcome.  Mechanism story (Dirichlet → gap) retained as descriptive** |
| H1 fails AND H2 fails | | **Theory falsified** |

A sign-flipped $\beta_1 > 0$ at significant $p$ is reported as
falsification, not "interesting alternative".

## 8. Robustness checks (confirmatory, do not alter decision)

These will be reported alongside the primary outcome:

- KDE bandwidth ∈ {0.5, 1.0, 1.5, 2.0, 3.0} × Scott's rule
  (replicate `examples/19`).
- $K_{\text{NN}} \in \{3, 7\}$ (vs 5 in primary).
- $K_{\text{spec}} \in \{5, 20\}$ (vs 10 in primary).
- Random-CV baseline: same regression on random-CV gain
  (expect smaller $|\beta|$ and larger $p$, illustrating the gap).

## 9. What this pre-registration does NOT cover

- The choice of unnormalised Laplacian for the Dirichlet energy
  (justified ex ante by the volume-conservation argument; tested
  in exploratory against $L_{\text{sym}}$, both give the same
  sign).
- The choice of Ridge ($\alpha = 1.0$) as the augmentation model;
  non-linear models (GBM, RF) are explicitly out of scope and
  reported in supplement only.
- Cross-mode generalisation (scooters, carsharing): deferred to
  Paper 2.
- Transfer learning matrix $W_1$-similarity → R²_drop: deferred.

## 10. Authors and contributions

Rohan Fossé conducted all analyses and will run the confirmatory
test.  External methodology reviewers shaped v1→v3:
- v1→v2: KDE-based null (round 1), GMM-BIC, strict inductive
  Nyström (round 2), Dirichlet + gain decomposition (round 3),
  unnormalised Laplacian + Ridge + H1b range restriction (round 4)
- v2→v3: spectral block-CV instead of k-means (round 5),
  R²_gap-Dirichlet hypothesis (round 6), eigenvector mediator,
  cluster-bootstrap robust estimator (round 6).

Reviewers are acknowledged in the manuscript but are not co-authors.

## Appendix A — Development networks (excluded from confirmatory test)

To be filled at OSF sealing time with exact parquet stems used in
scripts 14 through 22.

---

*End of pre-registration v3.*
