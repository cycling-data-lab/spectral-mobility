# Pre-registration v2: a two-axis applicability bound for graph-spectral priors on urban mobility networks

**Author**  Rohan Fossé (CESI LINEACT, Montpellier)
**Project** `spectral-mobility` — https://github.com/cycling-data-lab/spectral-mobility
**Code commit** to be inserted at OSF sealing time
**Date drafted** 2026-05-25
**Status**  v2 draft, supersedes [v1](preregistration_v1.md).  Sealed on OSF: pending.

This pre-registration replaces v1 (a single linear H1 on `z_kde`)
after exploratory analyses on 18 development cities revealed (a) a
much stronger candidate predictor of absolute applicability (the
Dirichlet energy of demand on the unnormalised Laplacian) and (b)
that `z_kde` predicts the *augmentation gain* rather than the
absolute ceiling.  The exploratory results are reported in section
5 below; the confirmatory test follows sections 6–8.

---

## 1. Context

A graph-spectral applicability bound was previously established for
urban mobility prediction:

$$
\mathbb{E}[L] \;\geq\; \bigl(1 - R^2_{\text{spec}}(S_K, y)\bigr)\cdot \mathrm{Var}(y),
$$

where $S_K$ is the span of the top-$K$ low-frequency eigenvectors
of the symmetric normalised Laplacian.  Empirically, augmentation
by these eigenvectors lifts the achievable $R^2$ from $0.05$ to
$0.46$ on Boston Bluebikes (cited in the prior cycling-data-lab
materials).  The open question is whether this gain is predictable
from network-level structural features.

## 2. Two-axis hypothesis

The exploratory data (n=18 cities with real demand) point to a
decomposition of applicability into two **orthogonal axes**:

- **Axis 1 — Network capacity**: the size-corrected localisation
  excess
  $$z_{\text{kde}}(c) = (\overline{\text{IPR}}_{\text{obs}}(c) - \mu_{\text{null}}(c))/\sigma_{\text{null}}(c)$$
  measured against a Gaussian-KDE spatial null (bandwidth = 1.5 ×
  Scott's rule, sampling $N$ points).  Computed once on the
  symmetric normalised Laplacian of the $k=5$-NN spatial graph.
- **Axis 2 — Signal compatibility**: the Dirichlet energy of demand
  on the *unnormalised* Laplacian
  $$
  E_D(y) = \frac{(y - \bar y)^\top L_{\text{un}} (y - \bar y)}{(y - \bar y)^\top (y - \bar y)},\quad
  L_{\text{un}} = D - W.
  $$
  This is the natural discrete-gradient norm of a node-valued
  count-like variable, justified ex ante by the conservation-of-mass
  argument: $y^\top L_{\text{un}} y$ does not normalise the signal by
  degree, preserving the absolute volume of demand at each station.

Empirically (n=18), these two axes have variance-inflation factors
below 3 and Pearson correlation $-0.12$, so they are jointly
identifiable.  The exploratory effect sizes are large
($r = -0.93$ for $E_D$ on $R^2_{\text{aug}}$;
$\beta = -0.010$ for $z_{\text{kde}}$ on the augmentation gain on
the $z_{\text{kde}} > -5$ subset, $p = 0.016$).

## 3. Confirmatory hypotheses

> **H1a (applicability ceiling).**  On networks unseen in the
> development panel, the OLS regression
> $$
> R^2_{\text{aug, Nyström}}(K=10) \;=\; \alpha_0 + \alpha_1 E_D(y) + \alpha_2 \log N + \varepsilon
> $$
> fitted on per-city averages over 10 random holdout splits has
> $\alpha_1 < 0$ with a two-tailed Wald $p \leq 0.01$.

> **H1b (augmentation utility).**  On networks unseen in the
> development panel **and with $z_{\text{kde}} > -5$**, the linear
> mixed-effects model
> $$
> \text{gain}_{c, s} \;=\; R^2_{\text{aug}} - R^2_{\text{base}} \;=\; \beta_0 + \beta_1 z_{\text{kde}}(c) + u_c + \varepsilon_{c,s}
> $$
> fitted on (city × split) observations, with city $c$ as random
> intercept, has $\beta_1 < 0$ with a two-tailed Wald $p \leq 0.05$.

Both hypotheses are tested on the same confirmatory data; no
α-spending adjustment is applied because they are pre-specified as
the two **primary outcomes** of distinct theoretical axes.

## 4. Anchor analyses (exploratory, not pre-registered)

These were performed before sealing and inform but do not test the
confirmatory hypotheses:

- 30-city panel bootstrap CIs + Mantel ranking
  (`examples/05_paper_ready_validation.py`,
  `examples/06_discover_structure.py`).  Mean IPR dominated
  cross-city similarity ($r = +0.63$); continent did not
  ($r = +0.006$).
- 124-city Atlas + KDE-null taxonomy + GMM-BIC selecting $k=3$
  types + finite-size falsification of type C
  (`examples/11_kde_null_and_gmm.py`).
- Linear pre-registered H1 (v1) falsified at $n=18$
  ($r = -0.017$, `examples/14_demand_validation.py`).
- Goldilocks quadratic alternative on absolute $R^2$ — supported in
  sign, underpowered ($p = 0.27$); falsified on Atlas-wide proxies
  (`examples/15_goldilocks_test.py`, `examples/16_goldilocks_atlas_proxy.py`).
- Dirichlet energy + augmentation gain decomposition
  (`examples/17_dirichlet_and_aug_gain.py`,
  `examples/18_pre_reg_v2_check.py`).  See section 5.

## 5. Exploratory effect sizes (n=18 development cities)

### H1a candidate (Dirichlet energy on $L_{\text{un}}$)
Per-city OLS, $R^2_{\text{aug}} \sim E_D(y) + \log N$:

```
                  coef     std err       t      P>|t|     [0.025    0.975]
const           +0.606     0.171      3.55     0.003     +0.240    +0.973
E_D             −0.206     0.032     −6.37    <0.001     −0.276    −0.137
log N           +0.028     0.024     +1.19     0.254     −0.023    +0.080
                                                          (n=18, R² = 0.88)
```

### H1b candidate ($z_{\text{kde}}$ on the gain, $z > -5$)
LMM, gain ~ z_kde + (1|city), n=15 cities × 10 splits = 150 obs:

```
                  coef     std err      z      P>|z|     [0.025    0.975]
Intercept       +0.033     0.010    +3.35     0.001     +0.014    +0.053
z_kde           −0.010     0.004    −2.42     0.016     −0.018    −0.002
Group Var       +0.001     0.016
```

### Orthogonality (VIF)
On n=18 per-city standardised regressors $(z_{\text{kde}}, E_D, \log N)$:
- VIF($z_{\text{kde}}$) = 1.71
- VIF($E_D$) = 1.84
- VIF($\log N$) = 2.35

All below 3; Pearson correlation of $z_{\text{kde}}$ with $E_D$ is $-0.117$.
The two axes are jointly identifiable.

## 6. Materials and confirmatory data

### 6.1 Test set (sealed; not yet inspected)

The confirmatory test will be conducted on bike-share networks in
`bikeshare-demand-forecasting/data_collection/imd_international/`
that satisfy ALL of the following:
1. Country (ISO) **not present** in the development set listed in
   appendix A (to be enumerated at sealing).
2. Demand data available either in `paper_demand/experiments/outputs/`
   or as a published per-station count for any continuous time
   window of at least 30 days.
3. At least 50 stations after de-duplication on (lat, lng).
4. Demand column non-constant.

A pre-registered enumeration of qualifying network filenames will be
appended to this document at OSF sealing.

### 6.2 Stop conditions

- If fewer than $n = 30$ qualifying networks have usable demand
  data after pre-processing, the confirmatory test is declared
  **underpowered** and reported as such.  The Dirichlet effect is
  expected to be detectable even at $n = 30$ because the exploratory
  effect size is $r = -0.93$, but the $z_{\text{kde}}$ effect may
  not be.
- If 30 ≤ $n$ ≤ 60, results are reported with an a-posteriori
  power-analysis caveat for H1b.

### 6.3 Pre-processing protocol (fixed)

For each confirmatory network $c$:
1. Load coordinates and demand; drop NA, de-duplicate on (lat, lng).
2. Cap at $N = 6000$ stations by uniform random sampling (seed=0)
   for the largest networks.
3. Build the symmetric normalised Laplacian on $k = 5$-NN spatial
   graph (Haversine + Gaussian RBF, $\sigma$ = auto).
4. Compute the top-$K = 10$ low-frequency eigenvectors.
5. Compute $z_{\text{kde}}(c)$ via 10 KDE-null replicates (bandwidth
   = 1.5 × Scott's rule).
6. Compute $E_D(y) = (y - \bar y)^\top L_{\text{un}} (y - \bar y) / (y-\bar y)^\top(y-\bar y)$
   where $y = \log(\text{trips} + 1)$ averaged per station.
7. For each of $S = 10$ holdout splits (random 70/30, seeds derived
   deterministically from `hash((city, split_id))`):
   a. Fit Ridge ($\alpha = 1.0$) on the in-sample 70% with the
      seven IMD features as covariates → $R^2_{\text{base}}$.
   b. Fit Ridge on the same features plus the Nyström-extended
      top-10 eigenvectors → $R^2_{\text{aug}}$.
   c. Record gain = $R^2_{\text{aug}} - R^2_{\text{base}}$.
8. Per-city: average $R^2_{\text{aug}}$ over splits → row of the
   H1a regression.  All (city × split) gains → rows of the H1b LMM.

### 6.4 Estimation

- **H1a** is fitted via `statsmodels.OLS` on per-city averages,
  classical heteroskedasticity-unadjusted standard errors.
- **H1b** is fitted via `statsmodels.MixedLM` with `groups=city`,
  REML, default LBFGS.

The python code that produces these models is in
`examples/18_pre_reg_v2_check.py` (sections H1a candidate and H1b
candidate).  No edits to this file are allowed after sealing.

## 7. Confirmatory decision rules

| Outcome on H1a | Outcome on H1b | Decision |
|---|---|---|
| $\alpha_1 < 0$, $p_{\alpha_1} \leq 0.01$ | $\beta_1 < 0$, $p_{\beta_1} \leq 0.05$ | **Theory supported**: both axes replicate |
| $\alpha_1 < 0$, $p_{\alpha_1} \leq 0.01$ | $\beta_1 \geq 0$ or $p_{\beta_1} > 0.05$ | **Partial**: signal-side axis ($E_D$) holds, taxonomy-side axis fails |
| $\alpha_1 \geq 0$ or $p_{\alpha_1} > 0.01$ | $\beta_1 < 0$, $p_{\beta_1} \leq 0.05$ | **Partial**: taxonomy-side axis ($z_{\text{kde}}$) holds, signal-side fails |
| Both fail | | **Theory falsified**: report the failure as primary outcome |

Sign-flipped outcomes ($\alpha_1 > 0$ or $\beta_1 > 0$ at significant
$p$) are reported as falsifications, not "interesting alternatives".

## 8. Robustness checks (also confirmatory)

The following will be reported alongside the primary outcomes, but
do **not** alter the decision tree:

- Re-fit H1a with $K \in \{5, 20\}$ instead of $10$.
- Re-fit H1a with $E_D$ replaced by Dirichlet on $L_{\text{sym}}$
  (the symmetric normalised Laplacian).
- Re-fit H1b on the full $z_{\text{kde}}$ range (no $z > -5$ cut).
- Re-fit H1b with $z_{\text{kde}}^2$ added (test of quadratic
  shape, exploratory).

## 9. What this pre-registration does NOT cover

The following are explicitly **post-hoc / exploratory** and will be
labelled as such in the manuscript:

- Choice of $k = 5$ for the spatial graph (sensitivity in
  `examples/10_taxonomy_robustness.py` is exploratory).
- Choice of $K = 10$ (the structural-bound papers use 8–10).
- Transfer-learning matrix on the 18 development cities → Paper #3.
- Random-walk Laplacian variant → Paper #3.

## 10. Authors and contributions

Rohan Fossé conducted all analyses and will run the confirmatory
test.  Methodological choices in v2 explicitly responded to:
- Round 1 external review: KDE-based null instead of bounding-box
  uniform; GMM-BIC instead of fixed k-means $k = 3$.
- Round 2 external review: strict inductive Nyström instead of
  semi-transductive OLS; raw $\log(y+1)$ instead of within-city
  z-scoring.
- Round 3 external review: Dirichlet energy as missing covariate
  (Idea #1); augmentation gain as alternative target (Idea #7).
- Round 4 external review: unnormalised Laplacian for $E_D$ (Q2);
  H1b restricted to $z_{\text{kde}} > -5$ (Q3); Ridge fixed as
  baseline (Q4); transfer test deferred (Q6).

External reviewers are not co-authors but are acknowledged in the
manuscript.

## Appendix A — Development networks (excluded from confirmatory test)

To be filled at sealing time with the exact list of parquet stems
in `data_collection/imd_international/` used in scripts 14–18.

---

*End of pre-registration v2.*
