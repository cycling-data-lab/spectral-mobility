# Pre-registration: a spectral applicability bound for urban mobility networks

**Author**  Rohan Fossé (CESI LINEACT, Montpellier)
**Project** `spectral-mobility` — https://github.com/cycling-data-lab/spectral-mobility
**Date**    2026-05-25 (to be sealed via OSF prior to extension to non-Western networks)
**Status**  Draft v1, awaiting external review before submission to OSF.

This document is a **confirmatory pre-registration**.  All analyses
described in sections 4–6 are to be run unchanged on the unseen
test set defined in section 5, with the hypothesis stated in section
3 and the decision rules in section 7.

---

## 1. Background and motivation

The structural applicability bound
$$\mathbb{E}[L] \;\geq\; (1 - R^2_{\text{spec}}(S, y))\cdot \mathrm{Var}(y)$$
predicts that for a graph-supervised regression problem, the
achievable test MSE on target $y$ is lower-bounded by the residual
that the leading-$K$ spectral subspace $S$ cannot explain.  We have
previously verified this bound on bike-share demand-forecasting
benchmarks (cycling-data-lab/papers, materials-applicability-bound).

An exploratory analysis on a curated 30-city panel revealed that
**the mean inverse participation ratio (IPR)** of the symmetric
normalised Laplacian eigenvectors is the single dominant predictor
of cross-city spectral similarity (Mantel $r = +0.634$,
$p < 0.0002$ vs $r = +0.006$ for continent).

A naïve extension to 124 networks of the international Atlas
revealed a strong size confound (smaller networks have mechanically
higher raw IPR).  We corrected this with a Monte-Carlo spatial null
based on a Gaussian KDE of the observed station locations
(bandwidth = 1.5 × Scott's rule).  The corrected coordinate
$$
z_{\text{kde}}(c) \;=\;
\frac{\text{IPR}_{\text{obs}}(c) - \mu_{\text{null}}(c)}
{\sigma_{\text{null}}(c)}
$$
is by construction independent of $N$.

We hereby pre-register the following confirmatory test on networks
that were not used in any of the development analyses (panel
curation, 30-city Mantel, 124-city KDE correction, GMM-BIC).

## 2. Anchor analyses (already completed, not pre-registered)

These are exploratory and disclosed in full:

- 30-city curated panel: bootstrap CIs on similarity matrix
  (`examples/05_paper_ready_validation.py`).
- Mantel ranking of 6 candidate metadata vs spectral similarity
  (`examples/06_discover_structure.py`).
- Atlas-wide bbox-null taxonomy (124 cities), found to be
  size-confounded (`examples/08`, `09`).
- KDE-null re-derivation + GMM-BIC model selection
  (`examples/11_kde_null_and_gmm.py`).

The continuous coordinate $z_{\text{kde}}$ and the extended fraction
are fixed by the code committed to the repository at the time this
pre-registration is sealed.

## 3. Hypothesis

> **H1.** In an unseen test panel of bike-share networks from Africa,
> Asia and Oceania, the slope $\beta$ of the mixed-effects
> regression
> $$R^2_{\text{spec}}(\text{Nyström}, K=10) \sim \beta \cdot z_{\text{kde}}
> + \gamma \cdot \log N + (1 \mid \text{country})$$
> is **negative and statistically significant** at $\alpha = 0.01$
> (two-tailed t-test on the slope estimate, with country-level
> random intercept).

Negative slope means: the stronger the localisation excess
$z_{\text{kde}}$, the lower the proportion of variance in demand that
the top-$K$ spectral subspace explains — i.e. the lower the
applicability of the structural prior, in line with the theory of
localised eigenmodes failing to support a global linear projection.

## 4. Materials and data

### 4.1 Test set (sealed; not yet inspected)

The test set is the union of bike-share parquets in
`bikeshare-demand-forecasting/data_collection/imd_international/`
whose ISO country code belongs to:

- Africa: any
- Asia: any except countries already represented in the development
  panel (currently none, so this is "any")
- Oceania: any

A pre-registered list of parquet filenames will be appended to this
document at the time of OSF sealing.  No new files will be added or
removed after sealing.

### 4.2 Inclusion criteria

A network is *included* in the confirmatory test iff:
- The parquet has at least 50 stations after de-duplication on
  (lat, lng).
- The parquet has a `demand` column or an equivalent target
  derivable per station (sum of trips, etc.); networks without
  demand data are excluded from H1 testing but may appear in
  descriptive figures.
- The spectral decomposition of the symmetric normalised Laplacian
  on its $k=5$-NN graph succeeds without numerical errors.

### 4.3 Exclusion criteria

The following are pre-specified exclusions; we will report them in
the manuscript even if they reduce $n_{\text{test}}$ to zero.

- Networks already present in the development panel
  (`output/05_panel.csv`).
- Networks whose demand column is empty, all-zero, or constant.
- Networks whose KDE null fails to produce $\geq 3$ valid replicates
  (i.e. degenerate spatial distributions).

## 5. Analysis plan

### 5.1 Per-city feature extraction

For each included city $c$:
1. Build the $k=5$-NN spatial graph with Haversine distance, Gaussian
   RBF weights, $\sigma$ chosen by `sigma="auto"` (median of $k$-th
   neighbour distance).
2. Compute the symmetric normalised Laplacian and its full eigen-
   decomposition.
3. Record `mean_ipr`, `extended_fraction`, `N`, $\sigma$.
4. Generate $n_{\text{null}}=10$ KDE-null layouts (Gaussian KDE,
   bandwidth = 1.5 × Scott's rule, sampling $N$ points) and compute
   $\mu_{\text{null}}, \sigma_{\text{null}}$.
5. Compute $z_{\text{kde}}(c) = (\text{IPR}_{\text{obs}} - \mu_{\text{null}})/\sigma_{\text{null}}$.

### 5.2 $R^2_{\text{spec}}$ via Nyström (inductive)

For each city $c$ with a demand target:
1. Hold out a random 30% of stations as a test set
   (`random_state=2026`).
2. On the training stations, compute the top-$K=10$ low-frequency
   eigenvectors of the in-sample Laplacian.
3. Extend them to the held-out stations via the Nyström formula
   (`spectral_mobility.SpectralAugmentedRegressor` with
   `mode="inductive"`).
4. Fit a linear projection from those 10 inductive features onto
   $z$-scored within-city demand.
5. Report $R^2_{\text{spec}}$ on the held-out stations.

### 5.3 Confirmatory model

Mixed-effects regression as in section 3, fit with `statsmodels.mixed_lm`,
maximum-likelihood, country as grouping variable.  We pre-specify:
- Fixed effects: $z_{\text{kde}}$, $\log N$.
- Random effect: country intercept.
- Decision metric: two-tailed Wald test on $\beta$
  ($z_{\text{kde}}$ slope), $\alpha = 0.01$.

## 6. Confirmatory criteria

H1 is supported iff:
- $\hat\beta < 0$ (negative slope), AND
- the two-tailed $p$-value of the Wald test is $\leq 0.01$, AND
- the slope's 99% CI excludes 0.

If any of the three fails, we will report H1 as *not supported*
and discuss the implications (taxonomy is continent-specific, KDE
null is insufficient, etc.) without re-fitting alternative models.
Any alternative analyses performed after seeing the test data will
be labelled *exploratory* in the manuscript.

## 7. Robustness checks (also confirmatory)

In addition to H1, we will report the following without changing the
decision rule:
- Slope of the same model with $K \in \{5, 20\}$ instead of $K=10$.
- Slope of the same model with $k_{\text{NN}} \in \{3, 7\}$ instead of $5$.
- Slope of the same model with `extended_fraction` substituted for
  $z_{\text{kde}}$.

These are descriptive and not used to update H1; they are pre-
registered only to ensure they are reported regardless of outcome.

## 8. Stop conditions and decision tree

| Outcome on H1 | Manuscript framing |
|---|---|
| Supported (β < 0, p ≤ 0.01) | Headline: "Localisation regimes predict applicability of structural priors" |
| Not supported, $\|\beta\|$ small | Headline reframed: "Spectral similarity is a size-corrected geometric signature; predictive utility limited to development panel" |
| Not supported, $\beta > 0$ (sign-flipped) | Critical re-analysis; manuscript reports the failure as a falsification of the panel-derived theory |

## 9. Reproducibility

All code is in `examples/11_kde_null_and_gmm.py` and the
`spectral_mobility` package, fixed at commit hash to be inserted at
OSF sealing.  Random seeds: as specified above.  Compute environment
is the project `.venv`, also pinned in `pyproject.toml`.

## 10. Authors and contributions

Rohan Fossé designed the study, wrote the code, and will conduct
the confirmatory analysis.  Co-authorship list will be finalised
before submission.  Reviewer feedback (acknowledged in section 11)
informed the methodological choices but reviewers are not co-authors.

## 11. Acknowledgements

Methodological choices that explicitly responded to external review:
- KDE-based null instead of bbox-uniform null (response to a 2026-05-25
  external methodology review pointing out that uniform sampling in
  the bounding box of coastal/lake-bordering cities places null
  stations in water and thereby biases $z_{\text{excess}}$).
- GMM with BIC instead of k-means with fixed $k=3$ (same review).
- Continuous $z_{\text{kde}}$ regression instead of categorical type
  ANOVA (same review).
- Inductive Nyström instead of transductive CV (response to
  earlier review pointing out the transductive overfit).

---

*End of pre-registration v1.  Sealed on OSF: <pending>.*
