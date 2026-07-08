# Methods — ODE growth-rate model for the TFMN4 barcode competition

This document derives the model implemented in `scripts/` and states precisely
what it can and cannot estimate. It is scoped to the **first assessment**: the
**11 Samples observed at all four transfers {3, 4, 6, 13}**.

---

## 1. The experiment and the data

`SeqCenter_2026-05-26_QUO1022807_exp2_extracted_barcodes.csv` is a **serial-transfer
barcode competition**. Each row is a read count for one enzyme variant in one
culture at one transfer.

| Concept | Column(s) | Meaning |
|---|---|---|
| Variant | `Candidate` = `verA`–`verB` | A unique enzyme pair (41 verA × 96 verB → 350 observed). The lineage whose growth rate we want. |
| Culture | `Sample` = `TFMN4.exp2.ACN3788.<DNA_construct>.<replicate>` | One competing population (24 total). |
| Time | `Transfer` ∈ {3, 4, 6, 13} | A serial-transfer snapshot. |
| Abundance | `Count` | Sequencing reads of that variant's barcode in that culture at that transfer. |

Two structural facts about the counts **determine the entire model**:

1. **The counts are compositional, not absolute cell numbers.** Total reads per
   `(Sample, Transfer)` are roughly fixed sequencing depth (min 101, median 190,
   max 274) and do **not** grow over transfers. A count therefore measures a
   *relative frequency* `f_i = N_i / Σ_j N_j`; the absolute population size is not
   in this file.

2. **A selective sweep is occurring.** The number of distinct variants per culture
   collapses over transfers (e.g. 48 → 27 → 17 → 9) while total reads stay flat —
   winners concentrate, losers fall to zero reads. This is competitive exclusion,
   exactly what a selection ODE predicts.

The selection pressure (from the OD file's `Condition`) is
**4 mM methoxybenzoate + kanamycin**: variants whose verA/verB enzymes metabolize
the substrate better out-compete the rest.

---

## 2. The ODE model

### 2.1 Latent (absolute) growth

Within a culture *c*, each variant *i* grows exponentially:

```
dN_{c,i}/dt = r_{c,i} · N_{c,i}        ⇒   N_{c,i}(t) = N_{c,i}(0) · exp(r_{c,i} t)
```

Serial dilution at each transfer multiplies **every** lineage by the same factor
1/D, so it cancels in frequencies and does not break the continuous-time picture.

### 2.2 What sequencing observes — the replicator equation

Reads see only frequencies `f_{c,i} = N_{c,i} / Σ_j N_{c,j}`. Differentiating:

```
df_{c,i}/dt = f_{c,i} · ( r_{c,i} − r̄_c(t) ),     r̄_c(t) = Σ_j f_{c,j} r_{c,j}
```

Adding a constant to **every** `r_i` leaves all frequencies unchanged. **This is
the crux: compositional data identifies only rate *differences*, i.e. relative
fitness — never absolute growth rate.**

### 2.3 The identifiable, fittable form

For any reference (we use the per-timepoint geometric mean = centered log-ratio,
CLR), the log-ratio is **linear in time**:

```
d/dt ln( f_{c,i} / f_ref ) = r_{c,i} − r_ref ≡ s_{c,i}        (selection coefficient)
⇒  CLR_{c,i}(t) = const + s_{c,i} · t
```

So **the growth rate of each variant is estimated as the slope of its
centered-log-ratio versus time.** When one variant dominates this reduces to the
logistic sweep `dg/dt = s·g(1−g)`, `logit(g)` linear in *t* — the observed sweep
shape.

### 2.4 Observation / noise model

Counts at each `(Sample, Transfer)` are Multinomial in the latent frequencies
(equivalently Poisson with `ln(depth)` as a fixed offset). Because depth is small
and 31 % of nonzero counts equal 1, sampling noise is large and is handled by
**weighting each CLR point by its count precision** `w = count + 0.5` in the
least-squares fit (down-weighting singletons and below-detection cells). Zeros are
treated as *left-censored* — frequency below ≈ 1/depth — not imputed to a value.
The fully rigorous upgrade is a hierarchical Dirichlet-Multinomial GLM (§6).

---

## 3. The time axis (an empirical correction)

The task described **1 transfer = 10 h**. The OD file's instrument timestamps
(File B, `series='exp2'`) instead show a **very steady ~26.4 h per transfer**
(see `outputs/intermediate/od_time_axis.csv`). The four sampled transfers fall at
≈ **53, 80, 133, 318 h** after T0, not 30/40/60/130 h.

This **only rescales rate units** — the sign, the ranking, and the per-cycle
coefficient are all unaffected. We therefore report three axes:

| Axis | Variable | Property |
|---|---|---|
| **per transfer-cycle** (primary) | `Transfer` index | gauge-robust; no time assumption |
| **per generation** | cumulative generations from the within-cycle OD sigmoid `Δg = log₂(plateau/inoculum)` ≈ 3.3 gen/cycle | canonical microbial-fitness unit |
| **per hour** | per-cycle ÷ cycle length | reported for **both** 26.4 h (measured) and 10 h (nominal) |

> Because barcodes are sampled at **end-of-cycle stationary phase** (OD pinned at
> carrying capacity), generations accrue only during the exponential portion of
> each cycle — which is why the per-*generation* axis (not calendar hours) is the
> defensible biological clock.

> **Per-hour caveat (corrected in `s08`/`s10`).** Selection acts only during the
> ~**7–10 h exponential phase** (`τ_exp`, measured from the within-cycle OD curve at
> T6/T13), not the full 26.4 h cycle. The correct per-hour conversion of a relative
> coefficient is `s / τ_exp`. The `s_per_hour_measured` / `s_per_hour_nominal`
> columns in `selection_coefficients_long.csv` are therefore **cycle-averaged** and
> understate exponential-phase rates by ~2.6× — treat them as cycle-level, and use
> `τ_exp` for true per-hour statements.

---

## 4. What is and is not identifiable

**Identifiable**
- Per-`(Sample, Candidate)` **selection coefficient** `s` = relative growth rate vs
  the community mean, for variants seen at ≥ 2 timepoints (237 of 460 pairs).
- The within-sample **fitness ranking** and **sweep-winner identity** (gauge-invariant).
- Reproducible **verA / verB marginal** selection coefficients, pooled across the
  11 cultures (the only signal that recurs — see §5).

**Not identifiable**
- **Absolute** Malthusian rates `r_i` — from this file *or* even with the OD file,
  because barcodes are sampled at stationary phase (OD = K, constant), so the OD
  multiplies frequencies by a per-well constant and adds no time-varying absolute
  information. Absolute rates would need exponential-phase sequencing or a
  spike-in / qPCR / CFU calibration.
- **True extinction vs sampling dropout** — both appear as zero reads.
- Any **per-Candidate carrying capacity** or time-varying fitness — only 4 uneven
  snapshots, with a ~185 h unobserved gap between T6 and T13.

---

## 5. Per-Candidate vs allele-level estimands

Per-Candidate coefficients are **noisy and largely irreproducible**: replicate
cultures fix on *different* full Candidates (the SpeI replicates' winners are
`A81-B8`, `A81-B174`, `A20-B151`, `A81-B151`, `A83-B42`), and 223/460 pairs are
seen at a single timepoint. The reproducible signal is at the **allele** level:

- The winning **verA is `A81` in 6 of 11 cultures**.
- But `A81`'s *marginal selection coefficient* (+0.020/cycle) is **inside the
  neutral-drift envelope** (drift p ≈ 0.64): it tends to win by **starting
  abundance / founder effect**, not a per-generation growth advantage.
- `A90` (+0.104/cycle, p < 0.001) and `A78` (+0.061, p = 0.01) show **robust,
  drift-confirmed advantages** despite rarely winning a sweep outright.

We therefore decompose `s_i ≈ a_{verA} + b_{verB} + interaction` and report the
verA/verB marginals (inverse-variance weighted across cultures) as the trustworthy
deliverable, each gated against a **neutral-drift null** (Multinomial resampling
at the observed depths from a constant frequency vector). This null also guards
against the **survivorship artifact**: at fixed depth a neutral survivor's
frequency rises merely because competitors drop below detection.

---

## 6. Estimation recipe (as implemented)

1. **`s01`** Restrict to the 11 four-transfer Samples; build the per-Sample variant
   union grid (absences = explicit zero cells).
2. **`s02`** Derive the time/generations axis from File B.
3. **`s03`** Per `(Sample, Candidate)`: CLR of counts, then weighted least-squares
   slope vs each time axis → selection coefficient `s`, with SE, R², monotonicity
   and a `quality` tier. Report only pairs with ≥ 2 observed timepoints.
4. **`s04`** Pivot to the **variant × Sample growth matrix** (the headline CSV);
   per-Sample summary with sweep winners.
5. **`s05`** Collapse to verA/verB marginals + neutral-drift null + robustness flag.
6. **`s06`/`s07`** Diagnostic + growth-matrix figures.
7. **`s08`–`s11`** Cross-sample integration and the OD absolute anchor — see §8.

**Recommended upgrade (not yet implemented — needs `numpyro`/`pymc`):** one
hierarchical Dirichlet-Multinomial GLM across all 11 Samples with partially-pooled
verA/verB main effects, a sparse verA×verB interaction, per-culture random
deviations, and **per-transfer-interval** coefficients (since ~76 % of all-4
trajectories are non-log-linear, a single global slope is only a first-order
summary — see the `quality`/`monotonic` flags).

---

## 7. Key caveats (read before using the numbers)

- **Relative, not absolute.** Every `s` is growth **relative to the community
  mean** in that culture. Do not compare `s` across cultures that share no
  variants without going through the allele level.
- **Per-hour numbers depend on the 26.4 h vs 10 h choice** (factor 2.64).
  Per-cycle and per-generation are unaffected.
- **Single-global-slope is approximate.** ~76 % of all-4 trajectories are
  non-monotone (the sweep mostly happens in the unobserved T6→T13 gap). Filter the
  matrix by `quality ∈ {high, good}` and `monotonic` for the cleanest estimates;
  treat `fair` cells (2 timepoints or low R²) as indicative only.
- **A positive slope can be survivorship, not selection.** Trust allele-level
  effects with `robust_selection = True`.

---

## 8. Cross-sample integration and absolute units (`s08`–`s11`)

To make a **variant-level statement across all samples**, and to ask whether growth
can be expressed in **OD/biomass per hour**, the pipeline adds four stages. The full
data-vs-identifiable-quantity map is in [`docs/EXPERIMENTAL_REQUIREMENTS.md`](EXPERIMENTAL_REQUIREMENTS.md).

**`s09` — cross-sample bridge model (relative scale).** Each sample's `s_{c,i}` is
gauged to its own community mean, so we tie the gauges together with a two-way model
`s_{c,i} = β_i + γ_c + ε`, fit by weighted alternating least squares with `Σγ_c = 0`.
`β_i` is the variant effect on **one common relative scale**; `γ_c` is the per-sample
gauge, pinned by **variants shared across samples (bridges)**. This is identifiable
because the bridge graph is **one connected component** (verified: 38 edges, 51
bridges). SEs are inflated by the observed overdispersion **φ = 2.53** (×1.59). The
lone degree-1 sample (`concX_largeLib_SpeI.3`) is flagged weakly anchored.
`s09b` shrinks the 117 single-sample variants toward additive verA+verB allele means
(empirical Bayes); the full crossed-random-effects version is the `numpyro` upgrade.

**`s08` — the one quantity OD legitimately supplies.** The within-cycle
exponential-phase slope `μ_bulk = d ln(OD)/dt = Σ_i f_i r_i = r̄` is the
**community-mean absolute rate** (~**0.3 h⁻¹**, doubling ~2 h; condition-dependent,
concX > concY). It also gives the exponential-phase clock `τ_exp ≈ 7–10 h`.

**Why per-variant absolute OD/h is NOT identifiable here (`s11` falsification).**
Three of four design approaches proposed `r_i = μ_bulk_c + s_{c,i}/τ_exp`. It fails
on this data:
1. **Go/no-go fails** — anchoring requires `corr(γ_c, μ_bulk_c) ≪ 0`; observed ≈ 0
   (panel a of `od_anchor_falsification.png`).
2. **μ_bulk can't discriminate samples** — across-sample SD ≲ within-sample noise; it
   is one shared constant + noise, not a per-sample gauge (panel b).
3. **No new information** — algebraically `s/τ_exp = s_per_generation × (μ_bulk/ln2)`,
   an existing column × a per-sample scalar; for the real alleles the correction is
   only **1.7–12 % of μ_bulk** (panel d). It is also circular for sweep winners.

**`s10` — graded deliverable.** Reports per-allele relative marginals (per cycle &
per generation, drift-gated) plus a clearly-flagged **derived** `R_allele_OD_clock`
= `μ_bulk + s/τ_exp` (interval, with assumption-flag booleans) — a unit relabel of
the relative effect, dominated by the shared community rate, **not** a per-variant
measurement.

**Bottom line.** What is genuinely absolute in OD/h is the **one community rate**.
Variant/allele results are **relative** (and reproducible at the allele level). True
per-variant absolute (biomass) rates need new data — within-exponential-phase
barcode sampling + an absolute abundance anchor (spike-in/qPCR/CFU), or monoculture
growth curves, plus an OD→biomass calibration — see `EXPERIMENTAL_REQUIREMENTS.md`.

---

## 9. Global static-rate optimization (`s12`/`s12b`)

A single global inverse problem that — under the explicit assumption that a
variant's biomass is its read-fraction × community OD — assigns **one mostly-static
growth rate `r_i` per variant, shared across all 11 samples**, and solves for the
combination that jointly reproduces every sample's relative counts **and** community
OD growth.

**Model (softmax replicator + OD-mean anchor).** Latent log-abundance
`η_{c,i,t} = logA_{c,i} + r_i·x_t` with `x_t` = cumulative exponential-phase hours
(`cumgen × ln2/μ_bar`); predicted frequency = softmax over a sample's variants;
predicted community rate `r̄_{c,t} = Σ_i f_{c,i,t} r_i`. Fit in **jax** (autodiff) +
scipy L-BFGS over ~314 global rates + ~460 per-(sample,variant) intercepts.
Loss = multinomial NLL (recreate counts) + `w_OD·(r̄−μ_bulk)²` (recreate OD growth)
+ ridge `w_r·(r_i−μ_bar)²` ("mostly static, minimally flexible") + data-anchored
`logA` conditioning.

**Result.** One static rate per variant reproduces the relative counts very well
(**freq pseudo-R² ≈ 0.95**); the community-rate match is looser (RMSE ≈ 0.15/h,
limited by noisy μ_bulk). Rates span ≈ **0.15–0.46 h⁻¹** around μ_bar ≈ 0.32 h⁻¹
(doubling ~2 h), fastest for the sweep-winner variants.

**Error in the rates (`s12b`), four layers:**
1. **Laplace SE** — inverse Hessian (Schur-complement over the `logA` nuisance);
   median ≈ 0.039/h.
2. **Bootstrap 95% CI** — resample multinomial counts at observed depths + jitter
   μ_bulk; median SE ≈ 0.020/h (agrees with Laplace to ~2×).
3. **LOSO heterogeneity** — leave-one-sample-out SD; median ≈ 0.010/h (small →
   the rates really are near-static across samples, validating the assumption).
4. **Weight sensitivity** — the absolute level shifts only ≈ 0.03/h as the OD
   anchor goes off→strong, and most rates move 0.02–0.05/h → the **relative
   structure is data-pinned but the absolute level is anchor-dependent** (the
   `identifiability` column flags data-pinned vs anchor/prior-driven variants).

**Honesty.** This delivers the requested per-variant absolute (OD/h) rates, but they
remain **community-dominated**: the relative ordering and spread are robust and
data-driven; the absolute level rides on the noisy community μ_bulk and the ridge
prior. Treat `r_global_per_h` as "community rate ± a data-driven relative offset,"
with the error columns quantifying both. Output: `global_variant_growth_rates.csv`,
figure `global_growth_fit.png`.

---

## 10. Joint WGS + amplicon fusion (`s13`/`s13b`/`s13c`)

The WGS experiment is uniformly **4MXB** (every focus well = `Methoxybenzoate 4 mM +
Kan`), the same environment as the amplicon 4MXB run, and WGS well **B4** is the same
physical culture as the amplicon B4. So both datasets measure each variant's **4MXB
growth rate**, and are fused into one fit: WGS supplies cross-well breadth (11 wells),
the amplicon supplies depth (~20× reads on B4).

Model = the `s12` softmax-replicator with **one global `r_i` per variant** across all
4MXB cells, a per-(physical-well, variant) `logA` gauge (WGS-B4 and amplicon-B4 share
one `logA_B4` block), and a single strongly-ridged assay-**differential** drift
`b_amp·x` (a constant assay offset cancels in the softmax). Counts enter at native
per-cell depth; the OD anchor is entered once per well; the 3 amplicon primer sets are
collapsed to one cell (freq corr 0.995–0.998).

**Error reduction** is measured by fitting the SAME model three ways (joint /
WGS-only / amplicon-B4-only) and reporting **per-variant** `se_reduction_ratio` and
`ESS_fold` (never a single global number), split into a sampling component (Laplace,
shrinks with depth) and a leave-one-well-out heterogeneity component (cross-culture,
must not shrink), gated by a WGS-B4-vs-amplicon-B4 same-culture concordance check.

**Result (honest):** `b_amp ≈ 0` (no systematic assay drift), concordance positive
(corr ≈ 0.42, noise-limited), cross-well heterogeneity small. The fusion **tightens
the 81 shared/deep-B4 variants modestly** (~12 % lower SE, ESS ~1.4×) and makes **131
variants newly estimable** from the deep amplicon, while leaving the 233 WGS-only
(not-in-B4) variants unchanged — i.e. it enhances coverage and modestly enhances
precision, not a dramatic global error drop (the amplicon's depth is concentrated:
174/212 B4 variants have <30 reads). Full narrative, decisions, and caveats in
[`docs/INTEGRATION.md`](INTEGRATION.md).

---

## 11. Complementary two-phase (breakpoint) regression (`s14`)

§2–§6 fit **one** slope per trajectory (a single selection coefficient `s`), and §9
fits **one** static rate per variant. But the README caveat #3 notes that ~76 % of
fully-sampled trajectories are **non-monotone** — a variant can sweep in fast and then
saturate, or rise then reverse, so a single slope is only a first-order summary.
`s14` offers a **complementary** view that lets the rate change once — and, by request,
**only in the increasing direction** (§11.3). It does **not** overwrite `s03` or `s12`;
it writes its own `segmented_growth_rates.csv`.

### 11.1 The four-parameter model

For each `(Sample, Candidate)` trajectory, the centered-log-ratio is fit as a
**continuous broken-stick** (hinge) in time:

```
CLR(t) = logA + r_init·(t − t0) + (r_final − r_init)·max(0, t − τ)
```

with **four** estimands (the user request):

| Symbol | Column | Meaning |
|---|---|---|
| `logA`     | `logA_init`            | **initial abundance** — fitted CLR (log relative abundance) at the first sampled transfer `t0` |
| `r_init`   | `r_init_per_cycle`     | **initial growth rate** — slope of the first segment |
| `r_final`  | `r_final_per_cycle`    | **final growth rate** — slope of the second segment |
| `τ`        | `breakpoint_transfer`  | **the inter-sample breakpoint** — the interior sampled transfer where `r_init` switches to `r_final` |

The segments join **continuously** at `τ` (no jump), so a variant that never changes
rate collapses exactly to the `s03` constant-slope line (`r_init == r_final`). Fit by
weighted least squares with the same Poisson-precision weights `w = count + ½` as
`s03`; on constant-model trajectories `s14`'s slope reproduces `s03`'s `s_per_cycle`
to rounding.

### 11.2 Why the breakpoint sits at a sampled transfer

With only 3–4 transfers per culture a **free** continuous breakpoint is not
identifiable. The estimable, biologically natural choice is a knot **at one of the
interior sampled transfers** — the points "between samples." `τ` is searched over
`TRANSFERS[1:-1]` (= {4, 6} for WGS) and the best-fitting knot is kept. The continuous
two-phase model has **3 linear parameters**, so a residual degree of freedom requires
**≥ 4 observed timepoints with ≥ 2 on each side of the knot**. Only such trajectories
are *breakpoint-testable* (`breakpoint_testable=True`); all others are reported with
the constant fit. In the WGS design that is exactly the **66 variants seen at all four
transfers**; the 3-sampled-transfer amplicon designs are never testable (correctly —
2 segments + intercept over 3 points has no residual d.o.f.).

### 11.3 When the rate is allowed to change (all gates must pass)

Default = constant. The rate switches to two-phase **only if all three** hold:

0. **Direction gate — increases only.** Only an **increase** in growth rate is
   permitted, `r_final > r_init`. A variant whose data prefer a growth-rate *decrease*
   is **not** allowed a two-phase model and keeps the constant fit. (Among the two
   candidate knots only increasing-rate fits are eligible; the would-be direction of
   the unconstrained best fit is still recorded in the `unconstrained_direction`
   column so the effect of this rule is auditable.) Note "increase" is on the growth
   *rate*: a still-negative rate that becomes less negative (e.g. −0.16 → −0.02) is an
   increase and is permitted.
1. **Fit gate.** The (increasing) two-phase fit must beat the constant fit on a
   small-sample **weighted BIC** by a *strong-evidence* margin,
   `ΔBIC = BIC_const − BIC_two > 6` (Kass–Raftery: 2–6 positive, 6–10 strong, >10 very
   strong). BIC already charges `k·ln n` for the extra parameter; the strong margin is
   deliberately conservative because at `n=4` the 3-parameter model has only 1 residual
   d.o.f. and is flexible. *(AICc is not usable here: at `n=4, k=3` its `n−k−1`
   correction divides by zero — itself a signal that a 4-point trajectory barely
   supports 3 parameters. BIC with a strong margin is the honest substitute.)* This
   encodes the requirement that the decision **depends on the fit before/after the
   change**, and that the rate **need not change if constant fits better**.
2. **Size gate.** The rate must **increase by more than 10 %**:
   `(r_final − r_init) / max(|r_init|, 0.05) > 0.10`. The floor only keeps the ratio
   finite when `r_init ≈ 0` (selection coefficients are centered near zero); the BIC
   gate does the real work of preventing trivial switches.

### 11.4 Result (WGS)

Of the 66 testable trajectories, **13 show strong evidence of an *increasing* two-phase
rate change** and are adopted; **47 are held at the constant fit because their best fit
was a decrease** (not permitted), and the remaining 6 keep the constant fit for want of
a strong-enough increasing fit. Where a switch is adopted the constant fit is genuinely
poor (median weighted R² ≈ 0.31 vs ≈ 0.96 two-phase), directly quantifying the "single
slope is only first-order" caveat. The adopted switches are **variants that were losing
ground early and then took off** — declining-then-surging trajectories that inflect
around T6 (e.g. `A78-B26`: `r −0.30 → +0.45/cyc` at T6; `A81-B151`: `−0.22 → +0.51`).
Tunable constants (`BIC_MARGIN`, `REL_RATE_CHANGE_MIN`, `MIN_OBS_FOR_BREAKPOINT`) sit at
the top of `s14`. Output: `segmented_growth_rates.csv`, figure
`segmented_growth_examples.png`.

**Caveat.** With 4 transfers the two-phase fit has only 1 residual d.o.f., so the exact
`τ`/`r_final` of any single variant is weakly determined; read `s14` as **which**
variants change rate and in **which direction**, not as a precise second-phase rate.
The recommended fuller upgrade remains a hierarchical per-interval Bayesian model (§6).

### 11.5 Communicating the results (`s14b` tables, `s14c` figures)

The segmented fit ships with the same table-plus-figure communication layer as the
single-slope assessment:

**Rollup tables (`s14b`).** `segmented_model_selection.csv` is the reproducible decision
audit (thresholds + classification counts + adopted-set medians);
`segmented_sample_summary.csv` gives per-culture acceleration counts, breakpoints, and
median phase rates; `segmented_allele_effects.csv` tallies acceleration per verA/verB
allele. The allele table surfaces a real cluster: **verB `B26` shows an adopted
increasing switch in 5/5 of its testable variants, and verA `A78` and `A81` in 4 each** —
the same alleles the single-slope assessment flags as robust, now seen to gain their
advantage as a *late acceleration* rather than a constant edge.

**Figure suite (`s14c`), built on the data-viz skill's validated CVD-safe palette:**
- `segmented_overview.png` — the model-selection story in four panels: the classification
  (estimable → testable → adopted/held/weak), the `r_init`-vs-`r_final` scatter with the
  `y=x` "increase-only" permit line (adopted points all sit above it), the two decision
  gates (ΔBIC and +Δrate, with adopted points in the pass quadrant), and constant-vs-
  two-phase fit quality (adopted points jump to high R²).
- `segmented_parameters.png` — the four fitted parameters: a **dumbbell** of `r_init →
  r_final` per adopted variant (the headline before/after), the breakpoint tally (T4 vs
  T6), initial abundance vs acceleration, and the normalized fitted **acceleration shapes**
  (the characteristic dip-then-rise, deeper for T6 switches).
- `segmented_gallery.png` — every adopted two-phase fit as small multiples (observed CLR,
  constant vs two-phase line, breakpoint), so each result is individually inspectable.
- `segmented_allele.png` — per-allele acceleration tallies for verA and verB.

**All datasets.** The tables and `segmented_overview.png` are produced for every dataset.
On the 3-transfer amplicon designs nothing is breakpoint-testable, so their overview is
the honest reduced form — a single "all constant" bar plus the single-slope rate
distribution — and the parameter/gallery/allele figures are skipped (there is nothing to
show). Only the 4-transfer WGS run yields the full suite.
