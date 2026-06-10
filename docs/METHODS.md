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
