# TFMN4 — variant growth rates from barcode competition counts

Estimate the **growth rate of each enzyme variant (verA–verB pair) in each culture**
from serial-transfer barcode sequencing counts, by fitting an ODE-based
relative-fitness model.

**First assessment scope:** the **11 Samples observed at all four transfers
{3, 4, 6, 13}** (908 rows, 314 distinct variants).

> The full model derivation, identifiability analysis, and caveats are in
> [`docs/METHODS.md`](docs/METHODS.md). Output column definitions are in
> [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

---

## The one-paragraph model

Sequencing counts are **compositional** (≈ fixed depth per timepoint), so they
measure *relative* frequencies, not absolute cell numbers. The latent ODE
`dN_i/dt = r_i·N_i` therefore collapses (via the replicator equation) to something
where only **rate differences** are identifiable. The estimable quantity is each
variant's **selection coefficient** `s` — the slope of its centered-log-ratio
(CLR) of counts versus time, i.e. its growth rate **relative to the community
mean**. Positive `s` ⇒ sweeps in; negative ⇒ sweeps out.

---

## Quick start

```bash
# Uses the project venv (see ~/.claude/CLAUDE.md)
~/Documents/py_venv/bin/python scripts/run_all.py            # WGS  -> outputs/

# Same pipeline on the deep verAB amplicon data (Plasmidsaurus), per carbon source:
TFMN4_DATASET=amplicon_4MXB     ~/Documents/py_venv/bin/python scripts/run_all.py   # -> outputs_amplicon_4MXB/
TFMN4_DATASET=amplicon_pyruvate ~/Documents/py_venv/bin/python scripts/run_all.py   # -> outputs_amplicon_pyruvate/
```

The default (no env var) is the original WGS run → `outputs/`. The `TFMN4_DATASET`
switch (see `config.py`) reuses the identical analysis code on the amplicon data,
writing to **separate** directories so nothing is overwritten. The amplicon design,
how it maps onto the pipeline, and its results are documented in
[`docs/AMPLICON.md`](docs/AMPLICON.md) — headline: one static rate per variant
reproduces the relative counts at **freq R² ≈ 0.995**, and **pyruvate grows ~3×
faster than 4-methoxybenzoate** at the community level (μ_bulk 0.83 vs 0.30 /h).

```bash
# Joint WGS+amplicon fusion for lower-error 4MXB per-variant rates -> outputs_joint_4MXB/
~/Documents/py_venv/bin/python scripts/s13_joint_4MXB.py
~/Documents/py_venv/bin/python scripts/s13b_joint_error.py
~/Documents/py_venv/bin/python scripts/s13c_error_reduction_fig.py
```

The WGS experiment is uniformly 4MXB, so it is fused with the amplicon 4MXB run (WGS
breadth + amplicon depth; WGS well B4 = the amplicon B4 culture) into one estimate of
each variant's 4MXB growth rate — see [`docs/INTEGRATION.md`](docs/INTEGRATION.md).
**Honest headline:** the fusion is valid (no systematic assay drift; positive
same-culture concordance) and **tightens the 81 shared/deep-B4 variants modestly
(~12 % lower SE, ESS ~1.4×) while making 131 variants newly estimable** from the deep
amplicon — an enhancement of *coverage* plus modest precision, not a dramatic global
error drop (the amplicon's depth is concentrated: 174/212 B4 variants have <30 reads).

## Repository layout

```
scripts/
  config.py            shared paths, constants, CLR + weighted-fit helpers
  s01_prepare_data.py  subset to 11 four-transfer Samples; tidy + depth tables
  s02_time_axis.py     transfer timing + generations/cycle from the OD file
  s03_estimate_growth.py   CLR weighted-slope -> per-(Sample,Candidate) selection coeff
  s04_growth_matrix.py     pivot to the variant x Sample growth matrix + summary
  s05_allele_effects.py    verA/verB marginals + neutral-drift null
  s06_figures.py           diagnostic plots
  s07_growth_matrix_figure.py   growth-matrix heatmap + distribution
  s08_od_bulk_rate.py      community mu_bulk + tau_exp from the OD curves
  s09_bridge_meta.py       cross-sample two-way bridge model (common relative scale)
  s09b_partial_pooling.py  EB shrinkage of variant effects toward allele means
  s10_allele_effects_graded.py  graded allele table + flagged OD-clock rate
  s11_falsification_figures.py  OD-anchor go/no-go + bridge graph + forest
  s12_global_growth_fit.py      global static per-variant rate (jax; counts + OD)
  s12b_growth_rate_error.py     Laplace + bootstrap + LOSO + sensitivity error
  s14_segmented_growth.py       COMPLEMENTARY two-phase (breakpoint) per-variant fit
  s14b_segmented_summary.py     segmented rollups (model-selection audit, sample, allele)
  s14c_segmented_figures.py     segmented figure suite (overview, params, gallery, allele)
  run_all.py               run the whole pipeline in order
docs/
  METHODS.md                model derivation, identifiability, cross-sample integration
  DATA_DICTIONARY.md        every output file + column
  EXPERIMENTAL_REQUIREMENTS.md   what data gives which growth-rate quantity (+ absolute)
outputs/
  growth_matrix_selection_per_cycle.csv        <-- HEADLINE: variants x samples x growth
  growth_matrix_selection_per_generation.csv
  selection_coefficients_long.csv              full per-(Sample,Candidate) table + flags
  sample_summary.csv                           per-Sample winners, richness, ranges
  allele_effects_verA.csv / allele_effects_verB.csv
  variant_bridged_relative.csv                 cross-sample variant effects (one scale)
  variant_pooled.csv                           partial-pooled variant effects
  sample_gauge.csv                             per-sample gauge gamma_c + anchor flags
  allele_growth_advantage.csv                  graded allele table (+ derived OD-clock)
  global_variant_growth_rates.csv              global static per-variant rate + error (s12/s12b)
  segmented_growth_rates.csv                   complementary two-phase breakpoint fit (s14)
  segmented_model_selection.csv                segmented decision audit + counts (s14b)
  segmented_sample_summary.csv                 per-culture acceleration rollup (s14b)
  segmented_allele_effects.csv                 per-allele acceleration rollup (s14b)
  intermediate/        tidy data, depth, OD time axis, OD bulk rate (mu_bulk)
  figures/             richness, sweeps, growth heatmap, OD-anchor falsification, global-growth fit,
                       segmented_{overview,parameters,gallery,allele}.png (s14c two-phase figures)
```

## The headline output

`outputs/growth_matrix_selection_per_cycle.csv` — **rows = variants (Candidate),
columns = Samples, cells = selection coefficient `s` per transfer-cycle.** A blank
cell = the variant is absent from that culture; the cell is omitted (NaN) when the
variant is seen at only one timepoint (not estimable). Read it alongside the
`quality` / `monotonic` flags in `selection_coefficients_long.csv`.

`s` is in **per-transfer-cycle** units (gauge-robust). Convert to per generation
(≈ ÷3.3) via the per-generation matrix, or to per hour via ÷26.4 h (measured) or
÷10 h (nominal) — see the time-axis note below.

## Key findings (first assessment)

- A clear **selective sweep** in every culture: richness collapses (e.g.
  48→27→17→9 variants) at fixed depth.
- Selection coefficients span roughly **−0.37 to +0.67 per cycle**; clean sweeps
  (e.g. `A83-B40`: freq 0.008→0.59, R²≈0.98) sit at the top.
- **verA `A81` wins 6/11 cultures** — but its marginal advantage is **within the
  neutral-drift envelope** (it wins by starting abundance / founder effect).
- **verA `A90` (+0.104/cycle) and `A78` (+0.061/cycle)** are the **robust,
  drift-confirmed** growth advantages, despite rarely winning a sweep outright.

### Cross-sample integration + absolute rates (`s08`–`s11`)
- The 11 samples form **one connected bridge graph**, so variant effects integrate
  onto a **common relative scale** (`variant_bridged_relative.csv`).
- The OD growth curves give **one absolute number**: the community bulk rate
  **μ_bulk ≈ 0.3 h⁻¹** (doubling ~2 h; concX > concY).
- **Absolute per-variant OD/h is not identifiable** — the OD-anchor fails its go/no-go
  test (`corr(γ_c, μ_bulk) ≈ 0`) and adds no per-variant information; the allele
  "advantage" is only 2–12 % of μ_bulk. See `od_anchor_falsification.png` and
  **`docs/EXPERIMENTAL_REQUIREMENTS.md`** for what data would unlock true biomass/h.

### Global static-rate fit (`s12`/`s12b`)
- A single global optimization assigns each variant **one static rate** (shared across
  all samples) that jointly reproduces the relative counts (**freq R²≈0.95**) and the
  community OD; rates span **0.15–0.46 h⁻¹** (doubling 1.5–4.5 h) around μ_bar≈0.32.
- Rates come **with error** (Laplace SE, bootstrap CI, leave-one-sample-out
  heterogeneity, OD-weight sensitivity). Cross-sample heterogeneity is small (~0.01/h
  → rates really are near-static), but the **absolute level is anchor-dependent**
  (most rates flagged `mixed`/`anchor-driven`) — the relative ordering is what's
  robust. See `global_growth_fit.png` and METHODS §9.

### Complementary two-phase / breakpoint fit (`s14`)
`s03` and `s12` fit **one** rate; `s14` is a **complementary** regression (it does not
replace them) that lets the rate change **once**, fitting **four** parameters per
variant: **initial abundance** (intercept), **initial growth rate**, **final growth
rate**, and the **inter-sample transfer at which the rate switches** (a continuous
broken-stick with the knot at an interior sampled transfer). **Only an *increase* in
growth rate is permitted** (`r_final > r_init`); a variant whose data prefer a decrease
keeps the constant fit. It adopts the change only when that increasing fit genuinely
improves the fit (weighted `ΔBIC>6`, strong evidence) **and** the rate increases by
**>10%**; otherwise it keeps the constant fit — so the rate does not have to change. Of
the 66 WGS trajectories seen at all four transfers, **13 show a strong *increasing*
two-phase change** (variants that lose ground early then surge, inflecting near T6),
**47 are held constant because their best fit was a decrease** (not permitted), and the
rest stay constant. Where adopted, the constant fit is genuinely poor (median weighted
R² ≈ 0.31 → ≈ 0.96 two-phase; caveat #3 below). Acceleration **clusters by allele** —
verB `B26` accelerates in 5/5 of its testable variants, verA `A78`/`A81` in 4 each.

This ships with a full table+figure communication layer (parallel to the single-variable
assessment): rollup tables `segmented_{model_selection,sample_summary,allele_effects}.csv`
(`s14b`) and a figure suite `segmented_{overview,parameters,gallery,allele}.png` (`s14c`,
on the data-viz skill's CVD-safe palette) — model-selection + the increasing-only
constraint, the four fitted parameters (incl. an `r_init`→`r_final` dumbbell), every
adopted fit as small multiples, and per-allele tallies. The tables + overview are emitted
for **all datasets**; the two 3-transfer amplicon runs are not breakpoint-testable (a
two-phase model needs ≥4 timepoints) so they report all-constant with a reduced overview.
See METHODS §11.

## ⚠️ Three things to know before using the numbers

1. **Relative, not absolute.** Each `s` is growth relative to the community mean of
   *its own culture*. Absolute per-variant growth rates are **not identifiable**
   from these data (compositional counts; OD is at stationary plateau when
   sampled). See METHODS §4.
2. **The time axis is ~26.4 h/transfer, not 10 h.** The OD instrument timestamps
   are unambiguous. This only rescales per-*hour* numbers (by 2.64×); per-cycle and
   per-generation are unaffected. **Tell me which convention you want for any
   per-hour reporting.**
3. **A single slope is a first-order summary.** ~76 % of fully-sampled trajectories
   are non-monotone (most of the sweep is in the unobserved T6→T13 gap). Filter to
   `quality ∈ {high, good}` for the cleanest estimates; the recommended upgrade is a
   hierarchical, per-interval Bayesian fit (METHODS §6).
