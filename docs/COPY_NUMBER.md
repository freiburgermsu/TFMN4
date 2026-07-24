# dgoA copy-number amplification — growth rates and the 1-var vs 4-var models

`copy-number-dgoA_.csv` is a **new, structurally different** dataset from the verAB
barcode competition: the **copy number of the `dgoA*` gene region** measured per
**lineage** across serial-transfer timepoints (amplicon-derived). It is a **direct
measurement**, not compositional read counts, which changes the modelling fundamentally:

> Barcode counts are compositional, so only *relative* growth rates are identifiable
> (CLR slopes). Copy number is absolute, so each lineage's **absolute amplification rate
> and its standard error are directly identifiable** — the "predicted growth rate + error
> per variant" quantity, with a real, data-driven error.

Pipeline: `scripts/cn01_prepare.py … cn05_figures.py`, run by `scripts/cn_run_all.py`;
outputs in `outputs_copynumber/`. **The headline numbers below were hardened by an
adversarial verification pass** (independent recomputation + statistical review) that
caught and fixed pseudoreplication, low-leverage, multiplicity, and labeling issues — see
§ *What verification changed*.

## The data

- **91 lineages** (+1 ancestral stock, dgoA copy_number ≈ 0.53), across **4 condition
  contexts**: `TFMN1 single-KO` (12: fba/pgi/sohB/tpiA), `TFMN1 double-KO` (17: gene
  pairs), `TFMN4 exp1` (55), `control (noDNA)` (7).
- Transfers span 1–32; each lineage has its own sampled transfers; **13 lineages have
  technical replicate measurements** at a timepoint (sample suffixes `.P/.S1/.S2/.L1/.L2`
  = re-measurements of the same population).
- **73 lineages estimable** (≥3 transfers); **24 CI-reliable** (≥4 transfers, dof≥2).

## The 1-variable model — the headline growth rates (`cn02`)

OLS of **ln(copy_number) on transfer**, fit on the **per-transfer means** (the unit of
replication — see below), giving `r` = amplification rate/transfer with its regression
**SE**, t(dof) **95% CI**, **p-value**, R², and per-transfer fold change `e^r`. Output:
`dgoA_growth_rates.csv`; figure `dgoA_growth_rates_forest.png`.

**Three honesty layers** (from the review):
- **Unit of replication** — technical replicates are collapsed to the per-transfer mean.
  Treating them as independent (the naive fit) makes SEs ~24 % and CI half-widths ~42 %
  too small, and manufactures false significance.
- **Low leverage** — a calibrated CI needs ≥4 distinct transfers (dof≥2). The 55 `exp1`
  lineages are sampled only at {8, 9, 30} (8,9 nearly coincident → dof=1, t≈12.7); they
  are rate-estimable but their CIs are **not calibrated** and are shown greyed, with no
  significance call.
- **Multiplicity** — significance (`amplifying`/`losing`) requires a **Benjamini-Hochberg
  FDR q < 0.05** among the CI-reliable lineages, not a raw per-lineage CI-excludes-0 test.

**Result (honest).** Rates span ≈ −0.02 to +0.14 /transfer; median SE ≈ 0.007. After
honest replicate handling, low-leverage flagging, and FDR correction, **exactly one
lineage is a robust dgoA amplifier**:

> **`TFMN1.pgi.1`: r = +0.071 ± 0.011 /transfer (×1.07/transfer, 95 % CI [0.043, 0.098],
> q = 0.019, R² 0.90, 8 transfers)** — copy number rises ~0.7 → ~10 over the experiment.

11 lineages have a *raw* CI that excludes 0, but the other 10 do **not** survive honest
replicate handling + FDR (they are pseudoreplication, dof=1 low-leverage, or multiplicity
artifacts — e.g. `pgi.2` flips to non-significant once its replicates are collapsed). dgoA
amplification is thus **real but rare** at this evidentiary bar.

> **Units.** Rates are **per transfer** (the gauge-robust clock). A per-hour conversion
> would need lineage-specific cycle lengths (the TFMN1 lineages are a different experiment
> from the OD-timed WGS), so per-transfer is the honest primary unit.

## The 4-variable model (`cn03`) and the comparison (`cn04`, `cn05`)

The 4-variable **increasing-only broken-stick** (identical policy to the barcode `s14`:
`ΔBIC>6` AND >10% rate increase AND `r_final>r_init`), on per-transfer-mean ln(CN), where
≥4 transfers allow it (24 lineages).

- **Coverage.** The *deployed* increasing-only 4-var equals the 1-var on **70/73** and
  refines only **3** lineages.
- **Copy number SATURATES.** Of 24 testable lineages, **20 are *decelerating*** — they
  amplify then plateau/decline (classic gene-amplification saturation), a *decrease* the
  increasing-only rule is not permitted to fit. A **bidirectional broken-stick FORM**
  captures it: in-sample median R² **0.40 → 0.93** (`fba.2`, `pgi.tpiA.1` rise to a peak
  then fall; the straight line reads them as ~flat).
- **Out-of-sample: a *modest* edge.** On leave-one-transfer-out CV the broken-stick FORM
  beats the line on **15/24** lineages, median RMSE **0.21 → 0.16** — but this is **not
  statistically significant** (Wilcoxon p ≈ 0.08). Unlike the barcode n=4 case (where the
  in-sample gain was pure optimism and out-of-sample was parity), here the ≥4-transfer
  designs give the FORM a real but weak predictive edge.

**Bottom line.** The per-variant *growth rate* is the 1-variable amplification rate ± SE
(directly identifiable; `pgi.1` the one robust amplifier). The deployed increasing-only
4-var rarely fires because dgoA copy number predominantly **saturates** — a two-phase
shape the increasing-only constraint blocks. A *bidirectional* breakpoint (same four
parameters, constraint relaxed) is the better description of the amplify-then-plateau
dynamics and would be the recommended 4-var variant for this dataset.

## What verification changed

An adversarial workflow independently recomputed every rate/SE/CI (statsmodels + hand-rolled
normal equations — **the core math reproduced exactly**) and stress-tested the statistics.
It corrected, before publication:
1. **Pseudoreplication** → fit on per-transfer means (was: all measurements independent).
2. **dof=1 low-leverage** `exp1` fits → tiered `low-leverage`, no significance calls.
3. **Uncontrolled multiplicity** → BH-FDR calls (raw 11 "significant" → **1** robust).
4. **Family mislabeling** → double-knockouts and no-DNA controls were being merged into
   single-KO / exp1; now parsed into a precise `genotype` + a coarse `context`.
5. **Deployed-vs-FORM conflation** in the comparison figures/narrative → now kept distinct,
   and the out-of-sample gain is reported as modest / non-significant.

## Caveats

1. `exp1`'s 3-timepoint {8,9,30} design cannot support calibrated rate CIs — treat its
   point rates as indicative only.
2. FDR is computed among the 24 CI-reliable lineages (the interpretable set); widening the
   test family only makes calls more conservative — `pgi.1` survives either way.
3. `r` is the rate of *dgoA copy-number* change (a specific adaptive readout), not
   whole-cell growth rate.
