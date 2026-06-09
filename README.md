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
~/Documents/py_venv/bin/python scripts/run_all.py
```

Reads the two CSVs in the repo root and writes everything under `outputs/`.

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
  intermediate/        tidy data, depth, OD time axis, OD bulk rate (mu_bulk)
  figures/             richness, sweeps, growth heatmap, OD-anchor falsification
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
