# Data dictionary — `outputs/`

Units: `s_per_cycle` = selection coefficient per **transfer-cycle** (primary,
gauge-robust); `s_per_generation` uses the OD-derived generations axis (≈3.3
gen/cycle); `s_per_hour_*` divide the per-cycle value by the cycle length
(`measured` = 26.4 h, `nominal` = 10 h). All `s` are **relative to the community
mean** of the culture (replicator gauge), so within a Sample they roughly center
on zero.

---

## `growth_matrix_selection_per_cycle.csv` — HEADLINE

Matrix: **rows = `Candidate` (variant), columns = `Sample`, cells = `s_per_cycle`.**
- Blank/empty cell → variant absent from that culture.
- `NaN` → present at only one timepoint (not estimable).
- Rows ordered by breadth (number of cultures with an estimate), then mean `s`.
- Column names have the `TFMN4.exp2.ACN3788.` prefix stripped.

## `growth_matrix_selection_per_generation.csv`
Same shape, cells = `s_per_generation`.

## `selection_coefficients_long.csv` — full per-(Sample, Candidate) table
| Column | Meaning |
|---|---|
| `Sample`, `DNA_construct`, `Replicate` | culture identifiers |
| `Candidate`, `verA`, `verB` | variant and its two enzyme parts |
| `n_obs_timepoints` | # transfers with count > 0 (1–4) |
| `total_reads`, `max_count` | summed / peak read count across transfers |
| `transfers_observed` | which of {3,4,6,13} had count > 0 |
| `freq_T3`, `freq_T13` | within-Sample frequency (count/depth) at first/last transfer |
| `s_per_cycle`, `se_per_cycle`, `r2_per_cycle` | selection coeff, its SE, weighted R² of the slope fit |
| `s_per_generation`, `se_per_generation` | same on the OD generations axis |
| `s_per_hour_measured`, `s_per_hour_nominal` | per-hour at 26.4 h / 10 h per cycle |
| `monotonic` | log-frequency moves one direction across observed timepoints |
| `quality` | `high` (4 pts, R²≥0.8) · `good` (≥3 pts, R²≥0.6) · `fair` (≥2 pts) · `none` |
| `estimable` | `True` if `n_obs_timepoints ≥ 2` |
| `sweep_winner` | most abundant variant in the Sample at T13 |

## `sample_summary.csv` — one row per culture
| Column | Meaning |
|---|---|
| `n_candidates`, `n_estimable` | variants total / with a fitted slope |
| `sweep_winner`, `winner_verA`, `winner_freq_T13` | dominant Candidate at T13, its verA, its frequency |
| `top_s_per_cycle`, `bottom_s_per_cycle` | fastest / slowest estimable variant |
| `n_variants_T3…T13` | distinct variants per transfer (richness collapse) |

## `allele_effects_verA.csv`, `allele_effects_verB.csv` — reproducible signal
| Column | Meaning |
|---|---|
| `allele` | verA (or verB) identifier |
| `marginal_s_per_cycle`, `se` | inverse-variance weighted mean selection coeff across cultures |
| `n_pairs` | # estimable (Sample, Candidate) pairs carrying this allele |
| `n_samples` | # cultures the allele appears in |
| `sweep_wins` | # cultures whose T13 winner carries this allele (verA file) |
| `drift_null_p` | two-sided p vs the neutral-drift (Multinomial resampling) null |
| `null_lo`, `null_hi` | 2.5/97.5 percentile of the null marginal |
| `exceeds_drift_null` | `drift_null_p < 0.05` |
| `robust_selection` | exceeds null **and** `n_pairs ≥ 5` **and** `n_samples ≥ 3` (trust these) |

---

## Cross-sample integration + OD anchor (`s08`–`s11`; see METHODS §8)

### `variant_bridged_relative.csv` — per-variant effect on one common scale
| Column | Meaning |
|---|---|
| `Candidate`, `verA`, `verB` | variant and parts |
| `n_samples` | cultures the variant is estimable in |
| `bridged` | appears in ≥2 samples (contributes to the gauge) |
| `beta_per_cycle` | cross-sample variant effect (common relative scale), per cycle |
| `beta_se` | SE, **overdispersion-inflated** by √φ (φ=2.53) |
| `beta_se_raw` | sampling SE (φ=1), used for the EB pooling in `s09b` |
| `beta_per_generation` | `beta_per_cycle` ÷ (generations/cycle ≈ 3.46) |
| `I2_heterogeneity` | across-sample heterogeneity (multi-sample variants) |
| `samples` | which cultures contributed |

### `sample_gauge.csv` — per-sample gauge from the bridge model
`Sample`, `gamma_c` (per-sample offset, Σ=0), `bridge_degree` (# samples sharing a
variant), `n_bridge_variants`, `weakly_anchored` (degree ≤ 1).

### `variant_pooled.csv` — partial-pooled variant effects (`s09b`)
Adds `allele_pred_per_cycle` (additive verA+verB prediction), `beta_pooled_per_cycle`
(EB blend), `shrinkage_lambda` (1 = keep own estimate, 0 = full pool to allele mean).

### `allele_growth_advantage.csv` — graded allele table (`s10`)
Per allele (verA & verB): `marginal_s_per_cycle`, `marginal_s_per_generation`, `se`,
`n_pairs`, `n_samples`, `drift_null_p`, `exceeds_drift_null`, `robust_selection`,
then the **derived, flagged** OD-clock columns `R_allele_OD_clock_per_h`,
`R_allele_lo/hi` (interval over the community μ_bulk), `correction_frac_of_mubulk`,
and assumption booleans `flag_anchor_is_community_level`,
`flag_correction_is_few_pct_of_mubulk`, `flag_OD_not_biomass`, `flag_derived_not_measured`.
**These OD/h numbers are community-dominated derivations, not per-variant measurements.**

### `figures/`
`growth_matrix_overview.png` (variant×sample heatmap + distribution),
`od_anchor_falsification.png` (go/no-go, μ_bulk vs transfer, bridge graph, allele forest).

---

## `intermediate/`
- **`barcode_4transfer_long.csv`** — tidy rows: `Sample, DNA_construct, Replicate,
  Microtiter_plate_well, Transfer, verA, verB, Candidate, Count, depth, freq`
  (`depth` = total reads in that Sample×Transfer; `freq` = Count/depth).
- **`depth_by_sample_transfer.csv`** — sequencing depth per Sample×Transfer (the
  multinomial exposure, **not** a population size).
- **`sample_well_map.csv`** — `Sample → Microtiter_plate_well` (+ construct,
  replicate); used to join the OD file.
- **`od_time_axis.csv`** — per transfer: `hours_from_T0` (instrument clock),
  `delta_g` (generations that cycle), `cumgen` (cumulative generations),
  `is_sampled_transfer`, `measured_hours_per_transfer` (≈26.4).
- **`od_bulk_rate.csv`** — per `(Sample, transfer)`: `mu_bulk_per_h` (within-cycle
  max exponential-phase specific growth rate = community r̄), `mu_se`, `fit_r2`,
  `plateau_K`, `inoculum_net`, `n_band_readings`, `doubling_h`, `tau_exp_h`,
  `reliable` (T6/T13 reliable; T3/T4 mostly not).
- **`od_bulk_condition_summary.csv`** — community μ_bulk by construct family
  (concX vs concY) at reliable transfers — **the one genuinely absolute OD/h rate**.
