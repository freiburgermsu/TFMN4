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

### `global_variant_growth_rates.csv` — global static-rate fit (`s12`/`s12b`; METHODS §9)
Per variant, one mostly-static growth rate shared across all samples, fit to jointly
reproduce relative counts + community OD, **with error**:
| Column | Meaning |
|---|---|
| `Candidate`, `verA`, `verB`, `n_samples` | variant, parts, # cultures present |
| `has_slope_info` | observed at ≥2 timepoints in ≥1 sample (else rate is prior-driven) |
| `r_global_per_h` | global static growth rate (per hour; community-dominated absolute) |
| `rel_advantage_per_h` | `r_global − μ_bar` (the data-driven relative offset) |
| `doubling_h` | ln2 / r_global |
| `se_laplace` | inverse-Hessian SE |
| `se_boot`, `ci_boot_lo/hi` | bootstrap SE + 95% CI (counts + μ_bulk resampling) |
| `loso_heterogeneity_sd` | leave-one-sample-out SD ("is the rate static?") |
| `sens_to_OD_weight` | \|Δr\| as the OD anchor goes 0→20 |
| `identifiability` | `data-pinned` / `mixed` / `anchor-prior-driven` |

### `segmented_growth_rates.csv` — complementary two-phase breakpoint fit (`s14`; METHODS §11)
One row per estimable `(Sample, Candidate)` trajectory. A **continuous broken-stick**
CLR fit that allows the growth rate to change **once**, at an interior sampled transfer.
Does **not** replace `s03`/`s12` — offered alongside them. **Only an INCREASE in rate is
permitted** (`r_final > r_init`); a variant whose data prefer a decrease keeps the
constant fit. Switches to two-phase only if that increasing fit improves on a
strong-evidence weighted BIC (`ΔBIC>6`) **and** the rate increases by `>10%`; otherwise
it reports the constant fit (then `r_init == r_final`, blank breakpoint). The **four
requested parameters** are the first four rows below.
| Column | Meaning |
|---|---|
| `logA_init` | **initial abundance**: fitted CLR (log rel. abundance) at the first sampled transfer |
| `r_init_per_cycle` | **initial growth rate**: slope of the first segment (per transfer-cycle) |
| `r_final_per_cycle` | **final growth rate**: slope of the second segment (= `r_init` if constant; else `> r_init`) |
| `breakpoint_transfer` | **inter-sample breakpoint**: transfer where `r_init`→`r_final` (blank if constant) |
| `model_selected` | `two_phase` or `constant` (the chosen model) |
| `direction` | `accelerating` or `constant` (only increases are adopted) |
| `unconstrained_direction` | direction the best fit would take if decreases were allowed (`accelerating`/`decelerating`/`none`); a `constant` row with `decelerating` here was held constant by the increasing-only rule |
| `breakpoint_testable` | ≥4 observed timepoints with ≥2 per segment (else constant is forced) |
| `delta_bic` | `BIC_const − BIC_two_phase` (>6 = strong evidence for the switch; blank if untestable) |
| `rel_rate_change`, `abs_rate_change` | \|Δrate\| relative to max(\|r_init\|, 0.05), and absolute |
| `r2_selected`, `r2_constant`, `r2_two_phase` | weighted R² of the chosen / constant / best two-phase fit |
| `wrss_constant`, `wrss_two_phase` | weighted residual sum of squares (the BIC inputs) |
| `s_constant_per_cycle` | the single-slope fit (matches `s03`'s `s_per_cycle`; sanity cross-check) |
| `r_init/r_final_per_generation` | the selected model's slopes on the cumulative-generation axis |
| `r_init/r_final_per_hour_measured` | per-cycle slopes ÷ 26.4 h (measured cycle length) |
| `n_obs_timepoints`, `transfers_observed`, `total_reads`, `freq_init`, `freq_final` | trajectory context |

### segmented rollups (`s14b`; METHODS §11.5)
Three communication tables aggregating `segmented_growth_rates.csv`:

- **`segmented_model_selection.csv`** — one `metric,value` row per audit item: the
  thresholds used (`bic_margin_strong_evidence`, `rel_rate_change_min`,
  `min_obs_for_breakpoint`, `increasing_only`, …) and the classification counts
  (`n_estimable_trajectories`, `n_breakpoint_testable`, `n_two_phase_adopted_increasing`,
  `n_held_constant_decrease_not_permitted`, `n_weak_constant`) plus the adopted-set
  medians (ΔBIC, R² constant vs two-phase, acceleration, `r_init`/`r_final`). The
  reproducible decision summary for the dataset.
- **`segmented_sample_summary.csv`** — one row per culture: `n_estimable`, `n_testable`,
  `n_accelerating` (adopted), `n_held_decrease`, `frac_accelerating`, breakpoint tally
  (`bp_T4`, `bp_T6`), and the median adopted `r_init`/`r_final`/acceleration + median
  `logA_init`. Which cultures show late speed-ups.
- **`segmented_allele_effects.csv`** — stacked per `level` (`verA`/`verB`), one row per
  allele: `n_variants`, `n_testable`, `n_accelerating`, `frac_of_testable_accelerating`,
  `mean_acceleration`, `mean_r_init_adopted`, `mean_r_final_adopted`, `modal_breakpoint`.
  Whether acceleration clusters in particular enzyme parts (e.g. verB `B26` accelerates
  in 5/5 testable variants; verA `A78`/`A81` in 4 each).

### `figures/`
`growth_matrix_overview.png` (variant×sample heatmap + distribution),
`od_anchor_falsification.png` (go/no-go, μ_bulk vs transfer, bridge graph, allele forest),
`global_growth_fit.png` (Laplace-vs-bootstrap SE, rate forest±CI, LOSO heterogeneity, OD-weight sensitivity),
`segmented_growth_examples.png` (`s14` broken-stick fits for the strongest rate switches: observed CLR, constant vs two-phase line, breakpoint).

Segmented figure suite (`s14c`; METHODS §11.5):
`segmented_overview.png` (model selection + the increasing-only constraint: classification bars, `r_init` vs `r_final` with the y=x permit line, the two decision gates, constant-vs-two-phase fit quality),
`segmented_parameters.png` (the four fitted parameters: `r_init`→`r_final` dumbbell per adopted variant, breakpoint tally, initial abundance vs acceleration, normalized acceleration shapes),
`segmented_gallery.png` (every adopted two-phase fit as small multiples),
`segmented_allele.png` (per-verA / per-verB acceleration tallies).
On 3-transfer datasets (amplicon), only a reduced `segmented_overview.png` is drawn (nothing is testable → the constant-slope distribution).

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

---

## Joint WGS+amplicon fusion (`s13`/`s13b`; see METHODS §10, docs/INTEGRATION.md)

### `outputs_joint_4MXB/joint_variant_growth_rates_4MXB.csv`
Per variant, the fused 4MXB growth rate with a full error/identifiability accounting:
| Column | Meaning |
|---|---|
| `Candidate`, `verA`, `verB` | variant and parts |
| `r_joint_per_h` | joint (WGS+amplicon) global 4MXB growth rate (per hour; community-dominated absolute) |
| `se_joint` | Laplace SE from the joint fit (sampling error) |
| `se_wgs_only`, `se_ampl_only` | Laplace SE from the WGS-only and amplicon-B4-only fits (same machinery) |
| `se_reduction_ratio` | `min(se_wgs_only, se_ampl_only) / se_joint` (>1 = fusion tightened it) |
| `ESS_fold_vs_wgs` | `(se_wgs_only/se_joint)²` — WGS-equivalent information gain |
| `se_heterogeneity_loso` | leave-one-WGS-well-out SD of the rate (cross-culture variation) |
| `se_combined` | `sqrt(se_joint² + se_heterogeneity_loso²)` — honest headline CI half-width |
| `r_sensitivity_to_b_amp` | \|Δr\| when the assay-drift ridge is relaxed vs off |
| `r_wgs_only`, `r_ampl_only` | rate from each single-assay fit |
| `evidence_tier` | `both-deep` (tightened) / `amplicon-B4-only` (newly estimable) / `WGS-only` (unchanged) |
| `in_B4` | variant present in the deep amplicon (well B4) |

### `outputs_joint_4MXB/b4_concordance.csv`
Same-culture gate: `Candidate`, `r_WGS_B4`, `r_ampl_B4` for variants estimable in both
the WGS-B4-only and amplicon-B4-only fits (corr ≈ 0.42).

### `outputs_joint_4MXB/intermediate/combined_4MXB_long.csv`
The fused input: WGS (11 wells) + amplicon-B4, all 4MXB, columns `Sample, assay,
Microtiter_plate_well, Transfer, verA, verB, Candidate, Count, depth, freq`.

### `outputs_joint_4MXB/figures/joint_error_reduction.png`
SE scatter (joint vs WGS-only), ESS histogram, top-tightened forest, B4 concordance.
