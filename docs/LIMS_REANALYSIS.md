# LIMS mirror re-analysis — every environment simulated separately

`lims_mirror.backup.db` (the lab LIMS mirror, SQLite, 445 MB, git-ignored) was evaluated and
the repo's methods were re-applied to its data **one environment at a time**: one full
barcode pipeline run (`s01`–`s14e`, every table and figure) per experiment × carbon source ×
host strain × assay, and one copy-number pipeline run (`cn01`–`cn05`) per genomic region ×
carbon source × platform. Environments are never pooled.

- Numbers in this document come from `outputs_lims/REPORT.md` (auto-generated) and the
  per-environment folders; regenerate everything with
  `~/Documents/py_venv/bin/python scripts/lims_run_all.py` (≈15 min on 8 parallel jobs).
- Per-environment results: `outputs_lims/barcode/<env>/`, `outputs_lims/copy_number/<env>/`
  (same file names as `outputs/` and `outputs_copynumber/`; the copy-number files are prefixed
  with the region, e.g. `verAB_growth_rates.csv`).
- Cross-environment tables/figures: `outputs_lims/comparison/`. The DB evaluation itself:
  `lims_data/inventory/`.

---

## 1. What the database contains

**31 tables.** Every LIMS table keeps soft-deleted row versions (`deleted = 1`; e.g. 1,478 of
2,338 `Robotic_ALE_samples` rows); only live rows are used. Six tables feed the pipelines:
`verAB_barcodes` (counts), `Copy_numbers`, `Robotic_OD`, `Robotic_ALE_samples`/`Samples`
(culture → condition, strain, plate well) and `Conditions` (carbon source, concentration).
The rest (`Mutations`: 278k live breseq calls, gene annotations, strain/plasmid stocks, dgoA
allele tables, …) has no consumer among this repo's methods and is only inventoried.

**Everything the repo already analysed is in the DB, unchanged:**

| repo input | DB source | check |
|---|---|---|
| `SeqCenter_…_extracted_barcodes.csv` (WGS counts) | `verAB_barcodes`, order SeqCenter_2026-05-26 | identical counts after normalising allele names (`A078`→`A78`); the DB adds 14 partial-barcode rows (`A_UNK`/`B_UNK`, 1–2 reads) that the CSV had dropped |
| `TFMN4_robotic_OD_processed_final.csv` | `Robotic_OD`, experiment TFMN4 | 54,336/54,336 rows identical |
| `copy-number-dgoA_.csv` | `Copy_numbers`, dgoA* region | identical up to the CSV's 3-decimal rounding |

**New in the DB:**

| data | what | used here |
|---|---|---|
| verAB **amplicon time series of the 11 TFMN4 4MXB cultures** | transfers 2, 3, 4, 5, 6, 8, 10, 13; ~3,900 reads/cell (WGS: ~190); 4 Plasmidsaurus orders | ✔ barcode env |
| **TFMN7verAB** (robotic run TFMN6, plate exp2) | guaiacol (host ACN3788 and ACN3882), isovanillate, 4-methoxybenzoate (sub-libraries), veratrate, all 4 mM + Kan; transfers 1, 2, 3, 4, 17; ~4,000 reads/cell | ✔ 5 barcode envs |
| robotic OD of TFMN5/TFMN6 | growth curves incl. every TFMN7 culture | ✔ (time axis, μ_bulk) |
| copy number: **EASy** long-term lineages | dgoA* and Kan-cassette CN, transfers 203–642, pyruvate | ✔ 2 CN envs |
| copy number: **verAB cassette** in TFMN2/TFMN3 | isovanillate / methoxybenzoate / vanillate (3 mM) | ✔ 3 CN envs |
| copy number: **verC** in TFMN4 exp2 | 4MXB, transfers 3, 4, 6, 13 | ✔ 1 CN env |
| TFMN1 dgoA at T32 (order KPF9J2) | 15 lineages; 14 have no other dgoA timepoint | in the dgoA env (adds one 2-point lineage) |
| Yusuke library selections, sub-library pools, LTA colony genotyping | 105 single-timepoint seqsamples | ✘ no time axis → no slope (inventoried) |

### 1.1 Data-quality findings (all in `lims_data/inventory/`)

- **Un-called alleles.** 47,814 reads (3.2%) carry `A_UNK`/`B_UNK`; median per-cell share
  1.5%. They are dropped, as the repo's CSV already did (log-ratios among the identified
  variants are unchanged; only each culture's gauge shifts). In a few cultures a half-called
  lineage is large and is flagged: TFMN7 4MXB `sub_9.1` is 98–99% `A81-B_UNK` from T2 on
  (excluded: <50 identified reads/cell), `ACN3882 guaiacol large.1` is 94% `A_UNK-B162` at T4
  (excluded), `sub_2.1` carries 37–45% `A88-B_UNK` (kept, flagged) — sweeps the allele caller
  cannot name.
- **Failed / superseded libraries.** 3 cells were re-sequenced in a later order (the deeper run
  is kept); 3 cells are failed libraries (4–127 reads); 5 more fall below 500 *identified* reads
  only because an un-called lineage dominates them (above).
- **Contamination.** The noDNA control `TFMN7verAB.ACN3788.Iva.noDNA.3` has 3,762 barcode reads
  at T4, 99.5% one variant (`A95-B123`) that is not dominant (>20%) in any culture of the DB —
  an unexplained contaminant of that well/library.
- **Suspected contamination in the TFMN4 amplicon: verA A183, mostly `A183-B63`** (§4.2) — the
  one variant whose amplicon and WGS readings of the same cultures disagree.
- `Conditions` labels `Vanillate 3 mM + Kan` with `Carbon_source = pyruvate` (corrected here).

## 2. How the methods were re-applied

`scripts/lims01_extract.py` reads the DB read-only, applies the QC above, and writes one
input set per environment (`lims_data/barcodes|od|copy_number/`) plus `manifest.json`.
`config.py` / `cn_common.py` gained a dataset switch for these (`TFMN4_DATASET=lims_<env>`,
`TFMN4_CN_DATASET=lims_cn_<env>`), exactly as the repo already switches between WGS and
amplicon data, so **the identical analysis code runs on every environment**.

**Environment = experiment × carbon source × host strain × assay.** Guaiacol is split by host
strain (ACN3788 vs ACN3882: a different genetic background would otherwise be fitted as one
population). The two 4MXB experiments (TFMN4, TFMN7) stay separate (different transfer
schedules, libraries and runs) and are compared afterwards. Within an environment, a transfer
joins the design if ≥50% of cultures passed QC there, and a culture is kept only with a
complete trajectory (the repo's rule).

**What had to be generalised** (each change leaves the original datasets' code path
untouched — verified by re-running the repo's own WGS, amplicon and copy-number pipelines
from a scratch copy: identical outputs apart from the s12b refits noted in §3):

| where | why |
|---|---|
| s09/s10 generations-per-cycle | used T3→T13 explicitly; now the first→last sampled transfer |
| s09 bridge model | TFMN7 4MXB cultures come from disjoint sub-libraries → 3 unconnected components; each now gets its own sum-to-zero gauge (`bridge_component` column) instead of an assertion failure |
| s12 global fit | 3,000–12,000 parameters: dense `jax.hessian` replaced by batched Hessian-vector products + an exact per-culture block Schur complement; L-BFGS hits its iteration cap on deep counts, so the optimum is polished with a trust-region Newton step (same loss). Every LIMS fit converges (‖grad‖ ≤ 2×10⁻⁴); rates moved ≤0.01/h except guaiacol ACN3882 (≤0.11/h) |
| s14d LOOCV | design matrices were hard-coded for 3 points (4 transfers) |
| s14b/s14c | breakpoint columns/colours were fixed to T4/T6; knots are now one ordinal colour ramp |
| figures | per-variant forests/galleries capped or drawn as ranked caterpillars when there are thousands of variants; sample-name prefixes generic |
| interpretation text | several titles/table sentences stated the WGS result as fixed text ("the in-sample edge is optimism…"); LIMS runs now generate these from their own numbers (on the WGS data they reproduce the original wording's conclusions) |
| copy number | region-tagged file names, data-driven contexts and colours, significance wording that states the direction |

## 3. Validation

The DB's WGS counts, run through the generalised code (`outputs_lims/barcode/TFMN4_methoxybenzoate_ACN3788_wgs/`),
reproduce the committed `outputs/`: **every number in 24 of 25 tables is identical**
(A90 +0.104/cycle, freq R² 0.946, 13/66 two-phase switches, LOOCV 1.707 vs 1.691, …). The
exception is s12b's bootstrap / leave-one-sample-out / sensitivity columns in
`global_variant_growth_rates.csv` (median 0.001/h, max 0.013/h): those are capped-iteration
L-BFGS refits, and they drift by the same amount when the repo's *unchanged* code is re-run on
this machine. The remaining text differences are the dataset label and the generated
interpretation sentences (`validation_wgs_db_vs_repo.csv`).

## 4. Barcode environments

| environment | cultures × transfers | variants | richness first → last (median) | μ_bulk (/h) | top sweep-winner verA | 4-var: adopted / testable | LOOCV 4-var/1-var (elbow learned) |
|---|---|--:|---|--:|---|---|---|
| TFMN4 · 4MXB · amplicon | 11 × 8 (T2–T13) | 1,909 | 159 → 44 | 0.31 | A81 (6/11) | 53 / 702 | **0.65** (50/53 better) |
| TFMN4 · 4MXB · WGS (validation) | 11 × 4 | 314 | 30 → 9 | 0.32 | A81 (6/11) | 13 / 66 | 0.92 (7/13) |
| TFMN7 · 4MXB · ACN3788 | 19 × 5 (T1–T4, T17) | 3,178 | 195 → 17 | 0.27 | A88 (4/19) | 83 / 769 | 0.51 (63/83) |
| TFMN7 · guaiacol · ACN3788 | 10 × 5 | 3,100 | 415 → 46 | 0.29 | **A183 (8/10)** | 2 / 527 | — (2 adopted) |
| TFMN7 · guaiacol · ACN3882 | 9 × 5 | 4,865 | 669 → 23 | 0.18 | **A183 (8/9)** | 0 / 13 | — |
| TFMN7 · isovanillate · ACN3788 | 4 × 5 | 1,272 | 297 → 170 | 0.26 | A186 (2/4) | 117 / 817 | 0.66 (87/117) |
| TFMN7 · veratrate · ACN3788 | 4 × 5 | 1,565 | 400 → 222 | 0.24 | A183 (2/4) | 228 / 974 | 0.60 (160/228) |

### 4.1 What the dense TFMN4 time series changes

- **Coverage:** 1,329 estimable trajectories vs 237 with WGS (×5.6); 702 are breakpoint-testable.
- **Assay agreement is good** on the same cultures and transfers: frequencies correlate
  r = 0.78 (log), the dominant variant agrees in 82% of cells, slopes on the identical four
  transfers correlate ρ = 0.60 (WGS depth-limited) — except for one variant (§4.2).
- **The 4-parameter model becomes predictive.** At 4 transfers the breakpoint was not
  identifiable (out-of-sample parity, METHODS §11.6); with 8 transfers the adopted two-phase
  fits predict held-out timepoints better in 50/53 cases (median RMSE ratio 0.65). Note what
  "increase" means here: most adopted switches are steep declines that level off (median
  r_init −0.85 → r_final −0.09/cycle; 20/53 end positive), whereas the WGS switches were
  dip-then-surge (11/13 end positive). The same holds in TFMN7, where the T1→T2 interval
  (the first cycle after transformation) produces a crash that later flattens.
- **A81 still wins 6/11 sweeps with only a negligible marginal advantage** (+0.016/cycle ≈
  +0.005/gen; drift-significant only because the deep data resolve tiny effects) — the repo's
  founder-effect reading holds.
- **The WGS-era "robust" A90/A78 advantages do not hold up.** On the same four transfers the
  amplicon gives A90 +0.043 (WGS +0.104) and A78 −0.006 (WGS +0.061); over the full series both
  are drift-confirmed *negative* (A90 −0.116, A78 −0.036/cycle) because A90 declines early
  (−0.17 over T2–T8). At ~200 reads the WGS slopes were inflated (the survivorship effect
  METHODS §5 warns about) and a single slope over a non-monotone trajectory depends on the
  sampling window. Treat the README's A90/A78 claim as superseded by the deeper data.

### 4.2 Suspected contamination: verA A183 (mostly `A183-B63`) in the TFMN4 amplicon

verA A183 is essentially absent from all 11 TFMN4 cultures through T8 (5 reads in 66
culture-transfers, ~260k reads) and from every WGS sample. Then, at T10 and T13, it appears in
**every** culture at once: `A183-B63` at 6–30% (T10) and 11–25% (T13), plus a dozen other A183
pairings at 1–6 reads. It occurs only in the two June 2026 amplicon orders (ZQT7PJ = T13,
KZGC3B = T6/T8/T10, whose T6/T8 samples stay clean), and the WGS of the same cultures at T13 has
**0 reads** where ~30 per culture are expected (Poisson P ≈ e⁻³⁰). A systematic scan in both
directions (a variant whose lower-bound frequency in one assay predicts reads that the other
assay lacks) flags `A183-B63` in all 11 cultures and otherwise a single cell (`A16-B117`, 3% at
T4 in one culture) (`cross_assay_discordance_TFMN4.csv`). A real sweep would have to rise from <0.03% to ~12% within
two cycles in eleven independent cultures simultaneously and then stall — not plausible. It is
therefore treated as a likely contamination of the T10/T13 amplicon material by an
A183-carrying population. (A183 itself is a genuine, fit library allele: it sweeps guaiacol and
wins 4MXB cultures in TFMN7 — §4.3.) A WGS allele-calling gap for A183 is the alternative;
checking the WGS extraction reference for A183, or re-sequencing a T13 sample, would settle it.

The main TFMN4 run keeps every read; a **sensitivity run without verA A183**
(`outputs_lims/barcode/TFMN4_methoxybenzoate_ACN3788_amplicon_excl_A183/`) shows that nothing
else depends on it (`comparison/sensitivity_excl_A183.csv`,
`comparison/figures/sensitivity_excl_A183.png`): allele marginals correlate ρ = 0.9995 (verA) and
0.9993 (verB) between the runs; the robust-allele sets differ only by A183 itself and B63 (B79
becomes robust); A90 (−0.112) and A78 (−0.032/cycle) stay disfavoured and A81 stays ≈0 (+0.021);
11 of the 12 fastest variants are unchanged; the same 53 two-phase switches are adopted; and the
global fit improves (freq R² 0.948 → 0.960). **Only A183's own TFMN4 effect (+0.21/cycle) and the
top rank of A183-B63 rest on the suspect reads.**

### 4.3 TFMN7: the carbon source sets the strength of selection

- **Guaiacol = hard sweeps.** Richness collapses from 415 to 46 (ACN3788) and 669 to 23
  (ACN3882) median variants; verA **A183** wins 8/10 and 8/9 cultures, rising from 39% (ACN3788)
  or 4% (ACN3882) at T1 to >90% by T4. Winners lead the community by 0.3–0.6/h. The ACN3882 host
  grows slower (μ_bulk 0.18 vs 0.29/h) and its sweep is the most extreme.
- **Isovanillate and veratrate = weak selection.** Richness stays high (297 → 170, 400 → 222);
  the fastest variants lead by only ~0.05–0.07/h; many variants decline early and then
  stabilise (most of the 117 and 228 adopted two-phase fits).
- **TFMN7 4MXB = intermediate, many winners.** Cultures seeded from different sub-libraries
  share few variants (3 bridge components); winners differ by sub-library (A88, A81, A196,
  A143, A183, A38). A183 is carried only by sub-library 1, and there it wins 2 of 4 cultures
  (A183-B57, A183-B79, rising to 50–90%) — so A183 is genuinely fit on 4MXB, but through other
  partners than the suspect A183-B63 of §4.2.

### 4.4 Across environments

- **Allele effects are mostly context-specific.** Pairwise Spearman correlations of the
  verA/verB marginals between environments are −0.13…+0.28 — including the same 4MXB condition in
  two experiments (ρ = 0.17, 101 shared alleles) and guaiacol in two host strains (ρ = 0.17).
  Among alleles that are drift-confirmed in both environments of a pair, the sign agrees in
  32–88% of cases (88% for the two guaiacol hosts, 62% for the two 4MXB experiments;
  `allele_concordance.csv`).
- **A183 is the one verA that is drift-confirmed favoured in every environment** (+0.006 to
  +0.060/gen), and the guaiacol sweeps are A183 sweeps; its TFMN4 value rests entirely on the
  suspect T10/T13 A183 reads (§4.2), and in TFMN7 4MXB the marginal is small (+0.016/gen)
  because only sub-library 1 carries A183.
- **Absolute per-hour rates are least trustworthy where selection is strongest.** The s12
  global fit reproduces the counts well (freq R² 0.82–0.99) but its community rate departs
  from the measured μ_bulk by 0.23–0.33/h in guaiacol (vs 0.06–0.15/h elsewhere), where some
  fitted per-hour rates are even negative: read those environments' rates as advantages over
  μ̄ (`rel_advantage_per_h`), as the cross-environment figure does.

## 5. Copy-number environments

| environment | lineages (CI-reliable) | FDR amplifying / losing | notes |
|---|---|---|---|
| dgoA · pyruvate · TFMN1+TFMN4 exp1 | 105 (24) | 1 / 0 | reproduces the repo: only `TFMN1.pgi.1` (+0.071 ± 0.011/transfer, q = 0.019); 20/24 testable saturate |
| dgoA · pyruvate · EASy | 22 (22) | 0 / 0 | long-term lineages (T203–T642), CN ≈ 1–2, no significant trend |
| Kan cassette · pyruvate · EASy | 22 (22) | 0 / 0 | starts at CN 15–27 (as high in the noDNA controls — consistent with a multicopy element), falls to ≈1 in some lineages; no FDR call |
| verAB · isovanillate · TFMN2 | 21 (21) | 1 / 0 | modest amplification (median +0.007/transfer) |
| verAB · vanillate · TFMN2 | 15 (15) | **5 / 0** | small but consistent verAB amplification (+0.004 to +0.010/transfer) |
| verAB · methoxybenzoate · TFMN2+TFMN3 | 26 (4) | 0 / 0 | TFMN2 4MXB lineages were only measured at T2 |
| verC · methoxybenzoate · TFMN4 exp2 | 34 (11) | 0 / 0 | flat |

EASy rates are per EASy transfer and are not comparable with robotic-transfer rates.

## 6. What to trust (caveats)

1. **Relative, not absolute** — as in the original analysis; per-cycle units differ between
   experiments (cycle 26.4 h for TFMN4 vs 26.7 h, and 2.7–3.4 generations per cycle), so
   compare environments per generation (the allele tables do).
2. **TFMN7 has a 13-transfer gap (T4 → T17)**; single slopes and breakpoints are dominated by
   the T1–T4 dynamics and the T17 endpoint.
3. **Many variants are seen once.** Deep amplicons reveal thousands of variants, but 32–95% of
   (culture, variant) pairs are present at a single timepoint (not estimable). Their s12 rates
   are prior-driven (hollow points in the caterpillar plots).
4. **s12b error bars in the large environments are approximate.** The bootstrap / LOSO /
   sensitivity refits keep the repo's 300-iteration L-BFGS. On these deep, ill-conditioned
   problems a diagnostic (one converged reference refit per replicate, TFMN4 amplicon) showed
   they recover ~80% of the true bootstrap displacement with a per-variant correlation of ~0.5,
   so `se_boot` / `loso_heterogeneity_sd` are conservative-low there; `se_laplace` (exact Hessian
   at the converged optimum) is the reliable per-variant error. The `identifiability` flag is
   "anchor/prior-driven" for almost every variant in these environments because the OD weight
   shifts the whole level by more than 0.05/h — the relative ordering is still data-pinned.
5. **Un-called alleles** are excluded; a few cultures carry a large half-called lineage (§1.1).
6. **A183 in TFMN4** (§4.2) and the **contaminated noDNA well** (§1.1) should be resolved with
   the wet lab before the TFMN4 A183 result is used; every other TFMN4 conclusion is unchanged
   without A183.
