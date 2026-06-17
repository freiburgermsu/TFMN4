# Joint WGS + amplicon integration for per-variant 4MXB growth rates

A comprehensive optimization that fuses the **WGS counts** and the deep **amplicon**
data into one estimate of each variant's **4-methoxybenzoate (4MXB) growth rate**,
to lower the per-variant error margins — with an honest accounting of how much is
actually gained. Scripts: `s13_joint_4MXB.py`, `s13b_joint_error.py`,
`s13c_error_reduction_fig.py`. Deliverables: `outputs_joint_4MXB/`.

---

## 1. Why the two datasets can be fused
- **Same selective environment.** The WGS experiment (exp2) is uniformly
  `Methoxybenzoate 4 mM + Kan` = **4MXB** (verified: every focus well's OD Condition).
  The amplicon 4MXB run is the same environment. So a variant's *4MXB* growth rate is
  a single quantity both datasets measure. (Amplicon **pyruvate** is a *different*
  environment and is **not** fused — see §7.)
- **A shared physical culture.** WGS well **B4** is the same culture as the amplicon
  B4 (WGS sample `concX_largeLib_SpeI.1` = well B4). So WGS-B4 (shallow, T3/4/6/13)
  and amplicon-B4 (deep, T3/4/13) are two assays of one timecourse.
- **Complementary strengths.** WGS gives **breadth** (11 wells, different founding
  subsets); the amplicon gives **depth** (~3,800 reads/cell vs ~190; ~10k/condition).

## 2. Three designs considered (and the choice)
A design workflow proposed and adversarially vetted three formulations:
1. **Unified replicator + assay batch** — one global `r_i` across all 4MXB cells,
   per-cell multinomial at native depth, an assay-differential drift term. *Chosen
   backbone* (best feasibility; extends the existing `s12`).
2. **Hierarchical two-condition** — 4MXB + pyruvate rates with cross-condition
   pooling. *Deferred* (pyruvate has only one well → prior-dominated; kept separate).
3. **Same-sample B4 fusion + meta-analysis** — fuse the two B4 assays, then
   inverse-variance meta-analyse across wells. *Grafted in* (B4 logA tie, the
   concordance gate, the rigorous error-reduction accounting).

## 3. The model (`s13`)
Extends the `s12` softmax-replicator. For cell *c* (a (physical-well, assay, transfer)),
variant *i*:

```
eta_{c,i} = logA_{well(c),i} + r_i · x_c + b_amp · x_c · 1[assay(c)=amplicon]
f_{c,i}   = softmax over the well's variant union           # per-cell, native depth
```
- **`r_i`** — one global 4MXB rate per variant (the deliverable), shared across all
  11 WGS wells + amplicon-B4.
- **`logA_{well,i}`** — per-(physical-well, variant) nuisance gauge; **WGS-B4 and
  amplicon-B4 share one `logA_B4` block** (this is what makes `b_amp` identifiable).
- **`b_amp`** — a single assay-**differential** rate drift. A *constant* assay offset
  cancels in the per-cell softmax, so only the `·x` drift is identifiable; it is
  **strongly ridged** so the shared biological slope is preferred over the batch term.
- **Loss** = multinomial NLL (native per-cell depth → deep amplicon tightens
  B4-present rates) + OD anchor (community rate = within-cycle OD slope μ_bulk,
  entered **once** per physical well) + ridge(`r`→μ_bar) + ridge(`logA`→data) +
  strong ridge(`b_amp`). Fit in jax + L-BFGS; Laplace SE via Schur complement
  marginalizing `logA` and `b_amp`.

## 4. Decisions log (with the verified numbers)
| Decision | Why | Evidence |
|---|---|---|
| Collapse the 3 amplicon primer sets → one cell/transfer | they are technical replicates | pairwise freq corr **0.995–0.998** |
| Enter B4's OD anchor **once** | WGS-B4 and amplicon-B4 are one culture | μ_bulk byte-identical (0.301@T4, 0.294@T13) |
| One reconciled B4 time axis | same physical culture | WGS vs amplicon cumgen differ ~11% at T4 |
| Tie `logA_B4` across assays | sole identifier of `b_amp` | drop the tie ⇒ `b_amp` unidentifiable |
| Build counts from `amplicons_reads.csv` | full resolution | 212 B4 variants vs 19 in the binned file |
| 4MXB-only fusion | pyruvate is a different environment | μ_bulk 0.83 vs 0.30 /h |

## 5. How the error reduction is quantified (`s13`, `s13b`)
The **same model machinery** is fit three ways so the SEs are comparable:
`joint`, `WGS-only`, `amplicon-B4-only`. Per variant:
- `se_reduction_ratio = min(se_wgs_only, se_ampl_only) / se_joint` (>1 = tightened).
  **No single global number** is reported — it would be meaningless.
- `ESS_fold_vs_wgs = (se_wgs_only / se_joint)²` — WGS-equivalent information gain.
- **Sampling vs heterogeneity split:** `se_joint` (Laplace, shrinks legitimately with
  depth) vs `se_heterogeneity_loso` (leave-one-WGS-well-out SD of `r_i` — cross-culture
  biological variation that must *not* shrink from one deep B4 culture). Honest
  combined CI = the two in quadrature (`se_combined`).
- **Concordance GATE** (go/no-go for the fusion): WGS-B4-only vs amplicon-B4-only `r`
  on the dual-estimable variants (`b4_concordance.csv`).
- **Evidence tiers:** `both-deep` (in WGS + amplicon-B4), `amplicon-B4-only` (newly
  estimable), `WGS-only` (unchanged).

## 6. Results and interpretation
| Tier | n | median se_joint | median reduction | median ESS | meaning |
|---|---|---|---|---|---|
| both-deep | 81 | 0.018 | **1.12×** | **1.4×** | genuinely tightened by fusion |
| amplicon-B4-only | 131 | 0.060 | — | — | **newly estimable** (WGS couldn't) |
| WGS-only | 233 | 0.039 | 1.01× | 1.0× | unchanged (variant not in B4) |

- **The fusion is valid:** `b_amp ≈ 0` under strong *and* weak ridge (no systematic
  WGS-vs-amplicon drift); same-culture concordance is **positive (corr ≈ 0.42)** with
  small bias (+0.045/h) — limited mainly by shallow-WGS noise, not by disagreement;
  cross-well heterogeneity is small (median 0.003/h, *below* the sampling SE), so the
  shared-rate assumption holds and the sampling-error reduction is real.
- **But the gain is modest, by design of the data:** the amplicon's depth is
  concentrated (top variant = 43 % of reads; **174/212 B4 variants have <30 total
  reads, median 2**), and it covers only well B4. So:
  - The **81 shared variants** tighten by ~12 % (ESS ~1.4×) — real but not dramatic.
  - **131 variants become newly estimable** from the deep amplicon — the largest
    concrete enhancement is *coverage*, not precision.
  - The **233 WGS-only variants** (absent from B4) are unchanged.
- The deep amplicon also fits its own counts far better (freq R² **0.996** vs WGS
  0.945), and the joint model retains R² ≈ 0.937 across both assays.
- **Bottom line:** fusing the two assays enhances the 4MXB per-variant predictions
  by (a) modestly tightening the variants present deep in B4 and (b) substantially
  expanding which variants are estimable — while honestly *not* manufacturing a large
  global error reduction. Relative ordering is robust; the absolute h⁻¹ level remains
  OD/community-dominated (carry the `s12b`/METHODS §8–9 caveat).

## 7. Caveats
1. **Error reduction is real only on the *sampling* component** and only for the ~81
   shared / deep-in-B4 variants — *not* the ~20× cell-depth ratio (depth is
   concentrated; median B4 variant has 2 reads).
2. **Deep single-culture precision ≠ global-rate certainty.** B4 depth does not reduce
   cross-well biological heterogeneity; `se_heterogeneity_loso` is reported alongside.
3. **Assay batch is structurally fragile** — identified from ~53 B4 variants, one
   well. A scalar drift cannot capture **variant/frequency-dependent PCR bias**;
   residual bias correlating with verA/verB could leak into allele effects. The
   concordance gate is the safeguard; here `b_amp≈0` and concordance is positive.
4. **WGS and amplicon are not independent** (amplicon PCRs the same B4 DNA WGS
   sequenced); the shared-`logA_B4` tie + single clock is the honest treatment, and
   the SE drop is correspondingly small.
5. **Absolute level stays OD/community-dominated** (METHODS §8–9); only relative
   rates and their error reduction are the defensible deliverable.
6. **Pyruvate** is a different, faster environment with one well — analysed separately
   (`outputs_amplicon_pyruvate/`); never fused into the 4MXB rate.

## 8. Deliverables / how to run
```bash
~/Documents/py_venv/bin/python scripts/s13_joint_4MXB.py        # joint fit + 3-way SE + gate
~/Documents/py_venv/bin/python scripts/s13b_joint_error.py      # LOSO heterogeneity + b_amp sweep
~/Documents/py_venv/bin/python scripts/s13c_error_reduction_fig.py
```
- `outputs_joint_4MXB/joint_variant_growth_rates_4MXB.csv` — per variant: `r_joint_per_h`,
  `se_joint`, `se_wgs_only`, `se_ampl_only`, `se_reduction_ratio`, `ESS_fold_vs_wgs`,
  `se_heterogeneity_loso`, `se_combined`, `r_sensitivity_to_b_amp`, `evidence_tier`,
  `in_B4`, `r_wgs_only`, `r_ampl_only` (column meanings in DATA_DICTIONARY.md).
- `outputs_joint_4MXB/b4_concordance.csv` — the same-culture gate data.
- `outputs_joint_4MXB/figures/joint_error_reduction.png` — SE scatter, ESS, forest, gate.
- `outputs_joint_4MXB/intermediate/combined_4MXB_long.csv` — the fused input.
