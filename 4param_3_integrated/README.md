# Analysis 3 — 4-parameter growth model on **integrated (amplicon + barcode) data**

Self-contained. The **4-parameter segmented (two-phase / broken-stick) growth model**
applied to a **fusion of both assays** on their shared 4MXB environment.

## The model (4 parameters per variant trajectory)
Identical to analyses 1 and 2:
```
CLR(t) = logA + r_init·(t − t0) + (r_final − r_init)·max(0, t − τ)
```
`logA`, `r_init`, `r_final`, `τ` — increasing switch only, adopted on strong evidence
(ΔBIC > 6 **and** > 10 % increase).

## The fusion (this is the only analysis that differs in *data*, not model)
`scripts/s01_prepare_data.py` builds the combined count table:
- **WGS breadth** — the 11 four-transfer wells `{3,4,6,13}` (as in analysis 2).
- **Amplicon depth** — well **B4** is the *same physical culture* as WGS sample
  `concX_largeLib_SpeI.1`. For B4 we **pool reads per (transfer, variant)**: amplicon +
  WGS. The 3 amplicon primer sets are collapsed to one count/transfer.

Consequences on B4, both intended:
- **T3 / T4 / T13** gain the amplicon's depth (≈ 190 → ≈ 10 000 reads) → far more precise
  frequencies, and many rare B4 variants become estimable.
- **T6** is present *only* in WGS (the amplicon never sampled it) — and it is exactly that
  WGS-only point that makes B4 **breakpoint-testable** (the 3-transfer amplicon never can be).

The other 10 wells are WGS-only, unchanged.

## The data
- `inputs/SeqCenter_2026-05-26_QUO1022807_exp2_extracted_barcodes.csv` — WGS barcode counts.
- `inputs/amplicons_reads.csv` — Plasmidsaurus amplicon reads (well B4, carbon 4MXB).
- `inputs/OD_4MXB.csv` — OD slice for the time / generations axis.

## Run (needs nothing outside this folder)
```bash
~/Documents/py_venv/bin/python run.py
```

## Key result — **strictly dominates** both single-assay runs
| metric | amplicon | barcode/WGS | **integrated** |
|---|--:|--:|--:|
| estimable trajectories | 186 | 237 | **308** |
| breakpoint-testable | 0 | 66 | **71** |
| two-phase adopted | 0 | 13 | **14** |

**On well B4 specifically** (the one shared culture): estimable **24 → 95**, testable
**6 → 11**, adopted **0 → 1**. The new B4 switch is `A81-B132` (r −0.77 → +0.36 /cycle @ T6,
ΔBIC = 15) — a genuine dip-then-surge that WGS's ~190-read depth alone could not resolve.

Integration is **purely additive** vs analysis 2 (it keeps all 13 WGS switches and adds the
B4 one). On B4 the fused single-slope rates track the **deep amplicon** (r = 0.94) and
**re-estimate** the noisy shallow-WGS values (r = 0.67, median shift 0.18/cycle) — variance
reduction on the one shallow culture, converging on the higher-confidence assay.

## Outputs (`outputs/`)
Same file set as analysis 2 (`segmented_growth_rates.csv`, rollups, `model_comparison_*`,
`model_fit_error*`, `figures/segmented_*`, `figures/model_*`). The `s01` log prints the fused
B4 per-transfer depth so the fusion is auditable.
