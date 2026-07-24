# Analysis 2 — 4-parameter growth model on **barcode (WGS) data only**

Self-contained. The **4-parameter segmented (two-phase / broken-stick) growth model**
applied to the **SeqCenter WGS extracted-barcode** counts alone.

## The model (4 parameters per variant trajectory)
```
CLR(t) = logA + r_init·(t − t0) + (r_final − r_init)·max(0, t − τ)
```
1. **logA** — initial (log) abundance   2. **r_init** — initial growth rate
3. **r_final** — final growth rate       4. **τ** — the transfer at which the rate switches

Increasing switch only (`r_final > r_init`), adopted only on strong evidence
(ΔBIC > 6 **and** > 10 % rate increase); otherwise the constant single-slope fit stands.

## The data
- `inputs/SeqCenter_2026-05-26_QUO1022807_exp2_extracted_barcodes.csv` — WGS barcode counts
  (the "LIMS" export). Restricted to the **11 samples/wells observed at all four transfers
  {3, 4, 6, 13}** (B4, B5, C4, C9, D10, D4, D5, D9, E4, E9, F4). Shallow (~200×, ≈ 190 reads/cell).
- `inputs/OD_4MXB.csv` — OD slice for the time / generations axis.

## Run (needs nothing outside this folder)
```bash
~/Documents/py_venv/bin/python run.py
```

## Key result — the breakpoint **is** identifiable
| metric | value |
|---|--:|
| sampled transfers | **4** `{3,4,6,13}` |
| estimable trajectories | 237 (across 11 wells) |
| **breakpoint-testable** | **66** |
| **two-phase switches adopted** | **13** |
| median ΔBIC (adopted) | 11.3 |
| median weighted R² constant → two-phase (testable) | 0.31 → 0.98 |

The fourth transfer (**T6**) gives the two-phase model a residual degree of freedom, so 66
trajectories are breakpoint-testable and **13 show a real *increasing* switch** — variants that
lose ground early, then surge (most inflect near T6). Acceleration clusters by allele
(verA **A81**, **A78**; verB **B26**). The other testable trajectories keep the constant fit —
47 because their best fit was a *decrease* (not permitted), 6 as weak.

**Honest caveat:** with only 4 transfers the two-phase fit has just one residual d.o.f., so the
large in-sample R² jump is partly optimism. Out-of-sample (LOOCV) the 4-var only clearly beats
the 1-var when the elbow location is supplied (see `model_comparison_summary.csv`,
`model_fit_error.csv`). The 4-var's value is **descriptive** — it exposes dip-then-rise dynamics
the single slope averages toward zero.

## Outputs (`outputs/`)
`segmented_growth_rates.csv` (per-trajectory 4-param fit),
`segmented_{model_selection,sample_summary,allele_effects}.csv` (rollups),
`model_comparison_{summary,by_trajectory}.csv` + `model_fit_error*` (fair 1-var vs 4-var),
`figures/segmented_{overview,parameters,gallery,allele}.png`,
`figures/model_comparison{,_blindspot}.png`, `figures/model_fit_error.png`,
`intermediate/` (tidy long table, depth, OD time axis).
