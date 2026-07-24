# Analysis 1 — 4-parameter growth model on **amplicon data only**

Self-contained. The **4-parameter segmented (two-phase / broken-stick) growth model**
applied to the **Plasmidsaurus amplicon** data alone (4-methoxybenzoate arm).

## The model (4 parameters per variant trajectory)
A variant's centered-log-ratio (CLR) of read counts vs time is fit as a continuous
broken stick:

```
CLR(t) = logA + r_init·(t − t0) + (r_final − r_init)·max(0, t − τ)
```

1. **logA** — initial (log) abundance at the first sampled transfer
2. **r_init** — initial growth rate (first segment slope)
3. **r_final** — final growth rate (second segment slope)
4. **τ** — the transfer at which the rate switches (searched over interior sampled transfers)

Only an *increasing* switch (`r_final > r_init`) is allowed, and it is adopted only on
strong evidence (ΔBIC > 6 **and** > 10 % rate increase); otherwise the constant fit stands.

## The data
- `inputs/amplicons_reads.csv` — per-read `verA`/`verB` calls (Plasmidsaurus X2VH94), **well B4**,
  carbon **M = 4MXB**, transfers **{3, 4, 13}**, 3 primer-set technical replicates (deep: ≈ 10 000
  reads/transfer).
- `inputs/OD_4MXB.csv` — the OD slice (exp2, 4MXB) used only to build the time / generations axis.

## Run (needs nothing outside this folder)
```bash
~/Documents/py_venv/bin/python run.py
```

## Key result — the 4th parameter is **not identifiable** here
| metric | value |
|---|--:|
| sampled transfers | **3** `{3,4,13}` |
| estimable trajectories | 186 (= 3 primer-set reps × ~62 variants of well B4) |
| **breakpoint-testable** | **0** |
| **two-phase switches adopted** | **0** |

A breakpoint needs **≥ 4 timepoints** (≥ 2 either side of the knot); the amplicon has only 3.
So every trajectory is reported as **constant** — the model's four parameters collapse to the
one-parameter single-slope fit (`s_constant_per_cycle`). This is the correct,
identifiability-respecting behaviour: the amplicon's **depth** sharpens each frequency but
cannot supply the missing timepoint. `model_comparison_summary.csv` / `model_fit_error.csv`
therefore report "4-var ≡ 1-var everywhere".

**Use this run for:** precise per-variant *relative rates / frequencies* on B4.
**Not for:** rate *dynamics* (dip-then-rise) — see analyses 2 and 3.

## Outputs (`outputs/`)
`segmented_growth_rates.csv` (per-trajectory 4-param fit; all constant here),
`segmented_{model_selection,sample_summary,allele_effects}.csv` (rollups),
`model_comparison_*` / `model_fit_error*` (1-var vs 4-var, identical here),
`figures/segmented_overview.png` (reduced constant-only overview + single-slope distribution),
`intermediate/` (tidy long table, depth, OD time axis).
