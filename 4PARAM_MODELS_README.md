# 4-parameter growth model — three data regimes + comparison

Four self-contained folders. Each holds **all of its own scripts, inputs, and outputs** and
runs standalone with the project venv (`~/Documents/py_venv/bin/python run.py`). The three model
runs are independent and were executed **in parallel**.

The **"4-parameter model"** is the segmented / two-phase (broken-stick) per-variant growth
regression: for each variant it fits an **initial abundance**, an **initial growth rate**, a
**final growth rate**, and the **transfer at which the rate switches** (increasing switches
only, adopted only on strong evidence). It is the project's `s14` model; here it is packaged
per data regime.

The two source datasets (both from the TFMN4 competition; the "LIMS" lab data):
- **Barcode / WGS** — `SeqCenter_..._extracted_barcodes.csv` (11 wells, transfers {3,4,6,13}, ~200×).
- **Amplicon** — Plasmidsaurus `amplicons_reads.csv` (well B4 only, transfers {3,4,13}, deep ≈10k).
All three model runs use the shared **4-methoxybenzoate (4MXB)** environment so they are comparable.

| Folder | Analysis | Transfers | Estimable | Breakpoint-testable | Two-phase adopted |
|---|---|--:|--:|--:|--:|
| [`4param_1_amplicon_only/`](4param_1_amplicon_only/README.md) | 4-param on **amplicon only** | 3 | 186 | **0** | **0** |
| [`4param_2_barcode_only/`](4param_2_barcode_only/README.md) | 4-param on **barcode/WGS only** | 4 | 237 | 66 | 13 |
| [`4param_3_integrated/`](4param_3_integrated/README.md) | 4-param on **integrated (both)** | 4 | **308** | **71** | **14** |
| [`4param_4_comparison/`](4param_4_comparison/COMPARISON_REPORT.md) | figures + report across the 3 | — | — | — | — |

## The story in one paragraph
The amplicon is **deep but short** (3 transfers) → the 4th parameter (the breakpoint) is
**mathematically not identifiable**, so all 186 amplicon trajectories collapse to a constant
slope. The WGS design adds **transfer T6**, making the breakpoint identifiable → 13 real
increasing switches. **Integrating** the deep amplicon onto WGS's shared culture (well **B4**)
gets the best of both: it lifts estimable coverage (237 → 308), makes more variants testable
(66 → 71), and resolves one extra B4 switch (13 → 14) that WGS's shallow depth missed —
while re-estimating B4's noisy rates onto the high-depth amplicon signal. Full write-up and
four figures: **[`4param_4_comparison/COMPARISON_REPORT.md`](4param_4_comparison/COMPARISON_REPORT.md)**.

## Reproduce
```bash
PY=~/Documents/py_venv/bin/python
( cd 4param_1_amplicon_only && $PY run.py ) &
( cd 4param_2_barcode_only  && $PY run.py ) &
( cd 4param_3_integrated    && $PY run.py ) &
wait
( cd 4param_4_comparison && $PY run.py )   # reads the three runs' outputs; refits nothing
```

## Notes
- The model code (`s02`, `s14*`) is byte-identical to the parent pipeline; only each folder's
  `config.py` (local paths + fixed dataset) and the integrated folder's `s01` (the fusion) are
  new. So these packages reuse the project's tested, dataset-aware code.
- Folder 4's `inputs/` are literally the three models' output CSVs ("the fed data"); the
  comparison computes every number from them and writes `COMPARISON_REPORT.md` + 4 figures.
