# Amplicon re-sequencing analysis (Plasmidsaurus X2VH94)

The same pipeline (local relative-fitness `s01`–`s07` + global static-rate
optimization `s12`/`s12b`/`s12c`) applied to the deep verAB **amplicon** data, with
deliverables written to **separate** output directories so the WGS results in
`outputs/` are never overwritten.

## The data
`amplicon_data/Plasmidsaurus_2026-06-10_X2VH94/` — `amplicons_reads.csv` gives a
per-read verA/verB assignment (the pipeline rebuilds full-resolution counts from it,
not the pre-binned 19-variant `counts_total.csv`).

- **One well, B4**, deeply re-sequenced (vs the 11-well WGS design).
- **Transfers {3, 4, 13}** (no T6).
- **Two carbon sources** (`metadata.xlsx`): **`4MXB`** = 4-methoxybenzoate (the
  selective substrate, sample tag `M`) and **`pyruvate`** (permissive control, tag `p`).
- **Three primer sets** (`fr`, `fbcr`, `bcfr`) = **technical replicates**.
- Depth ≈ **3,800 reads/cell** (per primerset) / ~10k per (carbon, transfer) —
  ~20–50× the WGS ~190, with a detection floor ~100× lower.

## How it maps onto the pipeline
Because 4MXB and pyruvate are **different selective environments** (growth rates
genuinely differ between them), the pipeline is run **per carbon source** — mirroring
the WGS design, which pooled across same-environment samples. Within a carbon source
the **three primer sets are the cross-sample (technical-replicate) units**, each a
timecourse over {3,4,13}. The OD anchor links well B4 to its within-cycle OD curve:
`4MXB → series exp2`, `pyruvate → series exp1` (both `Condition`-matched).

## Run it
```bash
TFMN4_DATASET=amplicon_4MXB     ~/Documents/py_venv/bin/python scripts/run_all.py
TFMN4_DATASET=amplicon_pyruvate ~/Documents/py_venv/bin/python scripts/run_all.py
# default (no env var) = the original WGS run -> outputs/
```
Outputs land in `outputs_amplicon_4MXB/` and `outputs_amplicon_pyruvate/` (same file
layout and column meanings as `outputs/`; see DATA_DICTIONARY.md).

## Results

| | 4MXB (selective) | pyruvate (permissive) |
|---|---|---|
| variants (with slope info) | 212 (82) | 251 (104) |
| community μ_bulk | **0.30 /h** (doubling 2.3 h) | **0.83 /h** (doubling 0.83 h) |
| global variant rate range | 0.16–0.39 /h | 0.60–1.09 /h |
| global fit freq R² | 0.996 | 0.995 |
| absolute-level identifiability | mostly `mixed` (weak anchor) | mostly `data-pinned` (clean anchor) |
| LOSO heterogeneity (median) | 0.013 /h | 0.011 /h |

- **One static rate per variant reproduces the relative counts almost perfectly**
  (freq R² ≈ 0.995 in both) — far tighter than the WGS R² ≈ 0.95, thanks to depth.
- **Pyruvate grows ~3× faster** than 4MXB at the community level, and every variant
  is correspondingly faster — the expected easy-vs-hard carbon-source contrast.
- **Same winners, different magnitudes:** `A81-B8` sweeps under both (≈87 % of T13
  reads on pyruvate, ≈67 % on 4MXB); `A186-B79` and `A143-B84` stay more competitive
  under 4MXB. Shared-variant rate correlation across conditions r ≈ 0.64 — the same
  players, with the selective environment modulating the rates.
- **Technical reproducibility is excellent:** leave-one-primerset-out heterogeneity
  ≈ 0.01 /h, i.e. the three primer sets agree closely (little primer/PCR bias in the
  rate estimates).
- **Absolute level is better anchored under pyruvate** (faster, cleaner exponential
  phase → better μ_bulk), so most pyruvate rates are `data-pinned`; under 4MXB the
  slow noisy growth leaves the absolute level weakly anchored (mostly `mixed`, and the
  near-threshold flag is borderline). As always, the **relative ordering is robust**;
  the absolute h⁻¹ level is community-dominated (see METHODS §8–9).

## Caveats
Same as the WGS analysis (METHODS §7–9): rates are relative-ordering-robust but
absolute-level depends on the OD anchor; the OD curves used are from the original
April experiment for well B4 under the matching `Condition` (assumed to describe the
re-grown cultures). The primer sets are technical, not biological, replicates — so the
cross-sample machinery here measures technical reproducibility, and biological
replication (more wells) would still be needed to generalize beyond lineage B4.
