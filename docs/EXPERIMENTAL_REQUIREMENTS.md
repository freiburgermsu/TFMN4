# Experimental requirements — what data gives which growth-rate quantity

This document maps **what you measure** to **what you can identify** about variant
growth, and states exactly what new data is required to reach an absolute,
per-variant **biomass-per-hour** rate. It is the companion to `docs/METHODS.md`.

---

## 1. Why the current data is relative-only

Two independent limits, both structural (not fixable by reprocessing):

1. **Barcode counts are compositional.** They give each variant's *fraction*
   `f_i = N_i / Σ_j N_j`, so only rate *differences* `s_i = r_i − r̄` are
   identifiable — never the absolute level `r̄`.
2. **OD is a single bulk signal, sampled at stationary phase.** It yields one
   community-average quantity per sample, and at the barcode-sampling times the
   culture is at carrying capacity (no growth signal). The within-cycle
   exponential slope recovers the bulk rate `μ_bulk = d ln(OD)/dt = Σ_i f_i r_i = r̄`
   — but that is the *community mean*, one number, not a per-variant rate.

## 2. Does converting OD → biomass fix it? No.

OD→biomass calibration is a **monotonic rescaling of one aggregate number**. It
re-expresses the single community rate from OD-doublings/h into g-dry-weight/h. It
does **not** create the per-variant information that is missing, because the
per-variant split still comes only from the compositional reads.

Worse, biomass introduces *additional* per-variant confounds:

- **Reads count genome copies → cells; biomass weights by cell mass.** If variants
  differ in cell size/morphology, the cell-number fraction (what barcodes see) ≠
  the biomass fraction, so cell-specific and biomass-specific per-variant rates
  genuinely differ and cannot be reconciled without per-variant cell mass.
- **Genome copies ≠ cells during fast growth** (replication-fork multiplicity), so
  even reads→cells is growth-state dependent. A single-copy amplicon locus
  mitigates but does not eliminate this.

**Conclusion:** OD→biomass upgrades the *units of the one community rate*; it does
not enable per-variant biomass/h.

---

## 3. Data → identifiable quantity map

| Data you have / add | Per-variant **relative** fitness `s_i` | Community **absolute** rate `r̄` | Per-variant **absolute** rate `r_i` | Units reachable |
|---|---|---|---|---|
| Barcode counts (compositional), end-of-cycle | ✅ (per-sample gauge) | ❌ | ❌ | log-ratio / cycle or / generation |
| + cross-sample bridging (this repo, `s09`) | ✅ (common relative scale) | ❌ | ❌ | / cycle, / generation |
| + within-cycle OD exponential slope (`s08`) | ✅ | ✅ (one number) | ❌ (fails go/no-go here) | adds **community** OD-specific h⁻¹ |
| + OD→biomass calibration curve | ✅ | ✅ (in biomass units) | ❌ | community **gDW** h⁻¹ |
| **+ absolute abundance anchor** (spike-in / qPCR-ddPCR / cell count) | ✅ | ✅ | ⚠️ only if also sampled during growth | absolute Nᵢ(t) |
| **+ within-exponential-phase barcode sampling** | ✅ | ✅ | ✅ (in the competition) | per-variant cells/h (→ biomass/h with calibration) |
| **+ per-variant cell mass** (Coulter/flow/microscopy) | — | — | converts cells/h ↔ biomass/h | per-variant **gDW** h⁻¹ |
| **Monoculture / arrayed growth curves** | n/a (no competition) | n/a | ✅ **intrinsic** μᵢ | per-variant OD or gDW h⁻¹ |

Legend: ✅ identifiable · ⚠️ conditional · ❌ not identifiable · — not applicable.

---

## 4. What to add, by goal

### Goal A — absolute per-variant rate *within the competition* (cells/h, then biomass/h)
Break **both** structural limits in the next experimental round:

1. **Within-exponential-phase barcode sampling** *(the linchpin)* — sample barcodes
   at several points *during* the growth phase of a cycle (e.g. h 2/4/6/8 of the
   ~26 h cycle), not only at the stationary end. Now frequency *and* density change
   during measured growth, so `d ln N_i/dt` is a real per-variant rate and the OD
   anchor is finally valid (same phase for both measurements).
2. **Absolute abundance anchor** at each timepoint — any one of:
   - **Spike-in standard**: a fixed amount of an external uniquely-barcoded
     reference per sample; reads-relative-to-spike-in → absolute `N_i(t)`.
   - **qPCR / ddPCR** of the verAB locus → absolute copies/mL × barcode fraction.
     ddPCR is absolute without a standard curve.
   - **Total cell count** (flow cytometry) or **CFU** × barcode fraction (CFU adds
     viability).
3. **OD→biomass calibration** (OD₆₀₀ vs gDW/L or vs cell count), once per
   strain/condition, to express the result in biomass rather than OD-equivalent.

(1) + (2) → absolute per-variant **cells/h** in context; + (3) and per-variant cell
mass → **biomass/h**.

### Goal B — *intrinsic* per-variant growth rate (the cleanest biomass/h)
- **Monoculture or low-complexity arrayed growth curves** for each variant (or the
  top candidates) under the same condition → intrinsic μᵢ directly from the OD/biomass
  curve, with no compositional confound. Add the OD→biomass calibration for gDW/h.
- Note: this measures *intrinsic* growth, not *competitive fitness*; the two differ
  when there is frequency dependence, cross-feeding, or resource competition. Use it
  to **validate** the competition-derived ranking, not replace it.

---

## 5. What the in-progress amplicon resequencing does (and doesn't)

The verAB amplicon run (replacing the ~200X WGS extraction) **sharpens the relative
analysis**: lower detection floor (~0.5–1% → ~0.01%), smaller sampling noise,
more variants observed at ≥2 timepoints (more bridges), and better-calibrated SEs.
It does **not** add absolute information — it cannot make per-variant rates
absolute. That still requires Goal-A or Goal-B data above.

**Two QC checks to run when amplicon data lands** (hooks already planned in the
pipeline): a WGS-vs-amplicon Bland–Altman on shared `(Sample, Candidate, Transfer)`
cells (a time-invariant PCR bias cancels in the slope; a composition-dependent bias
does **not** and masquerades as selection — a **mock community at known ratios**
through the same PCR detects it), and a re-run of bridge connectivity (deeper counts
should rescue the degree-1 sample).

---

## 6. One-line summary

> Relative per-variant fitness is fully identifiable (and reproducible at the allele
> level). One absolute community rate is measurable from the OD growth phase.
> **Absolute per-variant biomass/h is not reachable from competition + bulk-OD data
> at any unit choice** — it needs within-growth-phase sampling plus an absolute
> abundance anchor (in-competition), or monoculture growth curves (intrinsic), and a
> biomass calibration to leave OD-equivalent units.
