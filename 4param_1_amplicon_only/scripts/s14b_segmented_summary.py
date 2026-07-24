"""
s14b - Summary / rollup tables for the s14 segmented (two-phase) regression.

Aggregates the per-(Sample, Candidate) table `segmented_growth_rates.csv` into the
three communication tables that parallel the single-slope assessment's summaries:

  segmented_model_selection.csv  - one tidy metric-per-row audit of the decision:
                                    the thresholds used and how the trajectories were
                                    classified (estimable -> testable -> adopted /
                                    held-by-increasing-only / weak-constant).
  segmented_sample_summary.csv   - per culture: how many variants accelerate, the
                                    breakpoints used, and the median phase rates.
  segmented_allele_effects.csv   - per verA and per verB allele: how often a variant
                                    carrying it shows an adopted increasing switch,
                                    and the mean acceleration -- "does a late speed-up
                                    cluster in particular enzyme parts?"

All rows are DATASET-aware (config.OUT); a 3-transfer amplicon dataset simply reports
zero testable / zero adopted (segmentation is not identifiable on 3 timepoints).
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C
from s14_segmented_growth import (BIC_MARGIN, REL_RATE_CHANGE_MIN,
                                   MIN_OBS_FOR_BREAKPOINT, MIN_SEG_OBS)


def _classify(r):
    """Split into the four communicable classes used everywhere downstream."""
    testable = r[r.breakpoint_testable]
    nontest = r[~r.breakpoint_testable]
    adopted = testable[testable.model_selected == "two_phase"]
    held = testable[(testable.model_selected == "constant") &
                    (testable.unconstrained_direction == "decelerating")]
    weak = testable[(testable.model_selected == "constant") &
                    (testable.unconstrained_direction != "decelerating")]
    return dict(testable=testable, nontest=nontest, adopted=adopted, held=held, weak=weak)


def model_selection_table(r, cls):
    def med(s):
        return round(float(np.nanmedian(s)), 4) if len(s) and s.notna().any() else np.nan
    rows = [
        ("dataset", C.DATASET),
        ("bic_margin_strong_evidence", BIC_MARGIN),
        ("rel_rate_change_min", REL_RATE_CHANGE_MIN),
        ("min_obs_for_breakpoint", MIN_OBS_FOR_BREAKPOINT),
        ("min_seg_obs", MIN_SEG_OBS),
        ("increasing_only", True),
        ("n_transfers_sampled", len(C.FOUR_TRANSFER_SET)),
        ("n_estimable_trajectories", len(r)),
        ("n_breakpoint_testable", len(cls["testable"])),
        ("n_non_testable", len(cls["nontest"])),
        ("n_two_phase_adopted_increasing", len(cls["adopted"])),
        ("n_held_constant_decrease_not_permitted", len(cls["held"])),
        ("n_weak_constant", len(cls["weak"])),
        ("frac_testable_accelerating",
         round(len(cls["adopted"]) / len(cls["testable"]), 4) if len(cls["testable"]) else np.nan),
        ("median_delta_bic_adopted", med(cls["adopted"]["delta_bic"])),
        ("median_r2_constant_testable", med(cls["testable"]["r2_constant"])),
        ("median_r2_two_phase_testable", med(cls["testable"]["r2_two_phase"])),
        ("median_acceleration_adopted_per_cycle", med(cls["adopted"]["abs_rate_change"])),
        ("median_r_init_adopted_per_cycle", med(cls["adopted"]["r_init_per_cycle"])),
        ("median_r_final_adopted_per_cycle", med(cls["adopted"]["r_final_per_cycle"])),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def sample_summary_table(r, cls):
    adopted = cls["adopted"]
    out = []
    for s, g in r.groupby("Sample"):
        gt = g[g.breakpoint_testable]
        ga = gt[gt.model_selected == "two_phase"]
        gh = gt[(gt.model_selected == "constant") & (gt.unconstrained_direction == "decelerating")]
        bp = ga.breakpoint_transfer.value_counts()
        out.append(dict(
            Sample=s,
            DNA_construct=g.DNA_construct.iloc[0],
            n_estimable=len(g),
            n_testable=len(gt),
            n_accelerating=len(ga),
            n_held_decrease=len(gh),
            frac_accelerating=round(len(ga) / len(gt), 4) if len(gt) else np.nan,
            bp_T4=int(bp.get(4, 0)), bp_T6=int(bp.get(6, 0)),
            median_r_init_adopted=round(float(ga.r_init_per_cycle.median()), 4) if len(ga) else np.nan,
            median_r_final_adopted=round(float(ga.r_final_per_cycle.median()), 4) if len(ga) else np.nan,
            median_acceleration_adopted=round(float(ga.abs_rate_change.median()), 4) if len(ga) else np.nan,
            median_logA_init=round(float(g.logA_init.median()), 4),
        ))
    df = pd.DataFrame(out).sort_values("n_accelerating", ascending=False)
    return df


def allele_effects_table(r, cls):
    """Per-allele acceleration rollup, stacked for verA and verB (a `level` column)."""
    frames = []
    for level in ("verA", "verB"):
        rows = []
        for allele, g in r.groupby(level):
            gt = g[g.breakpoint_testable]
            ga = gt[gt.model_selected == "two_phase"]
            if len(g) == 0:
                continue
            rows.append(dict(
                level=level, allele=allele,
                n_variants=len(g),
                n_testable=len(gt),
                n_accelerating=len(ga),
                frac_of_testable_accelerating=round(len(ga) / len(gt), 4) if len(gt) else np.nan,
                mean_acceleration=round(float(ga.abs_rate_change.mean()), 4) if len(ga) else np.nan,
                mean_r_init_adopted=round(float(ga.r_init_per_cycle.mean()), 4) if len(ga) else np.nan,
                mean_r_final_adopted=round(float(ga.r_final_per_cycle.mean()), 4) if len(ga) else np.nan,
                modal_breakpoint=(int(ga.breakpoint_transfer.mode().iloc[0])
                                  if len(ga) and not ga.breakpoint_transfer.mode().empty else np.nan),
            ))
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["level", "n_accelerating", "n_testable"], ascending=[True, False, False])
    return df


def main():
    r = pd.read_csv(C.OUT / "segmented_growth_rates.csv")
    cls = _classify(r)

    msel = model_selection_table(r, cls)
    ssum = sample_summary_table(r, cls)
    alle = allele_effects_table(r, cls)

    msel.to_csv(C.OUT / "segmented_model_selection.csv", index=False)
    ssum.to_csv(C.OUT / "segmented_sample_summary.csv", index=False)
    alle.to_csv(C.OUT / "segmented_allele_effects.csv", index=False)

    n_ad = len(cls["adopted"]); n_te = len(cls["testable"])
    print(f"[s14b] model-selection audit: {len(r)} estimable, {n_te} testable, "
          f"{n_ad} adopted increasing, {len(cls['held'])} held (decrease), "
          f"{len(cls['weak'])} weak-constant.")
    if n_ad:
        topA = alle[(alle.level == 'verA')].head(3)
        al = ", ".join(f"{x.allele}({int(x.n_accelerating)})" for _, x in topA.iterrows())
        print(f"[s14b] verA alleles with most adopted accelerations: {al}")
        top_s = ssum[ssum.n_accelerating > 0].head(3)
        sl = ", ".join(f"{x.Sample.split('.')[-2]}({int(x.n_accelerating)})" for _, x in top_s.iterrows())
        print(f"[s14b] cultures with most accelerations: {sl}")
    print("[s14b] wrote segmented_model_selection.csv, segmented_sample_summary.csv, "
          "segmented_allele_effects.csv")
    return dict(model_selection=msel, sample_summary=ssum, allele_effects=alle)


if __name__ == "__main__":
    main()
