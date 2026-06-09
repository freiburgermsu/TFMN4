"""
s02 - Build the time / generations axis from the robotic OD file (File B).

File B links to the barcode samples on series=='exp2' + Microtiter_plate_well.
Two things come out of it:

  1. Real elapsed time. Instrument timestamps show ~26.4 h per transfer cycle
     (NOT the 10 h stated in the task). The barcode snapshot at transfer T is
     taken at the end of cycle T (cells at stationary plateau).

  2. Generations per cycle, Delta g = log2(plateau_OD / inoculum_OD), from the
     within-cycle OD sigmoid. Cumulative generations at transfer T set the
     biologically correct selection clock; the per-generation selection
     coefficient is the canonical barcode-fitness unit.

Output (outputs/intermediate/):
  od_time_axis.csv - per transfer: start hours from T0, Delta g (median over the
                     focus wells), and cumulative generations. Rows for the four
                     sampled transfers are flagged.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import config as C


def main():
    well_map = pd.read_csv(C.INTER / "sample_well_map.csv")
    focus_wells = set(well_map["Microtiter_plate_well"].astype(str))

    usecols = ["series", "transfer", "reading", "datetime",
               "Microtiter_plate_well", "od", "background"]
    od = pd.read_csv(C.OD_CSV, usecols=usecols)
    od = od[od["series"].astype(str) == "exp2"].copy()
    od["well"] = od["Microtiter_plate_well"].astype(str)

    # ---- (1) transfer timing from datetimes ---------------------------------
    od["dt"] = pd.to_datetime(od["datetime"], errors="coerce")
    starts = od.groupby("transfer")["dt"].min()
    hours_from_t0 = (starts - starts.min()).dt.total_seconds() / 3600.0
    gap = starts.diff().dt.total_seconds() / 3600.0
    measured_hpt = float(np.nanmedian(gap.values))
    print(f"[s02] measured cycle length = {measured_hpt:.2f} h/transfer "
          f"(task stated {C.HOURS_PER_TRANSFER_NOMINAL} h)")

    # ---- (2) generations per cycle from the within-cycle sigmoid ------------
    foc = od[od["well"].isin(focus_wells)].copy()
    foc["reading_n"] = pd.to_numeric(foc["reading"], errors="coerce")  # 'contam' -> NaN
    foc = foc.dropna(subset=["reading_n"])
    foc["net_od"] = (foc["od"] - foc["background"]).clip(lower=0.0)

    dg_rows = []
    for (t, w), g in foc.groupby(["transfer", "well"]):
        g = g.sort_values("reading_n")
        inoc = max(float(g["net_od"].iloc[0]), C.OD_INOCULUM_FLOOR)
        plateau = float(g["net_od"].max())
        if plateau <= inoc:
            continue
        dg = np.log2(plateau / inoc)
        dg = float(np.clip(dg, *C.DELTA_G_BOUNDS))
        dg_rows.append((t, w, inoc, plateau, dg))
    dgdf = pd.DataFrame(dg_rows, columns=["transfer", "well", "inoc", "plateau", "delta_g"])
    dg_by_t = dgdf.groupby("transfer")["delta_g"].median()

    # Cumulative generations to the end of transfer T = sum of Delta g over
    # cycles 1..T (cycle 0 is the founding inoculation).
    all_t = sorted(t for t in dg_by_t.index if t >= 1)
    cumgen = {}
    running = 0.0
    for t in range(1, max(all_t) + 1):
        running += float(dg_by_t.get(t, dg_by_t.median()))
        cumgen[t] = running

    rows = []
    for t in sorted(starts.index):
        rows.append(dict(
            transfer=int(t),
            hours_from_T0=round(float(hours_from_t0.get(t, np.nan)), 2),
            delta_g=round(float(dg_by_t.get(t, np.nan)), 4) if t in dg_by_t.index else np.nan,
            cumgen=round(float(cumgen.get(t, np.nan)), 4) if t in cumgen else np.nan,
            is_sampled_transfer=int(t in C.FOUR_TRANSFER_SET),
        ))
    axis = pd.DataFrame(rows)
    # carry the measured cycle length as metadata column (constant)
    axis["measured_hours_per_transfer"] = round(measured_hpt, 3)
    axis.to_csv(C.INTER / "od_time_axis.csv", index=False)

    samp = axis[axis.is_sampled_transfer == 1]
    print("[s02] sampled-transfer axis:")
    for _, r in samp.iterrows():
        print(f"        T{int(r.transfer):<2d}  {r.hours_from_T0:6.1f} h   "
              f"cumgen={r.cumgen:.2f}")
    print(f"[s02] median Delta g/cycle (focus wells) = {dg_by_t.median():.2f} generations")
    print("[s02] wrote od_time_axis.csv")
    return axis


if __name__ == "__main__":
    main()
