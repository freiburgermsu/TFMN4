"""
lims01 - Evaluate the LIMS mirror (lims_mirror.backup.db) and extract one tidy input
set PER ENVIRONMENT for the existing TFMN4 pipelines.

The database is opened READ-ONLY (sqlite URI mode=ro) and never modified. Every LIMS
table keeps soft-deleted row versions (`deleted = 1`); only live rows are used.

What is extracted (everything else in the DB is inventoried but not modelled):
  * verAB barcode counts (table verAB_barcodes) for time-series cultures, i.e. sample
    names of the form <culture>.T<transfer>.<suffix>. One BARCODE DATASET per
    (experiment, condition = carbon source, host strain, assay), so every environment is
    simulated separately and a culture's environment is never mixed with another's.
  * the robotic OD curves (Robotic_OD) of exactly those cultures' wells, for the
    time / generations axis (s02) and the community bulk rate (s08).
  * copy-number trajectories (Copy_numbers) per (genomic region, carbon source,
    platform) for the cn pipeline.

Barcode QC (reported cell by cell in inventory/barcode_cells.csv):
  - allele names are normalised to the repo convention (A078 -> A78, B003 -> B3);
  - partial reads with an un-called allele (A_UNK / B_UNK) are dropped, exactly as the
    repo's SeqCenter CSV export already did;
  - a cell (culture, transfer) re-sequenced in several orders keeps its DEEPEST run;
  - amplicon cells shallower than MIN_DEPTH_AMPLICON reads are failed libraries
    (e.g. 1, 2, 4, 29 reads) and are dropped; WGS keeps every cell (legacy behaviour);
  - noDNA cultures are controls (no library) and are never fitted; their reads are
    reported as a contamination check;
  - the dataset's transfer set = transfers that pass QC in >= half of its cultures
    (WGS is pinned to the repo's original {3,4,6,13}); a culture is kept only if it
    passes QC at every transfer in the set (the repo's complete-trajectory rule).

Outputs (lims_data/):
  manifest.json                      dataset definitions read by config.py / cn_common.py
  barcodes/<name>.csv                tidy counts (Sample, Transfer, verA, verB, Candidate, Count, ...)
  od/<name>.csv                      OD rows for the dataset's wells
  copy_number/<name>.csv             cn01-compatible copy-number table
  inventory/*.csv                    the database evaluation (tables, orders, cells, cultures,
                                     contamination, not-modelable sets, copy-number lineages)
"""
from __future__ import annotations
import json
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "lims_mirror.backup.db"
LIMS = ROOT / "lims_data"
INV = LIMS / "inventory"
OUT_ROOT = "outputs_lims"

MIN_DEPTH_AMPLICON = 500         # reads; below this an amplicon cell is a failed library
MIN_DEPTH_WGS = 0                # WGS (~200x) keeps every cell, as the original analysis did
TRANSFER_MAJORITY = 0.5          # a transfer joins the set if >= this share of cultures pass QC
WGS_TRANSFERS = (3, 4, 6, 13)    # the repo's original WGS design (validation dataset)
WGS_RELIABLE_OD = (6, 13)        # ... and its original reliable-OD transfers

# Variants whose signal is judged a likely artifact. The MAIN dataset keeps all data; a
# labelled SENSITIVITY copy with these candidates removed is written next to it, so every
# conclusion can be checked with and without them (evidence: docs/LIMS_REANALYSIS.md).
SENSITIVITY_EXCLUSIONS = {
    "TFMN4_methoxybenzoate_ACN3788_amplicon": dict(
        verA=["A183"], tag="excl_A183",
        reason=("verA A183 is essentially absent from all 11 TFMN4 cultures through T8 (5 reads in 66 "
                "culture-transfers, ~260k reads) and from every WGS sample (0 reads, incl. T13), then "
                "appears at T10/T13 in all 11 cultures at once - dominated by A183-B63 (6-30% at T10, "
                "11-25% at T13; ~30 WGS reads per culture expected at T13, 0 seen) plus a dozen other A183 "
                "pairings at 1-6 reads - and only in the two June 2026 amplicon orders (ZQT7PJ, KZGC3B) -> "
                "suspected contamination of the T10/T13 amplicon material by an A183-carrying population")),
}
CARBON_SHORT = {"methoxybenzoate": "methoxybenzoate", "guaiacol": "guaiacol",
                "isovanillate": "isovanillate", "veratrate": "veratrate",
                "vanillate": "vanillate", "pyruvate": "pyruvate"}
# copy-number regions modelled (lower-cased Region_name -> label used in file names)
CN_REGIONS = {"dgoa-star": "dgoA", "kanamycin": "KanR", "verab": "verAB", "verc": "verC"}
CN_SKIPPED = {"del_6kb_acn3560": "deletion marker (0 = deleted), not an amplification trajectory",
              "del_30kb_tfmn3": "deletion marker (0 = deleted), not an amplification trajectory",
              "ver_cassette": "same cassette as verAB (see inventory correlation); verAB is modelled",
              "rpoa": "housekeeping reference, single timepoint only",
              "dgoa": "ancestral-stock colonies only (no trajectory)",
              "kanr": "ancestral-stock colonies only (no trajectory)"}

TS_PAT = re.compile(r"^(?P<culture>.+)\.T(?P<transfer>\d+)\.(?P<suffix>[A-Za-z0-9]+)$")
CN_PAT = re.compile(r"^(?P<lineage>.+)\.T(?P<transfer>\d+)(?:\.(?P<suffix>[A-Za-z0-9]+))?$")


# --------------------------------------------------------------------------- helpers
def connect():
    if not DB.exists():
        raise SystemExit(f"[lims01] database not found: {DB}")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def q(con, sql):
    return pd.read_sql_query(sql, con)


def norm_allele(a):
    """A078 -> A78, B003 -> B3 (the repo's naming); anything else unchanged."""
    return re.sub(r"^([AB])0*(\d+)$", r"\1\2", str(a))


def common_prefix(names):
    """Longest common dot-delimited prefix (with trailing dot) of culture names."""
    parts = [n.split(".") for n in names]
    out = []
    for toks in zip(*parts):
        if len(set(toks)) == 1:
            out.append(toks[0])
        else:
            break
    out = out[:max(len(min(parts, key=len)) - 2, 0)]   # always keep construct + replicate
    return ".".join(out) + "." if out else ""


def condition_table(con):
    c = q(con, 'select Name, Carbon_source, "Carbon_concentration_(in_mM)" conc_mM, Antibiotics '
               'from Conditions where deleted = 0')
    c["Name"] = c["Name"].str.strip()
    return c.drop_duplicates("Name").set_index("Name")


def carbon_of(condition, conds):
    """Carbon source + concentration for a Condition name (falls back to parsing it)."""
    key = re.sub(r"\s+", " ", str(condition)).strip()
    if key in conds.index:
        r = conds.loc[key]
        carbon = str(r["Carbon_source"]).strip().lower()
        if carbon == "pyruvate" and key.lower().startswith("vanillate"):
            carbon = "vanillate"               # LIMS row 'Vanillate 3 mM + Kan' is mislabelled pyruvate
        return carbon, float(r["conc_mM"]) if pd.notna(r["conc_mM"]) else np.nan
    m = re.match(r"^([A-Za-z]+)\s+([\d.]+)\s*mM", key)
    return (m.group(1).lower(), float(m.group(2))) if m else (key.lower(), np.nan)


def culture_metadata(con):
    """Live culture rows (Robotic_ALE_samples) keyed by culture name."""
    ale = q(con, "select Name, Experiment, Type, Condition, Strain_name, Transforming_DNA, "
                 "Replicate_number, Robotic_run, Plate_name, Microtiter_plate_well "
                 "from Robotic_ALE_samples where deleted = 0 and Name is not null and Name != ''")
    ale["Condition"] = ale["Condition"].str.replace(r"\s+", " ", regex=True).str.strip()
    return ale.drop_duplicates("Name").set_index("Name")


def parse_culture(name):
    """Fallback metadata from a culture name when the LIMS row is missing.
    TFMN7verAB.<strain>.<carbon code>.<library>.<rep> / TFMN4.exp2.<strain>.<construct>.<rep>."""
    toks = name.split(".")
    out = dict(Experiment=toks[0], Strain_name=None, construct=toks[-2] if len(toks) >= 2 else None,
               Replicate_number=toks[-1])
    for t in toks:
        if re.match(r"^ACN\d+$", t):
            out["Strain_name"] = t
    return out


# --------------------------------------------------------------------------- inventory
def inventory_tables(con):
    names = q(con, "select name from sqlite_master where type='table' and name != 'sqlite_sequence'")["name"]
    rows = []
    for n in names:
        cols = q(con, f'pragma table_info("{n}")')["name"].tolist()
        tot = int(q(con, f'select count(*) n from "{n}"')["n"].iat[0])
        live = int(q(con, f'select count(*) n from "{n}" where deleted = 0')["n"].iat[0]) if "deleted" in cols else tot
        rows.append(dict(table=n, rows_total=tot, rows_live=live, rows_soft_deleted=tot - live,
                         n_columns=len(cols)))
    t = pd.DataFrame(rows).sort_values("rows_live", ascending=False)
    t.to_csv(INV / "tables.csv", index=False)
    return t


# --------------------------------------------------------------------------- barcodes
def load_barcodes(con):
    d = q(con, "select Seqsample, Seqorder, Transformation_library, verA, verB, Count "
               "from verAB_barcodes where deleted = 0")
    d["assay"] = np.where(d["Seqorder"].str.startswith("SeqCenter"), "wgs", "amplicon")
    d["unk"] = d["verA"].str.contains("UNK", na=True) | d["verB"].str.contains("UNK", na=True)
    m = d["Seqsample"].str.extract(TS_PAT)
    d = pd.concat([d, m], axis=1)
    return d


def inventory_orders(d):
    g = (d.groupby(["Seqorder", "assay"])
         .apply(lambda x: pd.Series(dict(
             n_seqsamples=x["Seqsample"].nunique(),
             n_time_series_seqsamples=x.loc[x["culture"].notna(), "Seqsample"].nunique(),
             reads=int(x["Count"].sum()),
             frac_reads_unk_allele=round(float(x.loc[x["unk"], "Count"].sum() / max(x["Count"].sum(), 1)), 4),
             example_sample=x["Seqsample"].iloc[0])), include_groups=False)
         .reset_index())
    g.to_csv(INV / "barcode_orders.csv", index=False)
    return g


def barcode_cells(d):
    """Per (assay, culture, transfer, order): depth (identified alleles only) + QC.
    Reads with an un-called allele are dropped from the counts (log-ratios among the
    identified variants are unaffected; only the per-culture gauge shifts), but their
    share and the dominant half-called lineage are recorded so an invisible sweep by
    an un-callable allele is reported rather than silently hidden."""
    allts = d[d["culture"].notna()].copy()
    allts["transfer"] = allts["transfer"].astype(int)
    unk = (allts.assign(unk_reads=np.where(allts["unk"], allts["Count"], 0),
                        lab=allts["verA"].map(norm_allele) + "-" + allts["verB"].map(norm_allele))
           .groupby(["assay", "culture", "transfer", "Seqorder"])
           .apply(lambda x: pd.Series(dict(
               depth_all_reads=int(x["Count"].sum()),
               frac_reads_unk_allele=round(float(x["unk_reads"].sum() / max(x["Count"].sum(), 1)), 4),
               top_unk_lineage=(x[x["unk"]].sort_values("Count").iloc[-1]["lab"] if x["unk"].any() else ""))),
                  include_groups=False).reset_index())
    ts = allts[~allts["unk"]].copy()
    cell = (ts.groupby(["assay", "culture", "transfer", "Seqorder"])
            .agg(depth=("Count", "sum"), n_variants=("Count", "size")).reset_index())
    cell = cell.merge(unk, on=["assay", "culture", "transfer", "Seqorder"], how="left")
    cell["n_orders_for_cell"] = cell.groupby(["assay", "culture", "transfer"])["Seqorder"].transform("nunique")
    cell["rank"] = cell.groupby(["assay", "culture", "transfer"])["depth"].rank(ascending=False, method="first")
    cell["selected_run"] = cell["rank"] == 1
    mind = np.where(cell["assay"] == "amplicon", MIN_DEPTH_AMPLICON, MIN_DEPTH_WGS)
    cell["min_depth"] = mind
    cell["qc_pass"] = cell["selected_run"] & (cell["depth"] >= cell["min_depth"])
    low = cell["depth"] < cell["min_depth"]
    unk_low = low & (cell["depth_all_reads"] >= cell["min_depth"])
    cell["qc_reason"] = np.select(
        [~cell["selected_run"], unk_low, low],
        ["superseded by a deeper re-sequencing of the same cell",
         "identified-allele depth below minimum: most reads carry an un-called allele "
         "(see top_unk_lineage) - a sweep the allele caller cannot resolve",
         "below minimum depth (failed library)"],
        default="pass")
    return ts, cell.drop(columns="rank")


def barcode_datasets(con, ts, cell, meta, conds):
    """Auto-discover one dataset per (experiment, condition, strain, assay)."""
    cult = pd.DataFrame({"culture": sorted(cell["culture"].unique())})
    rows = []
    for c in cult["culture"]:
        if c in meta.index:
            r = meta.loc[c]
            rows.append(dict(culture=c, Experiment=r.Experiment, Condition=r.Condition,
                             Strain_name=r.Strain_name, Transforming_DNA=r.Transforming_DNA,
                             Replicate=str(r.Replicate_number or c.split(".")[-1]),
                             Robotic_run=r.Robotic_run, Plate_name=r.Plate_name,
                             Microtiter_plate_well=r.Microtiter_plate_well, metadata_source="Robotic_ALE_samples"))
        else:
            p = parse_culture(c)
            rows.append(dict(culture=c, Experiment=p["Experiment"], Condition=None, Strain_name=p["Strain_name"],
                             Transforming_DNA=p["construct"], Replicate=p["Replicate_number"],
                             Robotic_run=None, Plate_name=None, Microtiter_plate_well=None,
                             metadata_source="parsed from name (no live LIMS row)"))
    cult = pd.DataFrame(rows)
    cult["construct"] = cult["culture"].str.split(".").str[-2]
    cult["is_control"] = (cult["Transforming_DNA"].astype(str).str.lower().eq("nodna")
                          | cult["culture"].str.contains(r"\.noDNA\.", regex=True))
    cb = cult["Condition"].map(lambda x: carbon_of(x, conds) if isinstance(x, str) else (None, np.nan))
    cult["carbon"] = [x[0] for x in cb]
    cult["carbon_mM"] = [x[1] for x in cb]

    datasets, cult_rows = [], []
    assays = cell[["culture", "assay"]].drop_duplicates()
    cult_a = cult.merge(assays, on="culture")
    ok = cult_a["Condition"].notna() & ~cult_a["is_control"]
    for (exp, cond, strain, assay), g in cult_a[ok].groupby(["Experiment", "Condition", "Strain_name", "assay"]):
        cells = cell[(cell["assay"] == assay) & cell["culture"].isin(g["culture"]) & cell["qc_pass"]]
        n_cult = g["culture"].nunique()
        share = cells.groupby("transfer")["culture"].nunique() / n_cult
        if assay == "wgs":
            tset = tuple(t for t in WGS_TRANSFERS if t in share.index)
        else:
            tset = tuple(int(t) for t in share[share >= TRANSFER_MAJORITY].index)
        have = cells.groupby("culture")["transfer"].apply(set)
        keep = sorted(c for c in g["culture"] if c in have.index and set(tset) <= have[c])
        for c in g["culture"]:
            miss = sorted(set(tset) - have.get(c, set()))
            cult_rows.append(dict(culture=c, assay=assay, dataset_transfers=",".join(map(str, tset)),
                                  included=c in keep,
                                  reason="complete trajectory" if c in keep else
                                  f"missing/failed QC at T{','.join(map(str, miss))}"))
        if len(keep) < 2 or len(tset) < 3:
            continue
        carbon, mM = g["carbon"].iloc[0], g["carbon_mM"].iloc[0]
        exp_short = "TFMN7" if str(exp).startswith("TFMN7") else str(exp)
        name = f"{exp_short}_{CARBON_SHORT.get(carbon, carbon)}_{strain}_{assay}"
        datasets.append(dict(
            id=f"lims_{name}", name=name, kind="barcode", experiment=exp, condition=cond,
            carbon=carbon, carbon_mM=mM, strain=strain, assay=assay,
            role="validation (same data as the repo's WGS CSV)" if assay == "wgs" else "new analysis",
            transfers=list(tset), cultures=keep,
            robotic_run=g["Robotic_run"].dropna().iloc[0] if g["Robotic_run"].notna().any() else None,
            plate=g["Plate_name"].dropna().iloc[0] if g["Plate_name"].notna().any() else None,
            min_depth=int(MIN_DEPTH_AMPLICON if assay == "amplicon" else MIN_DEPTH_WGS),
        ))
    cdf = cult.merge(pd.DataFrame(cult_rows), on="culture", how="left")
    mx = (cell[cell["selected_run"]].sort_values("frac_reads_unk_allele")
          .groupby(["culture", "assay"]).tail(1)[["culture", "assay", "frac_reads_unk_allele", "top_unk_lineage", "transfer"]]
          .rename(columns={"frac_reads_unk_allele": "max_frac_reads_unk", "transfer": "max_unk_transfer"}))
    cdf = cdf.merge(mx, on=["culture", "assay"], how="left")
    cdf["flag_unk_lineage_over_10pct"] = cdf["max_frac_reads_unk"] > 0.10
    return datasets, cdf


def write_barcode_dataset(ds, ts, cell, cult, con):
    keep = set(ds["cultures"]); tset = set(ds["transfers"])
    sel = cell[(cell["assay"] == ds["assay"]) & cell["culture"].isin(keep)
               & cell["transfer"].isin(tset) & cell["qc_pass"]][["culture", "transfer", "Seqorder"]]
    d = ts[ts["assay"] == ds["assay"]].merge(sel, on=["culture", "transfer", "Seqorder"])
    d["verA"] = d["verA"].map(norm_allele); d["verB"] = d["verB"].map(norm_allele)
    d["Candidate"] = d["verA"] + "-" + d["verB"]
    d = (d.groupby(["culture", "transfer", "Seqorder", "verA", "verB", "Candidate"], as_index=False)["Count"].sum())
    meta = cult.drop_duplicates("culture").set_index("culture")
    d["Sample"] = d["culture"]
    d["DNA_construct"] = d["culture"].map(meta["construct"])
    d["Replicate"] = d["culture"].map(meta["Replicate"])
    d["Microtiter_plate_well"] = d["culture"].map(meta["Microtiter_plate_well"])
    d["Transfer"] = d["transfer"].astype(int)
    d = d[["Sample", "DNA_construct", "Replicate", "Microtiter_plate_well", "Transfer",
           "verA", "verB", "Candidate", "Count", "Seqorder"]].sort_values(["Sample", "Transfer", "Candidate"])
    p = LIMS / "barcodes" / f"{ds['name']}.csv"
    d.to_csv(p, index=False)
    ds["barcodes_csv"] = str(p.relative_to(ROOT))
    ds["sample_prefix"] = common_prefix(sorted(keep))
    depth = d.groupby(["Sample", "Transfer"])["Count"].sum()
    ds.update(n_cultures=len(keep), n_rows=len(d), n_variants=int(d["Candidate"].nunique()),
              median_depth=int(depth.median()), min_depth_observed=int(depth.min()),
              wells=sorted(meta.loc[sorted(keep), "Microtiter_plate_well"].dropna().unique().tolist()),
              seqorders=sorted(d["Seqorder"].unique().tolist()))
    return d


def write_od_dataset(ds, con):
    """OD rows of the dataset's wells on its robotic run / plate (s02, s08)."""
    run, plate = ds["robotic_run"], ds["plate"]
    wells = ds["wells"]
    if not run or not plate or not wells:
        raise SystemExit(f"[lims01] {ds['name']}: no robotic run/plate/wells to link OD")
    wl = ",".join(f"'{w}'" for w in wells)
    od = q(con, f"select experiment, series, transfer, reading, datetime, od, background, "
                f"Microtiter_plate_well, sample_name, Condition from Robotic_OD where deleted = 0 "
                f"and experiment = '{run}' and series = '{plate}' and Microtiter_plate_well in ({wl})")
    if od.empty:
        raise SystemExit(f"[lims01] {ds['name']}: no OD rows for {run}/{plate}")
    p = LIMS / "od" / f"{ds['name']}.csv"
    od.to_csv(p, index=False)
    dt = pd.to_datetime(od["datetime"], errors="coerce")
    starts = dt.groupby(od["transfer"]).min().sort_index()
    gaps = starts.diff().dt.total_seconds().dropna() / 3600.0
    ds["od_csv"] = str(p.relative_to(ROOT))
    ds["od_rows"] = int(len(od))
    ds["od_transfers"] = int(od["transfer"].nunique())
    ds["hours_per_transfer_measured"] = round(float(np.median(gaps)), 1)
    # well <-> culture consistency between the barcode metadata and the OD table
    nm = od.dropna(subset=["sample_name"]).drop_duplicates("Microtiter_plate_well")
    ds["od_well_name_mismatches"] = int(sum(
        1 for _, r in nm.iterrows() if r.sample_name not in ds["cultures"]
        and str(r.sample_name).split(".")[0] == ds["cultures"][0].split(".")[0]))
    if ds["assay"] == "wgs":
        ds["reliable_od"] = list(WGS_RELIABLE_OD)
    else:
        ds["reliable_od"] = list(ds["transfers"])
    return od


def contamination_report(d, cult):
    ctl = cult[cult["is_control"]][["culture", "Experiment", "Condition", "Strain_name"]].drop_duplicates()
    x = d[d["culture"].isin(ctl["culture"]) & ~d["unk"]]
    if x.empty:
        out = pd.DataFrame(columns=["culture", "transfer", "Seqorder", "reads", "n_variants"])
    else:
        out = (x.groupby(["culture", "transfer", "Seqorder"])
               .agg(reads=("Count", "sum"), n_variants=("Count", "size")).reset_index()
               .merge(ctl, on="culture", how="left"))
        top = (x.sort_values("Count", ascending=False).groupby(["culture", "transfer"]).head(3)
               .assign(v=lambda z: z.verA.map(norm_allele) + "-" + z.verB.map(norm_allele) + ":" + z.Count.astype(str))
               .groupby(["culture", "transfer"])["v"].apply(", ".join).rename("top_variants").reset_index())
        out = out.merge(top, on=["culture", "transfer"], how="left")
        # where else is the control's dominant variant the top variant at that transfer?
        ok = d[d["culture"].notna() & ~d["unk"] & ~d["culture"].isin(ctl["culture"])].copy()
        ok["v"] = ok["verA"].map(norm_allele) + "-" + ok["verB"].map(norm_allele)
        tot = ok.groupby(["culture", "transfer"])["Count"].transform("sum")
        ok["freq"] = ok["Count"] / tot
        src = []
        for _, r in out.iterrows():
            v = str(r.get("top_variants", "")).split(":")[0]
            hit = ok[(ok["v"] == v) & (ok["transfer"] == r["transfer"]) & (ok["freq"] > 0.2)]
            src.append("; ".join(f"{h.culture} ({h.freq:.0%})" for h in hit.sort_values("freq", ascending=False).itertuples())[:400])
        out["cultures_where_top_variant_dominates_same_transfer"] = src
    out.to_csv(INV / "barcode_nodna_controls.csv", index=False)
    return out


def not_modelable(d):
    nts = d[d["culture"].isna()]
    g = (nts.groupby(["Seqorder", "Seqsample"]).agg(reads=("Count", "sum"), n_variants=("Count", "size"))
         .reset_index())
    def why(r):
        if r.Seqorder.startswith("LTA_"):
            return "isolate genotyping (1 read = the verAB identity of one colony); not a time series"
        if re.match(r"^verABLib_sub_\d+$", r.Seqsample):
            return "sub-library composition (input pool); no time axis"
        return "single end-point selection (no transfer index / no T0); slope not identifiable"
    g["reason"] = g.apply(why, axis=1)
    g.to_csv(INV / "barcode_not_modelable.csv", index=False)
    return g


# --------------------------------------------------------------------------- copy number
def cn_context(lineage, experiment, region):
    """Grouping label for the cn figures (generalises cn_common.context_of)."""
    lin = str(lineage)
    if "noDNA" in lin:
        return "control (noDNA)"
    toks = lin.split(".")
    if region == "dgoa-star" and lin.startswith(("TFMN1.", "TFMN4.exp1.")):
        import cn_common
        return cn_common.context_of(lin)
    if lin.startswith("EASy"):
        return f"{toks[0]} {toks[1]}"
    if lin.startswith("TFMN2."):
        return f"TFMN2 {toks[3]}"
    if lin.startswith("TFMN3."):
        return f"TFMN3 {toks[1]}"
    if lin.startswith("TFMN4.exp2."):
        c = toks[3]
        return "TFMN4 " + ("concX" if c.startswith("concX") else "concY" if c.startswith("concY") else c)
    return str(experiment)


def cn_genotype(lineage):
    stem = re.sub(r"\.\d+$", "", str(lineage))
    stem = re.sub(r"\.[A-J]$", "", stem)                      # EASy replicate letters
    for pre in ("TFMN1.", "TFMN4.exp1.", "TFMN4.exp2.", "TFMN2.", "TFMN3.", "TFMN4."):
        if stem.startswith(pre):
            return stem[len(pre):]
    return stem


def copy_number(con, meta, conds):
    cn = q(con, "select Seqsample, Seqorder, Region_name, Region_CN, Refgenome_avg_cov, Region_avg_cov "
                "from Copy_numbers where deleted = 0")
    cn["region"] = cn["Region_name"].str.lower()
    m = cn["Seqsample"].str.extract(CN_PAT)
    cn = pd.concat([cn, m], axis=1)
    smp = q(con, "select Name, Experiment, Condition, Strain_name, Transforming_DNA from Samples where deleted = 0")
    smp["Condition"] = smp["Condition"].str.replace(r"\s+", " ", regex=True).str.strip()
    smp = smp.drop_duplicates("Name").set_index("Name")

    def meta_for(r):
        for key, tab in ((r.lineage, meta), (r.Seqsample, smp)):
            if isinstance(key, str) and key in tab.index:
                t = tab.loc[key]
                return t["Experiment"], t["Condition"], t["Strain_name"]
        return None, None, None
    mm = cn.apply(meta_for, axis=1, result_type="expand")
    cn[["experiment", "condition", "strain"]] = mm
    cb = cn["condition"].map(lambda x: carbon_of(x, conds) if isinstance(x, str) else (None, np.nan))
    cn["carbon"] = [x[0] for x in cb]; cn["carbon_mM"] = [x[1] for x in cb]

    # inventory of every region x order x experiment (+ the stock colonies)
    inv = (cn.assign(is_traj=cn["transfer"].notna())
           .groupby(["region", "Seqorder", "experiment", "condition"], dropna=False)
           .agg(n_rows=("Seqsample", "size"), n_lineages=("lineage", "nunique"),
                transfers=("transfer", lambda s: ",".join(str(int(x)) for x in sorted(set(pd.to_numeric(s.dropna())))))
                ).reset_index())
    inv["modelled"] = inv["region"].map(lambda r: "yes" if r in CN_REGIONS else CN_SKIPPED.get(r, "no"))
    inv.to_csv(INV / "copy_number_inventory.csv", index=False)

    # verAB vs ver_cassette redundancy (same samples)
    va = cn[cn.region == "verab"].set_index("Seqsample")["Region_CN"]
    vc = cn[cn.region == "ver_cassette"].set_index("Seqsample")["Region_CN"]
    va = va[~va.index.duplicated()]; vc = vc[~vc.index.duplicated()]
    j = pd.concat([va, vc], axis=1, join="inner").dropna()
    red = dict(verAB_vs_ver_cassette_pearson=round(float(j.corr().iloc[0, 1]), 4) if len(j) > 2 else None,
               n_shared_samples=int(len(j)))

    traj = cn[cn["transfer"].notna() & cn["region"].isin(CN_REGIONS) & cn["carbon"].notna()].copy()
    traj["transfer"] = traj["transfer"].astype(int)
    traj["platform"] = np.where(traj["lineage"].str.startswith("EASy"), "EASy", "TFMN")
    datasets = []
    for (region, carbon, platform), g in traj.groupby(["region", "carbon", "platform"]):
        nt = g.groupby("lineage")["transfer"].nunique()
        if (nt >= 3).sum() < 2:
            continue
        exps = sorted(g["experiment"].dropna().unique())
        tag = CN_REGIONS[region]
        exps_short = "EASy" if platform == "EASy" else "-".join(
            (("TFMN4exp1" if region == "dgoa-star" else "TFMN4exp2") if e == "TFMN4" else e) for e in exps)
        name = f"{tag}_{carbon}_{exps_short}"
        out = pd.DataFrame(dict(
            region=g["Region_name"], lineage=g["lineage"], sample=g["Seqsample"], transfer=g["transfer"],
            copy_number=g["Region_CN"], seqorder=g["Seqorder"], experiment=g["experiment"],
            condition=g["condition"], strain=g["strain"]))
        out["context"] = [cn_context(l, e, region) for l, e in zip(out["lineage"], out["experiment"])]
        out["genotype"] = out["lineage"].map(cn_genotype)
        # ancestral stock colonies for the TFMN dgoA set (the legacy CSV carried ACN2586's)
        if region == "dgoa-star" and platform == "TFMN":
            st = cn[(cn["Seqsample"] == "ANLstock.ACN2586.colony3") & (cn["Region_name"] == "dgoA-star")]
            if len(st):
                out = pd.concat([pd.DataFrame(dict(
                    region=st["Region_name"], lineage=st["Seqsample"], sample=st["Seqsample"],
                    transfer=np.nan, copy_number=st["Region_CN"], seqorder=st["Seqorder"],
                    experiment="ANLstock", condition=None, strain="ACN2586",
                    context="ancestral stock", genotype="stock")), out], ignore_index=True)
        out = out.sort_values(["context", "lineage", "transfer"], na_position="first")
        p = LIMS / "copy_number" / f"{name}.csv"
        out.to_csv(p, index=False)
        conds_used = sorted(g["condition"].dropna().unique())
        datasets.append(dict(
            id=f"lims_cn_{name}", name=name, kind="copy_number", region=g["Region_name"].iloc[0],
            tag=tag, carbon=carbon, conditions=conds_used, platform=platform, experiments=exps,
            cn_csv=str(p.relative_to(ROOT)), out=f"{OUT_ROOT}/copy_number/{name}",
            n_lineages=int(g["lineage"].nunique()), n_rows=int(len(g)),
            n_lineages_ge3_transfers=int((nt >= 3).sum()), n_lineages_ge4_transfers=int((nt >= 4).sum()),
            transfers=sorted(int(t) for t in g["transfer"].unique()),
            seqorders=sorted(g["Seqorder"].unique().tolist())))
    lin = (traj.groupby(["region", "carbon", "platform", "experiment", "lineage"])
           .agg(n_rows=("Seqsample", "size"), n_transfers=("transfer", "nunique"),
                transfers=("transfer", lambda s: ",".join(str(x) for x in sorted(set(s)))),
                cn_first=("Region_CN", "first"), cn_last=("Region_CN", "last")).reset_index())
    lin.to_csv(INV / "copy_number_lineages.csv", index=False)
    return datasets, red


# --------------------------------------------------------------------------- main
def main():
    for d in (LIMS, INV, LIMS / "barcodes", LIMS / "od", LIMS / "copy_number"):
        d.mkdir(parents=True, exist_ok=True)
    con = connect()
    tables = inventory_tables(con)
    conds = condition_table(con)
    meta = culture_metadata(con)

    d = load_barcodes(con)
    orders = inventory_orders(d)
    ts, cell = barcode_cells(d)
    datasets, cult = barcode_datasets(con, ts, cell, meta, conds)
    cell.to_csv(INV / "barcode_cells.csv", index=False)
    cult.to_csv(INV / "barcode_cultures.csv", index=False)
    contam = contamination_report(d, cult)
    nm = not_modelable(d)

    for ds in datasets:
        write_barcode_dataset(ds, ts, cell, cult, con)
        write_od_dataset(ds, con)
        ds["out"] = f"{OUT_ROOT}/barcode/{ds['name']}"
    for ds in list(datasets):                          # sensitivity copies (suspect variants removed)
        ex = SENSITIVITY_EXCLUSIONS.get(ds["name"])
        if not ex:
            continue
        sd = dict(ds); name = f"{ds['name']}_{ex['tag']}"
        b = pd.read_csv(ROOT / ds["barcodes_csv"])
        b = b[~b["verA"].isin(ex.get("verA", [])) & ~b["Candidate"].isin(ex.get("candidates", []))]
        excluded = [f"{a} (all partners)" for a in ex.get("verA", [])] + ex.get("candidates", [])
        p = LIMS / "barcodes" / f"{name}.csv"
        b.to_csv(p, index=False)
        sd.update(id=f"lims_{name}", name=name, barcodes_csv=str(p.relative_to(ROOT)),
                  out=f"{OUT_ROOT}/barcode/{name}", role=f"sensitivity: {ds['name']} without "
                  f"{', '.join(excluded)}", excluded=excluded, excluded_verA=ex.get("verA", []),
                  exclusion_reason=ex["reason"], n_rows=len(b), n_variants=int(b["Candidate"].nunique()))
        dep = b.groupby(["Sample", "Transfer"])["Count"].sum()
        sd.update(median_depth=int(dep.median()), min_depth_observed=int(dep.min()))
        datasets.append(sd)

    cn_sets, redundancy = copy_number(con, meta, conds)

    od_runs = q(con, "select experiment, series, Condition, count(*) n_rows, count(distinct Microtiter_plate_well) n_wells, "
                     "count(distinct transfer) n_transfers, min(datetime) first_read, max(datetime) last_read "
                     "from Robotic_OD where deleted = 0 group by 1, 2, 3 order by 1, 2, 3")
    od_runs.to_csv(INV / "od_runs.csv", index=False)

    manifest = dict(
        source_db=DB.name, source_db_bytes=DB.stat().st_size,
        qc=dict(min_depth_amplicon=MIN_DEPTH_AMPLICON, min_depth_wgs=MIN_DEPTH_WGS,
                transfer_majority=TRANSFER_MAJORITY, wgs_transfers=list(WGS_TRANSFERS),
                unk_alleles="dropped", resequenced_cells="deepest run kept", controls="noDNA excluded"),
        barcode_datasets=datasets, copy_number_datasets=cn_sets,
        cn_region_redundancy=redundancy,
        n_reads_unk_dropped=int(d.loc[d["unk"], "Count"].sum()),
        n_cells_failed_depth=int(((cell["depth"] < cell["min_depth"]) & cell["selected_run"]).sum()),
        n_cells_superseded=int((~cell["selected_run"]).sum()),
    )
    (LIMS / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"[lims01] {DB.name}: {len(tables)} tables; verAB_barcodes {len(d)} live rows in "
          f"{d['Seqorder'].nunique()} orders ({int(d['Count'].sum())} reads; "
          f"{manifest['n_reads_unk_dropped']} UNK-allele reads dropped).")
    print(f"[lims01] barcode cells: {len(cell)} (culture x transfer x order); "
          f"{manifest['n_cells_superseded']} superseded by deeper re-sequencing, "
          f"{manifest['n_cells_failed_depth']} failed depth QC (<{MIN_DEPTH_AMPLICON} amplicon reads).")
    print(f"[lims01] noDNA control cells with barcode reads: {len(contam)}; "
          f"single-timepoint (not modelable) seqsamples: {len(nm)}.")
    print(f"[lims01] {len(datasets)} barcode datasets (one per experiment x environment x strain x assay):")
    for ds in datasets:
        print(f"        {ds['name']:42s} {ds['n_cultures']:2d} cultures  T{ds['transfers']}  "
              f"{ds['n_variants']:4d} variants  median depth {ds['median_depth']}  "
              f"{ds['hours_per_transfer_measured']} h/transfer")
    print(f"[lims01] {len(cn_sets)} copy-number datasets (region x carbon source x platform):")
    for ds in cn_sets:
        print(f"        {ds['name']:42s} {ds['n_lineages']:3d} lineages ({ds['n_lineages_ge4_transfers']} with >=4 transfers)")
    print(f"[lims01] wrote {LIMS.relative_to(ROOT)}/manifest.json + barcodes/ od/ copy_number/ inventory/")
    return manifest


if __name__ == "__main__":
    main()
