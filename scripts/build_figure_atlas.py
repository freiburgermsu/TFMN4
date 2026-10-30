"""
build_figure_atlas - one browsable HTML page of every figure the pipelines produce.

    ~/Documents/py_venv/bin/python scripts/build_figure_atlas.py      # -> figure_atlas.html

Writes figure_atlas.html at the repo root: open it in a browser from a checkout and every
figure loads from its own path (outputs*/, 4param_*/, outputs_lims/...). Sections follow the
datasets and the LIMS environments in lims_data/manifest.json; captions say what each figure
shows. Re-run after re-running the pipelines (the page is stamped with the current commit).
Only git-tracked figures are listed, minus the vendor charts under amplicon_data/.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(description="Build figure_atlas.html")
ap.add_argument("--out", default=str(ROOT / "figure_atlas.html"))
ap.add_argument("--commit", default=None, help="commit to stamp (default: current HEAD)")
ap.add_argument("--files-json", default=None, help="also write the figure list as [{path}] JSON")
_a = ap.parse_args()
OUT = Path(_a.out)
COMMIT = _a.commit or subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                     capture_output=True, text=True).stdout.strip()

CAPTION = {
    "richness_collapse.png": "Variant richness and dominance across transfers: how fast each culture collapses onto a few lineages.",
    "sweep_trajectories.png": "Frequency trajectories of the most abundant variants in each culture.",
    "growth_matrix_heatmap.png": "Per-cycle selection coefficients of the top 30 variants across cultures.",
    "growth_matrix_overview.png": "Variant × culture growth matrix of selection coefficients, with their distribution.",
    "od_anchor_falsification.png": "OD-anchor go/no-go test: community growth rate by transfer, cross-culture bridge graph, allele forest.",
    "global_growth_fit.png": "Errors of the global static-rate fit: Laplace vs bootstrap SE, rates with bootstrap 95% CIs, leave-one-culture-out heterogeneity, OD-weight sensitivity.",
    "global_growth_forest_all.png": "Every variant's global growth rate with its 95% CI, ranked; hollow points have no slope information (rate set by the prior).",
    "segmented_overview.png": "4-parameter (broken-stick) model selection: classification, rate before vs after with the increasing-only line, decision gates, fit quality.",
    "segmented_parameters.png": "The four fitted parameters of adopted two-phase variants: rate before and after, breakpoint tally, initial abundance vs acceleration.",
    "segmented_growth_examples.png": "Broken-stick fits for the strongest rate switches: observed CLR trajectory, constant vs two-phase line, breakpoint.",
    "segmented_gallery.png": "Every adopted two-phase fit as small multiples.",
    "segmented_allele.png": "Rate accelerations tallied by verA and verB allele.",
    "model_comparison.png": "1-variable (constant slope) vs 4-variable (broken-stick) model: coverage, in-sample fit, BIC funnel, leave-one-out prediction.",
    "model_comparison_blindspot.png": "Where a single slope misleads: single slope vs phase swing, with dip-then-rise examples.",
    "model_fit_error.png": "Fit error of the 1-variable line vs the 4-variable broken-stick: per-variant RMSE, residuals by timepoint, in- vs out-of-sample error.",
    "growth_rates_forest.png": "Copy-number amplification rate per lineage (ln copy number per transfer) ± SE; FDR-significant lineages flagged.",
    "growth_examples.png": "Copy-number trajectories behind the rates.",
    "joint_error_reduction.png": "Joint WGS + amplicon 4MXB fit: per-variant SE reduction, effective sample size, rate forest, same-culture concordance gate.",
    "counts_vs_amplicon_4MXB.png": "WGS counts only vs joint (counts + amplicon) 4MXB rates: the amplicon tightens the 81 B4 variants and shifts no rate.",
    "fig1_capability.png": "Same 4-parameter model on three data regimes: what each regime makes estimable.",
    "fig2_b4_integration.png": "What integration adds on well B4, the one culture measured by both assays.",
    "fig3_rate_concordance.png": "Per-variant rate on well B4: the integrated fit tracks the deep amplicon.",
    "fig4_adopted_switches.png": "Adopted rate switches: barcode/WGS vs integrated.",
    "environment_overview.png": "Every LIMS environment fitted separately: richness, community growth rate, selection strength.",
    "allele_effects_heatmap.png": "Enzyme-allele effects across carbon sources, each environment fitted separately.",
    "allele_concordance.png": "Do allele effects replicate across experiments, host strains and carbon sources?",
    "top_variants_by_environment.png": "Fastest variants per environment, as advantage over the community mean ± bootstrap 95% CI.",
    "wgs_vs_amplicon_TFMN4.png": "TFMN4 4MXB: whole-genome vs targeted amplicon sequencing of the same 11 cultures.",
    "copy_number_by_environment.png": "Copy-number amplification rates per lineage across regions, carbon sources and platforms.",
    "sensitivity_excl_A183.png": "Contamination check: TFMN4 amplicon results with and without verA A183.",
}
ORDER = ["richness_collapse", "sweep_trajectories", "growth_matrix_overview", "growth_matrix_heatmap",
         "od_anchor_falsification", "global_growth_fit", "global_growth_forest_all", "segmented_overview",
         "segmented_parameters", "segmented_growth_examples", "segmented_gallery", "segmented_allele",
         "model_comparison", "model_comparison_blindspot", "model_fit_error"]


def caption(name):
    if name in CAPTION:
        return CAPTION[name]
    for suffix in ("growth_rates_forest.png", "growth_examples.png"):
        if name.endswith(suffix):
            return CAPTION[suffix]
    return ""


def title(name):
    stem = name[:-4]
    for pre in ("dgoA_", "KanR_", "verAB_", "verC_"):
        if stem.startswith(pre):
            stem = stem[len(pre):] + f" ({pre[:-1]})"
    return stem.replace("_", " ").replace("excl A183", "without A183")


def rank(path):
    stem = Path(path).stem
    for i, o in enumerate(ORDER):
        if stem == o:
            return i
    return 100


files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")
pngs = sorted(f for f in files if f.endswith(".png") and not f.startswith("amplicon_data/"))
man = json.loads((ROOT / "lims_data/manifest.json").read_text())


def png_size(f):
    head = (ROOT / f).read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def figs(prefix):
    sel = [f for f in pngs if f.startswith(prefix)]
    out = []
    for f in sorted(sel, key=lambda f: (rank(f), f)):
        w, h = png_size(f)
        out.append(dict(path=f, title=title(Path(f).name), caption=caption(Path(f).name), w=w, h=h))
    return out


def span(ts):
    return f"T{ts[0]}–T{ts[-1]} ({len(ts)} sampled transfers)"


groups = []
orig = dict(id="original", title="Original analyses",
            blurb="The repo's own datasets: SeqCenter WGS barcodes of TFMN4 on 4-methoxybenzoate (11 cultures), "
                  "Plasmidsaurus amplicons of well B4 on 4-methoxybenzoate and pyruvate, their joint fit, dgoA copy number, "
                  "and the packaged 4-parameter model comparison.", sections=[])
orig["sections"] += [
    dict(id="wgs", title="WGS barcodes · 4-methoxybenzoate", meta="11 cultures · T3, T4, T6, T13 · 314 variants · ~200 reads per sample", figs=figs("outputs/figures/")),
    dict(id="amp-4mxb", title="Amplicon · 4-methoxybenzoate, well B4", meta="3 primer sets · T3, T4, T13 · 212 variants", figs=figs("outputs_amplicon_4MXB/")),
    dict(id="amp-pyr", title="Amplicon · pyruvate, well B4", meta="3 primer sets · T3, T4, T13 · 251 variants", figs=figs("outputs_amplicon_pyruvate/")),
    dict(id="joint", title="Joint WGS + amplicon fit · 4-methoxybenzoate", meta="445 variants · WGS breadth + amplicon depth on the shared culture B4", figs=figs("outputs_joint_4MXB/")),
    dict(id="cn-dgoa", title="Copy number · dgoA", meta="per-lineage amplification rates", figs=figs("outputs_copynumber/")),
    dict(id="fourparam", title="4-parameter model packages", meta="amplicon only · WGS only · integrated · three-way comparison",
         figs=figs("4param_1") + figs("4param_2") + figs("4param_3") + figs("4param_4")),
]
for s in orig["sections"]:
    if s["id"] == "fourparam":
        for f in s["figs"]:
            pkg = f["path"].split("/")[0].split("_", 2)[-1].replace("_", " ")
            f["title"] = f"{pkg} · {f['title']}"
groups.append(orig)

lims = dict(id="lims", title="LIMS re-analysis",
            blurb="Every environment in the lab's LIMS database simulated separately (experiment × carbon source × host strain × assay; "
                  "copy number: region × carbon source × platform).", sections=[])
lims["sections"].append(dict(id="lims-compare", title="Across environments", meta="cross-environment comparison of the runs below",
                             figs=figs("outputs_lims/comparison/")))
for d in man["barcode_datasets"]:
    carbon = d["carbon"].replace("methoxybenzoate", "4-methoxybenzoate")
    label = f"{d['experiment']} · {carbon} · {d['strain']} · {d['assay'].upper() if d['assay'] == 'wgs' else d['assay']}"
    if "excl" in d["name"]:
        label += " · without A183"
    meta = (f"{d['n_cultures']} cultures · {span(d['transfers'])} · {d['n_variants']:,} variants · "
            f"median depth {d['median_depth']:,} reads")
    lims["sections"].append(dict(id="bc-" + d["name"].lower(), title=label, meta=meta, figs=figs(d["out"] + "/")))
for d in man["copy_number_datasets"]:
    label = f"Copy number · {d['tag']} · {d['carbon']} · {d['platform']}"
    meta = f"{d['n_lineages']} lineages ({d['n_lineages_ge4_transfers']} with ≥4 transfers)"
    lims["sections"].append(dict(id="cn-" + d["name"].lower(), title=label, meta=meta, figs=figs(d["out"] + "/")))
groups.append(lims)

listed = {f["path"] for g in groups for s in g["sections"] for f in s["figs"]}
missing = sorted(set(pngs) - listed)
assert not missing, missing
n = len(listed)
data = dict(groups=groups, commit=COMMIT, n=n)

html = (Path(__file__).parent / "figure_atlas_template.html").read_text()
html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__N__", str(n)).replace("__COMMIT__", COMMIT)
OUT.write_text(html)
if _a.files_json:
    Path(_a.files_json).write_text(json.dumps([{"path": p} for p in sorted(listed)]))
print(f"{n} figures in {sum(len(g['sections']) for g in groups)} sections -> {OUT}")
