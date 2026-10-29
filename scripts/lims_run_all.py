"""
lims_run_all - Re-apply the repo's pipelines to the LIMS mirror DB, one environment at a
time, and save every environment's results separately.

    ~/Documents/py_venv/bin/python scripts/lims_run_all.py              # everything
    ~/Documents/py_venv/bin/python scripts/lims_run_all.py --only TFMN7_guaiacol_ACN3788_amplicon
    ~/Documents/py_venv/bin/python scripts/lims_run_all.py --skip-extract --jobs 4

Stages:
  lims01   evaluate the DB + extract per-environment inputs   -> lims_data/
  barcode  scripts/run_all.py per barcode environment (TFMN4_DATASET=lims_<name>)
                                                              -> outputs_lims/barcode/<name>/
  cn       scripts/cn_run_all.py per copy-number environment (TFMN4_CN_DATASET=lims_cn_<name>)
                                                              -> outputs_lims/copy_number/<name>/
  compare  scripts/lims_compare.py: cross-environment tables, figures, validation, report
                                                              -> outputs_lims/comparison/ + REPORT.md

Each environment runs in its own process (config.py / cn_common.py read the dataset from
the environment at import), so runs are independent and are executed in parallel. Every
run's full console output is kept in <output dir>/run.log.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
SCRIPTS = ROOT / "scripts"


def _env(**extra):
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")          # no GPU here; avoids the CUDA probe
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        env.setdefault(k, "8")
    env.update(extra)
    return env


def run_one(kind, ds):
    out = ROOT / ds["out"]
    out.mkdir(parents=True, exist_ok=True)
    if kind == "barcode":
        cmd, env = [PY, str(SCRIPTS / "run_all.py")], _env(TFMN4_DATASET=ds["id"])
    else:
        cmd, env = [PY, str(SCRIPTS / "cn_run_all.py")], _env(TFMN4_CN_DATASET=ds["id"])
    t0 = time.time()
    with open(out / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return kind, ds["name"], rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--skip-extract", action="store_true", help="reuse lims_data/ (skip lims01)")
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to run (default: all)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel environment runs")
    a = ap.parse_args()

    if not a.skip_extract:
        rc = subprocess.run([PY, str(SCRIPTS / "lims01_extract.py")], cwd=ROOT, env=_env()).returncode
        if rc:
            raise SystemExit(f"[lims] extraction failed (rc={rc})")
    man = json.loads((ROOT / "lims_data" / "manifest.json").read_text())
    jobs = [("barcode", d) for d in man["barcode_datasets"]] + \
           [("cn", d) for d in man["copy_number_datasets"]]
    if a.only:
        jobs = [(k, d) for k, d in jobs if d["name"] in set(a.only)]
    print(f"[lims] running {len(jobs)} environment pipelines ({a.jobs} in parallel) ...", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, k, d) for k, d in jobs]
        for f in as_completed(futs):
            kind, name, rc, dt = f.result()
            status = "ok" if rc == 0 else f"FAILED (rc={rc})"
            print(f"[lims] {kind:7s} {name:44s} {status:14s} {dt/60:5.1f} min", flush=True)
            if rc:
                failed.append(name)
    if not a.skip_compare:
        rc = subprocess.run([PY, str(SCRIPTS / "lims_compare.py")], cwd=ROOT, env=_env()).returncode
        if rc:
            failed.append("lims_compare")
    if failed:
        raise SystemExit(f"[lims] failures: {failed} (see each output dir's run.log)")
    print("[lims] DONE -> outputs_lims/ (REPORT.md, comparison/, barcode/<env>/, copy_number/<env>/)")


if __name__ == "__main__":
    main()
