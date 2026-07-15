"""
cn_run_all - Run the dgoA copy-number amplification pipeline end to end.

    ~/Documents/py_venv/bin/python scripts/cn_run_all.py

Stages:
  cn01 prepare        ingest copy-number-dgoA_.csv -> tidy trajectories + lineage index
  cn02 growth rates   1-variable amplification rate + error per variant   <-- HEADLINE
  cn03 segmented      4-variable (breakpoint) model per lineage
  cn04 compare        1-var vs 4-var comparison + fit error
  cn05 figures        growth-rate forest, trajectory examples, comparison, fit error
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cn01_prepare, cn02_growth_rates, cn03_segmented, cn04_compare, cn05_figures


def main():
    line = "-" * 70
    print("=" * 70)
    cn01_prepare.main(); print(line)
    cn02_growth_rates.main(); print(line)
    cn03_segmented.main(); print(line)
    cn04_compare.main(); print(line)
    try:
        cn05_figures.main()
    except Exception as e:
        print(f"[cn05] figures skipped ({type(e).__name__}: {e})")
    print("=" * 70)
    print("DONE. Headlines (outputs_copynumber/):")
    print("  dgoA_growth_rates.csv                     per-variant amplification rate ± error")
    print("  figures/dgoA_growth_rates_forest.png      forest of every variant's rate ± 95% CI")
    print("  dgoA_segmented.csv                        4-variable breakpoint model")
    print("  model_comparison_summary.csv + .png       1-var vs 4-var head-to-head")
    print("  model_fit_error.csv + figures/model_fit_error.png")


if __name__ == "__main__":
    main()
