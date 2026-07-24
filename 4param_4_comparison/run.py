"""
run - Build the three-way comparison (figures + report) of the 4-parameter growth model
across the amplicon-only, barcode/WGS-only, and integrated runs.

Reads only this folder's inputs/ (the three models' fed outputs); refits nothing.
Writes comparison tables + figures to outputs/ and COMPARISON_REPORT.md to the folder root.

Run:  ~/Documents/py_venv/bin/python run.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import compare

if __name__ == "__main__":
    compare.main()
