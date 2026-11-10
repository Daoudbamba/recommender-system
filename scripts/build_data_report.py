"""Genere reports/data_report.md et ses figures a partir de data/processed.

Usage : python scripts/build_data_report.py
"""

from __future__ import annotations

from recsys.analysis.report import build_report
from recsys.config import load_config


def main() -> None:
    print(f"Rapport ecrit : {build_report(load_config())}")


if __name__ == "__main__":
    main()
