"""Command-line entry point for the reproducible ETF experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_portfolio.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="ignore the local market-data cache and download the fixed period again",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output = run_experiment(project_root, refresh_data=args.refresh_data)
    print(output["performance"].round(4).to_string())
    print("\nResearch findings")
    for key, value in output["findings"].items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
