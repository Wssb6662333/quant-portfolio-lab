import json
import re
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "reports" / "tables"


def test_published_findings_are_derived_from_result_tables() -> None:
    performance = pd.read_csv(TABLES / "main_performance.csv", index_col="strategy")
    sensitivity = pd.read_csv(TABLES / "sensitivity.csv")
    diagnostics = pd.read_csv(TABLES / "covariance_diagnostics.csv")
    findings = json.loads((TABLES / "research_findings.json").read_text())

    sample = performance.loc["sample_min_variance"]
    shrinkage = performance.loc["ledoit_wolf_min_variance"]
    assert findings["evaluation_observations"] == sample["observations"]
    assert np.isclose(
        findings["ledoit_wolf_minus_sample_annual_turnover"],
        shrinkage["annualized_turnover"] - sample["annualized_turnover"],
    )

    median_condition = diagnostics.groupby("strategy")["condition_number"].median()
    expected_reduction = 100 * (
        1
        - median_condition["ledoit_wolf_min_variance"]
        / median_condition["sample_min_variance"]
    )
    assert np.isclose(
        findings["median_covariance_condition_number_reduction_pct"],
        expected_reduction,
    )

    paired = sensitivity.pivot_table(
        index=["lookback", "cost_bps"],
        columns="strategy",
        values="annual_volatility",
    )
    wins = int(
        (paired["ledoit_wolf_min_variance"] < paired["sample_min_variance"]).sum()
    )
    assert findings["sensitivity_scenarios_with_lower_volatility_than_sample"] == wins
    assert findings["sensitivity_scenarios"] == len(paired)


def test_public_research_notebook_executed_without_errors() -> None:
    path = PROJECT_ROOT / "notebooks" / "research" / "01_covariance_shrinkage_study.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert code_cells
    assert all(cell.execution_count is not None for cell in code_cells)
    assert all(
        output.output_type != "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )


def test_readme_local_links_exist() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"(?:!\[[^]]*\]|\[[^]]+\])\(([^)]+)\)", readme)

    for target in targets:
        if target.startswith(("http://", "https://", "#")):
            continue
        assert (PROJECT_ROOT / target).exists(), f"missing README target: {target}"
