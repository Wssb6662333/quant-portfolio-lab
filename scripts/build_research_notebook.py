"""Build and execute the concise public research notebook from verified artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tables = project_root / "reports" / "tables"
    notebook_path = (
        project_root / "notebooks" / "research" / "01_covariance_shrinkage_study.ipynb"
    )
    if not (tables / "research_findings.json").exists():
        raise FileNotFoundError("run scripts/run_experiment.py before building the notebook")

    findings = json.loads((tables / "research_findings.json").read_text(encoding="utf-8"))
    performance = pd.read_csv(tables / "main_performance.csv", index_col="strategy")
    sample = performance.loc["sample_min_variance"]
    shrinkage = performance.loc["ledoit_wolf_min_variance"]

    notebook = nbformat.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Covariance Shrinkage in Cross-Asset ETF Portfolios\n\n"
            "**Research question.** Does Ledoit-Wolf covariance shrinkage improve "
            "out-of-sample risk and weight stability relative to sample-covariance "
            "minimum variance?\n\n"
            f"I evaluate **{findings['evaluation_observations']:,}** daily "
            f"observations from **{findings['evaluation_start']}** to "
            f"**{findings['evaluation_end']}**. I keep the core logic in `src/` and use "
            "this notebook only to load and present reproducible artifacts."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'pyproject.toml').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parents[1]\n"
            "TABLES = PROJECT_ROOT / 'reports' / 'tables'\n"
            "performance = pd.read_csv(\n"
            "    TABLES / 'main_performance.csv', index_col='strategy'\n"
            ")\n"
            "sensitivity = pd.read_csv(TABLES / 'sensitivity.csv')\n"
            "var_backtest = pd.read_csv(\n"
            "    TABLES / 'var_backtest.csv', index_col='strategy'\n"
            ")\n"
            "findings = json.loads(\n"
            "    (TABLES / 'research_findings.json').read_text(encoding='utf-8')\n"
            ")\n"
            "print(f\"OOS: {findings['evaluation_start']} to {findings['evaluation_end']}\")"
        ),
        nbformat.v4.new_markdown_cell(
            "## Design\n\n"
            "I evaluate nine cross-asset ETFs with quarterly walk-forward decisions. "
            "I use a 252-day training window, 40% asset cap, and 10 bp costs on traded "
            "notional in the main configuration. I restrict every target to information "
            "available through the prior trading day. I use buy-and-hold 1/N, quarterly "
            "1/N, inverse volatility, and SPY as baselines."
        ),
        nbformat.v4.new_code_cell(
            "columns = ['cagr', 'annual_volatility', 'sharpe', 'max_drawdown', "
            "'annualized_turnover']\n"
            "main = performance[columns].copy()\n"
            "main[['cagr', 'annual_volatility', 'max_drawdown']] *= 100\n"
            "main.columns = ['CAGR (%)', 'Volatility (%)', 'Sharpe', "
            "'Max drawdown (%)', 'Annual turnover']\n"
            "display(main.round(3))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Finding\n\n"
            f"I find that Ledoit-Wolf reduced the median covariance condition number by "
            f"**{findings['median_covariance_condition_number_reduction_pct']:.1f}%** and "
            f"annual turnover by **{-findings['ledoit_wolf_minus_sample_annual_turnover']:.3f}**. "
            f"It did **not** improve main-setting risk: annual volatility was "
            f"{100 * shrinkage['annual_volatility']:.2f}% versus "
            f"{100 * sample['annual_volatility']:.2f}%, and maximum drawdown was "
            f"{100 * shrinkage['max_drawdown']:.2f}% versus "
            f"{100 * sample['max_drawdown']:.2f}%. I observe lower volatility in only "
            f"**{findings['sensitivity_scenarios_with_lower_volatility_than_sample']} of "
            f"{findings['sensitivity_scenarios']}** lookback-cost scenarios."
        ),
        nbformat.v4.new_markdown_cell(
            "## Performance and drawdowns\n\n"
            "![Net performance and drawdowns](../../reports/figures/performance_and_drawdowns.png)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Conditioning and concentration\n\n"
            "![Covariance diagnostics](../../reports/figures/covariance_diagnostics.png)"
        ),
        nbformat.v4.new_markdown_cell(
            "## Robustness\n\n"
            "I use the same 2013-2025 test dates for every 126/252/504-day lookback "
            "and 0/10/25 bp cost setting. Positive cells mean shrinkage had higher risk.\n\n"
            "![Sensitivity](../../reports/figures/sensitivity_heatmaps.png)"
        ),
        nbformat.v4.new_code_cell(
            "min_var = sensitivity[sensitivity['strategy'].isin([\n"
            "    'sample_min_variance', 'ledoit_wolf_min_variance'\n"
            "])][['lookback', 'cost_bps', 'strategy', 'annual_volatility', "
            "'max_drawdown', 'annualized_turnover']]\n"
            "display(min_var.round(5))"
        ),
        nbformat.v4.new_markdown_cell(
            "## VaR validation\n\n"
            "I calculate historical 95% VaR forecasts from the prior 252 strategy "
            "returns and exclude the current day. I use Kupiec tests to assess "
            "unconditional breach frequency."
        ),
        nbformat.v4.new_code_cell(
            "display(var_backtest[['forecast_observations', 'breaches', 'breach_rate', "
            "'kupiec_p_value']].round(4))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Limitations\n\n"
            "I select the ETF universe with hindsight. I use adjusted daily closes and "
            "omit intraday execution, spreads, market impact, taxes, and fund closure "
            "risk. My nine liquid ETFs are also a low-dimensional setting in which "
            "covariance shrinkage may have limited scope to help. I interpret the "
            "experiment as evidence of better conditioning, not a universal performance "
            "claim. I do not estimate statistical uncertainty around the small difference "
            "between the two minimum-variance portfolios."
        ),
    ]

    client = NotebookClient(notebook, timeout=600, kernel_name="python3")
    executed = client.execute(cwd=str(project_root))
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("not every research notebook code cell executed")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, notebook_path)
    print(f"Wrote executed notebook: {notebook_path}")


if __name__ == "__main__":
    main()
