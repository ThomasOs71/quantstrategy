"""Run the reproducible diagnostic workflow for Block 1, Article 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from series.from_assumptions_to_portfolios.block1_scenarios.article2_plots import (
    plot_autocorrelation,
    plot_joint_tail_event_timeline,
    plot_rolling_correlation,
    plot_rolling_volatility,
    plot_state_dependence,
)
from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    conditional_forward_return_summary,
    distribution_summary,
    joint_tail_event_summary,
    return_autocorrelation,
    squared_return_autocorrelation,
    tail_events,
    tail_thresholds,
)
from series.from_assumptions_to_portfolios.block1_scenarios.returns_building_blocks import (
    ARTICLE2_END,
    ARTICLE2_START,
    article2_panel_metadata,
    load_article2_panel,
)
from series.from_assumptions_to_portfolios.block1_scenarios.rolling_stats import (
    rolling_annualized_volatility,
    rolling_pairwise_correlation,
    trailing_state_features,
)


DEFAULT_ACF_DRIVERS = (
    "global_dm_ex_emu",
    "euro_high_yield",
    "commodities",
    "fx_eurusd",
)
DEFAULT_VOLATILITY_DRIVERS = (
    "global_dm_ex_emu",
    "euro_govt_bond_7_10",
    "euro_high_yield",
    "commodities",
)
DEFAULT_STATE_DRIVERS = (
    "global_dm_ex_emu",
    "euro_govt_bond_7_10",
    "euro_high_yield",
    "gold",
)
DEFAULT_CORRELATION_PAIRS = (
    ("global_dm_ex_emu", "euro_govt_bond_7_10"),
    ("global_dm_ex_emu", "euro_high_yield"),
    ("global_dm_ex_emu", "commodities"),
)


def run(output_dir: Path, start: str = ARTICLE2_START, end: str = ARTICLE2_END) -> None:
    """Create Article-2 diagnostic tables and figures from local source data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_article2_panel(start=start, end=end)

    summary = distribution_summary(panel)
    raw_acf = return_autocorrelation(panel)
    squared_acf = squared_return_autocorrelation(panel)
    thresholds = tail_thresholds(panel)
    individual_tails = tail_events(panel)
    joint_tails = joint_tail_event_summary(panel)
    rolling_volatility = rolling_annualized_volatility(panel)
    rolling_correlation = rolling_pairwise_correlation(panel, DEFAULT_CORRELATION_PAIRS)
    state_features = trailing_state_features(panel, "global_dm_ex_emu")
    state_summary = conditional_forward_return_summary(
        panel,
        state=state_features["trailing_volatility"],
        state_name="global_dm_ex_emu_12m_trailing_volatility",
    )

    _write_table(summary, output_dir / "distribution_summary.csv", index=False)
    _write_table(raw_acf, output_dir / "return_acf.csv", index=False)
    _write_table(squared_acf, output_dir / "squared_return_acf.csv", index=False)
    _write_table(thresholds, output_dir / "tail_thresholds.csv", index=False)
    _write_table(individual_tails, output_dir / "tail_events.csv", index=False)
    _write_table(joint_tails, output_dir / "joint_tail_events.csv", index=False)
    _write_table(rolling_volatility, output_dir / "rolling_volatility.csv", index=True)
    _write_table(rolling_correlation, output_dir / "rolling_correlations.csv", index=True)
    _write_table(state_features, output_dir / "state_features.csv", index=True)
    _write_table(state_summary, output_dir / "state_dependence.csv", index=False)

    _save_figure(
        plot_autocorrelation(raw_acf, DEFAULT_ACF_DRIVERS, "Raw return autocorrelation"),
        output_dir / "figure_return_acf.png",
    )
    _save_figure(
        plot_autocorrelation(
            squared_acf,
            DEFAULT_ACF_DRIVERS,
            "Squared demeaned return autocorrelation",
        ),
        output_dir / "figure_squared_return_acf.png",
    )
    _save_figure(
        plot_rolling_volatility(rolling_volatility, DEFAULT_VOLATILITY_DRIVERS),
        output_dir / "figure_rolling_volatility.png",
    )
    _save_figure(
        plot_rolling_correlation(rolling_correlation),
        output_dir / "figure_rolling_correlations.png",
    )
    if not joint_tails.empty:
        _save_figure(
            plot_joint_tail_event_timeline(joint_tails),
            output_dir / "figure_joint_tail_events.png",
        )
    _save_figure(
        plot_state_dependence(state_summary, DEFAULT_STATE_DRIVERS),
        output_dir / "figure_state_dependence.png",
    )

    metadata = article2_panel_metadata(panel) | {
        "article": "Block 1, Article 2: From Returns to Scenario Building Blocks",
        "tail_quantile": 0.05,
        "acf_lags": 12,
        "rolling_volatility_window_months": 12,
        "rolling_correlation_window_months": 24,
        "state_anchor": "global_dm_ex_emu",
        "state_feature": "12m trailing annualized volatility",
        "state_forward_horizon_months": 12,
        "state_bucket_count": 3,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _write_table(frame, path: Path, index: bool) -> None:
    frame.to_csv(path, index=index, index_label="date" if index else None)


def _save_figure(figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("articles/outputs/article2"),
        help="Directory for derived Article-2 tables and figures.",
    )
    parser.add_argument("--start", default=ARTICLE2_START)
    parser.add_argument("--end", default=ARTICLE2_END)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run(arguments.output_dir, start=arguments.start, end=arguments.end)
