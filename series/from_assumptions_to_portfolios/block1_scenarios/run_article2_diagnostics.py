"""Run the reproducible diagnostic workflow for Block 1, Article 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article2_plots import (
    plot_autocorrelation,
    plot_joint_tail_event_timeline,
    plot_rolling_correlation,
    plot_rolling_volatility,
    plot_scenario_architecture,
    plot_state_dependence,
)
from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    conditional_forward_return_summary,
    distribution_summary,
    joint_tail_event_summary,
    ljung_box_diagnostics,
    ljung_box_summary,
    return_autocorrelation,
    lagged_schweizer_wolff_diagnostics,
    schweizer_wolff_summary,
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
from series.from_assumptions_to_portfolios.flagship_plot_style import save_figure


DEFAULT_ACF_DRIVERS = (
    "global_dm_ex_emu",
    "euro_high_yield",
    "commodities",
    "fx_eurusd",
)
DEFAULT_SQUARED_ACF_DRIVERS = (
    "euro_govt_bond_7_10",
    "euro_ig_credit",
    "global_dm_ex_emu",
    "commodities",
)
DEFAULT_VOLATILITY_DRIVERS = (
    "global_dm_ex_emu",
    "euro_govt_bond_7_10",
    "euro_high_yield",
    "commodities",
)
DEFAULT_STATE_DRIVERS = (
    "global_dm_ex_emu",
    "euro_high_yield",
    "gold",
    "euro_govt_bond_7_10",
)
DEFAULT_CORRELATION_PAIRS = (
    ("global_dm_ex_emu", "euro_govt_bond_7_10"),
    ("global_dm_ex_emu", "euro_high_yield"),
    ("global_dm_ex_emu", "commodities"),
)


def run(
    output_dir: Path,
    start: str = ARTICLE2_START,
    end: str = ARTICLE2_END,
    *,
    sw_permutations: int = 999,
    sw_seed: int = 42,
    sw_grid_size: int = 100,
) -> None:
    """Create Article-2 diagnostic tables and figures from local source data."""
    panel = load_article2_panel(start=start, end=end)
    generate_article2_outputs(
        panel,
        output_dir,
        panel_metadata=article2_panel_metadata(panel),
        sw_permutations=sw_permutations,
        sw_seed=sw_seed,
        sw_grid_size=sw_grid_size,
    )


def generate_article2_outputs(
    panel: pd.DataFrame,
    output_dir: Path,
    *,
    panel_metadata: dict[str, object] | None = None,
    sw_permutations: int = 999,
    sw_seed: int = 42,
    sw_grid_size: int = 100,
) -> None:
    """Create Article-2 outputs from a prepared monthly log-return panel.

    Keeping this orchestration separate from data loading lets the plotting
    contract be exercised with deterministic synthetic panels in tests.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = distribution_summary(panel)
    raw_acf = return_autocorrelation(panel)
    squared_acf = squared_return_autocorrelation(panel)
    ljung_box = ljung_box_diagnostics(panel)
    ljung_box_q12 = ljung_box_summary(panel, lag=12)
    schweizer_wolff_lagged = lagged_schweizer_wolff_diagnostics(
        panel,
        max_lag=12,
        grid_size=sw_grid_size,
        n_permutations=sw_permutations,
        seed=sw_seed,
    )
    schweizer_wolff_by_driver = schweizer_wolff_summary(schweizer_wolff_lagged)
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
    footer = {
        "sample": (
            f"{panel.index.min():%b %Y}–{panel.index.max():%b %Y}"
        )
    }

    _write_table(summary, output_dir / "distribution_summary.csv", index=False)
    _write_table(raw_acf, output_dir / "return_acf.csv", index=False)
    _write_table(squared_acf, output_dir / "squared_return_acf.csv", index=False)
    _write_table(ljung_box, output_dir / "article2_ljung_box_diagnostics.csv", index=False)
    _write_table(ljung_box_q12, output_dir / "ljung_box.csv", index=False)
    _write_table(
        schweizer_wolff_lagged,
        output_dir / "schweizer_wolff_lagged.csv",
        index=False,
    )
    _write_table(
        schweizer_wolff_by_driver,
        output_dir / "schweizer_wolff_summary.csv",
        index=False,
    )
    _write_table(thresholds, output_dir / "tail_thresholds.csv", index=False)
    _write_table(individual_tails, output_dir / "tail_events.csv", index=False)
    _write_table(joint_tails, output_dir / "joint_tail_events.csv", index=False)
    _write_table(rolling_volatility, output_dir / "rolling_volatility.csv", index=True)
    _write_table(rolling_correlation, output_dir / "rolling_correlations.csv", index=True)
    _write_table(state_features, output_dir / "state_features.csv", index=True)
    _write_table(state_summary, output_dir / "state_dependence.csv", index=False)

    _save_figure(
        plot_autocorrelation(
            raw_acf,
            DEFAULT_ACF_DRIVERS,
            title="Raw returns show limited serial memory",
            subtitle="Monthly return autocorrelation, lags 1–12",
            sample_size=len(panel),
            footer=footer,
        ),
        output_dir / "figure_1a_raw_return_autocorrelation.png",
    )
    _save_figure(
        plot_autocorrelation(
            squared_acf,
            DEFAULT_SQUARED_ACF_DRIVERS,
            title="Volatility persistence is uneven across assets",
            subtitle="Autocorrelation of squared demeaned monthly returns",
            sample_size=len(panel),
            footer=footer,
        ),
        output_dir / "figure_1b_squared_return_autocorrelation.png",
    )
    _save_figure(
        plot_rolling_volatility(
            rolling_volatility,
            DEFAULT_VOLATILITY_DRIVERS,
            footer=footer,
        ),
        output_dir / "figure_2_rolling_volatility.png",
    )
    _save_figure(
        plot_rolling_correlation(rolling_correlation, footer=footer),
        output_dir / "figure_3_rolling_cross_asset_correlations.png",
    )
    _save_figure(
        plot_joint_tail_event_timeline(joint_tails, footer=footer),
        output_dir / "figure_4_joint_lower_tail_events.png",
    )
    _save_figure(
        plot_state_dependence(
            state_summary,
            DEFAULT_STATE_DRIVERS,
            footer=footer,
        ),
        output_dir / "figure_5_state_dependence.png",
    )
    _save_figure(plot_scenario_architecture(), output_dir / "figure_6_scenario_architecture.png")

    metadata = (panel_metadata or _generic_panel_metadata(panel)) | {
        "article": "Block 1, Article 2: From Returns to Scenario Building Blocks",
        "tail_quantile": 0.05,
        "acf_lags": 12,
        "ljung_box_lags": [6, 12],
        "ljung_box_lag": 12,
        "ljung_box_fdr_method": "benjamini-hochberg",
        "schweizer_wolff_p": 1,
        "schweizer_wolff_max_lag": 12,
        "schweizer_wolff_grid_size": sw_grid_size,
        "schweizer_wolff_permutations": sw_permutations,
        "schweizer_wolff_seed": sw_seed,
        "schweizer_wolff_permutation_reference": True,
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


def _generic_panel_metadata(panel: pd.DataFrame) -> dict[str, object]:
    """Describe a test or exploratory panel without imposing the Article-2 contract."""
    return {
        "start": panel.index.min().date().isoformat(),
        "end": panel.index.max().date().isoformat(),
        "n_observations": int(len(panel)),
        "n_drivers": int(panel.shape[1]),
        "driver_keys": list(panel.columns),
        "frequency": "monthly_month_end",
        "return_representation": "monthly log returns",
    }


def _write_table(frame, path: Path, index: bool) -> None:
    frame.to_csv(path, index=index, index_label="date" if index else None)


def _save_figure(figure, path: Path) -> None:
    save_figure(figure, path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("articles/outputs/article2_smoke"),
        help="Directory for derived Article-2 tables and figures.",
    )
    parser.add_argument("--start", default=ARTICLE2_START)
    parser.add_argument("--end", default=ARTICLE2_END)
    parser.add_argument(
        "--sw-permutations",
        type=int,
        default=999,
        help="Permutation draws per Schweizer-Wolff driver-lag diagnostic.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    run(
        arguments.output_dir,
        start=arguments.start,
        end=arguments.end,
        sw_permutations=arguments.sw_permutations,
    )
