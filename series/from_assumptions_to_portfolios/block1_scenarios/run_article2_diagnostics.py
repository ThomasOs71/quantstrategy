"""Run the reproducible diagnostic workflow for Block 1, Article 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.load_data import load_return_panel_download
from data.panel_profiles import (
    PanelFrequency,
    SourceProfile,
    coerce_panel_frequency,
    coerce_source_profile,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article2_plots import (
    plot_autocorrelation,
    plot_joint_tail_event_timeline,
    plot_rolling_correlation,
    plot_rolling_volatility,
    plot_scenario_architecture,
    plot_state_dependence,
)
from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    conditional_forward_period_return_summary,
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
from series.from_assumptions_to_portfolios.block1_scenarios.article2_frequency import (
    get_article2_frequency_spec,
)
from series.from_assumptions_to_portfolios.block1_scenarios.returns_building_blocks import (
    ARTICLE2_END,
    article2_panel_metadata,
    article2_research_panel_metadata,
    load_article2_panel,
    load_article2_research_panel,
    prepare_article2_research_panel,
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
    start: str | None = None,
    end: str = ARTICLE2_END,
    *,
    source_profile: str | SourceProfile = SourceProfile.MONTHLY_LEGACY,
    frequency: str | PanelFrequency = PanelFrequency.MONTHLY,
    input_csv: Path | None = None,
    input_manifest: Path | None = None,
    sw_permutations: int = 999,
    sw_seed: int = 42,
    sw_grid_size: int = 100,
) -> None:
    """Create a legacy or frequency-aware Article-2 diagnostic run."""
    parsed_profile = coerce_source_profile(source_profile)
    parsed_frequency = coerce_panel_frequency(frequency)
    spec = get_article2_frequency_spec(parsed_frequency)
    resolved_start = start or spec.start

    if parsed_profile is SourceProfile.MONTHLY_LEGACY:
        if parsed_frequency is not PanelFrequency.MONTHLY:
            raise ValueError("monthly_legacy Article-2 analysis supports monthly only")
        if input_csv is not None or input_manifest is not None:
            raise ValueError("input snapshots are supported only for daily_proxy_2011")
        panel = load_article2_panel(start=resolved_start, end=end)
        metadata = article2_panel_metadata(panel)
    else:
        if input_csv is not None:
            panel = load_return_panel_download(input_csv, input_manifest)
            panel = prepare_article2_research_panel(
                panel,
                frequency=parsed_frequency,
                start=resolved_start,
                end=end,
            )
        elif input_manifest is not None:
            raise ValueError("input_manifest requires input_csv")
        else:
            panel = load_article2_research_panel(
                frequency=parsed_frequency,
                start=resolved_start,
                end=end,
            )
        metadata = article2_research_panel_metadata(
            panel,
            frequency=parsed_frequency,
            start=resolved_start,
            end=end,
        )
    generate_article2_outputs(
        panel,
        output_dir,
        frequency=parsed_frequency,
        panel_metadata=metadata,
        sw_permutations=sw_permutations,
        sw_seed=sw_seed,
        sw_grid_size=sw_grid_size,
    )


def generate_article2_outputs(
    panel: pd.DataFrame,
    output_dir: Path,
    *,
    frequency: str | PanelFrequency = PanelFrequency.MONTHLY,
    panel_metadata: dict[str, object] | None = None,
    sw_permutations: int = 999,
    sw_seed: int = 42,
    sw_grid_size: int = 100,
) -> None:
    """Create Article-2 outputs from a prepared periodic log-return panel.

    Keeping this orchestration separate from data loading lets the plotting
    contract be exercised with deterministic synthetic panels in tests.
    """
    spec = get_article2_frequency_spec(frequency)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = distribution_summary(
        panel,
        periods_per_year=spec.periods_per_year,
        period_label=spec.period_adjective,
    )
    raw_acf = return_autocorrelation(panel, lags=spec.acf_lags)
    squared_acf = squared_return_autocorrelation(panel, lags=spec.acf_lags)
    ljung_box = ljung_box_diagnostics(panel, lags=spec.ljung_box_lags)
    ljung_box_summary_table = ljung_box_summary(
        panel,
        lag=spec.ljung_box_lags[-1],
    )
    schweizer_wolff_lagged = lagged_schweizer_wolff_diagnostics(
        panel,
        lags=spec.schweizer_wolff_lags,
        grid_size=sw_grid_size,
        n_permutations=sw_permutations,
        seed=sw_seed,
    )
    schweizer_wolff_by_driver = schweizer_wolff_summary(schweizer_wolff_lagged)
    thresholds = tail_thresholds(panel)
    individual_tails = tail_events(panel)
    joint_tails = joint_tail_event_summary(panel)
    rolling_volatility = rolling_annualized_volatility(
        panel,
        window=spec.rolling_volatility_window,
        periods_per_year=spec.periods_per_year,
    )
    rolling_correlation = rolling_pairwise_correlation(
        panel,
        DEFAULT_CORRELATION_PAIRS,
        window=spec.rolling_correlation_window,
    )
    state_features = trailing_state_features(
        panel,
        "global_dm_ex_emu",
        window=spec.state_window,
        periods_per_year=spec.periods_per_year,
    )
    state_summary = conditional_forward_period_return_summary(
        panel,
        state=state_features["trailing_volatility"],
        horizon_periods=spec.forward_horizon,
        period_unit=spec.period_unit_plural,
        state_name=(
            f"global_dm_ex_emu_{spec.state_window}{spec.period_unit[0]}_"
            "trailing_volatility"
        ),
    )
    footer = {
        "sample": f"{panel.index.min():%b %Y}–{panel.index.max():%b %Y}",
        "frequency_label": spec.display_name,
        "return_description": (
            f"EUR {spec.period_adjective} log returns where applicable"
        ),
    }

    _write_table(summary, output_dir / "distribution_summary.csv", index=False)
    _write_table(raw_acf, output_dir / "return_acf.csv", index=False)
    _write_table(squared_acf, output_dir / "squared_return_acf.csv", index=False)
    _write_table(
        ljung_box, output_dir / "article2_ljung_box_diagnostics.csv", index=False
    )
    _write_table(ljung_box_summary_table, output_dir / "ljung_box.csv", index=False)
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
    _write_table(
        rolling_correlation, output_dir / "rolling_correlations.csv", index=True
    )
    _write_table(state_features, output_dir / "state_features.csv", index=True)
    _write_table(state_summary, output_dir / "state_dependence.csv", index=False)

    _save_figure(
        plot_autocorrelation(
            raw_acf,
            DEFAULT_ACF_DRIVERS,
            title="Raw returns show limited serial memory",
            subtitle=(
                "Monthly return autocorrelation, lags 1–12"
                if spec.frequency is PanelFrequency.MONTHLY
                else "Weekly return autocorrelation, lags 1–52"
            ),
            sample_size=len(panel),
            lag_unit=spec.period_unit_plural,
            footer=footer,
        ),
        output_dir / "figure_1a_raw_return_autocorrelation.png",
    )
    _save_figure(
        plot_autocorrelation(
            squared_acf,
            DEFAULT_SQUARED_ACF_DRIVERS,
            title="Volatility persistence is uneven across assets",
            subtitle=(
                f"Autocorrelation of squared demeaned {spec.period_adjective} "
                + (
                    "returns, lags 1–12"
                    if spec.frequency is PanelFrequency.MONTHLY
                    else "returns, lags 1–52"
                )
            ),
            sample_size=len(panel),
            lag_unit=spec.period_unit_plural,
            footer=footer,
        ),
        output_dir / "figure_1b_squared_return_autocorrelation.png",
    )
    _save_figure(
        plot_rolling_volatility(
            rolling_volatility,
            DEFAULT_VOLATILITY_DRIVERS,
            window_label=spec.one_year_label,
            footer=footer,
        ),
        output_dir / "figure_2_rolling_volatility.png",
    )
    _save_figure(
        plot_rolling_correlation(
            rolling_correlation,
            window_label=spec.two_year_label,
            footer=footer,
        ),
        output_dir / "figure_3_rolling_cross_asset_correlations.png",
    )
    _save_figure(
        plot_joint_tail_event_timeline(
            joint_tails,
            period_label_plural=spec.period_unit_plural.title(),
            footer=footer,
        ),
        output_dir / "figure_4_joint_lower_tail_events.png",
    )
    _save_figure(
        plot_state_dependence(
            state_summary,
            DEFAULT_STATE_DRIVERS,
            forward_horizon_label=spec.one_year_label,
            state_window_label=spec.one_year_label,
            footer=footer,
        ),
        output_dir / "figure_5_state_dependence.png",
    )
    _save_figure(
        plot_scenario_architecture(), output_dir / "figure_6_scenario_architecture.png"
    )

    n_sw_hypotheses = len(panel.columns) * len(spec.schweizer_wolff_lags)
    metadata = (panel_metadata or _generic_panel_metadata(panel, spec.frequency)) | {
        "analysis_schema_version": 2,
        "article": "Block 1, Article 2: From Returns to Scenario Building Blocks",
        "analysis_frequency": spec.frequency.value,
        "periods_per_year": spec.periods_per_year,
        "tail_quantile": 0.05,
        "tail_threshold_policy": "full-sample empirical per-period quantile",
        "tail_period_unit": spec.period_unit,
        "acf_lags": list(spec.acf_lags),
        "acf_lag_unit": spec.period_unit,
        "ljung_box_lags": list(spec.ljung_box_lags),
        "ljung_box_lag": spec.ljung_box_lags[-1],
        "ljung_box_lag_unit": spec.period_unit,
        "ljung_box_fdr_method": "benjamini-hochberg",
        "schweizer_wolff_p": 1,
        "schweizer_wolff_lags": list(spec.schweizer_wolff_lags),
        "schweizer_wolff_lag_unit": spec.period_unit,
        "schweizer_wolff_hypotheses": n_sw_hypotheses,
        "schweizer_wolff_grid_size": sw_grid_size,
        "schweizer_wolff_permutations": sw_permutations,
        "schweizer_wolff_minimum_pvalue": 1.0 / (sw_permutations + 1.0),
        "schweizer_wolff_bh_resolution_limited": (
            sw_permutations + 1 < n_sw_hypotheses / 0.05
        ),
        "schweizer_wolff_seed": sw_seed,
        "schweizer_wolff_permutation_reference": True,
        "rolling_volatility_window_periods": spec.rolling_volatility_window,
        "rolling_volatility_window_unit": spec.period_unit,
        "rolling_volatility_calendar_label": spec.one_year_label,
        "rolling_correlation_window_periods": spec.rolling_correlation_window,
        "rolling_correlation_window_unit": spec.period_unit,
        "rolling_correlation_calendar_label": spec.two_year_label,
        "state_anchor": "global_dm_ex_emu",
        "state_feature": f"{spec.one_year_label} trailing annualized volatility",
        "state_window_periods": spec.state_window,
        "state_forward_horizon_periods": spec.forward_horizon,
        "state_period_unit": spec.period_unit,
        "state_bucket_boundary_policy": "full-sample qcut; retrospective descriptive",
        "state_bucket_count": 3,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _generic_panel_metadata(
    panel: pd.DataFrame,
    frequency: str | PanelFrequency,
) -> dict[str, object]:
    """Describe a test or exploratory panel without imposing the Article-2 contract."""
    spec = get_article2_frequency_spec(frequency)
    return {
        "start": panel.index.min().date().isoformat(),
        "end": panel.index.max().date().isoformat(),
        "n_observations": int(len(panel)),
        "n_drivers": int(panel.shape[1]),
        "driver_keys": list(panel.columns),
        "frequency": spec.frequency.value,
        "periods_per_year": spec.periods_per_year,
        "return_representation": f"{spec.period_adjective} log returns",
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
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=ARTICLE2_END)
    parser.add_argument(
        "--source-profile",
        choices=[profile.value for profile in SourceProfile],
        default=SourceProfile.MONTHLY_LEGACY.value,
    )
    parser.add_argument(
        "--frequency",
        choices=[frequency.value for frequency in PanelFrequency],
        default=PanelFrequency.MONTHLY.value,
    )
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--input-manifest", type=Path, default=None)
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
        source_profile=arguments.source_profile,
        frequency=arguments.frequency,
        input_csv=arguments.input_csv,
        input_manifest=arguments.input_manifest,
        sw_permutations=arguments.sw_permutations,
    )
