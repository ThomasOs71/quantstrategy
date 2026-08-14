"""Smoke tests for public Article-2 figure generation."""

from __future__ import annotations

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from data.asset_universe import get_driver_keys
from series.from_assumptions_to_portfolios.block1_scenarios.article2_plots import (
    plot_rolling_volatility,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article2_diagnostics import (
    generate_article2_outputs,
)
from series.from_assumptions_to_portfolios.flagship_plot_style import (
    add_empirical_footer,
    create_figure,
)


def _synthetic_panel() -> pd.DataFrame:
    """Create a deterministic monthly panel with several joint tail months."""
    dates = pd.date_range("2015-01-31", periods=96, freq="ME")
    drivers = get_driver_keys()
    generator = np.random.default_rng(42)
    values = generator.normal(0.003, 0.025, size=(len(dates), len(drivers)))
    values[[24, 62, 70], :] -= 0.11
    return pd.DataFrame(values, index=dates, columns=drivers)


def test_article2_public_figures_generate_at_specified_dimensions(tmp_path) -> None:
    generate_article2_outputs(_synthetic_panel(), tmp_path, sw_permutations=19)

    expected = {
        "figure_1a_raw_return_autocorrelation.png": (2400, 1600),
        "figure_1b_squared_return_autocorrelation.png": (2400, 1600),
        "figure_2_rolling_volatility.png": (1200, 1600),
        "figure_3_rolling_cross_asset_correlations.png": (1200, 1600),
        "figure_4_joint_lower_tail_events.png": (1200, 1600),
        "figure_5_state_dependence.png": (1200, 1600),
        "figure_6_scenario_architecture.png": (2800, 1600),
    }
    for filename, dimensions in expected.items():
        path = tmp_path / filename
        assert path.is_file()
        assert mpimg.imread(path).shape[:2] == dimensions

    diagnostics = pd.read_csv(tmp_path / "article2_ljung_box_diagnostics.csv")
    assert len(diagnostics) == len(get_driver_keys()) * 2 * 2
    assert len(pd.read_csv(tmp_path / "ljung_box.csv")) == len(get_driver_keys()) * 2
    assert len(pd.read_csv(tmp_path / "schweizer_wolff_lagged.csv")) == len(get_driver_keys()) * 12


def test_empirical_footer_accepts_the_actual_panel_sample() -> None:
    figure = create_figure()
    add_empirical_footer(figure, sample="Jan 2015–Dec 2022")

    labels = {text.get_text() for text in figure.texts}

    assert "Sample: Jan 2015–Dec 2022  |  Monthly  |  EUR investor perspective" in labels


def test_rolling_volatility_does_not_silently_drop_requested_series() -> None:
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    columns = [f"driver_{position}" for position in range(5)]
    frame = pd.DataFrame(
        np.arange(len(index) * len(columns), dtype=float).reshape(len(index), -1),
        index=index,
        columns=columns,
    )

    figure = plot_rolling_volatility(frame, columns)

    assert len(figure.axes[0].lines) == len(columns)
