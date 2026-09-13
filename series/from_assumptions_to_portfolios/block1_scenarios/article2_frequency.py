"""Explicit frequency semantics for Article-2 diagnostic reruns."""

from __future__ import annotations

from dataclasses import dataclass

from data.panel_profiles import PanelFrequency, coerce_panel_frequency


ARTICLE2_END = "2025-12-31"
WEEKLY_ACF_LAGS = tuple(range(1, 53))
MONTHLY_EQUIVALENT_WEEKLY_LAGS = (4, 9, 13, 17, 22, 26, 30, 35, 39, 43, 48, 52)


@dataclass(frozen=True)
class Article2FrequencySpec:
    """Calendar-consistent diagnostic settings for one sampling frequency."""

    frequency: PanelFrequency
    start: str
    requested_end: str
    periods_per_year: int
    period_adjective: str
    period_unit: str
    period_unit_plural: str
    display_name: str
    acf_lags: tuple[int, ...]
    ljung_box_lags: tuple[int, ...]
    schweizer_wolff_lags: tuple[int, ...]
    rolling_volatility_window: int
    rolling_correlation_window: int
    state_window: int
    forward_horizon: int
    one_year_label: str
    two_year_label: str


ARTICLE2_FREQUENCY_SPECS = {
    PanelFrequency.MONTHLY: Article2FrequencySpec(
        frequency=PanelFrequency.MONTHLY,
        start="2011-01-31",
        requested_end=ARTICLE2_END,
        periods_per_year=12,
        period_adjective="monthly",
        period_unit="month",
        period_unit_plural="months",
        display_name="Monthly",
        acf_lags=tuple(range(1, 13)),
        ljung_box_lags=(6, 12),
        schweizer_wolff_lags=tuple(range(1, 13)),
        rolling_volatility_window=12,
        rolling_correlation_window=24,
        state_window=12,
        forward_horizon=12,
        one_year_label="12-month",
        two_year_label="24-month",
    ),
    PanelFrequency.WEEKLY: Article2FrequencySpec(
        frequency=PanelFrequency.WEEKLY,
        start="2011-01-07",
        requested_end=ARTICLE2_END,
        periods_per_year=52,
        period_adjective="weekly",
        period_unit="week",
        period_unit_plural="weeks",
        display_name="Weekly (W-FRI)",
        acf_lags=WEEKLY_ACF_LAGS,
        ljung_box_lags=(26, 52),
        schweizer_wolff_lags=MONTHLY_EQUIVALENT_WEEKLY_LAGS,
        rolling_volatility_window=52,
        rolling_correlation_window=104,
        state_window=52,
        forward_horizon=52,
        one_year_label="52-week (~12-month)",
        two_year_label="104-week (~24-month)",
    ),
}


def get_article2_frequency_spec(
    frequency: str | PanelFrequency,
) -> Article2FrequencySpec:
    """Return the immutable Article-2 settings for ``frequency``."""
    return ARTICLE2_FREQUENCY_SPECS[coerce_panel_frequency(frequency)]
