"""Offline statistical contracts using explicitly synthetic fixtures."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests

from series.from_assumptions_to_portfolios.block1_scenarios.article3_block_length_diagnostics import (
    aggregate_log_paths,
    analytic_preservation_scores,
    automatic_block_length_diagnostics,
    expected_mbb_acf,
    four_week_endpoints,
    last_friday_month_endpoints,
    ljung_box_diagnostics,
    maximum_drawdown,
    resolution_diagnostics,
    source_uncertainty_diagnostics,
    stationary_resample_indices,
    terminal_correlation_rows,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import ensemble_acf
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import sample_moving_block_indices


@pytest.mark.parametrize("length,horizon", [(1, 5), (2, 5), (3, 6), (3, 5), (4, 3), (6, 5)])
@pytest.mark.parametrize("squared", [False, True])
def test_expected_acf_matches_exhaustive_start_enumeration(length, horizon, squared):
    # All possible start combinations, each with equal probability: an exact
    # finite-source oracle, including joins, nonzero edge means and truncation.
    history = np.array([[-.03, .01], [.02, -.01], [.04, .03], [-.02, .01], [.01, .02], [-.04, -.02]])
    signal = (history - history.mean(axis=0)) ** 2 if squared else history
    n_blocks = (horizon + length - 1) // length
    starts = np.array(list(product(range(len(history) - length + 1), repeat=n_blocks)))
    steps = np.arange(horizon)
    paths = signal[starts[:, steps // length] + steps % length]
    exact, _ = ensemble_acf(paths, max_lag=horizon - 1, transform="raw")
    actual = expected_mbb_acf(signal, horizon_steps=horizon, block_length_steps=length, max_lag=horizon - 1)
    np.testing.assert_allclose(actual, exact, atol=2e-14, rtol=2e-14)


def test_expected_acf_zero_variance_and_bad_inputs():
    assert np.isnan(expected_mbb_acf(np.ones((10, 2)), horizon_steps=8, block_length_steps=4, max_lag=3)).all()
    for kwargs in ({"block_length_steps": 11}, {"max_lag": 8}, {"block_length_steps": 0}):
        args = {"horizon_steps": 8, "block_length_steps": 4, "max_lag": 3} | kwargs
        with pytest.raises(ValueError):
            expected_mbb_acf(np.arange(20).reshape(10, 2), **args)


def test_stationary_outer_indices_reproducible_and_circular():
    a = stationary_resample_indices(100, mean_block_length=26, rng=np.random.default_rng(9))
    b = stationary_resample_indices(100, mean_block_length=26, rng=np.random.default_rng(9))
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 0 and a.max() < 100
    assert np.mean(a[1:] == (a[:-1] + 1) % 100) > .8
    continuous = stationary_resample_indices(100, mean_block_length=1e12, rng=np.random.default_rng(9))
    assert np.all(continuous[1:] == (continuous[:-1] + 1) % 100)
    independent = stationary_resample_indices(100, mean_block_length=1, rng=np.random.default_rng(9))
    assert np.mean(independent[1:] == (independent[:-1] + 1) % 100) < .1
    for bad in (0, .5, np.inf):
        with pytest.raises(ValueError):
            stationary_resample_indices(10, mean_block_length=bad, rng=np.random.default_rng(9))


def test_resolution_preserves_terminal_and_can_hide_all_drawdown():
    # Synthetic complete within-block recovery: coarse endpoints hide the loss.
    paths = np.array([[[-.1], [.1], [-.2], [.2]]])
    coarse = aggregate_log_paths(paths, [4])
    np.testing.assert_allclose(coarse, 0, atol=1e-15)
    assert maximum_drawdown(paths)[0, 0] == pytest.approx(-np.expm1(-.2))
    assert maximum_drawdown(coarse)[0, 0] == 0
    summary, _ = resolution_diagnostics(paths, ["synthetic_asset"], endpoints=[4], grid_name="four_week")
    assert summary.loc[summary.metric == "max_abs_terminal_log_difference", "value"].iloc[0] < 1e-15
    assert summary.loc[summary.metric == "fraction_mdd_understated_gt_1pp", "value"].iloc[0] == 1


def test_coarse_drawdown_never_exceeds_full_grid():
    paths = np.random.default_rng(42).normal(0, .02, (40, 13, 3))
    coarse = aggregate_log_paths(paths, four_week_endpoints(13))
    np.testing.assert_allclose(coarse.sum(axis=1), paths.sum(axis=1), atol=1e-14)
    assert np.all(maximum_drawdown(coarse) <= maximum_drawdown(paths) + 1e-14)


@pytest.mark.parametrize("ends", [[], [0, 4], [2, 2, 4], [2, 3], [2., 4.]])
def test_aggregation_rejects_invalid_endpoints(ends):
    with pytest.raises(ValueError):
        aggregate_log_paths(np.ones((2, 4, 2)), ends)


def test_four_week_and_calendar_month_labelled_grids_are_distinct():
    dates = pd.date_range("2026-01-02", periods=260, freq="W-FRI")
    four = four_week_endpoints(len(dates))
    month = last_friday_month_endpoints(dates)
    assert len(four) == 65
    assert 60 <= len(month) <= 61
    assert four[-1] == month[-1] == 260
    assert {4, 5}.issubset(set(np.diff(np.r_[0, month])[:-1]))
    with pytest.raises(ValueError, match="W-FRI"):
        last_friday_month_endpoints(dates.delete(10))


def test_terminal_correlation_distinguishes_log_and_simple_returns():
    paths = np.array([[[1., .4]], [[2., .8]], [[3., 1.2]], [[4., 1.6]]])
    table = terminal_correlation_rows(paths, ["a", "b"], horizons=[1])
    log = table.loc[table.return_type == "cumulative_log", "correlation"].iloc[0]
    simple = table.loc[table.return_type == "simple_compounded", "correlation"].iloc[0]
    assert log == pytest.approx(1)
    assert simple < .99
    with pytest.raises(ValueError):
        terminal_correlation_rows(paths, ["a", "b"], horizons=[2])


def test_ljung_box_uses_historical_sample_and_separate_corrected_families():
    values = np.random.default_rng(73).normal(size=(120, 3))
    table = ljung_box_diagnostics(values, ["a", "b", "cash"], lags=[4, 13])
    observed = table.loc[(table["transform"] == "raw") & (table.driver == "a")]
    reference = acorr_ljungbox(values[:, 0], lags=[4, 13], return_df=True)
    np.testing.assert_allclose(observed.p_value, reference.lb_pvalue)
    for transform in ("raw", "squared_demeaned"):
        group = table.loc[(table["transform"] == transform) & (table.family == "non_cash11")]
        expected = multipletests(group.p_value, method="holm")[1]
        np.testing.assert_allclose(group.p_holm, expected)
        assert (group.family_size == 4).all()
    assert (table.loc[table.driver == "cash", "family_size"] == 2).all()
    assert (table.n_observations == 120).all()


def test_ljung_box_undefined_constant_inputs_are_reported():
    values = np.column_stack([np.ones(100), np.random.default_rng(3).normal(size=100)])
    table = ljung_box_diagnostics(values, ["constant", "b"], lags=[4])
    constant = table.loc[table.driver == "constant"]
    assert constant.p_value.isna().all()
    assert not constant.reject_holm.any()
    assert (constant.status == "undefined_zero_variance").all()


def test_analytic_scores_l1_has_zero_expected_weekly_correlation_error():
    values = np.random.default_rng(19).normal(size=(100, 3))
    table = analytic_preservation_scores(values, ["a", "b", "cash"], lengths=[1, 4, 8], horizon_steps=40)
    l1 = table.loc[(table.L == 1) & table.metric.str.startswith("weekly_correlation")]
    np.testing.assert_allclose(l1.value, 0, atol=1e-15)


def test_outer_intervals_are_paired_reproducible_and_have_raw_draws():
    values = np.random.default_rng(19).normal(size=(80, 3))
    kwargs = dict(keys=["a", "b", "cash"], lengths=[1, 4, 8, 13], horizon_steps=40,
                  mean_block_lengths=[13], n_resamples=19, root_seed=73,
                  max_lag=4, contrasts=[(4, 1)])
    draws, intervals = source_uncertainty_diagnostics(values, **kwargs)
    repeated, _ = source_uncertainty_diagnostics(values, **kwargs)
    pd.testing.assert_frame_equal(draws, repeated)
    assert (intervals.n_finite == 19).all()
    assert (intervals.ci_lower <= intervals.ci_upper).all()
    metric = "squared_demeaned_acf_rmse_non_cash11_lags1_4"
    subset = draws.loc[draws.metric == metric]
    a = subset.loc[(subset.L == 4) & (subset.comparator_L == 0)].value.to_numpy()
    b = subset.loc[(subset.L == 1) & (subset.comparator_L == 0)].value.to_numpy()
    difference = subset.loc[(subset.L == 4) & (subset.comparator_L == 1)].value.to_numpy()
    np.testing.assert_allclose(difference, a - b, atol=1e-15)


def test_selector_uses_corrected_library_and_reports_no_mbb_optimum():
    pytest.importorskip("arch")
    from arch.bootstrap import optimal_block_length

    values = np.random.default_rng(99).normal(size=(200, 2))
    table = automatic_block_length_diagnostics(values, ["a", "cash"])
    actual = table.loc[(table["transform"] == "raw") & (table.driver_or_pair == "a")].iloc[0]
    reference = optimal_block_length(values[:, 0]).iloc[0]
    assert actual.stationary_length == pytest.approx(reference.stationary)
    assert actual.circular_length == pytest.approx(reference.circular)
    assert actual.cube_root_floor == 5
    assert "not a calibrated" in actual.interpretation
    constant = automatic_block_length_diagnostics(np.ones((50, 2)), ["a", "cash"])
    assert (constant.status == "undefined_zero_variance").all()


def test_sampler_regeneration_is_reference_prefix_for_l4():
    # The supplementary run must preserve the reference RNG's first paths.
    def sample(size):
        return sample_moving_block_indices(
            n_obs=782, n_scenarios=size, horizon_steps=260, block_length_steps=4,
            rng=np.random.default_rng(np.random.SeedSequence([20260913, 101, 4, 0]))
        ).source_indices
    np.testing.assert_array_equal(sample(64), sample(100)[:64])


def _synthetic_figure5_table():
    records = []
    assets = ("global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold")
    for asset in assets:
        for experiment, reference, count, offset in (
            ("historical_rolling", "historical_rolling", 523, .02),
            ("reference_L04", "reference_L04_L04_r0", 10000, -.03),
            ("synchronized_mbb_L04", "synchronized_mbb_L04_L04_r0", 3000, .50),
        ):
            for metric, value in (("q05", offset), ("median", offset + .04), ("q95", offset + .08)):
                records.append({"asset": asset, "experiment": experiment, "reference": reference,
                                "horizon_steps": 260, "return_kind": "annualized_geometric_return",
                                "metric": metric, "value": value, "n_paths": count})
    return pd.DataFrame.from_records(records)


def test_figure5_exports_and_renders_both_announced_series(tmp_path):
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_plots import figure05_terminal_outcomes

    data, paths = figure05_terminal_outcomes(_synthetic_figure5_table(), output_dir=tmp_path)
    assert len(data) == 8
    assert set(data.reference) == {"historical_rolling", "reference_L04"}
    mbb = data.loc[data.reference == "reference_L04"]
    assert (mbb.q05 == -.03).all()
    assert (mbb.n_paths == 10000).all()
    assert (mbb.source_reference == "reference_L04_L04_r0").all()
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)


@pytest.mark.parametrize("defect", ["missing_mbb", "missing_quantile", "duplicate", "nonfinite"])
def test_figure5_rejects_incomplete_or_ambiguous_series(defect):
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_plots import terminal_outcome_plot_data

    table = _synthetic_figure5_table()
    if defect == "missing_mbb":
        table = table.loc[table.experiment != "reference_L04"]
    elif defect == "missing_quantile":
        table = table.drop(table.loc[table.experiment == "reference_L04"].index[0])
    elif defect == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    else:
        table.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="Figure 5"):
        terminal_outcome_plot_data(table)


@pytest.mark.parametrize("override", [
    {"root_seed": 0}, {"outer_resamples": True}, {"outer_mean_block_lengths": ()},
    {"outer_mean_block_lengths": (13, 13)}, {"outer_confidence": 1.0},
    {"ljung_box_alpha": 0}, {"synthetic_calendar_start": "2026-01-03"},
    {"contrasts": ((4, 0),)}, {"spec_version": "2.0"},
])
def test_research_config_rejects_invalid_contract(override):
    from series.from_assumptions_to_portfolios.block1_scenarios.run_article3_block_length import BlockLengthResearchConfig

    with pytest.raises(ValueError):
        BlockLengthResearchConfig(**override)


def test_pearson_influence_selector_is_not_just_a_crossproduct():
    pytest.importorskip("arch")
    from arch.bootstrap import optimal_block_length

    values = np.random.default_rng(37).normal(0, .03, (150, 2))
    values[:, 1] += .8 * values[:, 0]
    keys = ["global_dm_ex_emu", "euro_govt_bond_7_10"]
    table = automatic_block_length_diagnostics(values, keys)
    row = table.loc[table["transform"] == "correlation_influence"].iloc[0]
    z = (values - values.mean(axis=0)) / values.std(axis=0, ddof=0)
    rho = (z[:, 0] * z[:, 1]).mean()
    influence = z[:, 0] * z[:, 1] - .5 * rho * (z[:, 0] ** 2 + z[:, 1] ** 2)
    expected = optimal_block_length(influence).iloc[0]
    assert row.circular_length == pytest.approx(expected.circular)
    assert row.stationary_length == pytest.approx(expected.stationary)


def test_resolution_can_change_correlation_without_changing_terminal_wealth():
    # Synthetic offset sequences: within-period cancellation removes information
    # from covariance. It is not legitimate to promise correlation invariance.
    paths = np.random.default_rng(12).normal(0, .02, (8, 12, 2))
    paths[:, 1::2, 1] += .03
    paths[:, ::2, 1] -= .03
    summary, pairs = resolution_diagnostics(paths, ["a", "b"], endpoints=[4, 8, 12], grid_name="four_week")
    assert summary.loc[summary.metric == "max_abs_terminal_log_difference", "value"].max() < 1e-14
    assert abs(pairs.coarse_minus_weekly.iloc[0]) > .01


def test_bundle_requires_complete_figure_sets(tmp_path):
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_block_length_reporting import build_research_bundle

    with pytest.raises(ValueError, match="all six original"):
        build_research_bundle(tmp_path, parent=tmp_path)
