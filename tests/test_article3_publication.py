"""Deterministic publication contracts using explicitly SYNTHETIC fixtures only."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest
import yaml

from data.asset_universe import get_driver_keys
from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import WeeklyInputProvenance

from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication import (
    ASSETS, EXHIBITS, LENGTHS, MEMORY_METRIC, PAIR, correlation_table_data,
    drawdown_plot_data, lineage_plot_data, memory_plot_data, publication_claims,
    resolution_plot_data, selector_table_data, tail_sequence_plot_data, wealth_plot_data,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_reporting import (
    build_publication_bundle, include_test_report,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import (
    sha256_file, write_artifact_manifest, write_json,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article3_publication import (
    PublicationConfig, resolve_runs, verify_artifact_tree, verify_publication_sources,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article3_block_length import BlockLengthResearchConfig
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import load_article3_config


def _tail_fixture():
    rows = []
    for lag, count in ((1, 22), (4, 23)):
        rows.append({"experiment": "historical_uniform", "group": "investable_ex_cash10",
                     "metric": "conditional_probability", "lag": lag, "value": count / 90,
                     "conditioning_events": 90, "L": np.nan, "replicate_id": np.nan})
    for length, values in ((1, (.1151, .1142)), (4, (.2126, .1154))):
        for lag, value in zip((1, 4), values):
            for replicate in range(5):
                rows.append({"experiment": "synchronized_iid_L01" if length == 1 else "synchronized_mbb_L04",
                             "group": "investable_ex_cash10", "metric": "conditional_probability",
                             "lag": lag, "value": value + (replicate - 2) * .0001,
                             "conditioning_events": 500, "L": length, "replicate_id": replicate})
    return pd.DataFrame(rows)


def _lineage_fixture():
    rows = []
    dates = pd.date_range("2000-01-07", periods=30, freq="W-FRI")
    for asset in (ASSETS[0], ASSETS[1], "gold"):
        for step in range(12):
            source = (step // 4) * 10 + step % 4
            rows.append({"scenario_id": 0, "asset": asset, "simulation_step": step,
                         "source_idx": source, "source_date": str(dates[source].date()),
                         "block_id": step // 4, "weekly_log_return": .01 * np.sin(step)})
    return pd.DataFrame(rows)


def _resolution_fixture():
    return pd.DataFrame([{"L": 4, "asset": asset, "grid": "four_week_endpoints", "metric": metric,
                          "replicate_id": replicate, "value": value + (replicate - 2) * .001}
                         for asset, values in zip(ASSETS, ((.3954, .38), (.191, .18), (.25, .24), (.30, .29)))
                         for metric, value in zip(("weekly_mdd_q95", "coarse_mdd_q95"), values)
                         for replicate in range(5)])


def _memory_fixture():
    return pd.DataFrame({"L": LENGTHS, "metric": MEMORY_METRIC,
                         "value": [.12, .10, .0828, .06, .05, .02, .01]})


def _drawdown_fixture():
    dm = {1: .3602, 2: .37, 4: .3954, 8: .38, 13: .35, 26: .32, 52: .2914}
    govt = {1: .1756, 2: .18, 4: .1910, 8: .20, 13: .21, 26: .23, 52: .2465}
    return pd.DataFrame([{"experiment": "synchronized_iid_L01" if length == 1 else f"synchronized_mbb_L{length:02d}",
                          "L": length, "asset": asset, "horizon_steps": 260,
                          "risk_kind": "maximum_drawdown", "metric": "q95", "replicate_id": replicate,
                          "value": values[length] + (replicate - 2) * .001}
                         for asset, values in zip(ASSETS[:2], (dm, govt))
                         for length in LENGTHS for replicate in range(5)])


def _selector_fixture():
    return pd.DataFrame([{"driver_or_pair": asset, "transform": transform, "circular_length": value}
                         for asset, values in zip(ASSETS, ((1.6, 18.3), (1., 50.), (4.7, 11.5), (2.1, 7.6)))
                         for transform, value in zip(("raw", "squared_demeaned"), values)])


def _correlation_fixtures():
    frames = [[], [], []]
    for length, values in ((1, (.145, .131, .139)), (4, (.144, .128, .192)), (52, (.156, .151, .373))):
        for replicate in range(5):
            common = {"L": length, "replicate_id": replicate, "driver_i": ASSETS[0], "driver_j": ASSETS[1]}
            frames[0].append(common | {"group": "all12", "correlation": values[0] + (replicate - 2) * .0001})
            frames[1].append(common | {"diagnostic": "rolling_correlation", "asset_or_pair": PAIR,
                                       "window_steps": 104, "metric": "q50", "value": values[1] + (replicate - 2) * .0001})
            frames[2].append(common | {"horizon_steps": 260, "return_type": "simple_compounded",
                                       "correlation": values[2] + (replicate - 2) * .0001})
    return tuple(pd.DataFrame(frame) for frame in frames)


def _publication_tables():
    # Contrived values exercise published-rounding assertions, not empirical findings.
    wealth, _, _ = wealth_plot_data(np.random.default_rng(7).normal(0, .01, (120, 12)), seed=[71, 401])
    return {"figure_a_workflow": pd.DataFrame({"order": range(5), "label": ["Synthetic\nfixture"] * 5}),
            "fig01_tail_sequence": tail_sequence_plot_data(_tail_fixture()),
            "fig02_block_lineage": lineage_plot_data(_lineage_fixture()),
            "fig03_time_resolution": resolution_plot_data(_resolution_fixture()),
            "fig04_volatility_memory": memory_plot_data(_memory_fixture()),
            "fig05_drawdown_sensitivity": drawdown_plot_data(_drawdown_fixture()),
            "fig06_wealth_paths": wealth,
            "table01_target_specific_lengths": selector_table_data(_selector_fixture()),
            "table_a1_correlations": correlation_table_data(*_correlation_fixtures())}


def test_tail_sequence_keeps_conditional_event_definition_and_exact_counts():
    table = tail_sequence_plot_data(_tail_fixture())
    assert len(table) == 6
    row = table.loc[(table.L == 0) & (table.lag == 1)].iloc[0]
    assert row.probability == pytest.approx(22 / 90)
    assert row.conditioning_events == 90 and row.joint_events == 22
    assert table.loc[(table.L == 4) & (table.lag == 1), "probability"].item() == pytest.approx(.2126)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite"])
@pytest.mark.parametrize("fixture,selector", [(_tail_fixture, tail_sequence_plot_data),
    (_resolution_fixture, resolution_plot_data), (_drawdown_fixture, drawdown_plot_data),
    (_memory_fixture, memory_plot_data), (_selector_fixture, selector_table_data)])
def test_publication_selectors_reject_incomplete_ambiguous_or_nonfinite_evidence(mutation, fixture, selector):
    frame = fixture()
    if mutation == "missing": frame = frame.drop(frame.index[-1])
    elif mutation == "duplicate": frame = pd.concat([frame, frame.iloc[[-1]]], ignore_index=True)
    else:
        column = "circular_length" if "circular_length" in frame else "value"
        frame.loc[frame.index[-1], column] = np.nan
    with pytest.raises(ValueError): selector(frame)


def test_resolution_is_mean_of_replicate_quantiles_and_keeps_units():
    table = resolution_plot_data(_resolution_fixture())
    assert table.loc[(table.asset == ASSETS[0]) & (table.metric == "weekly_mdd_q95"), "value"].item() == pytest.approx(.3954)
    bad = _resolution_fixture()
    bad.loc[bad.metric == "coarse_mdd_q95", "value"] = .99
    with pytest.raises(ValueError, match="cannot increase"): resolution_plot_data(bad)


def test_lineage_is_synchronized_and_does_not_silently_choose_another_scenario():
    frame = _lineage_fixture()
    assert len(lineage_plot_data(frame)) == 36
    frame.loc[0, "source_idx"] = 8
    with pytest.raises(ValueError, match="same source week"): lineage_plot_data(frame)
    frame = _lineage_fixture(); frame["scenario_id"] = 1
    with pytest.raises(ValueError, match="scenario zero"): lineage_plot_data(frame)


def test_correlation_table_does_not_confuse_rolling_log_and_terminal_estimands():
    weekly, rolling, terminal = _correlation_fixtures()
    extra = terminal.copy(); extra["return_type"] = "cumulative_log"; extra["correlation"] = -.9
    table = correlation_table_data(weekly, rolling, pd.concat([terminal, extra], ignore_index=True))
    row = table.loc[table.L == 4].iloc[0]
    assert row.same_week_correlation == pytest.approx(.144)
    assert row.median_104_week_rolling_correlation == pytest.approx(.128)
    assert row.five_year_outcome_correlation == pytest.approx(.192)
    with pytest.raises(ValueError): correlation_table_data(weekly.iloc[:-1], rolling, terminal)


def test_wealth_selection_is_reproducible_unique_and_includes_initial_wealth():
    returns = np.random.default_rng(12).normal(0, .01, (120, 9))
    a, ids, meta = wealth_plot_data(returns, seed=[20260913, 401])
    b, repeated_ids, _ = wealth_plot_data(returns, seed=[20260913, 401])
    pd.testing.assert_frame_equal(a, b); pd.testing.assert_frame_equal(ids, repeated_ids)
    assert len(ids) == ids.scenario_id.nunique() == 100
    assert len(a) == 101 * 10 and (a.loc[a.step == 0, "wealth"] == 1).all()
    assert meta["matches_original_published_selection"] is False
    assert meta["selection_source"] == "new_seeded_replacement"


def test_wealth_median_uses_all_paths_and_compounds_before_median():
    returns = np.log(np.array([[2.], [8.], [32.], [128.]]))
    table, ids, meta = wealth_plot_data(returns, seed=[71], n_paths=2, path_ids=[0, 1])
    assert table.loc[(table.series == "reference_median") & (table.step == 1), "wealth"].item() == pytest.approx(20)
    assert table.loc[(table.scenario_id == 1) & (table.step == 1), "wealth"].item() == pytest.approx(8)
    assert ids.scenario_id.tolist() == [0, 1]
    assert meta["median_population"] == "all_reference_paths_at_each_week"
    assert meta["selection_seed"] is None and meta["bit_generator"] is None
    assert not meta["matches_original_published_selection"]


@pytest.mark.parametrize("kwargs", [{"n_paths": True}, {"n_paths": 0}, {"n_paths": 5},
    {"n_paths": 2, "path_ids": [0, 0]}, {"n_paths": 2, "path_ids": [0., 1.]},
    {"n_paths": 2, "path_ids": [-1, 1]}, {"n_paths": 2, "path_ids": [0, 4]},
    {"n_paths": 2, "path_ids": [0]}, {"seed": [-1]}, {"seed": [True]}, {"seed": []}])
def test_wealth_selection_rejects_invalid_counts_ids_and_seeds(kwargs):
    arguments = {"seed": [19], "n_paths": 2} | kwargs
    with pytest.raises(ValueError): wealth_plot_data(np.zeros((4, 3)), **arguments)


@pytest.mark.parametrize("array", [np.ones(3), np.ones((3, 0)), np.array([[np.nan]])])
def test_wealth_paths_require_a_finite_nonempty_matrix(array):
    with pytest.raises(ValueError): wealth_plot_data(array, seed=[7], n_paths=1)


def test_publication_claims_check_all_34_published_roundings():
    tables = _publication_tables()
    claims = publication_claims(tables)
    assert len(claims) == 34 and claims.matches_published.all()
    tables["table_a1_correlations"].loc[0, "same_week_correlation"] = -.5
    with pytest.raises(ValueError, match="do not reconcile"): publication_claims(tables)


def test_manifest_verification_detects_tampering_and_escaping_paths(tmp_path):
    path = tmp_path / "synthetic_fixture.txt"; path.write_text("SYNTHETIC fixture", encoding="utf-8")
    write_artifact_manifest(tmp_path)
    assert "synthetic_fixture.txt" in verify_artifact_tree(tmp_path)
    path.write_text("changed SYNTHETIC fixture", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"): verify_artifact_tree(tmp_path)
    write_json(tmp_path / "artifact_manifest.json", {"entries": [
        {"relative_path": "../outside.txt", "sha256": "a" * 64}]})
    with pytest.raises(ValueError, match="escaping"): verify_artifact_tree(tmp_path)


def test_release_does_not_implicitly_build_missing_research(tmp_path):
    with pytest.raises(FileNotFoundError, match="--build-missing"):
        resolve_runs(root=tmp_path, parent=None, research=None, build_missing=False,
                     snapshot=Path("synthetic.csv"), manifest=Path("synthetic.json"), article2_metadata=None,
                     config_path=Path("synthetic.yaml"), research_config_path=Path("synthetic_research.yaml"), input_hash="a" * 64)


def test_explicit_build_missing_does_not_fabricate_article2_metadata(tmp_path, monkeypatch):
    from series.from_assumptions_to_portfolios.block1_scenarios import run_article3_publication as runner

    calls = []

    def synthetic_parent_builder(**kwargs):
        calls.append(kwargs)
        folder = tmp_path / "SYNTHETIC_NEW_PARENT"; folder.mkdir()
        write_json(folder / "run_metadata.json", {"run_id": "SYNTHETIC_NEW_PARENT"})
        return {"run_dir": folder}

    def synthetic_research_builder(**kwargs):
        calls.append(kwargs)
        folder = tmp_path / "SYNTHETIC_NEW_RESEARCH"; folder.mkdir()
        return folder

    monkeypatch.setattr(runner, "run_set_a", synthetic_parent_builder)
    monkeypatch.setattr(runner, "run_research", synthetic_research_builder)
    parent, research = resolve_runs(root=tmp_path, parent=None, research=None, build_missing=True,
        snapshot=Path("SYNTHETIC.csv"), manifest=Path("SYNTHETIC.manifest.json"), article2_metadata=None,
        config_path=Path("configs/article3_set_a_weekly.yaml"),
        research_config_path=Path("configs/article3_block_length_weekly.yaml"), input_hash="a" * 64)
    assert parent.name == "SYNTHETIC_NEW_PARENT" and research.name == "SYNTHETIC_NEW_RESEARCH"
    assert len(calls) == 2 and all(call["article2_metadata"] is None for call in calls)
    assert calls[0]["source_mode"] == "snapshot" and calls[0]["run_mode"] == calls[1]["mode"] == "full"


def test_publication_config_validation():
    config = PublicationConfig.load(Path("configs/article3_publication_weekly.yaml"))
    assert config.figure6_selection_seed == (20260913, 401)
    for change in ({"release_version": "9"}, {"figure6_n_paths": True}, {"figure6_n_paths": 100.0}, {"figure6_n_paths": 99},
                   {"published_snapshot_sha256": "bad"}, {"figure6_selection_seed": (-1,)}):
        with pytest.raises(ValueError): replace(config, **change)


def _synthetic_source_pair(tmp_path):
    """Metadata/checksum oracle; fake NPZ bytes are never used as financial paths."""
    parent, research = tmp_path / "synthetic_parent", tmp_path / "synthetic_research"
    parent.mkdir(); research.mkdir()
    config_path = Path("configs/article3_set_a_weekly.yaml")
    set_a = load_article3_config(config_path)
    config = replace(PublicationConfig.load(Path("configs/article3_publication_weekly.yaml")),
                     published_snapshot_sha256="a" * 64, published_canonical_panel_sha256="b" * 64)
    provenance = WeeklyInputProvenance(Path("SYNTHETIC.csv"), Path("SYNTHETIC.manifest.json"),
                                      "a" * 64, "c" * 64, "b" * 64, None, "not_checked", {})
    parent_meta = {"run_id": "SYNTHETIC_PARENT", "run_mode": "full", "input_file_sha256": "a" * 64,
                   "canonical_panel_sha256": "b" * 64, "H": 260, "L": 4, "S": 10000,
                   "root_seed": 20260913, "driver_keys": get_driver_keys()}
    write_json(parent / "run_metadata.json", parent_meta)
    write_json(parent / "run_status.json", {"status": "FULL_COMPLETE"})
    (parent / "resolved_config.yaml").write_text(yaml.safe_dump(set_a.as_dict()), encoding="utf-8")
    for name in ["tables/tail_sequence_dependence.csv", "tables/tail_thresholds.csv", "tables/path_risk_summary.csv",
                 "figures/plot_data/fig01_block_lineage.csv", "scenarios/set_a_L04.npz", "scenarios/set_a_L04_lineage.csv.gz"]:
        path = parent / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC checksum fixture, not financial data")
    write_artifact_manifest(parent)
    research_meta = {"run_mode": "full", "input_file_sha256": "a" * 64,
                     "canonical_panel_sha256": "b" * 64, "parent_run_id": "SYNTHETIC_PARENT",
                     "parent_manifest_sha256": sha256_file(parent / "artifact_manifest.json"),
                     "parent_metadata_sha256": sha256_file(parent / "run_metadata.json"),
                     "root_seed": 20260913, "grid_replicates": 5, "n_paths_per_grid_replicate": 3000,
                     "set_a_config": set_a.as_dict(),
                     "research_config": asdict(BlockLengthResearchConfig())}
    write_json(research / "run_metadata.json", research_meta)
    write_json(research / "run_status.json", {"status": "FULL_COMPLETE"})
    for name in ["resolution_summary.csv", "analytic_preservation.csv", "automatic_block_lengths.csv",
                 "weekly_correlation_matrices.csv", "rolling_diagnostics.csv", "terminal_correlations.csv",
                 "source_uncertainty_intervals.csv", "source_uncertainty_draws.csv.gz"]:
        path = research / "tables" / name; path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"SYNTHETIC checksum fixture, not financial data")
    write_artifact_manifest(research)
    return parent, research, provenance, config, config_path


def test_matching_serialized_tuple_and_list_contracts_are_accepted(tmp_path):
    parent, research, provenance, config, config_path = _synthetic_source_pair(tmp_path)
    checked = verify_publication_sources(parent, research, provenance, config, set_a_config_path=config_path)
    assert checked["parent_metadata"]["run_id"] == "SYNTHETIC_PARENT"


@pytest.mark.parametrize("defect", ["snapshot", "parent", "smoke", "grid", "set_a_config", "research_config", "manifest_coverage", "archive_hash"])
def test_release_rejects_incompatible_or_tampered_frozen_sources(tmp_path, defect):
    parent, research, provenance, config, config_path = _synthetic_source_pair(tmp_path)
    meta = json.loads((research / "run_metadata.json").read_text())
    if defect == "snapshot": provenance = replace(provenance, input_file_sha256="d" * 64)
    elif defect == "parent": meta["parent_run_id"] = "OTHER_SYNTHETIC_PARENT"
    elif defect == "smoke": write_json(research / "run_status.json", {"status": "SMOKE_COMPLETE"})
    elif defect == "grid": meta["grid_replicates"] = 1
    elif defect == "set_a_config": meta["set_a_config"]["block_length_steps"] = 13
    elif defect == "research_config": meta["research_config"]["outer_mean_block_lengths"] = [1]
    elif defect == "archive_hash": (parent / "scenarios/set_a_L04.npz").write_bytes(b"changed SYNTHETIC fixture")
    write_json(research / "run_metadata.json", meta)
    write_artifact_manifest(research)
    if defect == "manifest_coverage":
        manifest = json.loads((research / "artifact_manifest.json").read_text())
        manifest["entries"] = [row for row in manifest["entries"] if row["relative_path"] != "tables/analytic_preservation.csv"]
        write_json(research / "artifact_manifest.json", manifest)
    with pytest.raises(ValueError):
        verify_publication_sources(parent, research, provenance, config, set_a_config_path=config_path)


def test_test_report_cannot_claim_success_for_failed_or_missing_tests(tmp_path):
    report = tmp_path / "synthetic_report.xml"
    report.write_text('<testsuite><testcase name="synthetic_pass"/><testcase name="synthetic_skip"><skipped/></testcase></testsuite>')
    summary = include_test_report(report, tmp_path)
    assert summary["passed"] == summary["skipped"] == 1
    assert summary["report_sha256"] == sha256_file(report)
    report.write_text('<testsuite><testcase><failure/></testcase></testsuite>')
    with pytest.raises(ValueError): include_test_report(report, tmp_path)


def test_all_nine_exhibits_render_and_bundle_claim_references_resolve(tmp_path):
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_plots import render_publication

    tables = _publication_tables()
    render_publication(tables, tmp_path / "figures")
    data_dir = tmp_path / "figures/plot_data"; data_dir.mkdir()
    for stem, table in tables.items(): table.to_csv(data_dir / f"{stem}.csv", index=False)
    publication_claims(tables).to_csv(tmp_path / "article_claims.csv", index=False)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 9
    assert len(list((tmp_path / "figures").glob("*.svg"))) == 9
    assert all((tmp_path / f"figures/{stem}.png").stat().st_size > 1000 for stem in EXHIBITS)
    for name in ("article3_results.md", "figure_captions.md", "CHATGPT_HANDOVER.md", "START_PROMPT.txt",
                 "run_metadata.json", "run_status.json", "artifact_manifest.json", "figure6_selection.json",
                 "figure6_path_selection.csv", "article_exhibit_map.csv", "external_artifacts.json", "verification.json"):
        (tmp_path / name).write_text("SYNTHETIC bundle completeness fixture", encoding="utf-8")
    bundle = build_publication_bundle(tmp_path)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert "figures/plot_data/fig06_wealth_paths.csv" in archive.namelist()
    (tmp_path / "unexpected.npz").write_bytes(b"SYNTHETIC forbidden fixture")
    with pytest.raises(ValueError, match="cannot enter"): build_publication_bundle(tmp_path)


def test_publication_bundle_requires_every_article_exhibit(tmp_path):
    with pytest.raises(ValueError, match="required exhibit"): build_publication_bundle(tmp_path)


def test_lineage_panel_labels_do_not_overlap_repeated_x_tick_labels():
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_plots import _lineage

    figure = _lineage(lineage_plot_data(_lineage_fixture()))
    assert not figure.axes[0].get_xticklabels() and not figure.axes[1].get_xticklabels()
    assert len(figure.axes[2].get_xticklabels()) == 12
    figure.clear()
