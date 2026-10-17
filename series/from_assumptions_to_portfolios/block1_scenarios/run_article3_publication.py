"""Offline reproduction release for the published Article 3, with explicit lineage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import (
    WeeklyInputProvenance, load_verified_weekly_snapshot,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication import (
    correlation_table_data, drawdown_plot_data, lineage_plot_data, memory_plot_data,
    publication_claims, resolution_plot_data, selector_table_data, tail_sequence_plot_data,
    wealth_plot_data,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import (
    runtime_metadata, sha256_file, write_artifact_manifest, write_json,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article3_block_length import (
    BlockLengthResearchConfig, DEFAULT_MANIFEST, DEFAULT_OUTPUT, DEFAULT_SNAPSHOT,
    run_research, verify_parent_run,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article3_set_a import run as run_set_a
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import load_article3_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUBLICATION_CONFIG = Path("configs/article3_publication_weekly.yaml")


@dataclass(frozen=True)
class PublicationConfig:
    release_version: str
    article_url: str
    published_snapshot_sha256: str
    published_canonical_panel_sha256: str
    figure6_selection_seed: tuple[int, ...]
    figure6_n_paths: int = 100

    def __post_init__(self) -> None:
        if self.release_version != "1.0":
            raise ValueError("Unsupported Article-3 publication release version")
        if self.article_url != "https://quantstrategy.substack.com/p/how-much-history-should-a-scenario":
            raise ValueError("This exporter targets the published Article-3 URL only")
        for value in (self.published_snapshot_sha256, self.published_canonical_panel_sha256):
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError("Published input fingerprints must be lower-case SHA-256 strings")
        if not self.figure6_selection_seed or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.figure6_selection_seed
        ):
            raise ValueError("Figure-6 seed must contain nonnegative integers")
        if (isinstance(self.figure6_n_paths, bool) or not isinstance(self.figure6_n_paths, int)
                or self.figure6_n_paths != 100):
            raise ValueError("The publication illustration requires exactly 100 paths")

    @classmethod
    def load(cls, path: Path) -> PublicationConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("Publication configuration requires exactly the documented keys")
        payload["figure6_selection_seed"] = tuple(payload["figure6_selection_seed"])
        return cls(**payload)


def verify_artifact_tree(directory: Path) -> dict[str, dict]:
    """Verify immutable artifacts, rejecting duplicate and escaping manifest paths."""
    root = directory.resolve()
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    entries: dict[str, dict] = {}
    for row in manifest["entries"]:
        name = row["relative_path"]
        path = (root / name).resolve()
        if name in entries or Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Artifact manifest has a duplicate or escaping path")
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Artifact checksum mismatch or missing file: {name}")
        entries[name] = row
    return entries


def verify_publication_sources(
    parent: Path, research: Path, provenance: WeeklyInputProvenance, config: PublicationConfig,
    *, set_a_config_path: Path,
    research_config_path: Path = Path("configs/article3_block_length_weekly.yaml"),
) -> dict:
    if (provenance.input_file_sha256 != config.published_snapshot_sha256
            or provenance.canonical_panel_sha256 != config.published_canonical_panel_sha256):
        raise ValueError("Snapshot differs from the published fingerprint; do not label a provider-revised rerun as reproduction")
    set_a_config = load_article3_config(set_a_config_path)
    parent_meta = verify_parent_run(parent, input_hash=provenance.input_file_sha256,
                                    canonical_hash=provenance.canonical_panel_sha256, config=set_a_config)
    parent_entries = verify_artifact_tree(parent)
    research_entries = verify_artifact_tree(research)
    required_parent = {"run_metadata.json", "run_status.json", "resolved_config.yaml",
        "tables/tail_sequence_dependence.csv", "tables/tail_thresholds.csv", "tables/path_risk_summary.csv",
        "figures/plot_data/fig01_block_lineage.csv", "scenarios/set_a_L04.npz",
        "scenarios/set_a_L04_lineage.csv.gz"}
    if not required_parent.issubset(parent_entries):
        raise ValueError("Parent manifest must cover the publication tables, saved reference and full lineage")
    required_research = {"run_metadata.json", "run_status.json"} | {
        f"tables/{name}.csv" for name in ("resolution_summary", "analytic_preservation",
            "automatic_block_lengths", "weekly_correlation_matrices", "rolling_diagnostics",
            "terminal_correlations", "source_uncertainty_intervals")}
    required_research.add("tables/source_uncertainty_draws.csv.gz")
    if not required_research.issubset(research_entries):
        raise ValueError("Research manifest must cover every publication input table and metadata file")
    meta = json.loads((research / "run_metadata.json").read_text(encoding="utf-8"))
    status = json.loads((research / "run_status.json").read_text(encoding="utf-8"))
    expected = {"run_mode": "full", "input_file_sha256": provenance.input_file_sha256,
                "canonical_panel_sha256": provenance.canonical_panel_sha256,
                "parent_run_id": parent_meta["run_id"],
                "parent_manifest_sha256": sha256_file(parent / "artifact_manifest.json"),
                "parent_metadata_sha256": sha256_file(parent / "run_metadata.json"),
                "root_seed": 20260913, "grid_replicates": 5, "n_paths_per_grid_replicate": 3000}
    if status.get("status") != "FULL_COMPLETE" or any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError("Publication requires a FULL_COMPLETE research run from the matching frozen Set-A parent")
    if (parent_meta.get("S") != 10000 or parent_meta.get("H") != 260
            or meta.get("research_config", {}).get("outer_resamples") != 999):
        raise ValueError("Publication requires 10,000 reference paths, 260 weeks and 999 outer resamples")
    # JSON serializes configured tuples as lists; compare equivalent contracts.
    if meta.get("set_a_config") != json.loads(json.dumps(set_a_config.as_dict())):
        raise ValueError("Research Set-A configuration differs from the publication configuration")
    research_config = BlockLengthResearchConfig.load(research_config_path)
    if meta.get("research_config") != json.loads(json.dumps(asdict(research_config))):
        raise ValueError("Research diagnostic configuration differs from the publication configuration")
    return {"parent_metadata": parent_meta, "research_metadata": meta,
            "parent_entries": parent_entries, "research_entries": research_entries}


def _find_run(root: Path, prefix: str, **expected: object) -> Path | None:
    matches = []
    for metadata_path in root.glob(f"{prefix}*/run_metadata.json"):
        status_path = metadata_path.parent / "run_status.json"
        if not status_path.is_file():
            continue
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "FULL_COMPLETE" and all(meta.get(k) == v for k, v in expected.items()):
            matches.append(metadata_path.parent)
    return sorted(matches)[-1] if matches else None


def resolve_runs(
    *, root: Path, parent: Path | None, research: Path | None, build_missing: bool,
    snapshot: Path, manifest: Path, article2_metadata: Path | None,
    config_path: Path, research_config_path: Path, input_hash: str,
) -> tuple[Path, Path]:
    """Reuse full runs by default; expensive regeneration requires an explicit flag."""
    if parent is None and research is not None:
        meta = json.loads((research / "run_metadata.json").read_text(encoding="utf-8"))
        candidate = root / meta["parent_run_id"]
        if candidate.is_dir():
            parent = candidate
    parent = parent or _find_run(root, "article3_set_a_weekly_v2_", input_file_sha256=input_hash, run_mode="full")
    if parent is None:
        if not build_missing:
            raise FileNotFoundError("No full Set-A parent. Pass --build-missing or --set-a-run; no synthetic fallback is allowed")
        print("Building missing full Set-A run from the verified weekly snapshot", flush=True)
        parent = run_set_a(config_path=config_path, source_mode="snapshot", panel_snapshot=snapshot,
                           provenance_manifest=manifest, run_mode="full", output_root=root,
                           article2_metadata=article2_metadata)["run_dir"]
    parent_meta = json.loads((parent / "run_metadata.json").read_text(encoding="utf-8"))
    research = research or _find_run(root, "article3_block_length_v1_full_",
                                    parent_run_id=parent_meta["run_id"], input_file_sha256=input_hash)
    if research is None:
        if not build_missing:
            raise FileNotFoundError("No matching full block-length research. Pass --build-missing or --research-run")
        print("Building missing full block-length research from the existing Set-A parent", flush=True)
        research = run_research(config_path=config_path, research_config_path=research_config_path,
                                snapshot=snapshot, manifest=manifest, article2_metadata=article2_metadata,
                                parent_run=parent, output_root=root, mode="full")
    return parent.resolve(), research.resolve()


def _read_table(directory: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(directory / "tables" / f"{name}.csv", float_precision="round_trip")


def load_publication_tables(parent: Path, research: Path, config: PublicationConfig,
                            *, path_ids: list[int] | None = None) -> tuple[dict, pd.DataFrame, dict]:
    """Read the saved reference asset paths, not a newly drawn scenario set."""
    tables = {
        "figure_a_workflow": pd.DataFrame({"order": range(5), "label": [
            "Historical\nweekly returns\n782 × 12", "Synchronized\nfour-week\nblocks",
            "260-week\nasset paths", "Risk\ndiagnostics", "Later\nportfolio\nanalysis"]}),
        "fig01_tail_sequence": tail_sequence_plot_data(_read_table(parent, "tail_sequence_dependence")),
        "fig02_block_lineage": lineage_plot_data(pd.read_csv(
            parent / "figures/plot_data/fig01_block_lineage.csv", float_precision="round_trip")),
        "fig03_time_resolution": resolution_plot_data(_read_table(research, "resolution_summary")),
        "fig04_volatility_memory": memory_plot_data(_read_table(research, "analytic_preservation")),
        "fig05_drawdown_sensitivity": drawdown_plot_data(_read_table(parent, "path_risk_summary")),
        "table01_target_specific_lengths": selector_table_data(_read_table(research, "automatic_block_lengths")),
        "table_a1_correlations": correlation_table_data(
            _read_table(research, "weekly_correlation_matrices"), _read_table(research, "rolling_diagnostics"),
            _read_table(research, "terminal_correlations")),
    }
    with np.load(parent / "scenarios/set_a_L04.npz", allow_pickle=False) as archive:
        keys = archive["asset_keys"].tolist()
        probabilities = archive["p"]
        if len(probabilities) != 10000 or not np.allclose(probabilities, .0001, atol=1e-15, rtol=0):
            raise ValueError("Figure 6 requires the uniform 10,000-path reference distribution")
        returns = archive["asset_paths"][:, :, keys.index("global_dm_ex_emu")].copy()
    if returns.shape != (10000, 260):
        raise ValueError("Saved reference paths must contain 10,000 paths × 260 weeks")
    tables["fig06_wealth_paths"], ids, selection = wealth_plot_data(
        returns, seed=config.figure6_selection_seed, n_paths=config.figure6_n_paths, path_ids=path_ids)
    return tables, ids, selection


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def run_publication(
    *, publication_config_path: Path = DEFAULT_PUBLICATION_CONFIG,
    config_path: Path = Path("configs/article3_set_a_weekly.yaml"),
    research_config_path: Path = Path("configs/article3_block_length_weekly.yaml"),
    snapshot: Path = DEFAULT_SNAPSHOT, manifest: Path = DEFAULT_MANIFEST,
    output_root: Path = DEFAULT_OUTPUT, parent_run: Path | None = None,
    research_run: Path | None = None, article2_metadata: Path | None = None,
    build_missing: bool = False, path_ids: list[int] | None = None, test_report: Path | None = None,
) -> Path:
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_plots import render_publication
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_reporting import (
        build_publication_bundle, write_publication_handoff,
    )

    config = PublicationConfig.load(publication_config_path)
    output_root = output_root.resolve()
    if output_root.is_relative_to(REPO_ROOT) and not output_root.is_relative_to(REPO_ROOT / DEFAULT_OUTPUT):
        raise ValueError("In-repository release outputs must stay under ignored artifacts/article3_set_a")
    panel, provenance = load_verified_weekly_snapshot(snapshot, manifest, article2_metadata_path=article2_metadata)
    if provenance.input_file_sha256 != config.published_snapshot_sha256 or provenance.canonical_panel_sha256 != config.published_canonical_panel_sha256:
        raise ValueError("Published input fingerprint mismatch; use the frozen snapshot, not an unmarked replacement")
    if article2_metadata is not None and provenance.article2_snapshot_match_status != "matched":
        raise ValueError("Explicit Article-2 reconciliation must match the exact publication snapshot")
    parent, research = resolve_runs(root=output_root, parent=parent_run, research=research_run,
        build_missing=build_missing, snapshot=snapshot, manifest=manifest, article2_metadata=article2_metadata,
        config_path=config_path, research_config_path=research_config_path, input_hash=provenance.input_file_sha256)
    sources = verify_publication_sources(parent, research, provenance, config,
        set_a_config_path=config_path, research_config_path=research_config_path)
    tables, ids, selection = load_publication_tables(parent, research, config, path_ids=path_ids)
    claims = publication_claims(tables)
    run_id = f"article3_publication_v1_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    destination = output_root / run_id
    destination.mkdir(parents=True, exist_ok=False)
    source_files = sorted(Path(__file__).parent.glob("*.py")) + sorted((REPO_ROOT / "data").glob("*.py"))
    source_files += [REPO_ROOT / "series/from_assumptions_to_portfolios/flagship_plot_style.py",
                     config_path.resolve(), research_config_path.resolve(), publication_config_path.resolve()]
    metadata = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc), "article_url": config.article_url,
                "publication_config": asdict(config), "input_file_sha256": provenance.input_file_sha256,
                "canonical_panel_sha256": provenance.canonical_panel_sha256,
                "parent_run_id": sources["parent_metadata"]["run_id"],
                "research_run_id": sources["research_metadata"]["run_id"],
                "git_commit": _git("rev-parse", "HEAD"), "dirty_worktree": bool(_git("status", "--porcelain")),
                "runtime": runtime_metadata(), "source_code_sha256": {
                    path.relative_to(REPO_ROOT).as_posix(): sha256_file(path) for path in source_files},
                "article2_reconciliation": provenance.article2_snapshot_match_status,
                "input_shape": list(panel.shape), "figure6_selection": selection,
                "external_data_downloads": 0, "frozen_research_unchanged": True,
                "publication_rights_approved": False}
    try:
        write_json(destination / "run_metadata.json", metadata)
        data_dir = destination / "figures/plot_data"
        data_dir.mkdir(parents=True)
        for name, table in tables.items():
            table.to_csv(data_dir / f"{name}.csv", index=False, float_format="%.17g")
        ids.to_csv(destination / "figure6_path_selection.csv", index=False)
        write_json(destination / "figure6_selection.json", selection)
        claims.to_csv(destination / "article_claims.csv", index=False, float_format="%.17g")
        render_publication(tables, destination / "figures")
        write_publication_handoff(destination, parent=parent, research=research, sources=sources,
                                  metadata=metadata, claims=claims, test_report=test_report)
        write_json(destination / "run_status.json", {"status": "REPRODUCTION_COMPLETE",
            "published_numerical_claims_verified": len(claims), "figure6_original_selection_recovered": False,
            "figure6_replacement_exported": True, "publication_ready": False,
            "licensing_review_required_before_external_redistribution": True})
        write_artifact_manifest(destination, excluded_names=("artifact_manifest.json", "article3_author_bundle.zip"))
        bundle = build_publication_bundle(destination)
        write_json(destination / "bundle_integrity.json", {"bundle": bundle.name,
            "sha256": sha256_file(bundle), "size_bytes": bundle.stat().st_size})
        verify_artifact_tree(destination)
        print(json.dumps({"release": str(destination), "bundle": str(bundle), "verified_claims": len(claims)}, indent=2))
    except Exception as error:
        write_json(destination / "run_status.json", {"status": "FAILED", "publication_ready": False,
            "error_type": type(error).__name__, "error": str(error)})
        raise
    return destination


def main() -> None:
    # This CLI is a batch exporter, including any legacy-style research plots
    # generated by --build-missing. Do not require Tk/Qt or open GUI windows.
    import matplotlib
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-config", type=Path, default=DEFAULT_PUBLICATION_CONFIG)
    parser.add_argument("--config", type=Path, default=Path("configs/article3_set_a_weekly.yaml"))
    parser.add_argument("--research-config", type=Path, default=Path("configs/article3_block_length_weekly.yaml"))
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--set-a-run", type=Path)
    parser.add_argument("--research-run", type=Path)
    parser.add_argument("--article2-metadata", type=Path, help="Optional actual Article-2 metadata; never fabricated")
    parser.add_argument("--build-missing", action="store_true", help="Explicitly generate missing full research runs")
    parser.add_argument("--figure6-path-ids", type=Path, help="Optional CSV with scenario_id; does not prove original selection")
    parser.add_argument("--test-report", type=Path, help="Optional successful pytest JUnit XML to include in the bundle")
    args = parser.parse_args()
    ids = pd.read_csv(args.figure6_path_ids).scenario_id.tolist() if args.figure6_path_ids else None
    run_publication(publication_config_path=args.publication_config, config_path=args.config,
        research_config_path=args.research_config, snapshot=args.snapshot, manifest=args.manifest,
        output_root=args.output_root, parent_run=args.set_a_run, research_run=args.research_run,
        article2_metadata=args.article2_metadata, build_missing=args.build_missing, path_ids=ids,
        test_report=args.test_report)


if __name__ == "__main__":
    main()
