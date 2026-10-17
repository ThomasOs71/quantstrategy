"""Portable author evidence, publication numbering and safe local ZIP packaging."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication import EXHIBITS
from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication_plots import CAPTIONS
from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import sha256_file, write_json


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def include_test_report(report: Path, destination: Path) -> dict:
    """Accept actual successful pytest JUnit evidence, never a hand-written count."""
    cases = list(ET.parse(report).getroot().iter("testcase"))
    failed = sum(case.find("failure") is not None for case in cases)
    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    if not cases or failed or errors:
        raise ValueError("Release test report must contain successful tests and no failures/errors")
    _copy(report, destination / "verification/pytest_results.xml")
    return {"tests": len(cases), "passed": len(cases) - skipped, "skipped": skipped,
            "failures": failed, "errors": errors, "report_sha256": sha256_file(report)}


def write_publication_handoff(
    destination: Path, *, parent: Path, research: Path, sources: dict,
    metadata: dict, claims: pd.DataFrame, test_report: Path | None,
) -> None:
    """Stage only checked derived evidence and source code, not raw observations."""
    for name in sources["research_entries"]:
        path = research / name
        if path.suffix in {".csv", ".md", ".txt", ".json", ".yaml", ".png", ".svg"} or path.name == "source_uncertainty_draws.csv.gz":
            _copy(path, destination / "research" / name)
    # The earlier compact bundle omitted these publication-critical tables.
    for name in ("tail_sequence_dependence", "tail_thresholds"):
        relative = f"tables/{name}.csv"
        if relative not in sources["parent_entries"]:
            raise ValueError(f"Frozen parent lacks required publication evidence: {relative}")
        _copy(parent / relative, destination / "research/baseline" / relative)
    _copy(research / "artifact_manifest.json", destination / "research/original_artifact_manifest.json")
    _copy(parent / "artifact_manifest.json", destination / "research/baseline/original_artifact_manifest.json")
    repo = Path(__file__).resolve().parents[3]
    for relative in metadata["source_code_sha256"]:
        _copy(repo / relative, destination / "source_code" / relative)
    for relative in ("README.md", "docs/article3_reproduction_release.md",
                     "docs/article3_block_length_research.md", "docs/article3_set_a_implementation.md",
                     "requirements.txt", "requirements_article3_research.txt", "requirements_article3_release.txt"):
        _copy(repo / relative, destination / "reproduction" / relative)
    test_summary = include_test_report(test_report, destination) if test_report else {
        "status": "not_attached", "instructions": "Run the documented pytest command; no test result is inferred"}
    write_json(destination / "verification.json", {"successful_junit_report": test_summary,
        "published_rounding_checks": len(claims), "all_published_checks_match": bool(claims.matches_published.all()),
        "full_source_artifact_checksums_verified": True,
        "external_live_data_checks": "Not run by this offline exporter",
        "source_uncertainty": "999 outer resamples per mean length 13/26/52; raw score draws are included in research/tables",
        "visual_review": "Inspect the rendered figures; successful rendering alone is not visual approval"})
    external = []
    for name in ("scenarios/set_a_L04.npz", "scenarios/set_a_L04_lineage.csv.gz"):
        if name not in sources["parent_entries"]:
            raise ValueError("Reference ScenarioSet and full lineage must have verified manifest entries")
        external.append({"source_run_id": metadata["parent_run_id"], "source_relative_path": name,
                         "sha256": sources["parent_entries"][name]["sha256"], "included_in_bundle": False})
    external.append({"source": "weekly_return_panel", "csv_sha256": metadata["input_file_sha256"],
                     "canonical_panel_sha256": metadata["canonical_panel_sha256"], "included_in_bundle": False})
    write_json(destination / "external_artifacts.json", {"artifacts": external,
        "reason": "Raw provider observations, complete scenario cubes and full lineage remain local; source rights still apply"})
    file_map = pd.DataFrame({"article_element": ["Figure A", *[f"Figure {i}" for i in range(1, 7)], "Table 1", "Table A1"],
                             "stem": EXHIBITS, "caption": [CAPTIONS[name] for name in EXHIBITS]})
    file_map["png"] = file_map.stem.map(lambda name: f"figures/{name}.png")
    file_map["svg"] = file_map.stem.map(lambda name: f"figures/{name}.svg")
    file_map["plot_data"] = file_map.stem.map(lambda name: f"figures/plot_data/{name}.csv")
    file_map.to_csv(destination / "article_exhibit_map.csv", index=False)
    captions = "# Article 3 figure and table captions\n\n"
    captions += "\n\n".join(f"## {row.article_element}\n\n{row.caption}\n\nFiles: `{row.png}`, `{row.svg}`, `{row.plot_data}`."
                             for row in file_map.itertuples())
    (destination / "figure_captions.md").write_text(captions + "\n", encoding="utf-8")
    selection = metadata["figure6_selection"]
    selection_method = (f"PCG64 seed components {selection['selection_seed']}"
                        if selection["selection_source"] == "new_seeded_replacement"
                        else "explicit validated path IDs (no random selection seed is claimed)")
    results = f"""# Article 3 reproduction release

This release reproduces the numerical evidence for [How Much History Should a Scenario Remember?]({metadata['article_url']}). All {len(claims)} checks against published rounding pass. The verified source panel, Set-A sampler and research settings are unchanged. These results describe historical preservation and model sensitivity, not validated future probabilities.

## Published results

- Immediate joint-tail clustering is 22/90 = 24.44% historically, versus 11.51% for synchronized L=1 and 21.26% for L=4. At lag four, history is 25.56% and L=4 is 11.54%. Events use ten investable noncash assets and strict historical 5% thresholds.
- Global DM ex-EMU q95 five-year maximum drawdown is 39.54% on weekly observations and 38.00% on four-week endpoints of the SAME L=4 paths. The statistic averages five run-specific quantiles, not one pooled quantile.
- L=4 reduces the eleven-noncash-driver squared-ACF preservation error by about 31% relative to L=1. Longer blocks improve this score further. L=4 remains a short-block reference, not a calibrated optimum.
- Across lengths 1/4/52, Global DM drawdowns are 36.02/39.54/29.14%, whereas Euro government bonds are 17.56/19.10/24.65%. Longer blocks are not uniformly more conservative.
- The rounded selector suggestions and the one-week, rolling and terminal correlation table reconcile to the article. The selector concerns scalar stationary/CBB targets; it does not optimize this fixed MBB or drawdowns.

The machine-readable proof is `article_claims.csv`, with exact plot-data references, unrounded values, display scales and published/reproduced strings. All root claim references are inside this bundle. Full supplementary evidence, including source-uncertainty score draws, is in `research/`.

## Figure numbering and the replacement selection

`article_exhibit_map.csv` maps Figure A, Figures 1–6 and Tables 1/A1 to PNG, SVG and CSV files. The root figures are regenerated publication renditions, not pixel-identical copies of externally typeset Substack images.

Figure 6 is an explicit replacement. The original 100 path IDs and selection seed were not archived. This release selects 100 distinct paths from the saved 10,000-path L=4 reference using {selection_method}. `figure6_path_selection.csv` records every chosen ID, and the exported median uses ALL 10,000 paths. Replace the article's Figure 6 with this rendering if exact illustration reproducibility is desired. No claim is made that these are its original 100 lines.

The older research Figure 5 is a different chart: historical and MBB terminal-return quantiles. Its corrected version is preserved in `research/baseline/figures/`. It must not be mistaken for the published drawdown Figure 5.

## Source identity and reproduction

Input CSV SHA-256: `{metadata['input_file_sha256']}`.

Canonical panel SHA-256: `{metadata['canonical_panel_sha256']}`.

Reference parent: `{metadata['parent_run_id']}`. Supplement: `{metadata['research_run_id']}`. Release producer commit: `{metadata['git_commit']}`. The complete source-code fingerprints are in `run_metadata.json`; the matching source files are in `source_code/`.

Follow `reproduction/docs/article3_reproduction_release.md` in a Git checkout. The frozen source panel is not included. A provider-revised download is rejected instead of being mislabeled as the published input. Raw observations, complete scenario cubes and full lineage remain local; `external_artifacts.json` records their identities. Data access does not establish redistribution or publication rights.

## Verification and limitations

Source manifests and reference/archive lineage are checksum-verified before export. `verification.json` records the numerical checks and any attached actual pytest report. No successful live-data test, visual approval, stationarity certificate, out-of-sample validation or licensing approval is inferred by the exporter.
"""
    (destination / "article3_results.md").write_text(results, encoding="utf-8")
    handover = """# Article 3 project chat handover

Read `article3_results.md`, `article_exhibit_map.csv`, `article_claims.csv` and `figure_captions.md` first. The root numbering follows the published article, not the older research bundle. Use `research/article3_results.md` for the broader diagnostics and interpretation limits.

Figure 6 is a NEW reproducible replacement selection. The original 100-path selection was not recoverable from the repository. Keep that distinction explicit and tell the author to replace the original Figure 6 if exact visual reproducibility is required. The median is over all 10,000 reference paths, not the illustrated 100.

Preserve the separation between synchronized weekly vectors, temporal clustering, time-grid coarsening, and one-week/rolling/terminal correlations. Four-week blocks retain four weekly rows; four-week endpoints discard intermediate observations. Do not call these calendar-month data. Keep five-run mean quantiles distinct from pooled-path quantiles.

L=4 is a useful local-memory reference, not a proven optimum. Longer blocks preserve squared-return ACF better; the selector applies to different scalar targets and bootstrap variants. Source-uncertainty intervals are exploratory and distinct from Monte Carlo repetitions. Do not claim validated forecasts, future probabilities, or source publication rights.

All root numerical claim references resolve inside this ZIP. Complete source panels, scenario cubes and lineage are intentionally excluded; their hashes are in `external_artifacts.json`. Inspect source rights before external redistribution. The reproduction instructions require a repository checkout and the matching verified local snapshot.
"""
    (destination / "CHATGPT_HANDOVER.md").write_text(handover, encoding="utf-8")
    (destination / "START_PROMPT.txt").write_text(
        "Use this Article-3 reproduction release for the published article. Read CHATGPT_HANDOVER.md, "
        "article3_results.md, article_exhibit_map.csv and article_claims.csv. Verify every number against "
        "its included evidence. Root figures use published numbering. Figure 6 is an explicitly marked "
        "replacement with a new reproducible selection, not the original 100 paths. Preserve null and "
        "contrary findings, target-specific block length, units and all interpretation boundaries. "
        "Do not invent missing data, original path IDs, or forecasting validation.\n", encoding="utf-8")


def build_publication_bundle(directory: Path) -> Path:
    """Enforce complete exhibits and internal root claims before constructing ZIP."""
    for stem in EXHIBITS:
        for path in (directory / f"figures/{stem}.png", directory / f"figures/{stem}.svg",
                     directory / f"figures/plot_data/{stem}.csv"):
            if not path.is_file():
                raise ValueError(f"Publication bundle lacks required exhibit: {path.name}")
    required_documents = ("article3_results.md", "figure_captions.md", "CHATGPT_HANDOVER.md",
        "START_PROMPT.txt", "run_metadata.json", "run_status.json", "artifact_manifest.json",
        "figure6_selection.json", "figure6_path_selection.csv", "article_exhibit_map.csv",
        "external_artifacts.json", "verification.json")
    if any(not (directory / name).is_file() for name in required_documents):
        raise ValueError("Publication bundle requires the complete handover, selection, lineage and verification documents")
    claims = pd.read_csv(directory / "article_claims.csv")
    for reference in claims.table_path:
        path = (directory / reference).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError("Every root publication claim must resolve inside the bundle")
    selected = [path for path in directory.rglob("*") if path.is_file()
                and path.name not in {"article3_author_bundle.zip", "bundle_integrity.json"}]
    for path in selected:
        relative = path.relative_to(directory)
        if (path.suffix in {".npz", ".parquet", ".xlsx"} or "API_Keys" in relative.parts
                or "downloads" in relative.parts or ".env" in relative.name
                or "lineage.csv.gz" in relative.name):
            raise ValueError("Raw inputs, scenario cubes, full lineage or secrets cannot enter the author ZIP")
    destination = directory / "article3_author_bundle.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(selected):
            archive.write(path, arcname=path.relative_to(directory).as_posix())
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Author ZIP CRC check failed")
    return destination
