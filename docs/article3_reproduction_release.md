# Article 3 reproduction release

This release reproduces the numerical evidence and exports the exhibits for
[How Much History Should a Scenario Remember?](https://quantstrategy.substack.com/p/how-much-history-should-a-scenario).
It reuses verified weekly research rather than rebuilding the legacy monthly
pipeline. Figure A, Figures 1–6 and Tables 1/A1 have explicit PNG, SVG and CSV
exports. Figure 6 is a documented replacement selection, not a reconstruction
of the original unrecorded 100-path selection.

## Install from a fresh clone

Run from the repository root with Python 3.11. The optional release requirements
pin the direct dependencies used for the verified numerical environment:

```powershell
git clone https://github.com/ThomasOs71/quantstrategy.git
cd quantstrategy
```

Until the release PR is merged, run `git switch codex/article3-set-a-weekly-v2`
before the installation below. After merge, `main` contains the entrypoint.

```powershell
python -m venv .qs_venv
.\.qs_venv\Scripts\Activate.ps1
python -m pip install -r requirements_article3_release.txt
```

On Linux/macOS activate with `source .qs_venv/bin/activate`.
Numerical tolerances and published rounding are checked; ZIP bytes and rendered
font metrics are not promised to be identical across platforms.

## Obtain and verify the weekly input

No market observations or keys are stored in Git or the author ZIP. Configure
your own FRED key in `FRED_API_KEY` or the ignored `API_Keys/FredAPI.txt`. Do not
put the key in a shell command, script, commit or author handover.

If you already have the verified snapshot, keep it and its manifest in:

```text
data/downloads/daily_proxy_2011/weekly/
  daily_proxy_2011_weekly_2011-01-07_2025-12-26.csv
  daily_proxy_2011_weekly_2011-01-07_2025-12-26.manifest.json
```

Only when those files are absent, explicitly request a download:

```powershell
python -c "from data.load_data import download_return_panel; download_return_panel(frequency='weekly', start='2011-01-07', end='2025-12-31')"
```

This network-dependent step uses Yahoo proxies, the issuer NAV endpoint and
FRED. Free access is not an open-data licence, and endpoint availability and
redistribution rights are not guaranteed. See `data/SOURCES.md`. Existing
snapshots are protected; do not overwrite them to resolve a checksum mismatch.

The published input is 782 W-FRI rows from 2011-01-07 through 2025-12-26 with
12 ordered drivers: eleven EUR investable-asset log returns and auxiliary FX.
The required fingerprints are versioned in
`configs/article3_publication_weekly.yaml`:

- CSV SHA-256:
  `f9be25192bdfb9ce403a4099ce1134d4a218bc68fa94807ea73fdd4da2c52d79`.
- Canonical panel SHA-256:
  `19dec3047785021c9302518a08d908faac48ccf1c8af9574f5bdafc435662d20`.

Providers can revise old observations. A newly downloaded panel can therefore
pass its own provenance checks but differ from this published version. The
release exporter stops on that mismatch. Exact reproduction then needs the
matching frozen snapshot obtained through permitted source access; a revised
dataset belongs in a separately identified research rerun, not this release.
There is no synthetic fallback or invented provenance.

## Reproduce with one research command

With the verified input present:

```powershell
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_publication --build-missing
```

The command reuses matching `FULL_COMPLETE` runs. If none exist, the explicit
`--build-missing` flag first generates the full weekly Set-A run and then the
block-length research. It performs no implicit downloads. Full generation is
CPU- and memory-intensive; the saved reference archive alone is about 290 MB.

The publication entrypoint does not require an Article-2 rerun. Its verified
weekly-input contract is independent of that optional reconciliation. To check
an ACTUAL existing Article-2 run, pass
`--article2-metadata articles/outputs/article2/daily_proxy_2011/weekly/run_metadata.json`.
A missing or different explicitly supplied file is an error; without this flag,
the release records `article2_reconciliation=not_checked`. No stand-in Article-2
metadata is manufactured.

Existing full runs can also be selected explicitly:

```powershell
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_publication --set-a-run artifacts/article3_set_a/article3_set_a_weekly_v2_20260913T193159Z --research-run artifacts/article3_set_a/article3_block_length_v1_full_20261013T150735Z_d95e8a6c
```

Those example IDs identify the original local research. They are not folders
downloaded by cloning Git. A fresh clone uses `--build-missing` or paths to its
own verified full runs. Without that flag, missing runs cause an informative
error rather than an unexpected expensive calculation.

## Outputs and article mapping

Each invocation creates a new directory below ignored `artifacts/article3_set_a/`
with prefix `article3_publication_v1_`. No previous research output is changed.

| Article element | Export stem | Evidence |
|---|---|---|
| Figure A | `figure_a_workflow` | Code-native conceptual workflow |
| Figure 1 | `fig01_tail_sequence` | Conditional tail sequencing at lags 1 and 4 |
| Figure 2 | `fig02_block_lineage` | Recorded scenario zero and synchronized source weeks |
| Figure 3 | `fig03_time_resolution` | Same L=4 paths, weekly/four-week endpoint MDD |
| Figure 4 | `fig04_volatility_memory` | Exact expected squared-ACF RMSE, lags 1–13 |
| Figure 5 | `fig05_drawdown_sensitivity` | Five-run mean q95 MDD by block length |
| Figure 6 | `fig06_wealth_paths` | Documented replacement selection from saved paths |
| Table 1 | `table01_target_specific_lengths` | Corrected CBB suggestions, raw/squared targets |
| Table A1 | `table_a1_correlations` | Weekly, rolling and terminal correlations |

Every stem has PNG/SVG files under `figures/` and an unrounded CSV under
`figures/plot_data/`. `article_exhibit_map.csv` and `figure_captions.md` provide
the exact mapping. The new renderings preserve the numerical content but are
not pixel-identical copies of externally typeset Substack images.

The older research `fig05_terminal_outcomes` is NOT the published Figure 5.
Its corrected historical/MBB comparison remains in the supplemental evidence
under `research/baseline/figures/`.

## Figure 6 selection and article replacement

The original article's 100 selected scenario IDs and selection seed were not
archived. The release therefore uses a NEW deterministic illustration selection:
NumPy PCG64 with `SeedSequence([20260913, 401])`, sampling 100 distinct IDs
without replacement from the SAME saved 10,000-path reference. It does not
generate another scenario set. `figure6_path_selection.csv` lists IDs in draw
order; `figure6_selection.json` records the selection and its replacement status.

Wealth is `exp(cumsum(weekly_log_returns))`, including an initial value of one.
The dark line is the pointwise median of wealth across all 10,000 paths, not
the median of the selected 100 and not an individual scenario. The plot-data
CSV contains every illustrated value plus this full-reference median.

Replace the published Figure 6 with this exported rendering for exact
illustration reproducibility. This does not change any research result or the
underlying reference distribution. To replay a known selection, pass
`--figure6-path-ids <CSV-with-scenario_id-column>`; IDs are validated for count,
integrality, uniqueness and range. Supplying IDs does not independently prove
that they were the original published selection.

## Author bundle for the project chat

`article3_author_bundle.zip` includes:

- the seven article-numbered figures, two table images and their complete CSVs;
- `article3_results.md`, `article_claims.csv`, captions and the exhibit map;
- `CHATGPT_HANDOVER.md` and `START_PROMPT.txt`;
- Figure-6 IDs, seed and explicit replacement disclosure;
- baseline/supplementary evidence, including the previously omitted tail
  sequencing and threshold tables, and outer-bootstrap score draws;
- source-code fingerprints and copies, reproduction instructions, dependency
  pins, source-artifact checksums and an optional actual pytest report.

All root numerical claims point to files inside the ZIP. Raw market panels,
complete scenario cubes, full lineage rows and secrets are excluded. Their
source identities are recorded in `external_artifacts.json`. The bundle remains
a local author-review artifact until source/publication rights are reviewed.
It is not automatically pushed to GitHub or uploaded to another chat.

## Validation

```powershell
python -m pytest tests/test_article3_set_a.py tests/test_article3_block_length.py tests/test_article3_publication.py -q
python -m pytest -q -rs --junitxml=data/downloads/article3_release_tests.xml
git diff --check
```

On Windows, if default temporary-folder rename permissions prevent snapshot
tests, pass a fresh, nonexistent `--basetemp` path under ignored
`data/downloads/`. Do not reuse an existing directory: pytest may remove its
contents. Keep opt-in raw/licensed and network checks separate. The offline test
suite selects the noninteractive Agg backend in `tests/conftest.py`; no Tk/Qt
desktop installation is required. The publication CLI also renders headlessly.

An actual successful JUnit report can be attached to the final export:

```powershell
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_publication --test-report data/downloads/article3_release_tests.xml
```

The exporter checks source manifests, parent compatibility, all publication
rounding assertions, unique five-replicate coverage, archive dimensions,
probabilities, exhibit completeness, internal claim references, ZIP CRC and
SHA-256. Visually review all nine images before using them. Full research
reconstruction and an artifact-only re-export are distinct checks.

## Financial interpretation

The sampler, root seed, 52-step annualization, 260-step horizon and 52/156/260
terminal prefixes are unchanged. Returns are neither demeaned nor rescaled for
wealth calculations. Temporal preservation is target-specific: L=4 is a useful
short-block reference; longer blocks preserve the squared-ACF score better.
Selector suggestions target scalar stationary/CBB inference, not this fixed-MBB
drawdown problem. Five simulation runs measure Monte Carlo variation, whereas
the outer stationary bootstrap examines conditional historical-source
uncertainty. Neither supplies missing crises, calibrated future probabilities,
stationarity assurance or out-of-sample validation.
