# quantstrategy

Repository for the Substack "quantstrategy".

## Environment

This repo is set up so subscribers can run all article entrypoints from one shared environment.

```powershell
py -3.11 -m venv .qs_venv
.\.qs_venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.qs_venv\Scripts\python.exe -m pip install -r requirements.txt
```

Recommended interpreter baseline: Python 3.11.

## Repo Hygiene

- The local virtual environment lives in `.qs_venv/` and is ignored by git.
- Python cache files such as `__pycache__/` and `*.pyc` are ignored by git.
- Some article folders contain generated outputs that are useful for published posts. Keep committing those only when you explicitly want them in the repo.
- Scratch outputs such as `articles/outputs/`, the LinkedIn-specific sweep image, and the generated `articles/Optimization_Lexicographic_2/outputs/` directory are treated as local artifacts and ignored by default.

## Article Run Matrix

| Article | Entry Script | Required Local Inputs | Live Data | Default Outputs |
| --- | --- | --- | --- | --- |
| Covariance Part 1 | `articles/Covariance_Part1.py` | None | Yes, downloads market data from Yahoo Finance via `yfinance` | Interactive matplotlib figures only |
| Convex Risk Measures | `articles/Convex_RiskMeasures/Convex_RiskMeasureCombination.py` | `expected_returns.xlsx`, `covariance.xlsx`, `df.xlsx` in the same folder | No | Interactive matplotlib figures only |
| S&P 500 Constituents Helper | `articles/Convex_RiskMeasures/s&p500_download.py` | None | Yes, fetches the current S&P 500 table from Wikipedia | `articles/Convex_RiskMeasures/SP500_full_list.csv` by default, or `--output` |
| Optimization Lexicographic | `articles/Optimization_Lexicographic/lexicographic_mean_cvar_blog.py` | `articles/Optimization_Lexicographic/input/mc_scenarios.xlsx` | No | Files under `articles/Optimization_Lexicographic/output/` |
| Optimization Lexicographic 2 | `articles/Optimization_Lexicographic_2/run_experiment.py` | `articles/Optimization_Lexicographic_2/input/mc_scenarios.xlsx` | No | Files under `articles/Optimization_Lexicographic_2/outputs/` |
| Covariance Part 2 | `articles/cov_part2/run_part2_frequency_trap.py --config config.yaml` | `config.yaml`; existing CSVs in `data_raw/` are reused if present | Downloads Stooq CSV files only when local data is missing | Files under `articles/cov_part2/results/` and `articles/cov_part2/figures/` |

## Notes

- The older article scripts have been normalized to use paths relative to their own folders, so they work when launched from the repo root.
- Some article scripts fetch current market or reference data, so reruns can produce different numbers than the published post.
