# Streamlit Companion App

This app is a read-only interactive companion for the `Optimization_Lexicographic_2` article. It does not solve optimization problems live. Instead, it loads precomputed parquet/CSV files built from the existing experiment code.

Project links:

- Substack: `https://quantstrategy.substack.com/`
- GitHub: `https://github.com/ThomasOs71/quantstrategy`

## 1. Install app dependencies

```bash
pip install -r requirements_streamlit.txt
```

## 2. Generate the precomputed dataset

From the repo root:

```bash
python articles/Optimization_Lexicographic_2/build_streamlit_dataset.py
```

For a fast smoke build:

```bash
python articles/Optimization_Lexicographic_2/build_streamlit_dataset.py --smoke
```

This writes the Streamlit-ready files to:

```text
articles/Optimization_Lexicographic_2/streamlit_data/
```

## 3. Run the app locally

```bash
streamlit run streamlit_app/app.py
```

## Deployment notes

- The app uses repo-relative paths only.
- The committed dataset files make the app deployment-ready without a build step on the target host.
- If the parquet files are missing, the app falls back to CSV. If neither exists, it shows a user-friendly instruction message instead of crashing.

## Data files

The precompute pipeline exports:

- `sweep_results.parquet` and `sweep_results.csv`
- `portfolio_snapshots.parquet` and `portfolio_snapshots.csv`
- `summary_metrics.parquet` and `summary_metrics.csv`
- `data_manifest.json`

These files are keyed by the sidebar controls used in the app:

- Europe expected return
- portfolio risk factor
- TE-CVaR limit
- turnover limit

The sidebar also includes a compact asset reference table showing each asset's
expected return and scenario-based volatility. Europe expected return updates
with the slider selection; volatilities are fixed from the precomputed
scenario workbook build.

## Current production grid

The committed V1 dataset uses:

- Europe expected return from `7.0%` down to `6.0%` in `0.05` percentage-point steps
- portfolio risk factors: `0.95`, `1.00`, `1.05`, `1.10`
- TE-CVaR limits: `1%`, `2%`, `3%`, `4%`, `5%`
- turnover limits: `20%`, `30%`, `40%`
