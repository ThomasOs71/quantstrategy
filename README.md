# QuantStrategy

QuantStrategy is the code companion repository for the Substack flagship series *From Assumptions to Portfolios*, published at [quantstrategy.substack.com](https://quantstrategy.substack.com). The Substack also includes the no-code companion series *The Allocator's Toolkit*; because those practitioner articles are intentionally no-code, this repository does not maintain a separate Toolkit folder.

## Quick Start (data + tests)

- Ensure a `data/raw/msci/` folder with the required MSCI exports before running data tests.
  See [`data/raw/msci/README.md`](data/raw/msci/README.md) for download details.
- Keep raw downloaded files out of git (as described in that README).
- Make sure your FRED API key is available (required by parts of the pipeline):
  - `API_Keys/FredAPI.txt` (preferred for local setup), or
  - `FRED_API_KEY` environment variable.

### Verify local raw MSCI files

This repository includes a dedicated test that validates your manually downloaded MSCI files:

```powershell
$env:BLOCK1_CHECK_RAW_DATA = "1"
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

Linux/macOS:

```bash
export BLOCK1_CHECK_RAW_DATA=1
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

### Optional: full return-panel QA test

```powershell
python -m pytest -q tests/test_load_data.py -k return_panel_quality_smoke
```

### Expected target sample window

After all required sources are present, the intended complete Block-1 driver panel is:

- all 12 driver series (including `fx_eurusd`)
- no missing values for the monthly window **2011-01-31 to 2025-12-31**

Use this check to confirm:

```powershell
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c=p.index[complete]; print(f'complete_rows={len(c)}', f'first={c.min().date()}', f'last={c.max().date()}')"
```

Linux/macOS:

```bash
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c=p.index[complete]; print(f'complete_rows={len(c)}', f'first={c.min().date()}', f'last={c.max().date()}')"
```

Disable live-data QA if needed:

```powershell
$env:SKIP_DATA_QA = "1"
python -m pytest -q tests/test_load_data.py
```

Linux/macOS:

```bash
export SKIP_DATA_QA=1
python -m pytest -q tests/test_load_data.py
```
