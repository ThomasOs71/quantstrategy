# Data Layout

This directory contains only data code and documentation.
Downloaded raw data files must stay outside version control.

## Expected local data structure

- `data/raw/msci/`
  - Manual MSCI CSV exports for Block-1
  - See `data/raw/msci/README.md` for required filenames
- `data/raw/fred_cache/`
  - Optional local caches for FRED requests (ignored by git)

## Why no raw data in git

`load_data.py` needs local files for two inputs:
- MSCI monthly index CSVs (`MSCI EMU`, `MSCI World ex EMU`)
- optional API caches

These files are intentionally excluded from git because of licensing/redistribution
constraints and key hygiene.

## How to run locally

1. Put your FRED API key into environment variable `FRED_API_KEY`
   (or into `data/FredAPI.txt`, `API_Keys/FredAPI.txt`, or pass it into `build_return_panel`).
2. Download MSCI CSV files under `data/raw/msci/` per instructions there.
3. Run your data pipeline.

Example:

```bash
python - <<'PY'
from data.load_data import build_return_panel

panel = build_return_panel()
print(panel.shape)
PY
```

## Target sample expectation (Block-1)

For this repository, the expected full monthly panel target is:

- 12 driver series (including `fx_eurusd`)
- full coverage without gaps between:
  - **2011-01-31** (inclusion start)
  - **2025-12-31** (inclusion end)

You can verify this quickly:

```powershell
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c_idx=p.index[complete]; print(f'complete_rows={len(c_idx)}', f'first={c_idx.min().date()}', f'last={c_idx.max().date()}')"
```

Linux/macOS:

```bash
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c_idx=p.index[complete]; print(f'complete_rows={len(c_idx)}', f'first={c_idx.min().date()}', f'last={c_idx.max().date()}')"
```

Interpretation:

- If `first` is `2011-01-31` and `last` is `2025-12-31`, the target window is fully downloaded.
- If `first` is later or `last` is earlier, at least one source series has missing coverage at either start or end.
