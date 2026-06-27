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
   (or pass it into `build_return_panel`).
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
