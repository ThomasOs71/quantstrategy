# MSCI index data — manual download required

Do not place raw MSCI CSV files into git. Keep them in this folder only for
local execution.

1. Open https://app2.msci.com/products/index-data-search/
2. Export each required index:
   - MSCI EMU
   - MSCI World ex EMU
3. Settings:
   - Currency: USD
   - Level: Net Total Return
   - Frequency: Monthly
4. Save each file in `data/raw/msci/` with one of the following names (CSV or XLS are accepted):

Expected files:

- `msci_emu_ntr_usd.csv`
- `msci_world_ex_emu_ntr_usd.csv`

Also accepted:

- `msci_emu_ntr_usd.xls`
- `msci_world_ex_emu_ntr_usd.xls`
- `msci_emu_ntr_usd.xlsx`
- `msci_world_ex_emu_ntr_usd.xlsx`

Any additional header/metadata rows are supported by the parser as long as the
actual data rows contain:

- column 1: date
- column 2: index level

Typical examples:

- `Dec-86,100.00`
- `2023-12-31,1234.56`
- `31/12/1986,100.00`

## Quick verification

After placing both files, verify that they can be parsed with:

```powershell
$env:BLOCK1_CHECK_RAW_DATA = "1"
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

On Linux/macOS:

```bash
export BLOCK1_CHECK_RAW_DATA=1
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

## Target coverage note

The full monthly return panel (12 drivers incl. `fx_eurusd`) is expected to be complete
from **2011-01-31** to **2025-12-31** after both MSCI files and API-driven sources are available.
