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
4. Save each CSV in `data/raw/msci/` with the expected file names.

Expected files:

- `msci_emu_ntr_usd.csv`
- `msci_world_ex_emu_ntr_usd.csv`

Any additional header/metadata rows are supported by the parser as long as the
actual data rows contain:

- column 1: date
- column 2: index level

Typical examples:

- `Dec-86,100.00`
- `2023-12-31,1234.56`
- `31/12/1986,100.00`
