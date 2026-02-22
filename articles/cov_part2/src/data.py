from __future__ import annotations

from pathlib import Path
import pandas as pd
import urllib.request

def stooq_symbol(ticker: str, suffix: str = ".us") -> str:
    return f"{ticker.lower()}{suffix}"

def stooq_csv_url(symbol: str, interval: str = "d") -> str:
    return f"https://stooq.com/q/d/l/?i={interval}&s={symbol}"

def download_stooq_csv_if_missing(ticker: str, suffix: str, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    sym = stooq_symbol(ticker, suffix=suffix)
    fp = out_dir / f"{sym}.csv"
    if fp.exists():
        return fp
    url = stooq_csv_url(sym, interval="d")
    urllib.request.urlretrieve(url, fp)
    return fp

def load_close_panel(tickers: list[str], data_dir: Path, suffix: str = ".us") -> pd.DataFrame:
    series = []
    for t in tickers:
        sym = stooq_symbol(t, suffix=suffix)
        fp = data_dir / f"{sym}.csv"
        df = pd.read_csv(fp)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        series.append(df["Close"].astype(float).rename(t))
    px = pd.concat(series, axis=1).sort_index()
    return px.dropna(how="any")
