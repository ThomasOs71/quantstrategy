# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 14:57:00 2025

@author: thoma
"""

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "SP500_full_list.csv"

def get_sp500_constituents_from_wikipedia() -> pd.DataFrame:
    """
    Lädt die S&P-500-Liste von Wikipedia und gibt ein DataFrame mit:
    Ticker, Name, Sector, IndustryGroup, Ticker_yf, Date_added (datetime)
    zurück.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Alle Tabellen parsen
    tables = pd.read_html(StringIO(resp.text))

    # Tabelle mit GICS-Infos finden
    target = None
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("GICS" in c for c in cols):
            target = t
            break

    if target is None:
        # Debug-Ausgabe, wenn sich Wikipedia mal komplett ändert
        for i, t in enumerate(tables):
            print(f"Table {i} columns:", t.columns)
        raise ValueError("Keine Tabelle mit GICS-Spalten gefunden – Wikipedia-Struktur hat sich geändert?")

    df = target.copy()
    cols = list(df.columns)

    # Robust die relevanten Spalten finden
    symbol_col = next((c for c in cols if "Symbol" in str(c)), None)
    if symbol_col is None:
        symbol_col = cols[0]

    name_col = next((c for c in cols if "Security" in str(c)), None)
    if name_col is None:
        name_col = cols[1]

    sector_col = next(
        (c for c in cols if "GICS" in str(c) and "Sector" in str(c)),
        None
    )
    subind_col = next(
        (c for c in cols if "GICS" in str(c) and "Sub" in str(c)),
        None
    )

    # "Date added"-Spalte (Eintrittsdatum in den Index)
    date_added_col = next(
        (c for c in cols if "Date" in str(c) and "added" in str(c)),
        None
    )

    if sector_col is None or subind_col is None:
        print("Gefundene Spalten:", df.columns)
        raise ValueError("Konnte GICS Sector / GICS Sub-Industry nicht eindeutig finden.")

    if date_added_col is None:
        print("Warnung: Konnte 'Date added'-Spalte nicht eindeutig finden. Spalten sind:")
        print(df.columns)

    # Rename-Map aufbauen
    rename_map = {
        symbol_col: "Ticker",
        name_col: "Name",
        sector_col: "Sector",
        subind_col: "IndustryGroup",
    }
    if date_added_col is not None:
        rename_map[date_added_col] = "Date_added"

    # In saubere Spaltennamen umbenennen
    df = df.rename(columns=rename_map)

    # YFinance-Ticker (BRK.B -> BRK-B etc.)
    df["Ticker_yf"] = df["Ticker"].str.replace(".", "-", regex=False)

    # Date_added in datetime konvertieren (falls vorhanden)
    if "Date_added" in df.columns:
        df["Date_added"] = pd.to_datetime(df["Date_added"], errors="coerce")

    # Nur relevante Spalten für deine Pipeline
    cols_out = ["Ticker", "Name", "Sector", "IndustryGroup", "Ticker_yf"]
    if "Date_added" in df.columns:
        cols_out.append("Date_added")

    df_simple = df[cols_out].copy()

    return df_simple

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the current S&P 500 constituents table from Wikipedia."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path. Defaults to the article folder.",
    )
    args = parser.parse_args()

    sp500 = get_sp500_constituents_from_wikipedia()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Anzahl Constituents: {len(sp500)}")
    print(sp500.dtypes)  # Kontrollcheck: Date_added sollte datetime64[ns] sein

    sp500.to_csv(output_path, index=False)
    print(f"{output_path.resolve()} wurde gespeichert.")
