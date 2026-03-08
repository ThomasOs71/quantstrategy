from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from market_signal_suite import (
    add_expanding_regimes,
    build_etf_panel,
    compute_europe_aggregate,
    current_etf_signal,
    load_damodaran_snapshot,
    mapping_dataframe,
)
from multiples_research_pipeline import build_research_payload


OUTPUT_DIR = Path(__file__).with_name("research_outputs")
OUTPUT_PNG = OUTPUT_DIR / "usa_europe_regime_zones.png"
OUTPUT_MD = OUTPUT_DIR / "usa_europe_regime_zones.md"

ZONE_RED = "#d35454"
ZONE_GRAY = "#d7dbdd"
ZONE_GREEN = "#4caf50"
TEXT_DARK = "#17202a"
TEXT_MUTED = "#566573"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_inputs() -> dict[str, list[dict[str, object]]]:
    research = build_research_payload(save_outputs=False)
    us_live = research["us"]["live_df"].copy()
    us_quartile = us_live[us_live["method"].eq("quartile")].copy()

    us_rows = []
    for row in us_quartile.itertuples(index=False):
        us_rows.append(
            {
                "label": f"Shiller {row.signal}",
                "current": float(row.signal_value_pct),
                "low": float(row.low_cutoff_pct),
                "high": float(row.high_cutoff_pct),
                "regime": str(row.regime),
                "value_fmt": f"{row.signal_value_pct:.2f}%",
                "low_fmt": f"{row.low_cutoff_pct:.2f}%",
                "high_fmt": f"{row.high_cutoff_pct:.2f}%",
            }
        )

    snapshot = load_damodaran_snapshot()
    usa = snapshot["all_df"][snapshot["all_df"]["country_name"].eq("United States")].iloc[0]
    europe = compute_europe_aggregate(snapshot["all_df"], snapshot["spread_q1"], snapshot["spread_q3"])

    us_rows.append(
        {
            "label": "Damodaran EY-ERP",
            "current": float(usa["valuation_spread"] * 100.0),
            "low": float(snapshot["spread_q1"] * 100.0),
            "high": float(snapshot["spread_q3"] * 100.0),
            "regime": str(usa["damodaran_signal"]),
            "value_fmt": f"{usa['valuation_spread'] * 100.0:.2f} pts",
            "low_fmt": f"{snapshot['spread_q1'] * 100.0:.2f} pts",
            "high_fmt": f"{snapshot['spread_q3'] * 100.0:.2f} pts",
        }
    )

    mapping = mapping_dataframe()
    panel, _ = build_etf_panel(mapping)
    panel = add_expanding_regimes(panel)

    spy = current_etf_signal(panel, "USA", "SPY")
    vgk = current_etf_signal(panel, "Europe", "VGK")

    us_rows.append(
        {
            "label": "SPY Div-Yield",
            "current": float(spy["trailing_12m_dy_pct"]),
            "low": float(spy["q1_pct"]),
            "high": float(spy["q3_pct"]),
            "regime": str(spy["etf_regime"]),
            "value_fmt": f"{spy['trailing_12m_dy_pct']:.2f}%",
            "low_fmt": f"{spy['q1_pct']:.2f}%",
            "high_fmt": f"{spy['q3_pct']:.2f}%",
        }
    )

    eu_rows = [
        {
            "label": "Europe EY-ERP",
            "current": float(europe["valuation_spread"] * 100.0),
            "low": float(snapshot["spread_q1"] * 100.0),
            "high": float(snapshot["spread_q3"] * 100.0),
            "regime": str(europe["damodaran_signal"]),
            "value_fmt": f"{europe['valuation_spread'] * 100.0:.2f} pts",
            "low_fmt": f"{snapshot['spread_q1'] * 100.0:.2f} pts",
            "high_fmt": f"{snapshot['spread_q3'] * 100.0:.2f} pts",
        },
        {
            "label": "VGK Div-Yield",
            "current": float(vgk["trailing_12m_dy_pct"]),
            "low": float(vgk["q1_pct"]),
            "high": float(vgk["q3_pct"]),
            "regime": str(vgk["etf_regime"]),
            "value_fmt": f"{vgk['trailing_12m_dy_pct']:.2f}%",
            "low_fmt": f"{vgk['q1_pct']:.2f}%",
            "high_fmt": f"{vgk['q3_pct']:.2f}%",
        },
    ]

    return {"USA": us_rows, "Europe": eu_rows}


def position_on_gauge(current: float, low: float, high: float) -> float:
    if current <= low:
        if low == 0:
            return 16.5
        severity = min((low - current) / max(abs(low), 1e-9), 1.0)
        return max(4.0, 33.0 - 25.0 * severity)
    if current >= high:
        severity = min((current - high) / max(abs(high), 1e-9), 1.0)
        return min(96.0, 66.0 + 25.0 * severity)
    return 33.0 + 33.0 * (current - low) / max(high - low, 1e-9)


def regime_text(regime: str) -> str:
    mapping = {
        "low": "teuer",
        "middle": "neutral",
        "high": "guenstig",
        "underweight": "teuer",
        "neutral": "neutral",
        "overweight": "guenstig",
    }
    return mapping.get(regime, regime)


def draw_panel(ax: plt.Axes, title: str, rows: list[dict[str, object]]) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0, len(rows) - 0.05, title, fontsize=17, fontweight="bold", color=TEXT_DARK, va="top")
    ax.text(
        0,
        len(rows) - 0.45,
        "Links = teuer | Mitte = neutral | Rechts = guenstig",
        fontsize=10.5,
        color=TEXT_MUTED,
        va="top",
    )

    for idx, row in enumerate(rows):
        y = len(rows) - idx - 1.15

        ax.barh(y, 33, left=0, height=0.32, color=ZONE_RED, edgecolor="none")
        ax.barh(y, 33, left=33, height=0.32, color=ZONE_GRAY, edgecolor="none")
        ax.barh(y, 34, left=66, height=0.32, color=ZONE_GREEN, edgecolor="none")

        pos = position_on_gauge(float(row["current"]), float(row["low"]), float(row["high"]))
        ax.scatter([pos], [y], s=90, color="#111111", zorder=5)
        ax.scatter([pos], [y], s=26, color="white", zorder=6)

        ax.text(0, y + 0.38, str(row["label"]), fontsize=11.5, fontweight="bold", color=TEXT_DARK, va="bottom")
        ax.text(
            0,
            y - 0.42,
            (
                f"Aktuell {row['value_fmt']} | Teuer unter {row['low_fmt']} | "
                f"Guenstig ueber {row['high_fmt']} | Regime: {regime_text(str(row['regime']))}"
            ),
            fontsize=10.2,
            color=TEXT_MUTED,
            va="top",
        )


def write_markdown(data: dict[str, list[dict[str, object]]]) -> None:
    lines = [
        "# USA / Europe Regime Zones",
        "",
        "Die Grafik zeigt je Indikator:",
        "- schwarzer Marker = aktueller Stand",
        "- roter Bereich = teures Regime",
        "- grauer Bereich = neutral",
        "- gruener Bereich = guenstiges Regime",
        "",
    ]

    for market in ("USA", "Europe"):
        lines.append(f"## {market}")
        for row in data[market]:
            lines.append(
                f"- {row['label']}: aktuell {row['value_fmt']}, "
                f"teuer < {row['low_fmt']}, guenstig > {row['high_fmt']}, "
                f"Regime {regime_text(str(row['regime']))}"
            )
        lines.append("")

    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    data = build_inputs()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        "USA und Europe: Aktueller Stand vs. Bewertungsregime",
        fontsize=20,
        fontweight="bold",
        color=TEXT_DARK,
        y=1.02,
    )

    draw_panel(axes[0], "USA", data["USA"])
    draw_panel(axes[1], "Europe", data["Europe"])

    legend_handles = [
        Patch(color=ZONE_RED, label="Teures Regime"),
        Patch(color=ZONE_GRAY, label="Neutral"),
        Patch(color=ZONE_GREEN, label="Guenstiges Regime"),
    ]
    axes[1].legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=10,
    )

    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    write_markdown(data)

    print("Regime zone chart created.")
    print(str(OUTPUT_PNG))
    print(str(OUTPUT_MD))


if __name__ == "__main__":
    main()
