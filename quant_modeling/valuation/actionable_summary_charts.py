from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from market_signal_suite import build_suite


OUTPUT_DIR = Path(__file__).with_name("research_outputs")

SIGNAL_DE = {
    "underweight": "Untergewichten",
    "neutral": "Neutral",
    "overweight": "Uebergewichten",
    "expensive": "Teuer",
    "cheap": "Guenstig",
    "low": "Teueres Regime",
    "middle": "Mittelbereich",
    "high": "Guenstiges Regime",
}

SIGNAL_COLORS = {
    "underweight": "#c0392b",
    "neutral": "#7f8c8d",
    "overweight": "#1e8449",
}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def signal_color(signal: str) -> str:
    return SIGNAL_COLORS.get(signal, "#5d6d7e")


def as_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def as_mult(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}x"


def relative_preference(overview: pd.DataFrame) -> str:
    usa = overview[overview["market"].eq("USA")].iloc[0]
    eur = overview[overview["market"].eq("Europe")].iloc[0]
    if eur["damodaran_spread_pct"] > usa["damodaran_spread_pct"]:
        return "Europe vor USA im Relativvergleich"
    if eur["damodaran_spread_pct"] < usa["damodaran_spread_pct"]:
        return "USA vor Europe im Relativvergleich"
    return "USA und Europe sind fundamental aehnlich"


def plot_action_dashboard(overview: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(
        0.5,
        0.95,
        "Taktische Handlungsanweisung: Hauptmaerkte",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.90,
        relative_preference(overview) + ", aber beide derzeit defensiv",
        ha="center",
        va="top",
        fontsize=12,
        color="#34495e",
    )

    x_positions = [0.06, 0.54]
    card_width = 0.40
    card_height = 0.68

    for x, row in zip(x_positions, overview.itertuples(index=False)):
        border = signal_color(str(row.combined_signal))
        face = "#f7f9f9"
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.16),
                card_width,
                card_height,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=2.5,
                edgecolor=border,
                facecolor=face,
            )
        )

        ax.text(x + 0.03, 0.79, str(row.market), fontsize=18, fontweight="bold", color="#1f2d3d")
        ax.text(x + 0.03, 0.74, f"ETF Proxy: {row.proxy_ticker}", fontsize=11, color="#566573")

        badge = FancyBboxPatch(
            (x + 0.03, 0.66),
            0.24,
            0.06,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0,
            facecolor=border,
        )
        ax.add_patch(badge)
        ax.text(
            x + 0.15,
            0.69,
            SIGNAL_DE[str(row.combined_signal)],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )

        y = 0.61
        line_gap = 0.055
        details = [
            f"Forward P/E: {as_mult(row.damodaran_forward_pe)}",
            (
                "EY minus ERP: "
                f"{as_pct(row.damodaran_spread_pct)} ({SIGNAL_DE[str(row.damodaran_signal)]})"
            ),
            (
                "ETF Div-Yield: "
                f"{as_pct(row.etf_trailing_12m_dy_pct)} ({SIGNAL_DE[str(row.etf_regime)]})"
            ),
            f"ETF 12M Regime-Mittel: {as_pct(row.etf_expected_12m_total_return_pct)}",
        ]
        if pd.notna(row.us_shiller_composite_10y_real_pct):
            details.extend(
                [
                    (
                        "Shiller: "
                        f"{SIGNAL_DE[str(row.us_shiller_state)]}, "
                        f"{SIGNAL_DE[str(row.us_shiller_composite_signal)]}"
                    ),
                    f"Shiller 10Y real: {as_pct(row.us_shiller_composite_10y_real_pct)} p.a.",
                ]
            )
        else:
            details.append("Shiller: nicht direkt verfuegbar fuer Europe")

        for text in details:
            ax.text(x + 0.03, y, text, fontsize=11.5, color="#17202a")
            y -= line_gap

        if str(row.market) == "USA":
            takeaway = "Taktisch defensiv: teuer auf Shiller und ETF-Yield."
        else:
            takeaway = "Relativ attraktiver als USA, aber noch kein klares Kaufsignal."
        ax.text(x + 0.03, 0.22, takeaway, fontsize=11.5, color=border, fontweight="bold")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_main_market_gap(overview: pd.DataFrame, output_path: Path) -> None:
    labels = overview["market"].tolist()
    ey = overview["damodaran_forward_earnings_yield_pct"].to_numpy(dtype=float)
    erp = overview["damodaran_total_erp_pct"].to_numpy(dtype=float)
    spread = overview["damodaran_spread_pct"].to_numpy(dtype=float)

    x = np.arange(len(labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.bar(x - width / 2, ey, width=width, color="#1f618d", label="Forward Earnings Yield")
    ax.bar(x + width / 2, erp, width=width, color="#7d3c98", label="Total ERP")

    for i, label in enumerate(labels):
        ax.plot([x[i] - width / 2, x[i] + width / 2], [ey[i], erp[i]], color="#2c3e50", linewidth=1.5)
        ax.text(
            x[i],
            max(ey[i], erp[i]) + 0.25,
            f"Spread {spread[i]:.2f} pts",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#1b2631",
        )
        ax.text(
            x[i],
            0.35,
            SIGNAL_DE[str(overview.iloc[i]["combined_signal"])],
            ha="center",
            va="bottom",
            fontsize=11,
            color=signal_color(str(overview.iloc[i]["combined_signal"])),
            fontweight="bold",
        )

    ax.set_title("Bewertung vs. geforderte Rendite", fontsize=18, fontweight="bold")
    ax.set_ylabel("Prozent")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.5,
        -0.14,
        "Interpretation: Europe ist fundamental guenstiger als USA, aber beide bleiben taktisch defensiv.",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#34495e",
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_europe_internal_actions(mapped: pd.DataFrame, output_path: Path) -> None:
    focus = [
        "United Kingdom",
        "Germany",
        "France",
        "Spain",
        "Italy",
        "Switzerland",
        "Netherlands",
        "Sweden",
        "Austria",
    ]
    europe = mapped[
        mapped["market_label"].isin(focus)
        & mapped["valuation_spread"].notna()
    ].copy()
    europe = europe.sort_values("valuation_spread", ascending=True)
    strongest = europe.iloc[-2:]["market_label"].tolist()[::-1]
    weakest = europe.iloc[0]["market_label"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = [signal_color(str(item)) for item in europe["damodaran_signal"]]
    ax.barh(europe["market_label"], europe["valuation_spread"] * 100.0, color=colors)
    ax.axvline(0.0, color="#34495e", linewidth=1.0)

    for row in europe.itertuples(index=False):
        ax.text(
            row.valuation_spread * 100.0 + (0.12 if row.valuation_spread >= 0 else -0.12),
            row.market_label,
            SIGNAL_DE[str(row.damodaran_signal)],
            va="center",
            ha="left" if row.valuation_spread >= 0 else "right",
            fontsize=10,
            color="#1b2631",
        )

    ax.set_title("Europe intern: ETF-Allokation nach Bewertung", fontsize=18, fontweight="bold")
    ax.set_xlabel("Forward Earnings Yield minus ERP (Prozentpunkte)")
    ax.grid(axis="x", alpha=0.25)
    ax.text(
        0.5,
        -0.12,
        (
            "Praktisch: innerhalb Europe "
            f"{' und '.join(str(item) for item in strongest)} bevorzugen, "
            f"{weakest} tiefer gewichten."
        ),
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        color="#34495e",
    )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_notes(overview: pd.DataFrame, output_path: Path, chart_paths: dict[str, Path]) -> None:
    usa = overview[overview["market"].eq("USA")].iloc[0]
    eur = overview[overview["market"].eq("Europe")].iloc[0]

    lines = [
        "# Actionable Summary Charts",
        "",
        "## Klare Handlungsanweisung",
        f"- USA: {SIGNAL_DE[str(usa['combined_signal'])]}. Shiller ist teuer und der ETF-Yield liegt im teuren Regime.",
        f"- Europe: {SIGNAL_DE[str(eur['combined_signal'])]}. Fundamental guenstiger als USA, aber der ETF-Yield ist noch nicht im Kaufbereich.",
        "- Relative Reihenfolge: Europe vor USA.",
        "- Taktische Konsequenz: Kein aggressives Equity-Overweight in den Hauptmaerkten; Risiko nur selektiv erhoehen.",
        "",
        "## Charts",
        f"- Dashboard: {chart_paths['dashboard']}",
        f"- Value Gap: {chart_paths['gap']}",
        f"- Europe intern: {chart_paths['europe']}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    payload = build_suite(save_outputs=True)
    overview = payload["main_market_overview"].copy()
    mapped = payload["mapped_current"].copy()

    dashboard_path = OUTPUT_DIR / "action_dashboard_main_markets.png"
    gap_path = OUTPUT_DIR / "main_markets_value_gap.png"
    europe_path = OUTPUT_DIR / "europe_internal_actions.png"
    notes_path = OUTPUT_DIR / "actionable_chart_notes.md"

    plot_action_dashboard(overview, dashboard_path)
    plot_main_market_gap(overview, gap_path)
    plot_europe_internal_actions(mapped, europe_path)
    write_notes(
        overview,
        notes_path,
        {
            "dashboard": dashboard_path,
            "gap": gap_path,
            "europe": europe_path,
        },
    )

    print("Actionable charts created.")
    print(str(dashboard_path))
    print(str(gap_path))
    print(str(europe_path))
    print(str(notes_path))


if __name__ == "__main__":
    main()
