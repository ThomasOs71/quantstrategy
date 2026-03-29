from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .utils import ARCHITECTURE_NAMES, ASSET_COLUMNS, ARCHITECTURE_ORDER, SWEEP_CHART_ASSETS

ASSET_COLORS = {
    "US": "#1f77b4",
    "Europe": "#ff7f0e",
    "EM": "#2ca02c",
    "Japan": "#7f7f7f",
}
ARCHITECTURE_TITLES = ARCHITECTURE_NAMES


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        height=420,
        margin={"l": 12, "r": 12, "t": 36, "b": 18},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 13, "color": "#0f172a"},
    )
    return figure


def make_portfolio_pie_chart(row: pd.Series | None) -> go.Figure:
    if row is None:
        return _empty_figure("No precomputed portfolio available.")

    values = [float(row[f"w_{asset}"]) for asset in ASSET_COLUMNS if pd.notna(row[f"w_{asset}"])]
    labels = [asset for asset in ASSET_COLUMNS if pd.notna(row[f"w_{asset}"])]
    if not values:
        return _empty_figure("Portfolio weights unavailable for this combination.")

    figure = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                sort=False,
                marker={"colors": [ASSET_COLORS[asset] for asset in labels]},
                texttemplate="%{label}<br>%{percent}",
                textposition="inside",
                insidetextfont={"size": 13},
                hovertemplate="%{label}<br>Weight=%{percent}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        height=390,
        margin={"l": 8, "r": 8, "t": 26, "b": 8},
        showlegend=False,
        uniformtext_minsize=11,
        uniformtext_mode="hide",
        paper_bgcolor="white",
        font={"size": 13, "color": "#0f172a"},
    )
    return figure


def make_weight_sweep_figure(
    sweep_results: pd.DataFrame,
    *,
    architecture: str,
    selected_mu: float | None = None,
) -> go.Figure:
    subset = sweep_results[
        (sweep_results["architecture"] == architecture) & (sweep_results["is_feasible"])
    ].copy()
    if subset.empty:
        architecture_name = ARCHITECTURE_TITLES.get(architecture, architecture)
        return _empty_figure(f"No feasible sweep points for {architecture_name}.")

    subset = subset.sort_values("mu_europe", ascending=False)
    figure = go.Figure()
    for asset in SWEEP_CHART_ASSETS:
        figure.add_trace(
            go.Scatter(
                x=subset["mu_europe"],
                y=subset[f"w_{asset}"],
                mode="lines",
                name=asset,
                line={"width": 4, "color": ASSET_COLORS[asset]},
                hovertemplate=f"{asset}<br>Europe ER=%{{x:.2%}}<br>Weight=%{{y:.2%}}<extra></extra>",
            )
        )

    selected_subset = pd.DataFrame()
    if selected_mu is not None:
        selected_subset = subset[pd.Series(pd.to_numeric(subset["mu_europe"]), copy=False).sub(selected_mu).abs() < 1e-10]
        figure.add_vline(
            x=selected_mu,
            line_width=2.4,
            line_dash="dash",
            line_color="#0f172a",
            opacity=0.9,
        )
        figure.add_annotation(
            x=selected_mu,
            y=1.11,
            xref="x",
            yref="paper",
            text="Selected ER",
            showarrow=False,
            font={"size": 11, "color": "#0f172a"},
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=4,
        )
        if not selected_subset.empty:
            for asset in SWEEP_CHART_ASSETS:
                figure.add_trace(
                    go.Scatter(
                        x=selected_subset["mu_europe"],
                        y=selected_subset[f"w_{asset}"],
                        mode="markers",
                        showlegend=False,
                        marker={
                            "size": 12,
                            "color": ASSET_COLORS[asset],
                            "line": {"color": "#0f172a", "width": 1.4},
                        },
                        hovertemplate=(
                            f"{asset}<br>Selected Europe ER=%{{x:.2%}}<br>"
                            "Weight=%{y:.2%}<extra></extra>"
                        ),
                    )
                )

    figure.update_layout(
        title={
            "text": (
                f"{ARCHITECTURE_TITLES.get(architecture, architecture)}"
                "<br><span style='font-size:13px;color:#64748b'>"
                "Weight response as Europe expected return moves across the sweep"
                "</span>"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        height=500,
        margin={"l": 8, "r": 6, "t": 66, "b": 82},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.2,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 12},
        },
        hovermode="x unified",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 13, "color": "#0f172a"},
        hoverlabel={"font_size": 12},
    )
    figure.update_xaxes(
        title="Europe expected return",
        tickformat=".1%",
        autorange="reversed",
        showgrid=True,
        gridcolor="#E5E7EB",
        title_font={"size": 14},
        tickfont={"size": 12},
        zeroline=False,
    )
    figure.update_yaxes(
        title="Portfolio weight",
        tickformat=".0%",
        showgrid=True,
        gridcolor="#E5E7EB",
        title_font={"size": 14},
        tickfont={"size": 12},
        zeroline=False,
    )
    return figure


def sort_architecture_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        "architecture",
        key=lambda series: series.map({architecture: idx for idx, architecture in enumerate(ARCHITECTURE_ORDER)}),
    ).reset_index(drop=True)
