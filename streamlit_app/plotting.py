from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .utils import ASSET_COLUMNS, ARCHITECTURE_ORDER, SWEEP_CHART_ASSETS

ASSET_COLORS = {
    "US": "#1f77b4",
    "Europe": "#ff7f0e",
    "EM": "#2ca02c",
    "Japan": "#7f7f7f",
}
ARCHITECTURE_TITLES = {
    "A": "Architecture A",
    "B": "Architecture B",
    "C": "Architecture C",
}


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
        height=360,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="white",
        plot_bgcolor="white",
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
                textinfo="label+percent",
            )
        ]
    )
    figure.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        showlegend=False,
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
        return _empty_figure(f"No feasible sweep points for Architecture {architecture}.")

    subset = subset.sort_values("mu_europe", ascending=False)
    figure = go.Figure()
    for asset in SWEEP_CHART_ASSETS:
        figure.add_trace(
            go.Scatter(
                x=subset["mu_europe"],
                y=subset[f"w_{asset}"],
                mode="lines",
                name=asset,
                line={"width": 3, "color": ASSET_COLORS[asset]},
                hovertemplate=f"{asset}<br>Europe ER=%{{x:.2%}}<br>Weight=%{{y:.2%}}<extra></extra>",
            )
        )

    selected_subset = pd.DataFrame()
    if selected_mu is not None:
        selected_subset = subset[pd.Series(pd.to_numeric(subset["mu_europe"]), copy=False).sub(selected_mu).abs() < 1e-10]
        figure.add_vline(
            x=selected_mu,
            line_width=1.5,
            line_dash="dash",
            line_color="#111827",
            opacity=0.6,
        )
        figure.add_annotation(
            x=selected_mu,
            y=1.12,
            xref="x",
            yref="paper",
            text="Selected ER",
            showarrow=False,
            font={"size": 11, "color": "#334155"},
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
                            "size": 10,
                            "color": ASSET_COLORS[asset],
                            "line": {"color": "white", "width": 1.5},
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
                "<br><span style='font-size:12px;color:#64748b'>"
                "Weight response as Europe expected return moves across the sweep"
                "</span>"
            )
        },
        height=360,
        margin={"l": 10, "r": 10, "t": 50, "b": 70},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
        },
        hovermode="x unified",
    )
    figure.update_xaxes(
        title="Europe expected return",
        tickformat=".1%",
        autorange="reversed",
        showgrid=True,
        gridcolor="#E5E7EB",
    )
    figure.update_yaxes(
        title="Portfolio weight",
        tickformat=".0%",
        showgrid=True,
        gridcolor="#E5E7EB",
    )
    return figure


def sort_architecture_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        "architecture",
        key=lambda series: series.map({architecture: idx for idx, architecture in enumerate(ARCHITECTURE_ORDER)}),
    ).reset_index(drop=True)
