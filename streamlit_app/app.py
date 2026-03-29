from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from streamlit_app.load_data import StreamlitDataError, load_data_manifest, load_precomputed_tables
from streamlit_app.plotting import make_portfolio_pie_chart, make_weight_sweep_figure, sort_architecture_rows
from streamlit_app.utils import (
    ARCHITECTURE_ORDER,
    ARCHITECTURE_NAMES,
    ASSET_COLUMNS,
    GITHUB_REPO_URL,
    SUBSTACK_ARTICLE_URL,
    filter_parameter_slice,
    format_pct,
    format_status,
    get_sidebar_options,
    make_comparison_table,
    percent_display,
)

st.set_page_config(
    page_title="Portfolio Optimization Architectures",
    layout="wide",
)

ARCHITECTURE_EXPLAINERS = {
    "A": "Single-stage optimization with benchmark-relative portfolio CVaR and turnover constraints.",
    "B": "Adds an explicit TE-CVaR constraint to the stronger single-stage optimization setup.",
    "C": "Builds on the risk-constrained setup with a second-stage lexicographic step that stays close to the benchmark while preserving near-optimality.",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        .qs-link-row a {
            font-weight: 600;
            text-decoration: none;
        }
        .qs-link-row {
            margin-bottom: 0.8rem;
        }
        .qs-takeaway-box {
            background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #bfdbfe;
            border-radius: 18px;
            padding: 1rem 1.15rem;
            margin: 1rem 0 1.35rem 0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }
        .qs-takeaway-title {
            color: #1d4ed8;
            font-size: 0.84rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .qs-takeaway-body {
            color: #0f172a;
            font-size: 1rem;
            line-height: 1.55;
            margin: 0;
        }
        .qs-section-divider {
            border-top: 1px solid #e2e8f0;
            margin: 1.35rem 0 0.9rem 0;
        }
        .qs-section-lead {
            color: #475569;
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0.15rem 0 0.95rem 0;
            max-width: 68rem;
        }
        .qs-status-pill {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.55rem;
        }
        .qs-status-feasible {
            background: #e8f7ee;
            color: #166534;
        }
        .qs-status-infeasible {
            background: #fff4e5;
            color: #9a3412;
        }
        .qs-status-unknown {
            background: #eef2ff;
            color: #4338ca;
        }
        .qs-arch-note {
            color: #475569;
            font-size: 0.95rem;
            line-height: 1.45;
            margin-bottom: 0.55rem;
        }
        .qs-metric-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.55rem 0.8rem;
            margin-top: 0.85rem;
        }
        .qs-metric-item {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
        }
        .qs-metric-item-secondary {
            background: #ffffff;
        }
        .qs-metric-label {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.02em;
            margin-bottom: 0.18rem;
        }
        .qs-metric-value {
            color: #111827;
            font-size: 1.12rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .qs-table-caption {
            color: #64748b;
            font-size: 0.9rem;
            margin-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _status_badge(status: str) -> str:
    status_label = format_status(status)
    if status_label == "Feasible":
        css_class = "qs-status-feasible"
    elif status_label == "Infeasible":
        css_class = "qs-status-infeasible"
    else:
        css_class = "qs-status-unknown"
    return (
        f"<span class='qs-status-pill {css_class}'>"
        f"{status_label}"
        "</span>"
    )


def _render_metric_block(row) -> None:
    metric_items = [
        ("Expected Return", format_pct(row["expected_return"]), True),
        ("Portfolio CVaR", format_pct(row["portfolio_cvar"]), True),
        ("TE-CVaR", format_pct(row["te_cvar"]), True),
        ("Benchmark Distance", format_pct(row["bm_dist_l2"]), True),
        ("Turnover", format_pct(row["turnover"]), True),
        ("Herfindahl", f"{row['herfindahl']:.3f}" if row["herfindahl"] == row["herfindahl"] else "N/A", False),
    ]
    st.markdown(
        "".join(
            [
                "<div class='qs-metric-grid'>",
                *[
                    (
                        f"<div class='qs-metric-item{' qs-metric-item-secondary' if not is_primary else ''}'>"
                        f"<div class='qs-metric-label'>{label}</div>"
                        f"<div class='qs-metric-value'>{value}</div>"
                        "</div>"
                    )
                    for label, value, is_primary in metric_items
                ],
                "</div>",
            ]
        ),
        unsafe_allow_html=True,
    )


def _render_takeaway_box() -> None:
    st.markdown(
        """
        <div class="qs-takeaway-box">
            <div class="qs-takeaway-title">Key Takeaway</div>
            <p class="qs-takeaway-body">
                Standard Optimization is the most fragile architecture. TE-CVaR constraints improve
                portfolio stability, but the result remains sensitive to the TE-CVaR limit.
                Lexicographic Risk-Constrained Optimization provides the most stable and
                benchmark-consistent final portfolio selection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _build_asset_reference_table(
    manifest: dict[str, object],
    *,
    selected_mu: float,
) -> pd.DataFrame | None:
    asset_reference = manifest.get("asset_reference")
    if not isinstance(asset_reference, list):
        return None

    base_mu_lookup = (
        manifest.get("base_config", {}).get("mu_base", {})
        if isinstance(manifest.get("base_config"), dict)
        else {}
    )
    reference_lookup = {
        str(row.get("asset")): row
        for row in asset_reference
        if isinstance(row, dict) and row.get("asset") is not None
    }

    rows: list[dict[str, str]] = []
    for asset in ASSET_COLUMNS:
        row = reference_lookup.get(asset)
        if row is None:
            continue

        expected_return = selected_mu if asset == "Europe" else row.get("expected_return_base")
        if expected_return is None:
            expected_return = base_mu_lookup.get(asset)

        rows.append(
            {
                "Name of Asset": asset,
                "Expected Return": format_pct(expected_return),
                "Volatility": format_pct(row.get("volatility")),
            }
        )

    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    _inject_styles()
    st.title("Interactive Portfolio Optimization Architectures")
    st.markdown(
        "Explore how three portfolio construction architectures respond to changing Europe "
        "expected returns, benchmark-relative portfolio CVaR budgets, TE-CVaR caps, and "
        "turnover limits. The app reads a precomputed results library from the "
        "`Optimization_Lexicographic_2` research stack, so every interaction is instant and reproducible."
    )
    st.caption(
        "Standard Optimization is the lean baseline, Risk-Constrained Optimization adds active-risk "
        "discipline, and Lexicographic Risk-Constrained Optimization adds a second stage that stays closer "
        "to the benchmark without materially giving up the stage-one optimum."
    )
    st.markdown(
        f"<div class='qs-link-row'><a href='{SUBSTACK_ARTICLE_URL}'>Substack article</a>  |  "
        f"<a href='{GITHUB_REPO_URL}'>GitHub repository</a></div>",
        unsafe_allow_html=True,
    )
    _render_takeaway_box()

    try:
        tables = load_precomputed_tables()
        manifest = load_data_manifest()
    except StreamlitDataError as exc:
        st.error(str(exc))
        st.info(
            "Generate the dataset first with `python articles/Optimization_Lexicographic_2/build_streamlit_dataset.py` "
            "and then rerun `streamlit run streamlit_app/app.py`."
        )
        st.stop()

    sweep_results = tables["sweep_results"]
    portfolio_snapshots = tables["portfolio_snapshots"]
    summary_metrics = tables["summary_metrics"]
    options = get_sidebar_options(sweep_results)

    with st.sidebar:
        st.header("Parameters")
        selected_mu = st.select_slider(
            "Europe expected return",
            options=options["mu_europe"],
            value=0.065 if 0.065 in options["mu_europe"] else float(options["mu_europe"][len(options["mu_europe"]) // 2]),
            format_func=lambda value: percent_display(value, 2),
            help="Sets the exact portfolio snapshot used in the cards and the highlighted marker in the sweep charts.",
        )
        selected_risk_factor = st.select_slider(
            "Portfolio risk budget factor",
            options=options["portfolio_risk_factor"],
            value=1.05 if 1.05 in options["portfolio_risk_factor"] else options["portfolio_risk_factor"][0],
            help="Benchmark-relative portfolio CVaR budget factor. 1.00 is approximately the benchmark demeaned portfolio risk level.",
        )
        selected_te_limit = st.select_slider(
            "TE-CVaR downside-risk limit",
            options=options["te_cvar_limit"],
            value=0.02 if 0.02 in options["te_cvar_limit"] else options["te_cvar_limit"][0],
            format_func=lambda value: percent_display(value, 0),
            help="Controls the allowed level of benchmark-relative downside risk in the tail.",
        )
        selected_turnover_limit = st.select_slider(
            "Turnover / reallocation limit",
            options=options["turnover_limit"],
            value=0.30 if 0.30 in options["turnover_limit"] else options["turnover_limit"][0],
            format_func=lambda value: percent_display(value, 0),
            help="Limits portfolio reallocation intensity relative to the benchmark starting point.",
        )
        if manifest:
            st.caption(
                f"Precomputed rows: {manifest.get('row_count', 'N/A')} | "
                f"Built: {manifest.get('built_at_utc', 'N/A')}"
            )
            asset_reference_table = _build_asset_reference_table(
                manifest,
                selected_mu=selected_mu,
            )
            if asset_reference_table is None:
                st.caption("Asset reference inputs are unavailable in this dataset manifest.")
            else:
                st.markdown("**Asset Reference**")
                st.table(asset_reference_table)

    selected_snapshots = filter_parameter_slice(
        portfolio_snapshots,
        mu_europe=selected_mu,
        portfolio_risk_factor=selected_risk_factor,
        te_cvar_limit=selected_te_limit,
        turnover_limit=selected_turnover_limit,
    )
    selected_summary = filter_parameter_slice(
        summary_metrics,
        mu_europe=selected_mu,
        portfolio_risk_factor=selected_risk_factor,
        te_cvar_limit=selected_te_limit,
        turnover_limit=selected_turnover_limit,
    )
    filtered_sweep = filter_parameter_slice(
        sweep_results,
        portfolio_risk_factor=selected_risk_factor,
        te_cvar_limit=selected_te_limit,
        turnover_limit=selected_turnover_limit,
    )

    if selected_snapshots.empty:
        st.warning(
            "No precomputed portfolio snapshots matched the selected parameter combination. "
            "Try another combination or rebuild the dataset."
        )
    else:
        feasible_count = int(selected_snapshots["is_feasible"].sum())
        total_count = int(len(selected_snapshots))
        infeasible_architectures = selected_snapshots.loc[
            ~selected_snapshots["is_feasible"], "architecture"
        ].map(lambda value: ARCHITECTURE_NAMES.get(str(value), str(value))).tolist()
        if feasible_count == 0:
            st.warning(
                "This selected point is infeasible for all three optimization designs in the precomputed library. "
                "Try a higher risk factor or looser TE-CVaR / turnover budgets."
            )
        elif infeasible_architectures:
            st.info(
                "Some optimization designs are infeasible for this selected point: "
                + ", ".join(infeasible_architectures)
            )
        elif feasible_count == total_count:
            st.success("All three optimization designs are feasible for the selected parameter point.")

    st.markdown("<div class='qs-section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Selected Portfolio Comparison")
    st.markdown(
        "<div class='qs-section-lead'>"
        "Each card shows the final portfolio for the selected parameter point. The benchmark-relative "
        "risk budget factor, TE-CVaR limit, and turnover limit stay fixed while the selected Europe "
        "expected return pins down the exact portfolio snapshot."
        "</div>",
        unsafe_allow_html=True,
    )
    portfolio_columns = st.columns(3)
    for column, architecture in zip(portfolio_columns, ARCHITECTURE_ORDER):
        with column:
            with st.container(border=True):
                st.markdown(f"### {ARCHITECTURE_NAMES.get(architecture, architecture)}")
                st.markdown(
                    f"<div class='qs-arch-note'>{ARCHITECTURE_EXPLAINERS[architecture]}</div>",
                    unsafe_allow_html=True,
                )
                architecture_slice = selected_snapshots[selected_snapshots["architecture"] == architecture]
                if architecture_slice.empty:
                    st.warning("No precomputed result available.")
                    continue

                row = architecture_slice.iloc[0]
                st.markdown(_status_badge(row["status"]), unsafe_allow_html=True)
                if not bool(row["is_feasible"]):
                    st.caption(
                        "No feasible portfolio exists for this exact point in the precomputed library."
                    )
                    continue

                st.plotly_chart(make_portfolio_pie_chart(row), use_container_width=True)
                _render_metric_block(row)

    st.markdown("<div class='qs-section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Weight Sweep Across Europe Expected Return")
    st.markdown(
        "<div class='qs-section-lead'>"
        "These charts show how each architecture translates changes in Europe’s expected return into "
        "portfolio weights. A smoother response indicates greater robustness and more benchmark-consistent "
        "portfolio selection. The dashed marker highlights the currently selected Europe expected return."
        "</div>",
        unsafe_allow_html=True,
    )
    sweep_columns = st.columns(3)
    for column, architecture in zip(sweep_columns, ARCHITECTURE_ORDER):
        with column:
            st.plotly_chart(
                make_weight_sweep_figure(
                    filtered_sweep,
                    architecture=architecture,
                    selected_mu=selected_mu,
                ),
                use_container_width=True,
            )

    st.markdown("<div class='qs-section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Optimization Comparison Table")
    if selected_summary.empty:
        st.warning("No summary metrics are available for this selected parameter combination.")
    else:
        table = make_comparison_table(sort_architecture_rows(selected_summary))
        st.markdown(
            "<div class='qs-table-caption'>"
            "This table summarizes the selected point with consistently formatted percentages and compact "
            "stability diagnostics."
            "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=245,
            column_config={
                "Architecture": st.column_config.TextColumn("Architecture", width="large"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Expected Return": st.column_config.TextColumn("Expected Return", width="small"),
                "Portfolio CVaR": st.column_config.TextColumn("Portfolio CVaR", width="small"),
                "TE-CVaR": st.column_config.TextColumn("TE-CVaR", width="small"),
                "Benchmark Dist.": st.column_config.TextColumn("Benchmark Dist.", width="small"),
                "Turnover": st.column_config.TextColumn("Turnover", width="small"),
                "Herfindahl": st.column_config.TextColumn("Herfindahl", width="small"),
                "Effective N": st.column_config.TextColumn("Effective N", width="small"),
                "Instability": st.column_config.TextColumn("Instability", width="small"),
                "Max Jump": st.column_config.TextColumn("Max Jump", width="small"),
            },
        )


if __name__ == "__main__":
    main()
