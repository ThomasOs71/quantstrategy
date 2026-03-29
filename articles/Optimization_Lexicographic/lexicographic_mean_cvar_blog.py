import argparse
from dataclasses import dataclass
from pathlib import Path
import warnings

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "input" / "mc_scenarios.xlsx"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_CASE_RESULTS_CSV = DEFAULT_OUTPUT_DIR / "lexicographic_results.csv"
DEFAULT_CASE_SUMMARY_XLSX = DEFAULT_OUTPUT_DIR / "lexicographic_case_summary.xlsx"
DEFAULT_CASE_CHART = DEFAULT_OUTPUT_DIR / "case_weight_changes.png"
DEFAULT_SWEEP_RESULTS_CSV = DEFAULT_OUTPUT_DIR / "lexicographic_sweep_results.csv"
DEFAULT_SWEEP_SUMMARY_XLSX = DEFAULT_OUTPUT_DIR / "lexicographic_sweep_case_summary.xlsx"
DEFAULT_SWEEP_CHART = DEFAULT_OUTPUT_DIR / "sweep_weight_sensitivity.png"
DEFAULT_SWEEP_LINKEDIN_CHART = DEFAULT_OUTPUT_DIR / "sweep_weight_sensitivity_linkedin.png"


ASSETS = [
    "US Equities",
    "Europe Equities",
    "Emerging Market Equities",
    "Japan Equities",
]

SCENARIO_COLUMNS = [
    "MSCI USA",
    "MSCI Europe",
    "MSCI Emerging Markets",
    "MSCI Japan",
]

# Bounds follow the user specification (in portfolio weight units).
LOWER_BOUNDS = np.array([0.20, 0.20, 0.00, 0.00], dtype=float)
UPPER_BOUNDS = np.array([0.60, 0.60, 0.25, 0.20], dtype=float)
BENCHMARK = np.array([0.40, 0.40, 0.15, 0.05], dtype=float)
ASSET_COLORS = {
    "US Equities": "#1F3A5F",               # deep navy
    "Europe Equities": "#2E8B57",           # muted green
    "Emerging Market Equities": "#B77B2E",  # warm ochre
    "Japan Equities": "#7A4A6A",            # muted plum
}
PLOT_FONT_SIZES = {
    "title": 18,
    "axis_label": 14,
    "tick": 12,
    "legend": 12,
}


@dataclass(frozen=True)
class CaseDefinition:
    name: str
    expected_returns: np.ndarray
    use_lexicographic: bool


def empirical_cvar_from_returns(returns: np.ndarray, alpha: float) -> float:
    losses = -returns
    tail_count = max(1, int(np.ceil((1.0 - alpha) * losses.size)))
    worst_losses = np.partition(losses, -tail_count)[-tail_count:]
    return float(np.mean(worst_losses))


def solve_stage_one(
    scenario_returns: np.ndarray,
    expected_returns: np.ndarray,
    cvar_alpha: float,
    cvar_limit: float | None,
) -> dict:
    n_scenarios, n_assets = scenario_returns.shape

    w = cp.Variable(n_assets)

    constraints = [
        cp.sum(w) == 1.0,
        w >= LOWER_BOUNDS,
        w <= UPPER_BOUNDS,
    ]

    if cvar_limit is not None:
        t = cp.Variable()
        u = cp.Variable(n_scenarios)
        cvar_expr = t + (1.0 / ((1.0 - cvar_alpha) * n_scenarios)) * cp.sum(u)
        constraints += [
            u >= 0.0,
            u >= -scenario_returns @ w - t,
            cvar_expr <= cvar_limit,
        ]

    problem = cp.Problem(cp.Maximize(expected_returns @ w), constraints)

    solved = False
    last_error = None
    for solver, kwargs in [
        (cp.ECOS, {"abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10}),
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-5, "max_iters": 25000}),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(solver=solver, verbose=False, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solved = True
            break

    if not solved:
        if last_error is not None:
            raise RuntimeError(f"Stage 1 failed: {last_error}") from last_error
        raise RuntimeError(f"Stage 1 failed with status '{problem.status}'.")

    weights = np.asarray(w.value).reshape(-1)
    port_returns = scenario_returns @ weights
    realized_cvar = empirical_cvar_from_returns(port_returns, cvar_alpha)

    return {
        "weights": weights,
        "expected_return": float(expected_returns @ weights),
        "cvar": realized_cvar,
        "status": problem.status,
    }


def solve_stage_two_lexicographic(
    scenario_returns: np.ndarray,
    expected_returns: np.ndarray,
    cvar_alpha: float,
    cvar_limit: float | None,
    stage_one_utility: float,
    utility_floor_fraction: float,
) -> dict:
    n_scenarios, n_assets = scenario_returns.shape
    utility_floor = utility_floor_fraction * stage_one_utility

    w = cp.Variable(n_assets)

    constraints = [
        cp.sum(w) == 1.0,
        w >= LOWER_BOUNDS,
        w <= UPPER_BOUNDS,
        expected_returns @ w >= utility_floor,
    ]

    if cvar_limit is not None:
        t = cp.Variable()
        u = cp.Variable(n_scenarios)
        cvar_expr = t + (1.0 / ((1.0 - cvar_alpha) * n_scenarios)) * cp.sum(u)
        constraints += [
            u >= 0.0,
            u >= -scenario_returns @ w - t,
            cvar_expr <= cvar_limit,
        ]

    # Squared L2 has the same minimizer as plain L2 and yields a QP.
    objective = cp.Minimize(cp.sum_squares(w - BENCHMARK))
    problem = cp.Problem(objective, constraints)

    solved = False
    last_error = None
    for solver, kwargs in [
        (cp.OSQP, {"eps_abs": 1e-8, "eps_rel": 1e-8, "max_iter": 200000}),
        (cp.CLARABEL, {}),
        (cp.SCS, {"eps": 1e-5, "max_iters": 25000}),
    ]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                problem.solve(solver=solver, verbose=False, **kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            continue
        if problem.status in ("optimal", "optimal_inaccurate"):
            solved = True
            break

    if not solved:
        if last_error is not None:
            raise RuntimeError(f"Stage 2 failed: {last_error}") from last_error
        raise RuntimeError(f"Stage 2 failed with status '{problem.status}'.")

    weights = np.asarray(w.value).reshape(-1)
    port_returns = scenario_returns @ weights
    realized_cvar = empirical_cvar_from_returns(port_returns, cvar_alpha)

    return {
        "weights": weights,
        "expected_return": float(expected_returns @ weights),
        "cvar": realized_cvar,
        "l2_to_benchmark": float(np.linalg.norm(weights - BENCHMARK)),
        "utility_floor": float(utility_floor),
        "status": problem.status,
    }


def load_demeaned_scenarios(input_file: Path, sheet_name: str) -> np.ndarray:
    df = pd.read_excel(input_file, sheet_name=sheet_name)
    missing = [col for col in SCENARIO_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required scenario columns: {missing}")
    return df[SCENARIO_COLUMNS].to_numpy(dtype=float)


def run_cases(
    demeaned_scenarios: np.ndarray,
    cvar_alpha: float,
    cvar_limit: float | None,
    utility_floor_fraction: float,
) -> pd.DataFrame:
    cases = [
        CaseDefinition(
            name="Case 1 - Stage 1 only (US 7.0%, Europe 6.9%)",
            expected_returns=np.array([0.070, 0.069, 0.075, 0.055], dtype=float),
            use_lexicographic=False,
        ),
        CaseDefinition(
            name="Case 2 - Stage 1 only (US 6.9%, Europe 7.0%)",
            expected_returns=np.array([0.069, 0.070, 0.075, 0.055], dtype=float),
            use_lexicographic=False,
        ),
        CaseDefinition(
            name="Case 3 - Stage 1 + Stage 2 (US 6.9%, Europe 7.0%)",
            expected_returns=np.array([0.069, 0.070, 0.075, 0.055], dtype=float),
            use_lexicographic=True,
        ),
        CaseDefinition(
            name="Case 4 - Stage 1 + Stage 2 (US 7.0%, Europe 6.9%)",
            expected_returns=np.array([0.070, 0.069, 0.075, 0.055], dtype=float),
            use_lexicographic=True,
        ),
    ]

    rows = []
    stage_one_cache: dict[tuple[float, ...], dict] = {}

    for case in cases:
        mu = case.expected_returns
        scenario_returns = demeaned_scenarios + mu
        cache_key = tuple(np.round(mu, 8))

        if cache_key not in stage_one_cache:
            stage_one_cache[cache_key] = solve_stage_one(
                scenario_returns=scenario_returns,
                expected_returns=mu,
                cvar_alpha=cvar_alpha,
                cvar_limit=cvar_limit,
            )

        stage_one_result = stage_one_cache[cache_key]

        if not case.use_lexicographic:
            weights = stage_one_result["weights"]
            rows.append(
                {
                    "case": case.name,
                    "optimization": (
                        "Stage 1 (Mean-CVaR)"
                        if cvar_limit is not None
                        else "Stage 1 (Mean only)"
                    ),
                    "expected_return": stage_one_result["expected_return"],
                    "cvar": stage_one_result["cvar"],
                    "l2_to_benchmark": float(np.linalg.norm(weights - BENCHMARK)),
                    "utility_floor": np.nan,
                    "US Equities": weights[0],
                    "Europe Equities": weights[1],
                    "Emerging Market Equities": weights[2],
                    "Japan Equities": weights[3],
                }
            )
        else:
            stage_two_result = solve_stage_two_lexicographic(
                scenario_returns=scenario_returns,
                expected_returns=mu,
                cvar_alpha=cvar_alpha,
                cvar_limit=cvar_limit,
                stage_one_utility=stage_one_result["expected_return"],
                utility_floor_fraction=utility_floor_fraction,
            )
            weights = stage_two_result["weights"]
            rows.append(
                {
                    "case": case.name,
                    "optimization": (
                        "Stage 1 + Stage 2 (Lexicographic)"
                        if cvar_limit is not None
                        else "Stage 1 + Stage 2 (Lexicographic, no CVaR constraint)"
                    ),
                    "expected_return": stage_two_result["expected_return"],
                    "cvar": stage_two_result["cvar"],
                    "l2_to_benchmark": stage_two_result["l2_to_benchmark"],
                    "utility_floor": stage_two_result["utility_floor"],
                    "US Equities": weights[0],
                    "Europe Equities": weights[1],
                    "Emerging Market Equities": weights[2],
                    "Japan Equities": weights[3],
                }
            )

    return pd.DataFrame(rows)


def run_europe_return_sweep(
    demeaned_scenarios: np.ndarray,
    cvar_alpha: float,
    cvar_limit: float | None,
    utility_floor_fraction: float,
    us_return: float,
    europe_start: float,
    europe_end: float,
    europe_step: float,
    em_return: float,
    japan_return: float,
) -> pd.DataFrame:
    if europe_step <= 0.0:
        raise ValueError("europe_step must be positive.")
    if europe_start < europe_end:
        raise ValueError("europe_start must be >= europe_end for a downward sweep.")

    n_steps = int(round((europe_start - europe_end) / europe_step)) + 1
    europe_values = [europe_start - i * europe_step for i in range(n_steps)]
    europe_values = [max(europe_end, x) for x in europe_values]

    rows = []
    for idx, europe_return in enumerate(europe_values, start=1):
        print(
            f"Sweep step {idx}/{len(europe_values)}: Europe expected return = "
            f"{100.0 * europe_return:.2f}%"
        )

        mu = np.array(
            [us_return, europe_return, em_return, japan_return],
            dtype=float,
        )
        scenario_returns = demeaned_scenarios + mu

        stage_one_result = solve_stage_one(
            scenario_returns=scenario_returns,
            expected_returns=mu,
            cvar_alpha=cvar_alpha,
            cvar_limit=cvar_limit,
        )
        stage_two_result = solve_stage_two_lexicographic(
            scenario_returns=scenario_returns,
            expected_returns=mu,
            cvar_alpha=cvar_alpha,
            cvar_limit=cvar_limit,
            stage_one_utility=stage_one_result["expected_return"],
            utility_floor_fraction=utility_floor_fraction,
        )

        weights = stage_two_result["weights"]
        rows.append(
            {
                "case": f"Case {idx} - EU {100.0 * europe_return:.2f}%",
                "optimization": (
                    "Stage 1 + Stage 2 (Lexicographic)"
                    if cvar_limit is not None
                    else "Stage 1 + Stage 2 (Lexicographic, no CVaR constraint)"
                ),
                "expected_return": stage_two_result["expected_return"],
                "cvar": stage_two_result["cvar"],
                "l2_to_benchmark": stage_two_result["l2_to_benchmark"],
                "utility_floor": stage_two_result["utility_floor"],
                "US Equities": weights[0],
                "Europe Equities": weights[1],
                "Emerging Market Equities": weights[2],
                "Japan Equities": weights[3],
                "US Expected Return": us_return,
                "Europe Expected Return": europe_return,
                "EM Expected Return": em_return,
                "Japan Expected Return": japan_return,
            }
        )

    return pd.DataFrame(rows)


def build_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    case_numbers = results["case"].str.extract(r"Case\s+(\d+)")[0].astype(int)
    summary_data: dict[str, pd.Series] = {"Case": case_numbers}

    for col in [
        "US Expected Return",
        "Europe Expected Return",
        "EM Expected Return",
        "Japan Expected Return",
    ]:
        if col in results.columns:
            summary_data[col] = results[col]

    summary_data.update(
        {
            "US Weight": results["US Equities"],
            "Europe Weight": results["Europe Equities"],
            "EM Weight": results["Emerging Market Equities"],
            "Japan Weight": results["Japan Equities"],
            "Total PF Return": results["expected_return"],
            "Total PF CVaR": results["cvar"],
        }
    )

    summary = pd.DataFrame(summary_data)
    return summary.sort_values("Case").reset_index(drop=True)


def write_summary_excel(summary: pd.DataFrame, output_excel: Path) -> None:
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        ws = writer.book["Summary"]
        # All columns except "Case" are percentages.
        percent_col_idxs = [
            idx + 1 for idx, col_name in enumerate(summary.columns) if col_name != "Case"
        ]
        for col_idx in percent_col_idxs:
            for row_idx in range(2, len(summary) + 2):
                ws.cell(row=row_idx, column=col_idx).number_format = "0.00%"


def plot_case_weights(results: pd.DataFrame, output_path: Path) -> None:
    plot_df = results.copy()
    plot_df["Case Number"] = plot_df["case"].str.extract(r"Case\s+(\d+)")[0].astype(int)
    plot_df = plot_df.sort_values("Case Number").reset_index(drop=True)

    weight_cols = [
        "US Equities",
        "Europe Equities",
        "Emerging Market Equities",
        "Japan Equities",
    ]
    labels = [f"Case {x}" for x in plot_df["Case Number"]]
    weights = plot_df[weight_cols].to_numpy(dtype=float)

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, col in enumerate(weight_cols):
        ax.bar(
            x,
            weights[:, idx],
            bottom=bottom,
            color=ASSET_COLORS[col],
            edgecolor="white",
            linewidth=0.8,
            label=col,
        )
        bottom += weights[:, idx]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=PLOT_FONT_SIZES["tick"])
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(axis="y", labelsize=PLOT_FONT_SIZES["tick"])
    ax.set_ylabel("Portfolio Weight", fontsize=PLOT_FONT_SIZES["axis_label"])
    ax.set_title(
        "Portfolio Weight Allocation Across Four Cases",
        fontsize=PLOT_FONT_SIZES["title"],
        pad=12,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=PLOT_FONT_SIZES["legend"],
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_weights(results: pd.DataFrame, output_path: Path) -> None:
    required_cols = [
        "Europe Expected Return",
        "US Equities",
        "Europe Equities",
        "Emerging Market Equities",
        "Japan Equities",
    ]
    missing = [col for col in required_cols if col not in results.columns]
    if missing:
        raise ValueError(
            "Sweep plot requires columns not found in results: " + ", ".join(missing)
        )

    plot_df = results.copy()
    plot_df = plot_df.sort_values("Europe Expected Return", ascending=False).reset_index(drop=True)
    x = 100.0 * plot_df["Europe Expected Return"].to_numpy(dtype=float)

    line_specs = [
        ("US Equities", ASSET_COLORS["US Equities"]),
        ("Europe Equities", ASSET_COLORS["Europe Equities"]),
        ("Emerging Market Equities", ASSET_COLORS["Emerging Market Equities"]),
        ("Japan Equities", ASSET_COLORS["Japan Equities"]),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    for col, color in line_specs:
        ax.plot(
            x,
            100.0 * plot_df[col].to_numpy(dtype=float),
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=4.2,
            label=col,
        )

    ax.set_xlim(x.max(), x.min())
    ax.set_ylim(0.0, 60.0)
    ax.set_xlabel(
        "Expected Return Europe Equities (%)",
        fontsize=PLOT_FONT_SIZES["axis_label"],
    )
    ax.set_ylabel("Portfolio Weight (%)", fontsize=PLOT_FONT_SIZES["axis_label"])
    ax.set_title(
        "Lexicographic Optimization: Weight Sensitivity to Expected Return of European Equities",
        fontsize=PLOT_FONT_SIZES["title"],
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=PLOT_FONT_SIZES["legend"])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_weights_linkedin(results: pd.DataFrame, output_path: Path) -> None:
    required_cols = [
        "Europe Expected Return",
        "US Equities",
        "Europe Equities",
        "Emerging Market Equities",
    ]
    missing = [col for col in required_cols if col not in results.columns]
    if missing:
        raise ValueError(
            "LinkedIn sweep plot requires columns not found in results: " + ", ".join(missing)
        )

    plot_df = results.copy()
    # LinkedIn version: use ascending axis for fast readability (6.0 -> 7.0).
    plot_df = plot_df.sort_values("Europe Expected Return", ascending=True).reset_index(drop=True)
    x = 100.0 * plot_df["Europe Expected Return"].to_numpy(dtype=float)

    line_specs = [
        ("US Equities", ASSET_COLORS["US Equities"]),
        ("Europe Equities", ASSET_COLORS["Europe Equities"]),
        ("Emerging Market Equities", ASSET_COLORS["Emerging Market Equities"]),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    for col, color in line_specs:
        ax.plot(
            x,
            100.0 * plot_df[col].to_numpy(dtype=float),
            color=color,
            linewidth=2.6,
            marker="o",
            markersize=4.5,
            label=col,
        )

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 60.0)
    ax.set_xlabel(
        "Expected Return Europe Equities (%)",
        fontsize=PLOT_FONT_SIZES["axis_label"],
    )
    ax.set_ylabel("Portfolio Weight (%)", fontsize=PLOT_FONT_SIZES["axis_label"])
    ax.set_title(
        "Small Return Changes, Smooth Portfolio Adjustments",
        fontsize=PLOT_FONT_SIZES["title"],
        pad=12,
    )
    ax.tick_params(axis="both", labelsize=PLOT_FONT_SIZES["tick"])
    ax.grid(alpha=0.25)
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=PLOT_FONT_SIZES["legend"],
    )

    ax.text(
        0.01,
        0.99,
        "As Europe expected return declines from 7.0% to 6.0%,\n"
        "allocations adjust gradually rather than flipping abruptly.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PLOT_FONT_SIZES["tick"],
        color="#36454F",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#F4F7FA",
            "edgecolor": "#D4DCE5",
            "alpha": 0.95,
        },
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mean-CVaR and lexicographic portfolio optimization for blog cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to demeaned scenario workbook.",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default="Sheet1",
        help="Excel sheet name containing the scenarios.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.90,
        help="CVaR confidence level alpha (default 0.90).",
    )
    parser.add_argument(
        "--cvar-limit",
        type=float,
        default=0.25,
        help="CVaR upper bound (default 0.25 = 25%%).",
    )
    parser.add_argument(
        "--no-cvar-constraint",
        action="store_true",
        help="Disable the CVaR constraint in Stage 1 and Stage 2.",
    )
    parser.add_argument(
        "--utility-floor",
        type=float,
        default=0.995,
        help="Stage-2 utility floor as fraction of Stage-1 optimum (default 0.995).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_CASE_RESULTS_CSV,
        help="Where to save the case results CSV.",
    )
    parser.add_argument(
        "--output-excel",
        type=Path,
        default=DEFAULT_CASE_SUMMARY_XLSX,
        help="Where to save the formatted case summary Excel file.",
    )
    parser.add_argument(
        "--run-europe-sweep",
        action="store_true",
        help="Run lexicographic sweep: Europe return from 7.0%% down to 6.0%% (0.05%% steps).",
    )
    parser.add_argument(
        "--sweep-us-return",
        type=float,
        default=0.070,
        help="US expected return for sweep mode (default 0.070).",
    )
    parser.add_argument(
        "--sweep-europe-start",
        type=float,
        default=0.070,
        help="Sweep start for Europe expected return (default 0.070).",
    )
    parser.add_argument(
        "--sweep-europe-end",
        type=float,
        default=0.060,
        help="Sweep end for Europe expected return (default 0.060).",
    )
    parser.add_argument(
        "--sweep-europe-step",
        type=float,
        default=0.0005,
        help="Sweep decrement for Europe expected return (default 0.0005 = 0.05%%).",
    )
    parser.add_argument(
        "--sweep-em-return",
        type=float,
        default=0.075,
        help="EM expected return for sweep mode (default 0.075).",
    )
    parser.add_argument(
        "--sweep-japan-return",
        type=float,
        default=0.055,
        help="Japan expected return for sweep mode (default 0.055).",
    )
    parser.add_argument(
        "--output-case-chart",
        type=Path,
        default=DEFAULT_CASE_CHART,
        help="Where to save the four-case weight chart.",
    )
    parser.add_argument(
        "--output-sweep-chart",
        type=Path,
        default=DEFAULT_SWEEP_CHART,
        help="Where to save the sweep sensitivity chart.",
    )
    parser.add_argument(
        "--output-sweep-linkedin-chart",
        type=Path,
        default=DEFAULT_SWEEP_LINKEDIN_CHART,
        help="Where to save the LinkedIn sweep sensitivity chart.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    args = parser.parse_args()

    demeaned = load_demeaned_scenarios(args.input, args.sheet)
    cvar_limit = None if args.no_cvar_constraint else args.cvar_limit

    if args.run_europe_sweep:
        if args.output_csv == DEFAULT_CASE_RESULTS_CSV:
            args.output_csv = DEFAULT_SWEEP_RESULTS_CSV
        if args.output_excel == DEFAULT_CASE_SUMMARY_XLSX:
            args.output_excel = DEFAULT_SWEEP_SUMMARY_XLSX
        results = run_europe_return_sweep(
            demeaned_scenarios=demeaned,
            cvar_alpha=args.cvar_alpha,
            cvar_limit=cvar_limit,
            utility_floor_fraction=args.utility_floor,
            us_return=args.sweep_us_return,
            europe_start=args.sweep_europe_start,
            europe_end=args.sweep_europe_end,
            europe_step=args.sweep_europe_step,
            em_return=args.sweep_em_return,
            japan_return=args.sweep_japan_return,
        )
    else:
        results = run_cases(
            demeaned_scenarios=demeaned,
            cvar_alpha=args.cvar_alpha,
            cvar_limit=cvar_limit,
            utility_floor_fraction=args.utility_floor,
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_csv, index=False)
    summary = build_summary_table(results)
    write_summary_excel(summary, args.output_excel)

    display = results.copy()
    percent_cols = [
        "expected_return",
        "cvar",
        "l2_to_benchmark",
        "utility_floor",
        "US Equities",
        "Europe Equities",
        "Emerging Market Equities",
        "Japan Equities",
        "US Expected Return",
        "Europe Expected Return",
        "EM Expected Return",
        "Japan Expected Return",
    ]
    for col in percent_cols:
        if col in display.columns:
            display[col] = display[col].map(
                lambda x: "" if pd.isna(x) else f"{100.0 * float(x):.2f}%"
            )

    print("\nOptimization results:\n")
    print(display.to_string(index=False))
    print(f"\nSaved CSV to: {args.output_csv.resolve()}")
    print(f"Saved Excel summary to: {args.output_excel.resolve()}")

    if not args.no_plots:
        if args.run_europe_sweep:
            plot_sweep_weights(results, args.output_sweep_chart)
            plot_sweep_weights_linkedin(results, args.output_sweep_linkedin_chart)
            print(f"Saved sweep chart to: {args.output_sweep_chart.resolve()}")
            print(
                "Saved LinkedIn sweep chart to: "
                f"{args.output_sweep_linkedin_chart.resolve()}"
            )
        else:
            plot_case_weights(results, args.output_case_chart)
            print(f"Saved case chart to: {args.output_case_chart.resolve()}")

    if not args.run_europe_sweep:
        case3 = results[results["case"].str.startswith("Case 3")].iloc[0]
        case4 = results[results["case"].str.startswith("Case 4")].iloc[0]
        weight_diff = np.abs(
            case3[ASSETS].to_numpy(dtype=float) - case4[ASSETS].to_numpy(dtype=float)
        )
        print(
            "\nCase 3 vs Case 4 absolute weight differences (%-points): "
            + ", ".join(
                f"{asset}: {100.0 * diff:.2f}" for asset, diff in zip(ASSETS, weight_diff)
            )
        )


if __name__ == "__main__":
    main()
