from __future__ import annotations

import pandas as pd
from pathlib import Path

from portfolio_experiment.config import apply_overrides, build_default_config
from portfolio_experiment.experiment import align_config_to_scenario_feasibility, run_sweep_experiment
from portfolio_experiment.metrics import compute_instability_metrics
from portfolio_experiment.reporting import (
    export_outputs,
    make_plots,
    make_summary_tables,
    plot_te_cvar_sensitivity_sweep,
)
from portfolio_experiment.scenarios import generate_or_load_scenarios

SCRIPT_DIR = Path(__file__).resolve().parent

# Small top-level config dictionary for quick research tuning.
CONFIG_OVERRIDES = {
    "alpha": 0.90,
    "cvar_portfolio_limit": 0.25,
    "cvar_portfolio_limit_mode": "benchmark_factor",
    "cvar_portfolio_benchmark_factor": 1.05,
    "cvar_te_limit": 0.06,
    "turnover_limit": 0.40,
    "epsilon": 0.005,
    "enable_turnover_constraint": True,
    "auto_adjust_portfolio_cvar_to_benchmark": False,
    "n_scenarios": 3000,
    "seed": 42,
}
TE_CVAR_SENSITIVITY_VALUES = (0.02, 0.03, 0.04)


def build_te_cvar_sensitivity_results(
    *,
    config,
    R,
    q,
    scenario_metadata,
) -> pd.DataFrame:
    sensitivity_frames: list[pd.DataFrame] = []
    for te_cvar_limit in TE_CVAR_SENSITIVITY_VALUES:
        sensitivity_config = apply_overrides(config, {"cvar_te_limit": te_cvar_limit})
        sensitivity_results = run_sweep_experiment(
            config=sensitivity_config,
            R=R,
            q=q,
            scenario_metadata=scenario_metadata,
        )
        sensitivity_results = sensitivity_results[
            sensitivity_results["architecture"].isin(["B", "C"])
        ].copy()
        sensitivity_results["te_cvar_limit"] = float(te_cvar_limit)
        sensitivity_frames.append(sensitivity_results)

    return pd.concat(sensitivity_frames, ignore_index=True)


def main() -> None:
    config = apply_overrides(
        build_default_config(output_dir=SCRIPT_DIR / "outputs"),
        CONFIG_OVERRIDES,
    )
    R, q, scenario_metadata = generate_or_load_scenarios(config=config)
    config, feasibility_metadata = align_config_to_scenario_feasibility(
        config=config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
    )
    scenario_metadata = {**scenario_metadata, **feasibility_metadata}

    raw_results = run_sweep_experiment(config=config, R=R, q=q, scenario_metadata=scenario_metadata)
    raw_results = compute_instability_metrics(raw_results, config.assets)

    summary_df, snapshot_df = make_summary_tables(raw_results, config)
    plot_paths = make_plots(
        results_df=raw_results,
        summary_df=summary_df,
        snapshot_df=snapshot_df,
        config=config,
    )
    te_cvar_sensitivity_results = build_te_cvar_sensitivity_results(
        config=config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
    )
    plot_paths.append(
        plot_te_cvar_sensitivity_sweep(
            results_df=te_cvar_sensitivity_results,
            config=config,
            te_cvar_values=TE_CVAR_SENSITIVITY_VALUES,
        )
    )
    output_paths = export_outputs(
        raw_results=raw_results,
        summary_df=summary_df,
        snapshot_df=snapshot_df,
        config=config,
        scenario_metadata=scenario_metadata,
        plot_paths=plot_paths,
    )

    print("Portfolio architecture comparison complete.")
    print(f"Output directory: {config.output_dir.resolve()}")
    if scenario_metadata.get("portfolio_cvar_limit_adjusted"):
        print(
            "Portfolio CVaR limit adjusted to keep the benchmark feasible: "
            f"requested={scenario_metadata['requested_portfolio_cvar_limit']:.4f}, "
            f"effective={scenario_metadata['effective_portfolio_cvar_limit']:.4f}."
        )
    elif config.optimization.cvar_portfolio_limit_mode == "benchmark_factor":
        print(
            "Portfolio CVaR limit mode: benchmark_factor "
            f"(demeaned benchmark CVaR x {config.optimization.cvar_portfolio_benchmark_factor:.2f})."
        )
    print()
    print(
        summary_df[
            [
                "architecture",
                "average_expected_return",
                "average_portfolio_cvar",
                "average_te_cvar",
                "average_l2_benchmark_distance",
                "average_turnover",
                "average_instability_score",
                "max_jump",
            ]
        ].to_string(index=False)
    )
    print()
    print("Exported files:")
    for name, path in output_paths.items():
        print(f"  {name}: {path}")
    for path in plot_paths:
        print(f"  plot: {path}")


if __name__ == "__main__":
    main()
