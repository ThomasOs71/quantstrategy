"""Scenario-based portfolio architecture comparison experiment."""

from .config import (
    ASSETS,
    BENCHMARK_WEIGHTS,
    DEFAULT_CONFIG,
    build_default_config,
    build_expected_return_vector,
)
from .experiment import run_sweep_experiment
from .metrics import compute_instability_metrics
from .reporting import export_outputs, make_plots, make_summary_tables
from .scenarios import generate_or_load_scenarios

__all__ = [
    "ASSETS",
    "BENCHMARK_WEIGHTS",
    "DEFAULT_CONFIG",
    "build_default_config",
    "build_expected_return_vector",
    "compute_instability_metrics",
    "export_outputs",
    "generate_or_load_scenarios",
    "make_plots",
    "make_summary_tables",
    "run_sweep_experiment",
]
