from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

ASSETS: tuple[str, ...] = ("US", "Europe", "EM", "Japan")
BENCHMARK_WEIGHTS = np.array([0.40, 0.40, 0.15, 0.05], dtype=float)
LOWER_BOUNDS = np.array([0.20, 0.20, 0.00, 0.00], dtype=float)
UPPER_BOUNDS = np.array([0.60, 0.60, 0.25, 0.20], dtype=float)

MU_BASE: dict[str, float | None] = {
    "US": 0.070,
    "Europe": None,
    "EM": 0.075,
    "Japan": 0.055,
}


def build_inclusive_grid(start: float, end: float, step: float) -> np.ndarray:
    """Create an inclusive grid without floating-point drift."""

    if step == 0:
        raise ValueError("step must be non-zero.")
    if np.sign(step) != np.sign(end - start):
        raise ValueError("step direction must move from start toward end.")

    n_points = int(round((end - start) / step)) + 1
    grid = start + step * np.arange(n_points, dtype=float)
    grid[-1] = end
    return np.round(grid, 10)


DEFAULT_MU_EUROPE_GRID = build_inclusive_grid(0.070, 0.060, -0.0005)


def _default_excel_path() -> Path:
    return Path(__file__).resolve().parents[1] / "input" / "mc_scenarios.xlsx"


def _default_synthetic_means() -> np.ndarray:
    return np.array([0.070, 0.065, 0.075, 0.055], dtype=float)


def _default_volatilities() -> np.ndarray:
    # The default calibration intentionally leaves some room for the TE-CVaR
    # constraint to become relevant while still looking equity-like.
    return np.array([0.20, 0.15, 0.24, 0.14], dtype=float)


def _default_correlation() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.68, 0.60, 0.48],
            [0.68, 1.00, 0.55, 0.42],
            [0.60, 0.55, 1.00, 0.38],
            [0.48, 0.42, 0.38, 1.00],
        ],
        dtype=float,
    )


@dataclass(frozen=True)
class ScenarioConfig:
    n_scenarios: int = 3000
    seed: int = 42
    distribution: str = "multivariate_normal"
    excel_path: Path | None = field(default_factory=_default_excel_path)
    excel_sheet_name: str | None = None
    probability_column: str | None = None
    excel_scenarios_are_demeaned: bool = True
    synthetic_means: np.ndarray = field(default_factory=_default_synthetic_means)
    volatilities: np.ndarray = field(default_factory=_default_volatilities)
    correlation: np.ndarray = field(default_factory=_default_correlation)

    @property
    def covariance(self) -> np.ndarray:
        return np.diag(self.volatilities) @ self.correlation @ np.diag(self.volatilities)


@dataclass(frozen=True)
class OptimizationConfig:
    alpha: float = 0.90
    cvar_portfolio_limit: float = 0.25
    cvar_portfolio_limit_mode: str = "benchmark_factor"
    cvar_portfolio_benchmark_factor: float = 1.05
    cvar_te_limit: float = 0.06
    turnover_limit: float = 0.40
    epsilon: float = 0.005
    enable_turnover_constraint: bool = True
    auto_adjust_portfolio_cvar_to_benchmark: bool = False
    portfolio_cvar_adjustment_buffer: float = 1e-4
    solver_preference: tuple[str, ...] = ("CLARABEL", "ECOS", "OSQP", "SCS")
    solver_verbose: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    assets: tuple[str, ...] = ASSETS
    benchmark_weights: np.ndarray = field(default_factory=lambda: BENCHMARK_WEIGHTS.copy())
    lower_bounds: np.ndarray = field(default_factory=lambda: LOWER_BOUNDS.copy())
    upper_bounds: np.ndarray = field(default_factory=lambda: UPPER_BOUNDS.copy())
    mu_base: dict[str, float | None] = field(default_factory=lambda: dict(MU_BASE))
    mu_europe_grid: np.ndarray = field(default_factory=lambda: DEFAULT_MU_EUROPE_GRID.copy())
    selected_sweep_points: tuple[float, ...] = (0.070, 0.065, 0.060)
    output_dir: Path = Path("outputs")
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


DEFAULT_CONFIG = ExperimentConfig()


def build_default_config(output_dir: str | Path | None = None) -> ExperimentConfig:
    config = DEFAULT_CONFIG
    if output_dir is None:
        return config
    return replace(config, output_dir=Path(output_dir))


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply flat overrides to the experiment, scenario, or optimization config."""

    exp_fields = set(ExperimentConfig.__dataclass_fields__)
    scenario_fields = set(ScenarioConfig.__dataclass_fields__)
    optimization_fields = set(OptimizationConfig.__dataclass_fields__)

    experiment_updates: dict[str, Any] = {}
    scenario_updates: dict[str, Any] = {}
    optimization_updates: dict[str, Any] = {}

    for key, value in overrides.items():
        if key in scenario_fields:
            scenario_updates[key] = value
        elif key in optimization_fields:
            optimization_updates[key] = value
        elif key in exp_fields:
            experiment_updates[key] = value
        else:
            raise KeyError(f"Unknown configuration override: {key}")

    updated_scenario = replace(config.scenario, **scenario_updates)
    updated_optimization = replace(config.optimization, **optimization_updates)
    return replace(
        config,
        scenario=updated_scenario,
        optimization=updated_optimization,
        **experiment_updates,
    )


def build_expected_return_vector(mu_europe: float, config: ExperimentConfig) -> np.ndarray:
    """Build the objective expected-return vector in asset order."""

    mu_vector = []
    for asset in config.assets:
        if asset == "Europe":
            mu_vector.append(float(mu_europe))
        else:
            mu_value = config.mu_base[asset]
            if mu_value is None:
                raise ValueError(f"Missing baseline expected return for asset: {asset}")
            mu_vector.append(float(mu_value))
    return np.array(mu_vector, dtype=float)


def to_serializable_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Convert the nested dataclass config into a JSON-friendly dictionary."""

    def _convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: _convert(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(item) for item in value]
        return value

    return _convert(asdict(config))
