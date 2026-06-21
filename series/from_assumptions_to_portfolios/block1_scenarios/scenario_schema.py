"""ScenarioSet container and validation for Block 1.

Blueprint reference: block1_blueprint_v4_3_nonparametric_no_japan.md
Section 15 (ScenarioSet object).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


N_DRIVERS = 12
N_ASSETS = 11
HORIZONS = ("12m", "36m", "60m")


@dataclass
class ScenarioSet:
    """Container for a complete multi-period scenario set (Set A, B, or C)."""

    label: str
    type: str
    driver_paths: np.ndarray  # shape (S, H, N_DRIVERS)
    asset_paths: np.ndarray  # shape (S, H, N_ASSETS)
    asset_terminal: Dict[str, np.ndarray]  # {"12m": (S, N_ASSETS), ...}
    p: np.ndarray  # shape (S,)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.driver_paths = np.asarray(self.driver_paths)
        self.asset_paths = np.asarray(self.asset_paths)
        self.p = np.asarray(self.p)
        self.asset_terminal = {
            key: np.asarray(value) for key, value in self.asset_terminal.items()
        }
        self.validate()

    def validate(self) -> None:
        """Validate internal consistency of shapes, probabilities and terminal map."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")

        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("type must be a non-empty string")

        if self.driver_paths.ndim != 3:
            raise ValueError(
                f"driver_paths ndim {self.driver_paths.ndim} != expected 3"
            )

        if self.asset_paths.ndim != 3:
            raise ValueError(
                f"asset_paths ndim {self.asset_paths.ndim} != expected 3"
            )

        if self.driver_paths.shape[2] != N_DRIVERS:
            raise ValueError(
                f"driver_paths shape {self.driver_paths.shape} != expected ...x...x{N_DRIVERS}"
            )

        if self.asset_paths.shape[2] != N_ASSETS:
            raise ValueError(
                f"asset_paths shape {self.asset_paths.shape} != expected ...x...x{N_ASSETS}"
            )

        if self.driver_paths.shape[0] != self.asset_paths.shape[0]:
            raise ValueError(
                "driver_paths and asset_paths must have same scenario dimension: "
                f"{self.driver_paths.shape[0]} vs {self.asset_paths.shape[0]}"
            )

        if self.driver_paths.shape[1] != self.asset_paths.shape[1]:
            raise ValueError(
                "driver_paths and asset_paths must have same horizon dimension: "
                f"{self.driver_paths.shape[1]} vs {self.asset_paths.shape[1]}"
            )

        if self.p.ndim != 1 or self.p.shape[0] != self.asset_paths.shape[0]:
            raise ValueError(
                f"p shape {self.p.shape} != expected ({self.asset_paths.shape[0]},)"
            )

        if not np.all(self.p >= 0):
            raise ValueError("p must be non-negative")

        if not np.isclose(self.p.sum(), 1.0, atol=1e-8):
            raise ValueError(f"p sum {self.p.sum()} != 1.0")

        if set(self.asset_terminal.keys()) != set(HORIZONS):
            raise ValueError(
                "asset_terminal keys must be exactly {'12m', '36m', '60m'}"
            )

        scenario_count = self.asset_paths.shape[0]
        for horizon in HORIZONS:
            terminal = self.asset_terminal[horizon]
            if terminal.shape != (scenario_count, N_ASSETS):
                raise ValueError(
                    f"asset_terminal['{horizon}'] shape {terminal.shape} != expected "
                    f"{(scenario_count, N_ASSETS)}"
                )

        if not np.isfinite(self.driver_paths).all():
            raise ValueError("driver_paths must be finite")

        if not np.isfinite(self.asset_paths).all():
            raise ValueError("asset_paths must be finite")

        if not np.isfinite(self.p).all():
            raise ValueError("p must be finite")

        for horizon in HORIZONS:
            terminal = self.asset_terminal[horizon]
            if not np.isfinite(terminal).all():
                raise ValueError(
                    f"asset_terminal['{horizon}'] must be finite"
                )

    @property
    def n_scenarios(self) -> int:
        return int(self.asset_paths.shape[0])

    @property
    def horizon_months(self) -> int:
        return int(self.asset_paths.shape[1])
