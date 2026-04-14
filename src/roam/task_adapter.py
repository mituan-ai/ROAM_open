from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class RoamTaskDefaults:
    observation_bias_magnitude: float | None = None
    observation_scale_delta: float | None = None
    observation_update_interval_steps: int | None = None
    observation_update_interval_minutes: int | None = None
    process_update_interval_steps: int | None = None
    process_update_interval_minutes: int | None = None
    label_delay_steps: int | None = None
    label_delay_minutes: int | None = None


@dataclass(frozen=True)
class PriorTaskContext:
    task_goal: str
    environment_boundary_lines: tuple[str, ...]
    hard_rule_lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class PriorKnownContext:
    summary_lines: tuple[str, ...] = ()
    payload: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "summary_lines": list(self.summary_lines),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class RoamTaskAdapter:
    name: str
    feature_names: tuple[str, ...]
    observation_feature_names: tuple[str, ...]
    summary_feature_names: tuple[str, ...]
    load_feature_names: tuple[str, ...]
    dynamics_feature_names: tuple[str, ...]
    readout_feature_names: tuple[str, ...]
    prior_task_context: PriorTaskContext
    reliability_delay_steps: int
    target_index_field_name: str
    target_time_field_name: str | None
    supervised_indices_fn: Callable[[object, int], np.ndarray]
    target_value_fn: Callable[[object, int], float]
    target_position_fn: Callable[[object, int], int]
    target_time_value_fn: Callable[[object, int], float | None]
    build_prior_known_context_fn: Callable[[object, int, np.ndarray, Any], PriorKnownContext]
    build_diagnostics_fn: Callable[[Any, Any, np.ndarray], Any]
    build_prior_diagnostics_payload_fn: Callable[[Any], dict[str, object]]
    observation_feature_vector_fn: Callable[[Any], np.ndarray]
    summary_feature_vector_fn: Callable[[Any], np.ndarray]
    load_context_vector_fn: Callable[[Any, np.ndarray, Any], np.ndarray]
    readout_context_vector_fn: Callable[[Any, np.ndarray, Any], np.ndarray]
    load_feature_vector_from_context_fn: Callable[[np.ndarray, float], np.ndarray]
    readout_feature_vector_from_context_fn: Callable[[np.ndarray, float], np.ndarray]
    dynamics_feature_vector_fn: Callable[[np.ndarray], np.ndarray]
    apply_observation_bias_fn: Callable[[np.ndarray, float, Any], np.ndarray]
    apply_observation_scale_fn: Callable[[np.ndarray, float], np.ndarray]
    apply_reliability_delay_fn: Callable[[np.ndarray, int], np.ndarray]
    apply_dynamics_scaling_fn: Callable[[np.ndarray, float], np.ndarray]
    roam_defaults: RoamTaskDefaults = field(default_factory=RoamTaskDefaults)

    def supervised_indices(self, episode_grid: object, *, window_size: int) -> np.ndarray:
        return np.asarray(self.supervised_indices_fn(episode_grid, window_size), dtype=np.int64)

    def target_value(self, episode_grid: object, *, target_index: int) -> float:
        return float(self.target_value_fn(episode_grid, target_index))

    def target_position(self, episode_grid: object, *, target_index: int) -> int:
        return int(self.target_position_fn(episode_grid, target_index))

    def target_time_value(self, episode_grid: object, *, target_index: int) -> float | None:
        return self.target_time_value_fn(episode_grid, target_index)

    def build_prior_known_context(
        self,
        episode_grid: object,
        *,
        target_index: int,
        feature_window: np.ndarray,
        diagnostics: Any,
    ) -> PriorKnownContext:
        return self.build_prior_known_context_fn(
            episode_grid,
            target_index,
            np.asarray(feature_window, dtype=np.float64),
            diagnostics,
        )

    def build_diagnostics(self, model: Any, reference_stats: Any, feature_window: np.ndarray) -> Any:
        return self.build_diagnostics_fn(model, reference_stats, feature_window)

    def build_prior_diagnostics_payload(self, diagnostics: Any) -> dict[str, object]:
        return self.build_prior_diagnostics_payload_fn(diagnostics)

    def observation_feature_vector(self, diagnostics: Any) -> np.ndarray:
        return np.asarray(self.observation_feature_vector_fn(diagnostics), dtype=np.float64)

    def summary_feature_vector(self, diagnostics: Any) -> np.ndarray:
        return np.asarray(self.summary_feature_vector_fn(diagnostics), dtype=np.float64)

    def load_context_vector(
        self,
        diagnostics: Any,
        feature_window: np.ndarray,
        reference_stats: Any,
    ) -> np.ndarray:
        return np.asarray(
            self.load_context_vector_fn(diagnostics, feature_window, reference_stats),
            dtype=np.float64,
        )

    def readout_context_vector(
        self,
        diagnostics: Any,
        feature_window: np.ndarray,
        reference_stats: Any,
    ) -> np.ndarray:
        return np.asarray(
            self.readout_context_vector_fn(diagnostics, feature_window, reference_stats),
            dtype=np.float64,
        )

    def load_feature_vector_from_context(self, load_context: np.ndarray, *, residual: float) -> np.ndarray:
        return np.asarray(
            self.load_feature_vector_from_context_fn(np.asarray(load_context, dtype=np.float64), residual),
            dtype=np.float64,
        )

    def readout_feature_vector_from_context(self, readout_context: np.ndarray, *, residual: float) -> np.ndarray:
        return np.asarray(
            self.readout_feature_vector_from_context_fn(np.asarray(readout_context, dtype=np.float64), residual),
            dtype=np.float64,
        )

    def dynamics_feature_vector(self, summary_residual: np.ndarray) -> np.ndarray:
        return np.asarray(self.dynamics_feature_vector_fn(summary_residual), dtype=np.float64)

    def apply_observation_bias(
        self,
        feature_window: np.ndarray,
        magnitude: float,
        *,
        reference_stats: Any,
    ) -> np.ndarray:
        return np.asarray(
            self.apply_observation_bias_fn(np.asarray(feature_window, dtype=np.float64), magnitude, reference_stats),
            dtype=np.float64,
        )

    def apply_observation_scale(self, feature_window: np.ndarray, scale: float) -> np.ndarray:
        return np.asarray(
            self.apply_observation_scale_fn(np.asarray(feature_window, dtype=np.float64), scale),
            dtype=np.float64,
        )

    def apply_reliability_delay(self, feature_window: np.ndarray, delay_steps: int) -> np.ndarray:
        return np.asarray(
            self.apply_reliability_delay_fn(np.asarray(feature_window, dtype=np.float64), delay_steps),
            dtype=np.float64,
        )

    def apply_dynamics_scaling(self, feature_window: np.ndarray, factor: float) -> np.ndarray:
        return np.asarray(
            self.apply_dynamics_scaling_fn(np.asarray(feature_window, dtype=np.float64), factor),
            dtype=np.float64,
        )
