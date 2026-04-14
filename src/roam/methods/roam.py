from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from roam.methods.sequence_regressor import SequenceReferenceStats, SequenceRegressorProtocol
from roam.prior_engine import (
    EvidenceBundle,
    OpenEvidence,
    PriorEngineInput,
    PriorEngineResult,
    SemanticAxis,
    SemanticPriorEngine,
)
from roam.sequence import SequenceRegressionDataset
from roam.task_adapter import RoamTaskAdapter


def _corrective_axes() -> tuple[SemanticAxis, ...]:
    return (
        SemanticAxis.BIAS,
        SemanticAxis.SCALE,
        SemanticAxis.LOAD,
        SemanticAxis.DYNAMICS,
        SemanticAxis.READOUT,
    )


def _all_axes() -> tuple[SemanticAxis, ...]:
    return _corrective_axes()


def _prior_axes() -> tuple[SemanticAxis, ...]:
    return tuple(SemanticAxis)


def _observation_axes() -> tuple[SemanticAxis, ...]:
    return (SemanticAxis.BIAS, SemanticAxis.SCALE)


def _process_axes() -> tuple[SemanticAxis, ...]:
    return (SemanticAxis.LOAD, SemanticAxis.DYNAMICS, SemanticAxis.READOUT)


SUPPORTED_UPDATE_SPACES = ("subspace", "full")


@dataclass(frozen=True)
class LatentAdapterConfig:
    ridge_lambda: float = 1e-3
    latent_l2: float = 1e-2
    joint_fit_iterations: int = 3
    synthetic_anchor_weight: float = 1.0
    real_residual_weight: float = 2.0
    observation_anchor_weight: float = 1.0
    observation_real_weight: float = 1.0
    process_anchor_weight: float = 1.0
    process_real_weight: float = 1.0
    nominal_summary_weight: float = 1.0
    observation_bias_magnitude: float = 8e-4
    observation_scale_delta: float = 0.04
    dynamics_slowdown_factor: float = 0.6
    dynamics_speedup_factor: float = 1.4

    def validate(self) -> None:
        if self.ridge_lambda < 0.0:
            raise ValueError("ridge_lambda must be non-negative")
        if self.latent_l2 < 0.0:
            raise ValueError("latent_l2 must be non-negative")
        if self.joint_fit_iterations < 1:
            raise ValueError("joint_fit_iterations must be at least 1")
        if self.synthetic_anchor_weight <= 0.0:
            raise ValueError("synthetic_anchor_weight must be positive")
        if self.real_residual_weight <= 0.0:
            raise ValueError("real_residual_weight must be positive")
        if self.observation_anchor_weight <= 0.0:
            raise ValueError("observation_anchor_weight must be positive")
        if self.observation_real_weight <= 0.0:
            raise ValueError("observation_real_weight must be positive")
        if self.process_anchor_weight <= 0.0:
            raise ValueError("process_anchor_weight must be positive")
        if self.process_real_weight <= 0.0:
            raise ValueError("process_real_weight must be positive")
        if self.nominal_summary_weight <= 0.0:
            raise ValueError("nominal_summary_weight must be positive")
        if self.observation_bias_magnitude <= 0.0:
            raise ValueError("observation_bias_magnitude must be positive")
        if not (0.0 < self.observation_scale_delta < 1.0):
            raise ValueError("observation_scale_delta must be in (0, 1)")
        if not (0.0 < self.dynamics_slowdown_factor < 1.0):
            raise ValueError("dynamics_slowdown_factor must be in (0, 1)")
        if self.dynamics_speedup_factor <= 1.0:
            raise ValueError("dynamics_speedup_factor must be > 1")


@dataclass(frozen=True)
class OnlinePosteriorConfig:
    label_delay_steps: int = 1
    label_delay_minutes: int | None = None
    label_observation_noise_variance: float = 4e-4
    enable_trust_gate: bool = True

    @property
    def effective_label_delay_steps(self) -> int:
        return int(
            self.label_delay_steps
            if self.label_delay_minutes is None
            else self.label_delay_minutes
        )

    def validate(self) -> None:
        if self.effective_label_delay_steps < 1:
            raise ValueError("label_delay_steps must be at least 1")
        if self.label_observation_noise_variance <= 0.0:
            raise ValueError("label_observation_noise_variance must be positive")


@dataclass(frozen=True)
class ObservationEvidenceConfig:
    enabled: bool = True
    observation_variance: float = 1.0
    update_interval_steps: int = 30
    update_interval_minutes: int | None = None
    variance_floor: float = 0.25

    @property
    def effective_update_interval_steps(self) -> int:
        return int(
            self.update_interval_steps
            if self.update_interval_minutes is None
            else self.update_interval_minutes
        )

    def validate(self) -> None:
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if self.effective_update_interval_steps < 1:
            raise ValueError("update_interval_steps must be at least 1")
        if self.variance_floor < 0.0:
            raise ValueError("variance_floor must be non-negative")


@dataclass(frozen=True)
class ProcessEvidenceConfig:
    enabled: bool = True
    observation_variance: float = 1.2
    update_interval_steps: int = 30
    update_interval_minutes: int | None = None
    variance_floor: float = 0.35

    @property
    def effective_update_interval_steps(self) -> int:
        return int(
            self.update_interval_steps
            if self.update_interval_minutes is None
            else self.update_interval_minutes
        )

    def validate(self) -> None:
        if self.observation_variance <= 0.0:
            raise ValueError("observation_variance must be positive")
        if self.effective_update_interval_steps < 1:
            raise ValueError("update_interval_steps must be at least 1")
        if self.variance_floor < 0.0:
            raise ValueError("variance_floor must be non-negative")


@dataclass(frozen=True)
class SupportGeometryConfig:
    enable_risk_gate: bool = True
    gamma_uncertainty_scale_obs: float = 0.12
    gamma_uncertainty_scale_proc: float = 0.12
    gamma_support_scale_obs: float = 0.08
    gamma_support_scale_proc: float = 0.08

    def validate(self) -> None:
        if self.gamma_uncertainty_scale_obs < 0.0:
            raise ValueError("gamma_uncertainty_scale_obs must be non-negative")
        if self.gamma_uncertainty_scale_proc < 0.0:
            raise ValueError("gamma_uncertainty_scale_proc must be non-negative")
        if self.gamma_support_scale_obs < 0.0:
            raise ValueError("gamma_support_scale_obs must be non-negative")
        if self.gamma_support_scale_proc < 0.0:
            raise ValueError("gamma_support_scale_proc must be non-negative")


@dataclass(frozen=True)
class RoamLatentConfig:
    name: str = "roam_v1"
    axes: tuple[SemanticAxis, ...] = field(default_factory=_corrective_axes)
    update_space: str = "subspace"
    prior_variance_min: float = 0.15
    prior_variance_max: float = 1.5
    adapter: LatentAdapterConfig = field(default_factory=LatentAdapterConfig)
    observation_evidence: ObservationEvidenceConfig = field(default_factory=ObservationEvidenceConfig)
    process_evidence: ProcessEvidenceConfig = field(default_factory=ProcessEvidenceConfig)
    online: OnlinePosteriorConfig = field(default_factory=OnlinePosteriorConfig)
    support_geometry: SupportGeometryConfig = field(default_factory=SupportGeometryConfig)

    def validate(self) -> None:
        if not self.axes:
            raise ValueError("axes must not be empty")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("axes must not contain duplicates")
        invalid_axes = tuple(axis for axis in self.axes if axis not in _corrective_axes())
        if invalid_axes:
            raise ValueError(f"axes must only contain corrective axes: {invalid_axes}")
        if self.update_space not in SUPPORTED_UPDATE_SPACES:
            raise ValueError(f"update_space must be one of {SUPPORTED_UPDATE_SPACES}")
        if self.prior_variance_min <= 0.0:
            raise ValueError("prior_variance_min must be positive")
        if self.prior_variance_max < self.prior_variance_min:
            raise ValueError("prior_variance_max must be >= prior_variance_min")
        self.adapter.validate()
        self.observation_evidence.validate()
        self.process_evidence.validate()
        self.online.validate()
        self.support_geometry.validate()


@dataclass(frozen=True)
class ModelDiagnostics:
    prediction: float
    prediction_zscore: float
    feature_drift_score: float
    hidden_drift_score: float
    sensitivity_shift_score: float
    range_anomaly_score: float
    anomaly_score: float
    feature_drift_by_channel: dict[str, float]
    channel_sensitivity: dict[str, float]
    channel_sensitivity_delta: dict[str, float]
    hidden_state_norm: float
    summary_metrics: dict[str, float]
    summary_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction": self.prediction,
            "prediction_zscore": self.prediction_zscore,
            "feature_drift_score": self.feature_drift_score,
            "hidden_drift_score": self.hidden_drift_score,
            "sensitivity_shift_score": self.sensitivity_shift_score,
            "range_anomaly_score": self.range_anomaly_score,
            "anomaly_score": self.anomaly_score,
            "feature_drift_by_channel": self.feature_drift_by_channel,
            "channel_sensitivity": self.channel_sensitivity,
            "channel_sensitivity_delta": self.channel_sensitivity_delta,
            "hidden_state_norm": self.hidden_state_norm,
            "summary_metrics": self.summary_metrics,
        }

    def to_summary(self) -> str:
        return self.summary_text


@dataclass(frozen=True)
class LatentAdapterModel:
    axes: tuple[SemanticAxis, ...]
    weight_matrix: np.ndarray
    bias_vector: np.ndarray
    hidden_mean: np.ndarray
    hidden_std: np.ndarray
    observation_feature_names: tuple[str, ...]
    observation_feature_mean: np.ndarray
    observation_feature_std: np.ndarray
    observation_weight_matrix: np.ndarray
    observation_bias_vector: np.ndarray
    reliability_weight_vector: np.ndarray
    reliability_bias: float
    summary_feature_names: tuple[str, ...]
    nominal_summary_weight_matrix: np.ndarray
    nominal_summary_bias_vector: np.ndarray
    load_feature_names: tuple[str, ...]
    load_feature_mean: np.ndarray
    load_feature_std: np.ndarray
    load_weight_vector: np.ndarray
    load_bias: float
    dynamics_feature_names: tuple[str, ...]
    dynamics_feature_mean: np.ndarray
    dynamics_feature_std: np.ndarray
    dynamics_weight_vector: np.ndarray
    dynamics_bias: float
    readout_feature_names: tuple[str, ...]
    readout_feature_mean: np.ndarray
    readout_feature_std: np.ndarray
    readout_weight_vector: np.ndarray
    readout_bias: float
    latent_mean: np.ndarray
    latent_std: np.ndarray
    train_episode_latents: np.ndarray
    anchor_latents: np.ndarray
    training_metrics: dict[str, float]

    @property
    def support_latents(self) -> np.ndarray:
        if self.train_episode_latents.size == 0:
            return np.asarray(self.anchor_latents, dtype=np.float64)
        return np.vstack(
            (
                np.asarray(self.anchor_latents, dtype=np.float64),
                np.asarray(self.train_episode_latents, dtype=np.float64),
            )
        )

    def axis_coefficients(
        self,
        hidden_state: np.ndarray,
        *,
        load_features: np.ndarray | None = None,
        dynamics_features: np.ndarray | None = None,
        readout_features: np.ndarray | None = None,
    ) -> np.ndarray:
        normalized_hidden = (
            np.asarray(hidden_state, dtype=np.float64) - self.hidden_mean
        ) / self.hidden_std
        return np.asarray(self.weight_matrix @ normalized_hidden + self.bias_vector, dtype=np.float64)

    def standardize_latent(self, raw_latent: np.ndarray) -> np.ndarray:
        return (np.asarray(raw_latent, dtype=np.float64) - self.latent_mean) / self.latent_std

    def reliability_score(self, observation_features: np.ndarray) -> float:
        standardized_features = (
            np.asarray(observation_features, dtype=np.float64) - self.observation_feature_mean
        ) / self.observation_feature_std
        raw_score = float(self.reliability_weight_vector @ standardized_features + self.reliability_bias)
        return float(np.clip(raw_score, 0.0, 1.0))

    def nominal_summary(self, hidden_state: np.ndarray) -> np.ndarray:
        normalized_hidden = (
            np.asarray(hidden_state, dtype=np.float64) - self.hidden_mean
        ) / self.hidden_std
        return self.nominal_summary_weight_matrix @ normalized_hidden + self.nominal_summary_bias_vector

    def observation_latent_observation(self, observation_features: np.ndarray) -> np.ndarray:
        standardized_features = (
            np.asarray(observation_features, dtype=np.float64) - self.observation_feature_mean
        ) / self.observation_feature_std
        observation_latent = (
            self.observation_weight_matrix @ standardized_features + self.observation_bias_vector
        )
        full_latent = np.zeros(len(self.axes), dtype=np.float64)
        full_latent[np.asarray(_subspace_indices(self.axes, _observation_axes()), dtype=np.int64)] = observation_latent
        return full_latent

    def load_latent_observation(self, load_features: np.ndarray) -> np.ndarray:
        full_latent = np.zeros(len(self.axes), dtype=np.float64)
        standardized_features = (
            np.asarray(load_features, dtype=np.float64) - self.load_feature_mean
        ) / self.load_feature_std
        full_latent[self.axes.index(SemanticAxis.LOAD)] = float(
            self.load_weight_vector @ standardized_features + self.load_bias
        )
        return full_latent

    def dynamics_latent_observation(self, dynamics_features: np.ndarray) -> np.ndarray:
        full_latent = np.zeros(len(self.axes), dtype=np.float64)
        standardized_features = (
            np.asarray(dynamics_features, dtype=np.float64) - self.dynamics_feature_mean
        ) / self.dynamics_feature_std
        full_latent[self.axes.index(SemanticAxis.DYNAMICS)] = float(
            self.dynamics_weight_vector @ standardized_features + self.dynamics_bias
        )
        return full_latent

    def readout_latent_observation(self, readout_features: np.ndarray) -> np.ndarray:
        full_latent = np.zeros(len(self.axes), dtype=np.float64)
        standardized_features = (
            np.asarray(readout_features, dtype=np.float64) - self.readout_feature_mean
        ) / self.readout_feature_std
        full_latent[self.axes.index(SemanticAxis.READOUT)] = float(
            self.readout_weight_vector @ standardized_features + self.readout_bias
        )
        return full_latent

    def as_dict(self) -> dict[str, object]:
        standardized_support = self.standardize_latent(self.support_latents)
        latent_covariance = (
            np.cov(standardized_support, rowvar=False)
            if standardized_support.shape[0] > 1
            else np.eye(len(self.axes), dtype=np.float64)
        )
        return {
            "axes": [axis.value for axis in self.axes],
            "num_train_episode_latents": int(self.train_episode_latents.shape[0]),
            "num_anchor_latents": int(self.anchor_latents.shape[0]),
            "observation_feature_names": list(self.observation_feature_names),
            "summary_feature_names": list(self.summary_feature_names),
            "load_feature_names": list(self.load_feature_names),
            "dynamics_feature_names": list(self.dynamics_feature_names),
            "readout_feature_names": list(self.readout_feature_names),
            "latent_support_mean": np.asarray(standardized_support.mean(axis=0), dtype=np.float64).tolist(),
            "latent_support_covariance": np.asarray(latent_covariance, dtype=np.float64).tolist(),
        }


@dataclass(frozen=True)
class LatentPrior:
    mean: np.ndarray
    covariance: np.ndarray
    axis_semantics: dict[str, dict[str, object]]
    reliability: dict[str, object]
    global_uncertainty: float
    summary: str
    raw_response_text: str
    api_reliable: bool
    error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64).tolist(),
            "covariance_diag": np.asarray(np.diag(self.covariance), dtype=np.float64).tolist(),
            "axis_semantics": self.axis_semantics,
            "reliability": self.reliability,
            "global_uncertainty": self.global_uncertainty,
            "summary": self.summary,
            "raw_response_text": self.raw_response_text,
            "api_reliable": self.api_reliable,
            "error": self.error,
        }


@dataclass(frozen=True)
class LatentPosterior:
    mean: np.ndarray
    covariance: np.ndarray
    posterior_trace: float
    posterior_trace_obs: float
    posterior_trace_proc: float
    support_distance: float
    support_distance_obs: float
    support_distance_proc: float
    gamma: float
    gamma_obs: float
    gamma_proc: float
    abstain: bool
    dominant_obs_axis: str
    dominant_proc_axis: str
    num_updates: int

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": np.asarray(self.mean, dtype=np.float64).tolist(),
            "covariance_diag": np.asarray(np.diag(self.covariance), dtype=np.float64).tolist(),
            "posterior_trace": self.posterior_trace,
            "posterior_trace_obs": self.posterior_trace_obs,
            "posterior_trace_proc": self.posterior_trace_proc,
            "support_distance": self.support_distance,
            "support_distance_obs": self.support_distance_obs,
            "support_distance_proc": self.support_distance_proc,
            "gamma": self.gamma,
            "gamma_obs": self.gamma_obs,
            "gamma_proc": self.gamma_proc,
            "abstain": self.abstain,
            "dominant_obs_axis": self.dominant_obs_axis,
            "dominant_proc_axis": self.dominant_proc_axis,
            "num_updates": self.num_updates,
        }


@dataclass(frozen=True)
class LatentPredictionRecord:
    split_name: str
    scenario_name: str
    scenario_zh_name: str
    domain_bucket: str
    episode_index: int
    target_minute: int
    target_time_h: float | None
    baseline_prediction: float
    adapted_prediction: float
    target_value: float
    posterior_mean: np.ndarray
    posterior_trace: float
    posterior_trace_obs: float
    posterior_trace_proc: float
    support_distance: float
    support_distance_obs: float
    support_distance_proc: float
    gamma: float
    gamma_obs: float
    gamma_proc: float
    trust_score: float
    abstain: bool
    dominant_obs_axis: str
    dominant_proc_axis: str

    def as_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "scenario_name": self.scenario_name,
            "scenario_zh_name": self.scenario_zh_name,
            "domain_bucket": self.domain_bucket,
            "episode_index": self.episode_index,
            "target_minute": self.target_minute,
            "target_time_h": self.target_time_h,
            "baseline_prediction": self.baseline_prediction,
            "adapted_prediction": self.adapted_prediction,
            "target_value": self.target_value,
            "posterior_mean": np.asarray(self.posterior_mean, dtype=np.float64).tolist(),
            "posterior_trace": self.posterior_trace,
            "posterior_trace_obs": self.posterior_trace_obs,
            "posterior_trace_proc": self.posterior_trace_proc,
            "support_distance": self.support_distance,
            "support_distance_obs": self.support_distance_obs,
            "support_distance_proc": self.support_distance_proc,
            "gamma": self.gamma,
            "gamma_obs": self.gamma_obs,
            "gamma_proc": self.gamma_proc,
            "trust_score": self.trust_score,
            "abstain": self.abstain,
            "dominant_obs_axis": self.dominant_obs_axis,
            "dominant_proc_axis": self.dominant_proc_axis,
        }


@dataclass(frozen=True)
class LatentEpisodeTrace:
    split_name: str
    scenario_name: str
    scenario_zh_name: str
    domain_bucket: str
    episode_index: int
    evidence: dict[str, str]
    initial_diagnostics: dict[str, object]
    raw_prior_response: str
    prior: dict[str, object]
    reliability: dict[str, object]
    final_posterior: dict[str, object]
    dominant_prior_axis: str
    dominant_posterior_axis: str
    dominant_obs_axis: str
    dominant_proc_axis: str
    avg_gamma: float
    avg_posterior_trace: float
    avg_posterior_trace_obs: float
    avg_posterior_trace_proc: float
    avg_support_distance: float
    avg_support_distance_obs: float
    avg_support_distance_proc: float
    abstain_count: int
    num_samples: int
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "split_name": self.split_name,
            "scenario_name": self.scenario_name,
            "scenario_zh_name": self.scenario_zh_name,
            "domain_bucket": self.domain_bucket,
            "episode_index": self.episode_index,
            "evidence": self.evidence,
            "initial_diagnostics": self.initial_diagnostics,
            "raw_prior_response": self.raw_prior_response,
            "prior": self.prior,
            "reliability": self.reliability,
            "final_posterior": self.final_posterior,
            "dominant_prior_axis": self.dominant_prior_axis,
            "dominant_posterior_axis": self.dominant_posterior_axis,
            "dominant_obs_axis": self.dominant_obs_axis,
            "dominant_proc_axis": self.dominant_proc_axis,
            "avg_gamma": self.avg_gamma,
            "avg_posterior_trace": self.avg_posterior_trace,
            "avg_posterior_trace_obs": self.avg_posterior_trace_obs,
            "avg_posterior_trace_proc": self.avg_posterior_trace_proc,
            "avg_support_distance": self.avg_support_distance,
            "avg_support_distance_obs": self.avg_support_distance_obs,
            "avg_support_distance_proc": self.avg_support_distance_proc,
            "abstain_count": self.abstain_count,
            "num_samples": self.num_samples,
            "error": self.error,
        }


@dataclass(frozen=True)
class RoamEpisodeResult:
    sample_records: tuple[LatentPredictionRecord, ...]
    decision_trace: LatentEpisodeTrace


@dataclass(frozen=True)
class _EpisodeResidualSeries:
    normalized_hidden: np.ndarray
    residuals: np.ndarray
    observation_features: np.ndarray
    summary_features: np.ndarray
    load_context_features: np.ndarray
    readout_context_features: np.ndarray


@dataclass(frozen=True)
class _EpisodeBootstrap:
    supervised_indices: np.ndarray
    prior_input: PriorEngineInput
    initial_diagnostics: ModelDiagnostics


def _round_for_prior(value: float) -> float:
    return round(float(value), 4)


def _top_channels(mapping: dict[str, float], *, limit: int = 3) -> list[dict[str, float | str]]:
    ranked = sorted(mapping.items(), key=lambda item: abs(float(item[1])), reverse=True)[:limit]
    return [
        {"name": channel_name, "value": _round_for_prior(channel_value)}
        for channel_name, channel_value in ranked
    ]


def _prediction_range_penalty(prediction: float, reference_stats: SequenceReferenceStats) -> float:
    target_mean = float(reference_stats.target_mean[0])
    target_std = float(reference_stats.target_std[0])
    lower_bound = target_mean - 3.0 * target_std
    upper_bound = target_mean + 3.0 * target_std
    if lower_bound <= prediction <= upper_bound:
        return 0.0
    distance = lower_bound - prediction if prediction < lower_bound else prediction - upper_bound
    return float(distance / max(target_std, 1e-6))


def _phase_mean_delta(values: np.ndarray, phase_mask: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    aligned_mask = np.asarray(phase_mask[1:] > 0.5, dtype=bool)
    selected = deltas[aligned_mask]
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _subspace_indices(
    axes: tuple[SemanticAxis, ...],
    subspace_axes: tuple[SemanticAxis, ...],
) -> tuple[int, ...]:
    return tuple(axes.index(axis) for axis in subspace_axes if axis in axes)


def _summary_feature_names() -> tuple[str, ...]:
    return (
        "pressurizing_ratio",
        "phase_switch_rate",
        "p3_delta",
        "p2_delta",
        "p3_pressurizing_slope",
        "p3_discharging_slope",
        "cycle_duration_proxy",
    )


def _summary_metric(diagnostics: ModelDiagnostics, key: str) -> float:
    return float(diagnostics.summary_metrics.get(key, 0.0))


def _dynamics_feature_names() -> tuple[str, ...]:
    return (
        "phase_switch_rate_residual",
        "p3_delta_residual",
        "p2_delta_residual",
        "p3_pressurizing_slope_residual",
        "p3_discharging_slope_residual",
        "cycle_duration_proxy_residual",
    )


def _load_feature_names() -> tuple[str, ...]:
    return (
        "residual_mean",
        "q_in_level_z",
        "p2_level_z",
        "p3_level_z",
        "hidden_drift_score",
        "pressurizing_ratio",
    )


def _readout_feature_names() -> tuple[str, ...]:
    return (
        "residual_mean",
        "residual_abs_mean",
        "residual_q_in_coupling",
        "residual_p3_coupling",
        "prediction_zscore",
    )


def _observation_feature_names() -> tuple[str, ...]:
    return (
        "prediction_zscore",
        "feature_drift_score",
        "hidden_drift_score",
        "sensitivity_shift_score",
        "range_anomaly_score",
        "anomaly_score",
        "q_in_drift",
        "p2_drift",
        "p3_drift",
        "q_in_sensitivity_delta",
        "p2_sensitivity_delta",
        "p3_sensitivity_delta",
    )


def _summary_feature_vector(diagnostics: ModelDiagnostics) -> np.ndarray:
    return np.asarray(
        (
            _summary_metric(diagnostics, "pressurizing_ratio"),
            _summary_metric(diagnostics, "phase_switch_rate"),
            _summary_metric(diagnostics, "p3_delta"),
            _summary_metric(diagnostics, "p2_delta"),
            _summary_metric(diagnostics, "p3_pressurizing_slope"),
            _summary_metric(diagnostics, "p3_discharging_slope"),
            _summary_metric(diagnostics, "cycle_duration_proxy"),
        ),
        dtype=np.float64,
    )


def _level_z(feature_window: np.ndarray, *, feature_index: int, reference_stats: SequenceReferenceStats) -> float:
    return float(
        (
            float(np.asarray(feature_window, dtype=np.float64)[:, feature_index].mean())
            - float(reference_stats.feature_mean[feature_index])
        )
        / max(float(reference_stats.feature_std[feature_index]), 1e-6)
    )


def _load_context_vector(
    diagnostics: ModelDiagnostics,
    feature_window: np.ndarray,
    reference_stats: SequenceReferenceStats,
    *,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    q_in_index = feature_names.index("q_in")
    p2_index = feature_names.index("p2")
    p3_index = feature_names.index("p3")
    return np.asarray(
        (
            _level_z(feature_window, feature_index=q_in_index, reference_stats=reference_stats),
            _level_z(feature_window, feature_index=p2_index, reference_stats=reference_stats),
            _level_z(feature_window, feature_index=p3_index, reference_stats=reference_stats),
            diagnostics.hidden_drift_score,
            _summary_metric(diagnostics, "pressurizing_ratio"),
        ),
        dtype=np.float64,
    )


def _readout_context_vector(
    diagnostics: ModelDiagnostics,
    feature_window: np.ndarray,
    reference_stats: SequenceReferenceStats,
    *,
    feature_names: tuple[str, ...],
) -> np.ndarray:
    q_in_index = feature_names.index("q_in")
    p3_index = feature_names.index("p3")
    return np.asarray(
        (
            _level_z(feature_window, feature_index=q_in_index, reference_stats=reference_stats),
            _level_z(feature_window, feature_index=p3_index, reference_stats=reference_stats),
            diagnostics.prediction_zscore,
        ),
        dtype=np.float64,
    )


def _load_feature_vector_from_context(load_context: np.ndarray, *, residual_mean: float) -> np.ndarray:
    context_array = np.asarray(load_context, dtype=np.float64)
    return np.asarray(
        (
            residual_mean,
            context_array[0],
            context_array[1],
            context_array[2],
            context_array[3],
            context_array[4],
        ),
        dtype=np.float64,
    )


def _dynamics_feature_vector(summary_residual: np.ndarray) -> np.ndarray:
    summary_array = np.asarray(summary_residual, dtype=np.float64)
    return np.asarray(summary_array[1:], dtype=np.float64)


def _readout_feature_vector_from_context(readout_context: np.ndarray, *, residual: float) -> np.ndarray:
    context_array = np.asarray(readout_context, dtype=np.float64)
    q_in_level_z = float(context_array[0])
    p3_level_z = float(context_array[1])
    prediction_zscore = float(context_array[2])
    return np.asarray(
        (
            residual,
            abs(residual),
            residual * q_in_level_z,
            residual * p3_level_z,
            prediction_zscore,
        ),
        dtype=np.float64,
    )


def _observation_feature_vector(diagnostics: ModelDiagnostics) -> np.ndarray:
    return np.asarray(
        (
            diagnostics.prediction_zscore,
            diagnostics.feature_drift_score,
            diagnostics.hidden_drift_score,
            diagnostics.sensitivity_shift_score,
            diagnostics.range_anomaly_score,
            diagnostics.anomaly_score,
            float(diagnostics.feature_drift_by_channel.get("q_in", 0.0)),
            float(diagnostics.feature_drift_by_channel.get("p2", 0.0)),
            float(diagnostics.feature_drift_by_channel.get("p3", 0.0)),
            float(diagnostics.channel_sensitivity_delta.get("q_in", 0.0)),
            float(diagnostics.channel_sensitivity_delta.get("p2", 0.0)),
            float(diagnostics.channel_sensitivity_delta.get("p3", 0.0)),
        ),
        dtype=np.float64,
    )


def _diagnostic_score_components(
    *,
    prediction_zscore: float,
    feature_drift_score: float,
    hidden_drift_score: float,
    sensitivity_shift_score: float,
    range_anomaly_score: float,
) -> float:
    return float(
        0.30 * prediction_zscore
        + 0.20 * feature_drift_score
        + 0.25 * hidden_drift_score
        + 0.15 * sensitivity_shift_score
        + 0.25 * range_anomaly_score
    )


def _baseline_prediction(model: SequenceRegressorProtocol, window: np.ndarray) -> float:
    return float(model.predict(window[None, :, :])[0, 0])


def _baseline_hidden_state(model: SequenceRegressorProtocol, window: np.ndarray) -> np.ndarray:
    return np.asarray(model.encode_hidden(window[None, :, :])[0], dtype=np.float64)


def build_model_diagnostics(
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    feature_window: np.ndarray,
    *,
    feature_names: tuple[str, ...],
) -> ModelDiagnostics:
    prediction = _baseline_prediction(model, feature_window)
    hidden_state = _baseline_hidden_state(model, feature_window)

    feature_mean = np.asarray(feature_window.mean(axis=0), dtype=np.float64)
    feature_drift_array = np.abs(feature_mean - reference_stats.feature_mean) / reference_stats.feature_std
    feature_drift_by_channel = {
        feature_name: float(feature_drift_array[index])
        for index, feature_name in enumerate(feature_names)
    }
    feature_drift_score = float(np.mean(feature_drift_array))

    hidden_drift_array = np.abs(hidden_state - reference_stats.hidden_mean) / reference_stats.hidden_std
    hidden_drift_score = float(np.mean(hidden_drift_array))

    channel_sensitivity: dict[str, float] = {}
    channel_sensitivity_delta: dict[str, float] = {}
    for feature_index, feature_name in enumerate(feature_names):
        masked_window = np.asarray(feature_window, dtype=np.float64).copy()
        masked_window[:, feature_index] = reference_stats.feature_mean[feature_index]
        masked_prediction = _baseline_prediction(model, masked_window)
        sensitivity = abs(masked_prediction - prediction)
        channel_sensitivity[feature_name] = float(sensitivity)
        baseline_sensitivity = float(reference_stats.feature_sensitivity_baseline[feature_index])
        channel_sensitivity_delta[feature_name] = float(
            abs(sensitivity - baseline_sensitivity) / max(baseline_sensitivity, 1e-6)
        )

    sensitivity_shift_score = float(np.mean(list(channel_sensitivity_delta.values())))
    prediction_zscore = float(
        abs(prediction - float(reference_stats.prediction_mean[0])) / max(float(reference_stats.prediction_std[0]), 1e-6)
    )
    range_anomaly_score = _prediction_range_penalty(prediction, reference_stats)
    phase_pressurizing = np.asarray(feature_window[:, feature_names.index("phase_pressurizing")], dtype=np.float64)
    phase_switch_rate = float(np.mean(np.abs(np.diff(phase_pressurizing)))) if phase_pressurizing.size > 1 else 0.0
    p3_series = np.asarray(feature_window[:, feature_names.index("p3")], dtype=np.float64)
    p2_series = np.asarray(feature_window[:, feature_names.index("p2")], dtype=np.float64)
    pressurizing_ratio = float(np.mean(phase_pressurizing))
    p3_delta = float(p3_series[-1] - p3_series[0]) if p3_series.size else 0.0
    p2_delta = float(p2_series[-1] - p2_series[0]) if p2_series.size else 0.0
    p3_pressurizing_slope = _phase_mean_delta(p3_series, phase_pressurizing)
    p3_discharging_slope = _phase_mean_delta(
        p3_series,
        np.asarray(feature_window[:, feature_names.index("phase_discharging")], dtype=np.float64),
    )
    cycle_duration_proxy = float(1.0 / max(phase_switch_rate, 1e-3))
    anomaly_score = _diagnostic_score_components(
        prediction_zscore=prediction_zscore,
        feature_drift_score=feature_drift_score,
        hidden_drift_score=hidden_drift_score,
        sensitivity_shift_score=sensitivity_shift_score,
        range_anomaly_score=range_anomaly_score,
    )
    summary_metrics = {
        "pressurizing_ratio": float(pressurizing_ratio),
        "phase_switch_rate": float(phase_switch_rate),
        "p3_delta": float(p3_delta),
        "p2_delta": float(p2_delta),
        "p3_pressurizing_slope": float(p3_pressurizing_slope),
        "p3_discharging_slope": float(p3_discharging_slope),
        "cycle_duration_proxy": float(cycle_duration_proxy),
    }

    return ModelDiagnostics(
        prediction=prediction,
        prediction_zscore=prediction_zscore,
        feature_drift_score=feature_drift_score,
        hidden_drift_score=hidden_drift_score,
        sensitivity_shift_score=sensitivity_shift_score,
        range_anomaly_score=range_anomaly_score,
        anomaly_score=anomaly_score,
        feature_drift_by_channel=feature_drift_by_channel,
        channel_sensitivity=channel_sensitivity,
        channel_sensitivity_delta=channel_sensitivity_delta,
        hidden_state_norm=float(np.linalg.norm(hidden_state)),
        summary_metrics=summary_metrics,
        summary_text=(
            "prediction={prediction:.6f}, prediction_zscore={prediction_zscore:.3f}, "
            "feature_drift={feature_drift_score:.3f}, hidden_drift={hidden_drift_score:.3f}, "
            "sensitivity_shift={sensitivity_shift_score:.3f}, range_anomaly={range_anomaly_score:.3f}, "
            "anomaly_score={anomaly_score:.3f}, press_ratio={pressurizing_ratio:.3f}, "
            "phase_switch_rate={phase_switch_rate:.3f}, p3_delta={p3_delta:.6f}, "
            "p3_discharge_slope={p3_discharging_slope:.6f}, cycle_duration={cycle_duration_proxy:.3f}"
        ).format(
            prediction=prediction,
            prediction_zscore=prediction_zscore,
            feature_drift_score=feature_drift_score,
            hidden_drift_score=hidden_drift_score,
            sensitivity_shift_score=sensitivity_shift_score,
            range_anomaly_score=range_anomaly_score,
            anomaly_score=anomaly_score,
            pressurizing_ratio=summary_metrics["pressurizing_ratio"],
            phase_switch_rate=summary_metrics["phase_switch_rate"],
            p3_delta=summary_metrics["p3_delta"],
            p3_discharging_slope=summary_metrics["p3_discharging_slope"],
            cycle_duration_proxy=summary_metrics["cycle_duration_proxy"],
        ),
    )


def build_prior_diagnostics_payload(diagnostics: ModelDiagnostics) -> dict[str, object]:
    rounded_feature_drift_by_channel = {
        channel_name: _round_for_prior(channel_value)
        for channel_name, channel_value in diagnostics.feature_drift_by_channel.items()
    }
    rounded_channel_sensitivity = {
        channel_name: _round_for_prior(channel_value)
        for channel_name, channel_value in diagnostics.channel_sensitivity.items()
    }
    rounded_channel_sensitivity_delta = {
        channel_name: _round_for_prior(channel_value)
        for channel_name, channel_value in diagnostics.channel_sensitivity_delta.items()
    }
    return {
        "prediction": _round_for_prior(diagnostics.prediction),
        "prediction_zscore": _round_for_prior(diagnostics.prediction_zscore),
        "feature_drift_score": _round_for_prior(diagnostics.feature_drift_score),
        "hidden_drift_score": _round_for_prior(diagnostics.hidden_drift_score),
        "sensitivity_shift_score": _round_for_prior(diagnostics.sensitivity_shift_score),
        "range_anomaly_score": _round_for_prior(diagnostics.range_anomaly_score),
        "anomaly_score": _round_for_prior(diagnostics.anomaly_score),
        "hidden_state_norm": _round_for_prior(diagnostics.hidden_state_norm),
        "pressurizing_ratio": _round_for_prior(_summary_metric(diagnostics, "pressurizing_ratio")),
        "phase_switch_rate": _round_for_prior(_summary_metric(diagnostics, "phase_switch_rate")),
        "p3_delta": _round_for_prior(_summary_metric(diagnostics, "p3_delta")),
        "p2_delta": _round_for_prior(_summary_metric(diagnostics, "p2_delta")),
        "p3_pressurizing_slope": _round_for_prior(_summary_metric(diagnostics, "p3_pressurizing_slope")),
        "p3_discharging_slope": _round_for_prior(_summary_metric(diagnostics, "p3_discharging_slope")),
        "cycle_duration_proxy": _round_for_prior(_summary_metric(diagnostics, "cycle_duration_proxy")),
        "summary_metrics": {
            metric_name: _round_for_prior(metric_value)
            for metric_name, metric_value in diagnostics.summary_metrics.items()
        },
        "feature_drift_by_channel": rounded_feature_drift_by_channel,
        "channel_sensitivity": rounded_channel_sensitivity,
        "channel_sensitivity_delta": rounded_channel_sensitivity_delta,
        "top_feature_drift_channels": _top_channels(diagnostics.feature_drift_by_channel),
        "top_channel_sensitivity": _top_channels(diagnostics.channel_sensitivity),
        "top_channel_sensitivity_delta": _top_channels(diagnostics.channel_sensitivity_delta),
    }


def _solve_ridge_regression(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    xtx = design_matrix.T @ design_matrix
    xty = design_matrix.T @ targets
    regularized = xtx + ridge_lambda * np.eye(design_matrix.shape[1], dtype=np.float64)
    return np.linalg.solve(regularized, xty)


def _solve_multitarget_ridge_regression(
    design_matrix: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float,
) -> np.ndarray:
    xtx = design_matrix.T @ design_matrix
    xty = design_matrix.T @ targets
    regularized = xtx + ridge_lambda * np.eye(design_matrix.shape[1], dtype=np.float64)
    return np.linalg.solve(regularized, xty)


def _axis_anchor_latents(axes: tuple[SemanticAxis, ...]) -> np.ndarray:
    support_raw = [np.zeros(len(axes), dtype=np.float64)]
    for axis_index in range(len(axes)):
        positive = np.zeros(len(axes), dtype=np.float64)
        negative = np.zeros(len(axes), dtype=np.float64)
        positive[axis_index] = 1.0
        negative[axis_index] = -1.0
        support_raw.extend((positive, negative))
    return np.asarray(support_raw, dtype=np.float64)


def _weighted_examples(
    design_rows: list[np.ndarray],
    targets: list[float],
    *,
    weight: float,
) -> tuple[list[np.ndarray], list[float]]:
    sqrt_weight = float(np.sqrt(weight))
    return (
        [sqrt_weight * np.asarray(row, dtype=np.float64) for row in design_rows],
        [sqrt_weight * float(target) for target in targets],
    )


def _build_real_episode_series(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    episode_grids: list,
    window_size_minutes: int,
    task_adapter: RoamTaskAdapter,
) -> list[_EpisodeResidualSeries]:
    adapter = task_adapter
    hidden_mean = np.asarray(reference_stats.hidden_mean, dtype=np.float64)
    hidden_std = np.asarray(reference_stats.hidden_std, dtype=np.float64)
    episode_series: list[_EpisodeResidualSeries] = []
    for episode_grid in episode_grids:
        normalized_hidden_rows: list[np.ndarray] = []
        residuals: list[float] = []
        observation_features: list[np.ndarray] = []
        summary_features: list[np.ndarray] = []
        load_context_features: list[np.ndarray] = []
        readout_context_features: list[np.ndarray] = []
        for target_index in adapter.supervised_indices(episode_grid, window_size=window_size_minutes).tolist():
            start_minute = target_index - window_size_minutes + 1
            feature_window = np.asarray(
                episode_grid.feature_matrix[start_minute : target_index + 1],
                dtype=np.float64,
            )
            hidden_state = _baseline_hidden_state(model, feature_window)
            normalized_hidden = (hidden_state - hidden_mean) / hidden_std
            prediction = _baseline_prediction(model, feature_window)
            residual = float(adapter.target_value(episode_grid, target_index=target_index) - prediction)
            diagnostics = adapter.build_diagnostics(model, reference_stats, feature_window)
            normalized_hidden_rows.append(np.asarray(normalized_hidden, dtype=np.float64))
            residuals.append(residual)
            observation_features.append(adapter.observation_feature_vector(diagnostics))
            summary_features.append(adapter.summary_feature_vector(diagnostics))
            load_context_features.append(adapter.load_context_vector(diagnostics, feature_window, reference_stats))
            readout_context_features.append(
                adapter.readout_context_vector(diagnostics, feature_window, reference_stats)
            )
        episode_series.append(
            _EpisodeResidualSeries(
                normalized_hidden=np.asarray(normalized_hidden_rows, dtype=np.float64),
                residuals=np.asarray(residuals, dtype=np.float64),
                observation_features=np.asarray(observation_features, dtype=np.float64),
                summary_features=np.asarray(summary_features, dtype=np.float64),
                load_context_features=np.asarray(load_context_features, dtype=np.float64),
                readout_context_features=np.asarray(readout_context_features, dtype=np.float64),
            )
        )
    return episode_series


def fit_latent_adapter(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    train_dataset: SequenceRegressionDataset,
    config: RoamLatentConfig,
    task_adapter: RoamTaskAdapter,
    train_episode_grids: list | None = None,
) -> LatentAdapterModel:
    config.validate()
    adapter = task_adapter
    features = np.asarray(train_dataset.features, dtype=np.float64)
    targets = np.asarray(train_dataset.targets[:, 0], dtype=np.float64)
    baseline_predictions = np.asarray(model.predict(features)[:, 0], dtype=np.float64)
    base_hidden = np.asarray(model.encode_hidden(features), dtype=np.float64)
    hidden_dim = base_hidden.shape[1]
    latent_dim = len(config.axes)
    obs_indices = _subspace_indices(config.axes, _observation_axes())

    target_std = float(reference_stats.target_std[0])
    normalized_hidden_mean = np.asarray(reference_stats.hidden_mean, dtype=np.float64)
    normalized_hidden_std = np.asarray(reference_stats.hidden_std, dtype=np.float64)

    load_axis_index = config.axes.index(SemanticAxis.LOAD)
    dynamics_axis_index = config.axes.index(SemanticAxis.DYNAMICS)
    readout_axis_index = config.axes.index(SemanticAxis.READOUT)
    reliability_delay_steps = int(adapter.reliability_delay_steps)

    anchor_design_rows: list[np.ndarray] = []
    anchor_residual_targets: list[float] = []
    base_hidden_norm_rows: list[np.ndarray] = []
    base_observation_features: list[np.ndarray] = []
    base_summary_features: list[np.ndarray] = []
    observation_anchor_features: list[np.ndarray] = []
    observation_anchor_targets: list[np.ndarray] = []
    reliability_anchor_features: list[np.ndarray] = []
    reliability_anchor_targets: list[float] = []
    dynamics_anchor_hidden_rows: list[np.ndarray] = []
    dynamics_anchor_summary_rows: list[np.ndarray] = []
    dynamics_anchor_targets: list[float] = []
    readout_anchor_features: list[np.ndarray] = []
    readout_anchor_targets: list[float] = []
    load_anchor_features: list[np.ndarray] = []
    load_anchor_targets: list[float] = []
    episode_series: list[_EpisodeResidualSeries] = []

    def _obs_target(*, bias: float = 0.0, scale: float = 0.0) -> np.ndarray:
        return np.asarray((bias, scale), dtype=np.float64)

    def _standardize_features(
        feature_rows: list[np.ndarray],
        *,
        feature_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if feature_rows:
            feature_matrix = np.asarray(feature_rows, dtype=np.float64)
        else:
            feature_matrix = np.zeros((1, feature_dim), dtype=np.float64)
        feature_mean = np.asarray(feature_matrix.mean(axis=0), dtype=np.float64)
        feature_std = np.asarray(np.clip(feature_matrix.std(axis=0), 1e-6, None), dtype=np.float64)
        return feature_mean, feature_std

    def _fit_multitarget_head(
        *,
        anchor_features: list[np.ndarray],
        anchor_targets: list[np.ndarray],
        real_features: list[np.ndarray],
        real_targets: list[np.ndarray],
        feature_dim: int,
        target_dim: int,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        anchor_weight: float,
        real_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        if feature_mean is None or feature_std is None:
            feature_mean, feature_std = _standardize_features(
                anchor_features + real_features,
                feature_dim=feature_dim,
            )
        design_rows: list[np.ndarray] = []
        target_rows: list[np.ndarray] = []
        for feature_row, target_row in zip(anchor_features, anchor_targets, strict=True):
            standardized = (np.asarray(feature_row, dtype=np.float64) - feature_mean) / feature_std
            design_rows.append(
                np.concatenate(
                    (
                        np.sqrt(anchor_weight) * standardized,
                        [np.sqrt(anchor_weight)],
                    )
                )
            )
            target_rows.append(np.sqrt(anchor_weight) * np.asarray(target_row, dtype=np.float64))
        for feature_row, target_row in zip(real_features, real_targets, strict=True):
            standardized = (np.asarray(feature_row, dtype=np.float64) - feature_mean) / feature_std
            design_rows.append(
                np.concatenate(
                    (
                        np.sqrt(real_weight) * standardized,
                        [np.sqrt(real_weight)],
                    )
                )
            )
            target_rows.append(np.sqrt(real_weight) * np.asarray(target_row, dtype=np.float64))
        if not design_rows:
            design_rows = [np.concatenate((np.zeros(feature_dim, dtype=np.float64), [1.0]))]
            target_rows = [np.zeros(target_dim, dtype=np.float64)]
        design_matrix = np.asarray(design_rows, dtype=np.float64)
        target_matrix = np.asarray(target_rows, dtype=np.float64)
        solution = _solve_multitarget_ridge_regression(
            design_matrix,
            target_matrix,
            ridge_lambda=config.adapter.ridge_lambda,
        )
        fitted = design_matrix @ solution
        return (
            np.asarray(feature_mean, dtype=np.float64),
            np.asarray(feature_std, dtype=np.float64),
            np.asarray(solution[:-1, :].T, dtype=np.float64),
            np.asarray(solution[-1, :], dtype=np.float64),
            float(np.mean(np.abs(fitted - target_matrix))),
            float(np.sqrt(np.mean(np.square(fitted - target_matrix)))),
        )

    def _fit_scalar_head(
        *,
        anchor_features: list[np.ndarray],
        anchor_targets: list[float],
        real_features: list[np.ndarray],
        real_targets: list[float],
        feature_dim: int,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        anchor_weight: float,
        real_weight: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        if feature_mean is None or feature_std is None:
            feature_mean, feature_std = _standardize_features(
                anchor_features + real_features,
                feature_dim=feature_dim,
            )
        design_rows: list[np.ndarray] = []
        target_values: list[float] = []
        for feature_row, target_value in zip(anchor_features, anchor_targets, strict=True):
            standardized = (np.asarray(feature_row, dtype=np.float64) - feature_mean) / feature_std
            design_rows.append(
                np.concatenate(
                    (
                        np.sqrt(anchor_weight) * standardized,
                        [np.sqrt(anchor_weight)],
                    )
                )
            )
            target_values.append(np.sqrt(anchor_weight) * float(target_value))
        for feature_row, target_value in zip(real_features, real_targets, strict=True):
            standardized = (np.asarray(feature_row, dtype=np.float64) - feature_mean) / feature_std
            design_rows.append(
                np.concatenate(
                    (
                        np.sqrt(real_weight) * standardized,
                        [np.sqrt(real_weight)],
                    )
                )
            )
            target_values.append(np.sqrt(real_weight) * float(target_value))
        if not design_rows:
            design_rows = [np.concatenate((np.zeros(feature_dim, dtype=np.float64), [1.0]))]
            target_values = [0.0]
        design_matrix = np.asarray(design_rows, dtype=np.float64)
        target_array = np.asarray(target_values, dtype=np.float64)
        solution = _solve_ridge_regression(
            design_matrix,
            target_array,
            ridge_lambda=config.adapter.ridge_lambda,
        )
        fitted = design_matrix @ solution
        return (
            np.asarray(feature_mean, dtype=np.float64),
            np.asarray(feature_std, dtype=np.float64),
            np.asarray(solution[:-1], dtype=np.float64),
            float(solution[-1]),
            float(np.mean(np.abs(fitted - target_array))),
            float(np.sqrt(np.mean(np.square(fitted - target_array)))),
        )

    def _fit_raw_basis(
        episode_latents: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        real_design_rows: list[np.ndarray] = []
        real_targets: list[float] = []
        if episode_series and episode_latents.size:
            for episode_latent, series in zip(episode_latents, episode_series, strict=True):
                for normalized_hidden, residual in zip(series.normalized_hidden, series.residuals, strict=True):
                    real_design_rows.append(_build_design_row(normalized_hidden, episode_latent))
                    real_targets.append(float(residual))
        weighted_anchor_rows, weighted_anchor_targets = _weighted_examples(
            anchor_design_rows,
            anchor_residual_targets,
            weight=config.adapter.synthetic_anchor_weight,
        )
        weighted_real_rows, weighted_real_targets = _weighted_examples(
            real_design_rows,
            real_targets,
            weight=config.adapter.real_residual_weight,
        )
        design_matrix = np.asarray(weighted_anchor_rows + weighted_real_rows, dtype=np.float64)
        target_array = np.asarray(weighted_anchor_targets + weighted_real_targets, dtype=np.float64)
        solution = _solve_ridge_regression(
            design_matrix,
            target_array,
            ridge_lambda=config.adapter.ridge_lambda,
        )
        weight_matrix = np.asarray(solution[: hidden_dim * latent_dim].reshape(latent_dim, hidden_dim), dtype=np.float64)
        bias_vector = np.asarray(solution[hidden_dim * latent_dim :], dtype=np.float64)
        if not real_design_rows:
            return weight_matrix, bias_vector, 0.0, 0.0
        real_design_matrix = np.asarray(real_design_rows, dtype=np.float64)
        real_target_array = np.asarray(real_targets, dtype=np.float64)
        real_prediction_array = real_design_matrix @ np.concatenate(
            (weight_matrix.reshape(latent_dim * hidden_dim), bias_vector),
            axis=0,
        )
        return (
            weight_matrix,
            bias_vector,
            float(np.mean(np.abs(real_prediction_array - real_target_array))),
            float(np.sqrt(np.mean(np.square(real_prediction_array - real_target_array)))),
        )

    for feature_window, target_value, baseline_prediction, hidden_state in zip(
        features,
        targets,
        baseline_predictions,
        base_hidden,
        strict=True,
    ):
        base_diagnostics = adapter.build_diagnostics(model, reference_stats, feature_window)
        base_hidden_norm = (np.asarray(hidden_state, dtype=np.float64) - normalized_hidden_mean) / normalized_hidden_std
        base_observation_feature = adapter.observation_feature_vector(base_diagnostics)
        base_summary_feature = adapter.summary_feature_vector(base_diagnostics)
        base_hidden_norm_rows.append(np.asarray(base_hidden_norm, dtype=np.float64))
        base_observation_features.append(base_observation_feature)
        base_summary_features.append(base_summary_feature)

        transformed_windows = np.stack(
            (
                adapter.apply_observation_bias(
                    feature_window,
                    config.adapter.observation_bias_magnitude,
                    reference_stats=reference_stats,
                ),
                adapter.apply_observation_bias(
                    feature_window,
                    -config.adapter.observation_bias_magnitude,
                    reference_stats=reference_stats,
                ),
                adapter.apply_observation_scale(feature_window, 1.0 + config.adapter.observation_scale_delta),
                adapter.apply_observation_scale(feature_window, 1.0 - config.adapter.observation_scale_delta),
                adapter.apply_dynamics_scaling(feature_window, config.adapter.dynamics_slowdown_factor),
                adapter.apply_dynamics_scaling(feature_window, config.adapter.dynamics_speedup_factor),
                adapter.apply_reliability_delay(feature_window, reliability_delay_steps),
            ),
            axis=0,
        )
        transformed_predictions = np.asarray(model.predict(transformed_windows)[:, 0], dtype=np.float64)
        transformed_hidden = np.asarray(model.encode_hidden(transformed_windows), dtype=np.float64)
        transformed_diagnostics = [
            adapter.build_diagnostics(model, reference_stats, transformed_window)
            for transformed_window in transformed_windows
        ]
        transformed_hidden_norms = [
            (np.asarray(hidden_row, dtype=np.float64) - normalized_hidden_mean) / normalized_hidden_std
            for hidden_row in transformed_hidden
        ]
        transformed_observation_features = [
            adapter.observation_feature_vector(diagnostics)
            for diagnostics in transformed_diagnostics
        ]
        transformed_summary_features = [
            adapter.summary_feature_vector(diagnostics)
            for diagnostics in transformed_diagnostics
        ]

        synthetic_specs = (
            (SemanticAxis.BIAS, 1.0, transformed_hidden_norms[0], float(target_value - transformed_predictions[0])),
            (SemanticAxis.BIAS, -1.0, transformed_hidden_norms[1], float(target_value - transformed_predictions[1])),
            (SemanticAxis.SCALE, 1.0, transformed_hidden_norms[2], float(target_value - transformed_predictions[2])),
            (SemanticAxis.SCALE, -1.0, transformed_hidden_norms[3], float(target_value - transformed_predictions[3])),
            (
                SemanticAxis.DYNAMICS,
                -1.0,
                transformed_hidden_norms[4],
                float(target_value - transformed_predictions[4]),
            ),
            (
                SemanticAxis.DYNAMICS,
                1.0,
                transformed_hidden_norms[5],
                float(target_value - transformed_predictions[5]),
            ),
            (SemanticAxis.READOUT, 1.0, base_hidden_norm, float(target_std)),
            (SemanticAxis.READOUT, -1.0, base_hidden_norm, float(-target_std)),
        )
        for axis, value, normalized_hidden, residual_target in synthetic_specs:
            anchor_design_rows.append(
                _build_design_row(normalized_hidden, _raw_axis_vector(config.axes, axis, value))
            )
            anchor_residual_targets.append(float(residual_target))

        observation_anchor_features.append(base_observation_feature)
        observation_anchor_targets.append(_obs_target())
        observation_anchor_features.append(transformed_observation_features[0])
        observation_anchor_targets.append(_obs_target(bias=1.0))
        observation_anchor_features.append(transformed_observation_features[1])
        observation_anchor_targets.append(_obs_target(bias=-1.0))
        observation_anchor_features.append(transformed_observation_features[2])
        observation_anchor_targets.append(_obs_target(scale=1.0))
        observation_anchor_features.append(transformed_observation_features[3])
        observation_anchor_targets.append(_obs_target(scale=-1.0))
        observation_anchor_features.append(transformed_observation_features[4])
        observation_anchor_targets.append(_obs_target())
        observation_anchor_features.append(transformed_observation_features[5])
        observation_anchor_targets.append(_obs_target())
        observation_anchor_features.append(transformed_observation_features[6])
        observation_anchor_targets.append(_obs_target())

        reliability_anchor_features.append(base_observation_feature)
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[0])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[1])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[2])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[3])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[4])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[5])
        reliability_anchor_targets.append(0.0)
        reliability_anchor_features.append(transformed_observation_features[6])
        reliability_anchor_targets.append(1.0)

        dynamics_anchor_hidden_rows.append(base_hidden_norm)
        dynamics_anchor_summary_rows.append(base_summary_feature)
        dynamics_anchor_targets.append(0.0)
        dynamics_anchor_hidden_rows.append(np.asarray(transformed_hidden_norms[4], dtype=np.float64))
        dynamics_anchor_summary_rows.append(transformed_summary_features[4])
        dynamics_anchor_targets.append(-1.0)
        dynamics_anchor_hidden_rows.append(np.asarray(transformed_hidden_norms[5], dtype=np.float64))
        dynamics_anchor_summary_rows.append(transformed_summary_features[5])
        dynamics_anchor_targets.append(1.0)

        load_anchor_features.append(
            adapter.load_feature_vector_from_context(
                adapter.load_context_vector(base_diagnostics, feature_window, reference_stats),
                residual=0.0,
            )
        )
        load_anchor_targets.append(0.0)

        readout_anchor_features.append(
            adapter.readout_feature_vector_from_context(
                adapter.readout_context_vector(base_diagnostics, feature_window, reference_stats),
                residual=0.0,
            )
        )
        readout_anchor_targets.append(0.0)
        readout_anchor_features.append(
            adapter.readout_feature_vector_from_context(
                adapter.readout_context_vector(base_diagnostics, feature_window, reference_stats),
                residual=target_std,
            )
        )
        readout_anchor_targets.append(1.0)
        readout_anchor_features.append(
            adapter.readout_feature_vector_from_context(
                adapter.readout_context_vector(base_diagnostics, feature_window, reference_stats),
                residual=-target_std,
            )
        )
        readout_anchor_targets.append(-1.0)

    anchor_latents_raw = _axis_anchor_latents(config.axes)
    raw_weight_matrix, raw_bias_vector, real_fit_mae, real_fit_rmse = _fit_raw_basis(
        np.zeros((0, latent_dim), dtype=np.float64)
    )

    base_hidden_norm_matrix = np.asarray(base_hidden_norm_rows, dtype=np.float64)
    base_summary_matrix = np.asarray(base_summary_features, dtype=np.float64)
    nominal_summary_design = np.hstack(
        (
            np.sqrt(config.adapter.nominal_summary_weight) * base_hidden_norm_matrix,
            np.sqrt(config.adapter.nominal_summary_weight)
            * np.ones((base_hidden_norm_matrix.shape[0], 1), dtype=np.float64),
        )
    )
    nominal_summary_targets = np.sqrt(config.adapter.nominal_summary_weight) * base_summary_matrix
    nominal_summary_solution = _solve_multitarget_ridge_regression(
        nominal_summary_design,
        nominal_summary_targets,
        ridge_lambda=config.adapter.ridge_lambda,
    )
    nominal_summary_weight_matrix = np.asarray(nominal_summary_solution[:-1, :].T, dtype=np.float64)
    nominal_summary_bias_vector = np.asarray(nominal_summary_solution[-1, :], dtype=np.float64)

    def _predict_nominal_summary(normalized_hidden: np.ndarray) -> np.ndarray:
        return nominal_summary_weight_matrix @ np.asarray(normalized_hidden, dtype=np.float64) + nominal_summary_bias_vector

    observation_real_features: list[np.ndarray] = []
    observation_real_targets: list[np.ndarray] = []
    reliability_real_features: list[np.ndarray] = []
    reliability_real_targets: list[float] = []
    dynamics_real_features: list[np.ndarray] = []
    dynamics_real_targets: list[float] = []
    readout_real_features: list[np.ndarray] = []
    readout_real_targets: list[float] = []
    load_real_features: list[np.ndarray] = []
    load_real_targets: list[float] = []

    observation_feature_mean = np.zeros(len(adapter.observation_feature_names), dtype=np.float64)
    observation_feature_std = np.ones(len(adapter.observation_feature_names), dtype=np.float64)
    observation_weight_matrix = np.zeros((len(obs_indices), len(adapter.observation_feature_names)), dtype=np.float64)
    observation_bias_vector = np.zeros(len(obs_indices), dtype=np.float64)
    reliability_weight_vector = np.zeros(len(adapter.observation_feature_names), dtype=np.float64)
    reliability_bias = 0.0

    dynamics_feature_mean = np.zeros(len(adapter.dynamics_feature_names), dtype=np.float64)
    dynamics_feature_std = np.ones(len(adapter.dynamics_feature_names), dtype=np.float64)
    dynamics_weight_vector = np.zeros(len(adapter.dynamics_feature_names), dtype=np.float64)
    dynamics_bias = 0.0

    readout_feature_mean = np.zeros(len(adapter.readout_feature_names), dtype=np.float64)
    readout_feature_std = np.ones(len(adapter.readout_feature_names), dtype=np.float64)
    readout_weight_vector = np.zeros(len(adapter.readout_feature_names), dtype=np.float64)
    readout_bias = 0.0

    load_feature_mean = np.zeros(len(adapter.load_feature_names), dtype=np.float64)
    load_feature_std = np.ones(len(adapter.load_feature_names), dtype=np.float64)
    load_weight_vector = np.zeros(len(adapter.load_feature_names), dtype=np.float64)
    load_bias = 0.0

    def _predict_observation_latent(observation_features: np.ndarray) -> np.ndarray:
        standardized = (
            np.asarray(observation_features, dtype=np.float64) - observation_feature_mean
        ) / observation_feature_std
        return np.asarray(observation_weight_matrix @ standardized + observation_bias_vector, dtype=np.float64)

    def _predict_reliability(observation_features: np.ndarray) -> float:
        standardized = (
            np.asarray(observation_features, dtype=np.float64) - observation_feature_mean
        ) / observation_feature_std
        raw_score = float(reliability_weight_vector @ standardized + reliability_bias)
        return float(np.clip(raw_score, 0.0, 1.0))

    def _predict_dynamics_latent(dynamics_features: np.ndarray) -> float:
        standardized = (
            np.asarray(dynamics_features, dtype=np.float64) - dynamics_feature_mean
        ) / dynamics_feature_std
        return float(dynamics_weight_vector @ standardized + dynamics_bias)

    def _predict_readout_latent(readout_features: np.ndarray) -> float:
        standardized = (
            np.asarray(readout_features, dtype=np.float64) - readout_feature_mean
        ) / readout_feature_std
        return float(readout_weight_vector @ standardized + readout_bias)

    def _fit_provisional_heads() -> tuple[float, float, float, float, float, float, float, float]:
        nonlocal observation_feature_mean
        nonlocal observation_feature_std
        nonlocal observation_weight_matrix
        nonlocal observation_bias_vector
        nonlocal reliability_weight_vector
        nonlocal reliability_bias
        nonlocal dynamics_feature_mean
        nonlocal dynamics_feature_std
        nonlocal dynamics_weight_vector
        nonlocal dynamics_bias
        nonlocal readout_feature_mean
        nonlocal readout_feature_std
        nonlocal readout_weight_vector
        nonlocal readout_bias

        shared_observation_mean, shared_observation_std = _standardize_features(
            observation_anchor_features + reliability_anchor_features,
            feature_dim=len(adapter.observation_feature_names),
        )
        (
            observation_feature_mean,
            observation_feature_std,
            observation_weight_matrix,
            observation_bias_vector,
            observation_fit_mae,
            observation_fit_rmse,
        ) = _fit_multitarget_head(
            anchor_features=observation_anchor_features,
            anchor_targets=observation_anchor_targets,
            real_features=[],
            real_targets=[],
            feature_dim=len(adapter.observation_feature_names),
            target_dim=len(obs_indices),
            feature_mean=shared_observation_mean,
            feature_std=shared_observation_std,
            anchor_weight=config.adapter.observation_anchor_weight,
            real_weight=config.adapter.observation_real_weight,
        )
        (
            _,
            _,
            reliability_weight_vector,
            reliability_bias,
            reliability_fit_mae,
            reliability_fit_rmse,
        ) = _fit_scalar_head(
            anchor_features=reliability_anchor_features,
            anchor_targets=reliability_anchor_targets,
            real_features=[],
            real_targets=[],
            feature_dim=len(adapter.observation_feature_names),
            feature_mean=shared_observation_mean,
            feature_std=shared_observation_std,
            anchor_weight=config.adapter.observation_anchor_weight,
            real_weight=config.adapter.observation_real_weight,
        )

        dynamics_anchor_features = [
            adapter.dynamics_feature_vector(
                np.asarray(summary_row, dtype=np.float64) - _predict_nominal_summary(hidden_row)
            )
            for summary_row, hidden_row in zip(dynamics_anchor_summary_rows, dynamics_anchor_hidden_rows, strict=True)
        ]
        (
            dynamics_feature_mean,
            dynamics_feature_std,
            dynamics_weight_vector,
            dynamics_bias,
            dynamics_fit_mae,
            dynamics_fit_rmse,
        ) = _fit_scalar_head(
            anchor_features=dynamics_anchor_features,
            anchor_targets=dynamics_anchor_targets,
            real_features=[],
            real_targets=[],
            feature_dim=len(adapter.dynamics_feature_names),
            anchor_weight=config.adapter.process_anchor_weight,
            real_weight=config.adapter.process_real_weight,
        )
        (
            readout_feature_mean,
            readout_feature_std,
            readout_weight_vector,
            readout_bias,
            readout_fit_mae,
            readout_fit_rmse,
        ) = _fit_scalar_head(
            anchor_features=readout_anchor_features,
            anchor_targets=readout_anchor_targets,
            real_features=[],
            real_targets=[],
            feature_dim=len(adapter.readout_feature_names),
            anchor_weight=config.adapter.process_anchor_weight,
            real_weight=config.adapter.process_real_weight,
        )
        return (
            observation_fit_mae,
            observation_fit_rmse,
            reliability_fit_mae,
            reliability_fit_rmse,
            dynamics_fit_mae,
            dynamics_fit_rmse,
            readout_fit_mae,
            readout_fit_rmse,
        )

    provisional_metrics = _fit_provisional_heads()

    episode_series = (
        _build_real_episode_series(
            model=model,
            reference_stats=reference_stats,
            episode_grids=train_episode_grids,
            window_size_minutes=features.shape[1],
            task_adapter=adapter,
        )
        if train_episode_grids
        else []
    )

    def _series_basis_matrix(
        series: _EpisodeResidualSeries,
        current_weight_matrix: np.ndarray,
        current_bias_vector: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            (series.normalized_hidden @ current_weight_matrix.T) + current_bias_vector[None, :],
            dtype=np.float64,
        )

    def _series_summary_residuals(series: _EpisodeResidualSeries) -> np.ndarray:
        return np.asarray(
            [
                np.asarray(summary_row, dtype=np.float64) - _predict_nominal_summary(hidden_row)
                for summary_row, hidden_row in zip(series.summary_features, series.normalized_hidden, strict=True)
            ],
            dtype=np.float64,
        )

    def _series_readout_feature_rows(
        series: _EpisodeResidualSeries,
        residuals: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                adapter.readout_feature_vector_from_context(
                    readout_context,
                    residual=float(residual),
                )
                for readout_context, residual in zip(
                    series.readout_context_features,
                    np.asarray(residuals, dtype=np.float64),
                    strict=True,
                )
            ],
            dtype=np.float64,
        )

    def _series_load_feature_row(
        series: _EpisodeResidualSeries,
        *,
        residual_mean: float,
    ) -> np.ndarray:
        if series.load_context_features.size == 0:
            context_mean = np.zeros(max(len(adapter.load_feature_names) - 1, 0), dtype=np.float64)
        else:
            context_mean = np.asarray(
                np.mean(np.asarray(series.load_context_features, dtype=np.float64), axis=0),
                dtype=np.float64,
            )
        return adapter.load_feature_vector_from_context(
            context_mean,
            residual=residual_mean,
        )

    def _derive_episode_latents(
        current_weight_matrix: np.ndarray,
        current_bias_vector: np.ndarray,
    ) -> np.ndarray:
        if not episode_series:
            return np.zeros((0, latent_dim), dtype=np.float64)
        episode_latents: list[np.ndarray] = []
        for series in episode_series:
            episode_latent = np.zeros(latent_dim, dtype=np.float64)
            basis_matrix = _series_basis_matrix(series, current_weight_matrix, current_bias_vector)
            if series.observation_features.size:
                observation_predictions = np.asarray(
                    [_predict_observation_latent(row) for row in series.observation_features],
                    dtype=np.float64,
                )
                observation_mean_target = np.asarray(observation_predictions.mean(axis=0), dtype=np.float64)
                episode_latent[list(obs_indices)] = observation_mean_target
                residual_after_observation = np.asarray(
                    series.residuals
                    - _subspace_correction(
                        basis_matrix,
                        episode_latent,
                        active_indices=obs_indices,
                    ),
                    dtype=np.float64,
                )
            else:
                residual_after_observation = np.asarray(series.residuals, dtype=np.float64)

            summary_residuals = _series_summary_residuals(series)
            dynamics_predictions = np.asarray(
                [_predict_dynamics_latent(adapter.dynamics_feature_vector(row)) for row in summary_residuals],
                dtype=np.float64,
            )
            episode_latent[dynamics_axis_index] = float(dynamics_predictions.mean()) if dynamics_predictions.size else 0.0
            residual_after_dynamics = np.asarray(
                residual_after_observation
                - _subspace_correction(
                    basis_matrix,
                    episode_latent,
                    active_indices=(dynamics_axis_index,),
                ),
                dtype=np.float64,
            )

            readout_feature_rows = _series_readout_feature_rows(series, residual_after_dynamics)
            readout_predictions = np.asarray(
                [_predict_readout_latent(row) for row in readout_feature_rows],
                dtype=np.float64,
            )
            episode_latent[readout_axis_index] = float(readout_predictions.mean()) if readout_predictions.size else 0.0
            residual_after_readout = np.asarray(
                residual_after_dynamics
                - _subspace_correction(
                    basis_matrix,
                    episode_latent,
                    active_indices=(readout_axis_index,),
                ),
                dtype=np.float64,
            )
            slow_residual = float(np.mean(residual_after_readout))
            episode_latent[load_axis_index] = slow_residual / max(target_std, 1e-6)
            episode_latents.append(episode_latent)
        episode_latent_array = np.asarray(episode_latents, dtype=np.float64)
        return episode_latent_array - episode_latent_array.mean(axis=0, keepdims=True)

    train_episode_latents_raw = np.zeros((0, latent_dim), dtype=np.float64)
    if episode_series:
        for _ in range(config.adapter.joint_fit_iterations):
            train_episode_latents_raw = _derive_episode_latents(raw_weight_matrix, raw_bias_vector)
            raw_weight_matrix, raw_bias_vector, real_fit_mae, real_fit_rmse = _fit_raw_basis(train_episode_latents_raw)
        train_episode_latents_raw = _derive_episode_latents(raw_weight_matrix, raw_bias_vector)

        observation_real_features.clear()
        observation_real_targets.clear()
        reliability_real_features.clear()
        reliability_real_targets.clear()
        dynamics_real_features.clear()
        dynamics_real_targets.clear()
        readout_real_features.clear()
        readout_real_targets.clear()
        load_real_features.clear()
        load_real_targets.clear()

        for episode_latent, series in zip(train_episode_latents_raw, episode_series, strict=True):
            observation_target = np.asarray(episode_latent, dtype=np.float64)[list(obs_indices)]
            basis_matrix = _series_basis_matrix(series, raw_weight_matrix, raw_bias_vector)
            summary_residuals = _series_summary_residuals(series)
            dynamics_target = float(episode_latent[dynamics_axis_index])
            readout_target = float(episode_latent[readout_axis_index])
            load_target = float(episode_latent[load_axis_index])
            residual_after_observation = np.asarray(
                series.residuals
                - _subspace_correction(
                    basis_matrix,
                    episode_latent,
                    active_indices=obs_indices,
                ),
                dtype=np.float64,
            )
            residual_after_dynamics = np.asarray(
                residual_after_observation
                - _subspace_correction(
                    basis_matrix,
                    episode_latent,
                    active_indices=(dynamics_axis_index,),
                ),
                dtype=np.float64,
            )
            readout_feature_rows = _series_readout_feature_rows(series, residual_after_dynamics)
            residual_after_readout = np.asarray(
                residual_after_dynamics
                - _subspace_correction(
                    basis_matrix,
                    episode_latent,
                    active_indices=(readout_axis_index,),
                ),
                dtype=np.float64,
            )
            for observation_row in series.observation_features:
                observation_real_features.append(np.asarray(observation_row, dtype=np.float64))
                observation_real_targets.append(observation_target)
                reliability_real_features.append(np.asarray(observation_row, dtype=np.float64))
                reliability_real_targets.append(0.0)
            for summary_row in summary_residuals:
                dynamics_real_features.append(adapter.dynamics_feature_vector(summary_row))
                dynamics_real_targets.append(dynamics_target)
            for readout_row in readout_feature_rows:
                readout_real_features.append(np.asarray(readout_row, dtype=np.float64))
                readout_real_targets.append(readout_target)
            load_real_features.append(
                _series_load_feature_row(
                    series,
                    residual_mean=float(np.mean(residual_after_readout)),
                )
            )
            load_real_targets.append(load_target)

    shared_observation_mean, shared_observation_std = _standardize_features(
        observation_anchor_features
        + observation_real_features
        + reliability_anchor_features
        + reliability_real_features,
        feature_dim=len(adapter.observation_feature_names),
    )
    (
        observation_feature_mean,
        observation_feature_std,
        observation_weight_matrix,
        observation_bias_vector,
        observation_fit_mae,
        observation_fit_rmse,
    ) = _fit_multitarget_head(
        anchor_features=observation_anchor_features,
        anchor_targets=observation_anchor_targets,
        real_features=observation_real_features,
        real_targets=observation_real_targets,
        feature_dim=len(adapter.observation_feature_names),
        target_dim=len(obs_indices),
        feature_mean=shared_observation_mean,
        feature_std=shared_observation_std,
        anchor_weight=config.adapter.observation_anchor_weight,
        real_weight=config.adapter.observation_real_weight,
    )
    (
        _,
        _,
        reliability_weight_vector,
        reliability_bias,
        reliability_fit_mae,
        reliability_fit_rmse,
    ) = _fit_scalar_head(
        anchor_features=reliability_anchor_features,
        anchor_targets=reliability_anchor_targets,
        real_features=reliability_real_features,
        real_targets=reliability_real_targets,
        feature_dim=len(adapter.observation_feature_names),
        feature_mean=shared_observation_mean,
        feature_std=shared_observation_std,
        anchor_weight=config.adapter.observation_anchor_weight,
        real_weight=config.adapter.observation_real_weight,
    )

    dynamics_anchor_features = [
        adapter.dynamics_feature_vector(
            np.asarray(summary_row, dtype=np.float64) - _predict_nominal_summary(hidden_row)
        )
        for summary_row, hidden_row in zip(dynamics_anchor_summary_rows, dynamics_anchor_hidden_rows, strict=True)
    ]
    (
        dynamics_feature_mean,
        dynamics_feature_std,
        dynamics_weight_vector,
        dynamics_bias,
        dynamics_fit_mae,
        dynamics_fit_rmse,
    ) = _fit_scalar_head(
        anchor_features=dynamics_anchor_features,
        anchor_targets=dynamics_anchor_targets,
        real_features=dynamics_real_features,
        real_targets=dynamics_real_targets,
        feature_dim=len(adapter.dynamics_feature_names),
        anchor_weight=config.adapter.process_anchor_weight,
        real_weight=config.adapter.process_real_weight,
    )
    (
        readout_feature_mean,
        readout_feature_std,
        readout_weight_vector,
        readout_bias,
        readout_fit_mae,
        readout_fit_rmse,
    ) = _fit_scalar_head(
        anchor_features=readout_anchor_features,
        anchor_targets=readout_anchor_targets,
        real_features=readout_real_features,
        real_targets=readout_real_targets,
        feature_dim=len(adapter.readout_feature_names),
        anchor_weight=config.adapter.process_anchor_weight,
        real_weight=config.adapter.process_real_weight,
    )
    (
        load_feature_mean,
        load_feature_std,
        load_weight_vector,
        load_bias,
        load_fit_mae,
        load_fit_rmse,
    ) = _fit_scalar_head(
        anchor_features=load_anchor_features,
        anchor_targets=load_anchor_targets,
        real_features=load_real_features,
        real_targets=load_real_targets,
        feature_dim=len(adapter.load_feature_names),
        anchor_weight=config.adapter.process_anchor_weight,
        real_weight=config.adapter.process_real_weight,
    )

    if episode_series:
        train_episode_latents_raw = _derive_episode_latents(raw_weight_matrix, raw_bias_vector)
        raw_weight_matrix, raw_bias_vector, real_fit_mae, real_fit_rmse = _fit_raw_basis(train_episode_latents_raw)
        train_episode_latents_raw = _derive_episode_latents(raw_weight_matrix, raw_bias_vector)

    support_raw = np.vstack((anchor_latents_raw, train_episode_latents_raw)) if train_episode_latents_raw.size else anchor_latents_raw
    latent_mean = np.asarray(support_raw.mean(axis=0), dtype=np.float64)
    latent_std = np.asarray(np.clip(support_raw.std(axis=0, ddof=1), 1e-3, None), dtype=np.float64)

    anchor_design_matrix = np.asarray(anchor_design_rows, dtype=np.float64)
    anchor_target_array = np.asarray(anchor_residual_targets, dtype=np.float64)
    anchor_fitted_residuals = anchor_design_matrix @ np.concatenate(
        (
            raw_weight_matrix.reshape(latent_dim * hidden_dim),
            raw_bias_vector,
        ),
        axis=0,
    )
    training_metrics = {
        "num_examples": int(anchor_design_matrix.shape[0]),
        "num_real_examples": int(sum(series.residuals.size for series in episode_series)),
        "num_train_episode_latents": int(train_episode_latents_raw.shape[0]),
        "reliability_positive_anchors": int(sum(target > 0.5 for target in reliability_anchor_targets)),
        "load_real_examples": int(len(load_real_features)),
        "dynamics_real_examples": int(len(dynamics_real_features)),
        "readout_real_examples": int(len(readout_real_features)),
        "joint_fit_iterations": int(config.adapter.joint_fit_iterations),
        "nominal_summary_fit_mae": float(
            np.mean(
                np.abs(
                    (base_hidden_norm_matrix @ nominal_summary_weight_matrix.T + nominal_summary_bias_vector[None, :])
                    - base_summary_matrix
                )
            )
        ),
        "anchor_fit_mae": float(np.mean(np.abs(anchor_fitted_residuals - anchor_target_array))),
        "anchor_fit_rmse": float(np.sqrt(np.mean(np.square(anchor_fitted_residuals - anchor_target_array)))),
        "real_fit_mae": real_fit_mae,
        "real_fit_rmse": real_fit_rmse,
        "observation_fit_mae": observation_fit_mae,
        "observation_fit_rmse": observation_fit_rmse,
        "reliability_fit_mae": reliability_fit_mae,
        "reliability_fit_rmse": reliability_fit_rmse,
        "load_fit_mae": load_fit_mae,
        "load_fit_rmse": load_fit_rmse,
        "dynamics_fit_mae": dynamics_fit_mae,
        "dynamics_fit_rmse": dynamics_fit_rmse,
        "readout_fit_mae": readout_fit_mae,
        "readout_fit_rmse": readout_fit_rmse,
        "provisional_observation_fit_mae": provisional_metrics[0],
        "provisional_observation_fit_rmse": provisional_metrics[1],
        "provisional_reliability_fit_mae": provisional_metrics[2],
        "provisional_reliability_fit_rmse": provisional_metrics[3],
        "provisional_dynamics_fit_mae": provisional_metrics[4],
        "provisional_dynamics_fit_rmse": provisional_metrics[5],
        "provisional_readout_fit_mae": provisional_metrics[6],
        "provisional_readout_fit_rmse": provisional_metrics[7],
    }

    return LatentAdapterModel(
        axes=config.axes,
        weight_matrix=np.asarray(raw_weight_matrix, dtype=np.float64),
        bias_vector=np.asarray(raw_bias_vector, dtype=np.float64),
        hidden_mean=np.asarray(normalized_hidden_mean, dtype=np.float64),
        hidden_std=np.asarray(normalized_hidden_std, dtype=np.float64),
        observation_feature_names=adapter.observation_feature_names,
        observation_feature_mean=np.asarray(observation_feature_mean, dtype=np.float64),
        observation_feature_std=np.asarray(observation_feature_std, dtype=np.float64),
        observation_weight_matrix=np.asarray(observation_weight_matrix, dtype=np.float64),
        observation_bias_vector=np.asarray(observation_bias_vector, dtype=np.float64),
        reliability_weight_vector=np.asarray(reliability_weight_vector, dtype=np.float64),
        reliability_bias=float(reliability_bias),
        summary_feature_names=adapter.summary_feature_names,
        nominal_summary_weight_matrix=np.asarray(nominal_summary_weight_matrix, dtype=np.float64),
        nominal_summary_bias_vector=np.asarray(nominal_summary_bias_vector, dtype=np.float64),
        load_feature_names=adapter.load_feature_names,
        load_feature_mean=np.asarray(load_feature_mean, dtype=np.float64),
        load_feature_std=np.asarray(load_feature_std, dtype=np.float64),
        load_weight_vector=np.asarray(load_weight_vector, dtype=np.float64),
        load_bias=float(load_bias),
        dynamics_feature_names=adapter.dynamics_feature_names,
        dynamics_feature_mean=np.asarray(dynamics_feature_mean, dtype=np.float64),
        dynamics_feature_std=np.asarray(dynamics_feature_std, dtype=np.float64),
        dynamics_weight_vector=np.asarray(dynamics_weight_vector, dtype=np.float64),
        dynamics_bias=float(dynamics_bias),
        readout_feature_names=adapter.readout_feature_names,
        readout_feature_mean=np.asarray(readout_feature_mean, dtype=np.float64),
        readout_feature_std=np.asarray(readout_feature_std, dtype=np.float64),
        readout_weight_vector=np.asarray(readout_weight_vector, dtype=np.float64),
        readout_bias=float(readout_bias),
        latent_mean=np.asarray(latent_mean, dtype=np.float64),
        latent_std=np.asarray(latent_std, dtype=np.float64),
        train_episode_latents=np.asarray(train_episode_latents_raw, dtype=np.float64),
        anchor_latents=np.asarray(anchor_latents_raw, dtype=np.float64),
        training_metrics=training_metrics,
    )


def _raw_axis_vector(
    axes: tuple[SemanticAxis, ...],
    axis: SemanticAxis,
    value: float,
) -> np.ndarray:
    vector = np.zeros(len(axes), dtype=np.float64)
    vector[axes.index(axis)] = float(value)
    return vector


def _apply_p3_bias(feature_window: np.ndarray, bias: float) -> np.ndarray:
    transformed = np.asarray(feature_window, dtype=np.float64).copy()
    transformed[:, 2] = transformed[:, 2] + float(bias)
    return transformed


def _apply_q_in_scale(feature_window: np.ndarray, scale: float) -> np.ndarray:
    transformed = np.asarray(feature_window, dtype=np.float64).copy()
    transformed[:, 0] = transformed[:, 0] * float(scale)
    return transformed


def _apply_p3_observation_delay(feature_window: np.ndarray, delay_minutes: int) -> np.ndarray:
    transformed = np.asarray(feature_window, dtype=np.float64).copy()
    if delay_minutes <= 0:
        return transformed
    p3_history = np.asarray(transformed[:, 2], dtype=np.float64).copy()
    for minute_index in range(transformed.shape[0]):
        source_index = max(0, minute_index - delay_minutes)
        transformed[minute_index, 2] = p3_history[source_index]
    return transformed


def _apply_dynamics_scaling(feature_window: np.ndarray, factor: float) -> np.ndarray:
    transformed = np.asarray(feature_window, dtype=np.float64).copy()
    for feature_index in (1, 2):
        for minute_index in range(1, transformed.shape[0]):
            delta = transformed[minute_index, feature_index] - transformed[minute_index - 1, feature_index]
            transformed[minute_index, feature_index] = transformed[minute_index - 1, feature_index] + factor * delta
    return transformed


def _default_supervised_indices(episode_grid: object, window_size: int) -> np.ndarray:
    target_size = int(_episode_target_series(episode_grid).shape[0])
    if hasattr(episode_grid, "supervision_indices"):
        indices = np.asarray(getattr(episode_grid, "supervision_indices"), dtype=np.int64)
    elif hasattr(episode_grid, "supervision_mask"):
        indices = np.flatnonzero(np.asarray(getattr(episode_grid, "supervision_mask"), dtype=bool))
    else:
        indices = np.arange(target_size, dtype=np.int64)
    return np.asarray(indices[indices >= window_size - 1], dtype=np.int64)


def _episode_target_series(episode_grid: object) -> np.ndarray:
    if hasattr(episode_grid, "target_values"):
        return np.asarray(getattr(episode_grid, "target_values"), dtype=np.float64)
    if hasattr(episode_grid, "target_concentration"):
        return np.asarray(getattr(episode_grid, "target_concentration"), dtype=np.float64)
    raise AttributeError("episode_grid must expose target_values or target_concentration")


def _default_target_value(episode_grid: object, target_index: int) -> float:
    return float(_episode_target_series(episode_grid)[target_index])


def _default_target_position(episode_grid: object, target_index: int) -> int:
    if hasattr(episode_grid, "minutes"):
        return int(np.asarray(getattr(episode_grid, "minutes"), dtype=np.int64)[target_index])
    return int(target_index)


def _default_target_time_value(episode_grid: object, target_index: int) -> float | None:
    del episode_grid, target_index
    return None


def _build_design_row(normalized_hidden: np.ndarray, standardized_latent: np.ndarray) -> np.ndarray:
    cross_terms = np.concatenate(
        [standardized_latent[axis_index] * normalized_hidden for axis_index in range(standardized_latent.size)],
        axis=0,
    )
    return np.concatenate((cross_terms, standardized_latent), axis=0)


def _normalize_prior_direction(axis: SemanticAxis, direction: float) -> float:
    if axis == SemanticAxis.TRUST and direction < 0.0:
        return 0.0
    return direction


def _subspace_correction(
    basis: np.ndarray,
    latent: np.ndarray,
    *,
    active_indices: tuple[int, ...],
) -> np.ndarray | float:
    if not active_indices:
        basis_array = np.asarray(basis, dtype=np.float64)
        if basis_array.ndim == 1:
            return 0.0
        return np.zeros(basis_array.shape[0], dtype=np.float64)
    basis_array = np.asarray(basis, dtype=np.float64)
    latent_array = np.asarray(latent, dtype=np.float64)
    if basis_array.ndim == 1:
        return float(np.dot(basis_array[list(active_indices)], latent_array[list(active_indices)]))
    return np.asarray(
        basis_array[:, list(active_indices)] @ latent_array[list(active_indices)],
        dtype=np.float64,
    )


def build_latent_prior(
    prior_result: PriorEngineResult,
    *,
    config: RoamLatentConfig,
) -> LatentPrior:
    raw_mean = np.zeros(len(config.axes), dtype=np.float64)
    variances: list[float] = []
    semantics_by_axis = prior_result.axis_mapping()
    trust_semantic = semantics_by_axis[SemanticAxis.TRUST.value]
    trust_direction = _normalize_prior_direction(
        SemanticAxis.TRUST,
        float(trust_semantic["direction"]),
    )
    trust_strength = float(trust_semantic["strength"])
    trust_score = float(np.clip(max(trust_direction, 0.0) * trust_strength, 0.0, 1.0))

    for axis_index, axis in enumerate(config.axes):
        semantic_payload = semantics_by_axis[axis.value]
        direction = _normalize_prior_direction(axis, float(semantic_payload["direction"]))
        strength = float(semantic_payload["strength"])
        confidence = float(semantic_payload["confidence"])
        raw_mean[axis_index] = direction * strength
        uncertainty_score = float(
            np.clip(
                0.5 * ((1.0 - confidence) + prior_result.global_uncertainty),
                0.0,
                1.0,
            )
        )
        raw_variance = config.prior_variance_min + (config.prior_variance_max - config.prior_variance_min) * uncertainty_score
        variances.append(float(raw_variance))

    covariance = np.diag(np.asarray(variances, dtype=np.float64))
    return LatentPrior(
        mean=np.asarray(raw_mean, dtype=np.float64),
        covariance=covariance,
        axis_semantics=semantics_by_axis,
        reliability={
            "trust_score": trust_score,
            "semantic": trust_semantic,
        },
        global_uncertainty=prior_result.global_uncertainty,
        summary=prior_result.summary,
        raw_response_text=prior_result.raw_response_text,
        api_reliable=prior_result.api_reliable,
        error=prior_result.error,
    )


def _dominant_axis(
    axes: tuple[SemanticAxis, ...],
    vector: np.ndarray,
    covariance: np.ndarray,
) -> str:
    if vector.size == 0:
        return "none"
    vector_array = np.asarray(vector, dtype=np.float64)
    uncertainty = float(np.sqrt(max(float(np.trace(covariance)), 0.0)))
    if float(np.linalg.norm(vector_array)) <= uncertainty:
        return "none"
    return axes[int(np.argmax(np.abs(vector_array)))].value


def _dominant_axis_in_subspace(
    axes: tuple[SemanticAxis, ...],
    vector: np.ndarray,
    covariance: np.ndarray,
    subspace_axes: tuple[SemanticAxis, ...],
) -> str:
    subspace_indices = _subspace_indices(axes, subspace_axes)
    if not subspace_indices:
        return "none"
    subspace_vector = np.asarray(vector, dtype=np.float64)[list(subspace_indices)]
    if subspace_vector.size == 0:
        return "none"
    subspace_covariance = np.asarray(covariance, dtype=np.float64)[np.ix_(subspace_indices, subspace_indices)]
    uncertainty = float(np.sqrt(max(float(np.trace(subspace_covariance)), 0.0)))
    if float(np.linalg.norm(subspace_vector)) <= uncertainty:
        return "none"
    dominant_index = int(np.argmax(np.abs(subspace_vector)))
    return subspace_axes[dominant_index].value


def _update_posterior_once(
    mean: np.ndarray,
    covariance: np.ndarray,
    basis_vector: np.ndarray,
    residual: float,
    *,
    observation_noise_variance: float,
    variance_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    innovation_variance = float(basis_vector @ covariance @ basis_vector + observation_noise_variance)
    kalman_gain = (covariance @ basis_vector) / innovation_variance
    innovation = float(residual - basis_vector @ mean)
    updated_mean = mean + kalman_gain * innovation
    updated_covariance = covariance - np.outer(kalman_gain, basis_vector) @ covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    if variance_floor is not None and variance_floor > 0.0:
        updated_covariance = _apply_covariance_floor(updated_covariance, variance_floor)
    return np.asarray(updated_mean, dtype=np.float64), np.asarray(updated_covariance, dtype=np.float64)


def _update_posterior_from_subspace_observation(
    mean: np.ndarray,
    covariance: np.ndarray,
    latent_observation: np.ndarray,
    *,
    active_indices: tuple[int, ...],
    observation_variance: float,
    variance_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not active_indices:
        return np.asarray(mean, dtype=np.float64), np.asarray(covariance, dtype=np.float64)
    observation_matrix = np.zeros((len(active_indices), mean.size), dtype=np.float64)
    for row_index, axis_index in enumerate(active_indices):
        observation_matrix[row_index, axis_index] = 1.0
    observation_covariance = observation_variance * np.eye(len(active_indices), dtype=np.float64)
    innovation_covariance = (
        observation_matrix @ covariance @ observation_matrix.T + observation_covariance
    )
    kalman_gain = covariance @ observation_matrix.T @ np.linalg.inv(innovation_covariance)
    innovation = (
        np.asarray(latent_observation, dtype=np.float64)[list(active_indices)]
        - observation_matrix @ np.asarray(mean, dtype=np.float64)
    )
    updated_mean = np.asarray(mean, dtype=np.float64) + kalman_gain @ innovation
    updated_covariance = covariance - kalman_gain @ observation_matrix @ covariance
    updated_covariance = 0.5 * (updated_covariance + updated_covariance.T)
    if variance_floor is not None and variance_floor > 0.0:
        updated_covariance = _apply_covariance_floor(updated_covariance, variance_floor)
    return np.asarray(updated_mean, dtype=np.float64), np.asarray(updated_covariance, dtype=np.float64)


def _update_posterior_from_latent_observation(
    mean: np.ndarray,
    covariance: np.ndarray,
    latent_observation: np.ndarray,
    *,
    observation_variance: float,
    variance_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _update_posterior_from_subspace_observation(
        mean,
        covariance,
        latent_observation,
        active_indices=tuple(range(mean.size)),
        observation_variance=observation_variance,
        variance_floor=variance_floor,
    )


def _apply_covariance_floor(covariance: np.ndarray, variance_floor: float) -> np.ndarray:
    if variance_floor <= 0.0:
        return np.asarray(covariance, dtype=np.float64)
    symmetric_covariance = 0.5 * (np.asarray(covariance, dtype=np.float64) + np.asarray(covariance, dtype=np.float64).T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)
    floored_eigenvalues = np.maximum(eigenvalues, variance_floor)
    return np.asarray(
        eigenvectors @ np.diag(floored_eigenvalues) @ eigenvectors.T,
        dtype=np.float64,
    )


def _support_distance(mean: np.ndarray, support_latents: np.ndarray) -> float:
    deltas = np.asarray(support_latents, dtype=np.float64) - np.asarray(mean, dtype=np.float64)[None, :]
    distances = np.sum(np.square(deltas), axis=1)
    return float(np.min(distances))


def _subspace_support_distance(
    mean: np.ndarray,
    support_latents: np.ndarray,
    *,
    active_indices: tuple[int, ...],
) -> float:
    projected_mean = np.asarray(mean, dtype=np.float64)[list(active_indices)]
    projected_support = np.asarray(support_latents, dtype=np.float64)[:, list(active_indices)]
    return _support_distance(projected_mean, projected_support)


def _subspace_trace(covariance: np.ndarray, *, active_indices: tuple[int, ...]) -> float:
    return float(np.trace(np.asarray(covariance, dtype=np.float64)[np.ix_(active_indices, active_indices)]))


def _subspace_effect(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    active_indices: tuple[int, ...],
) -> float:
    if not active_indices:
        return 0.0
    subspace_mean = np.asarray(mean, dtype=np.float64)[list(active_indices)]
    subspace_trace = _subspace_trace(covariance, active_indices=active_indices)
    mean_norm = float(np.linalg.norm(subspace_mean))
    uncertainty = float(np.sqrt(max(subspace_trace, 0.0)))
    return float(mean_norm / (mean_norm + uncertainty + 1e-6))


def _gamma_from_posterior(
    covariance: np.ndarray,
    *,
    support_distance_obs: float,
    support_distance_proc: float,
    axes: tuple[SemanticAxis, ...],
    config: SupportGeometryConfig,
) -> float:
    trace_obs = _subspace_trace(covariance, active_indices=_subspace_indices(axes, _observation_axes()))
    trace_proc = _subspace_trace(covariance, active_indices=_subspace_indices(axes, _process_axes()))
    return float(
        np.exp(
            -config.gamma_uncertainty_scale_obs * trace_obs
            -config.gamma_uncertainty_scale_proc * trace_proc
            -config.gamma_support_scale_obs * support_distance_obs
            -config.gamma_support_scale_proc * support_distance_proc
        )
    )


def _build_latent_posterior(
    axes: tuple[SemanticAxis, ...],
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    support_distance: float,
    support_distance_obs: float,
    support_distance_proc: float,
    num_updates: int,
    trust_score: float,
    config: SupportGeometryConfig,
) -> LatentPosterior:
    posterior_trace = float(np.trace(covariance))
    posterior_trace_obs = _subspace_trace(covariance, active_indices=_subspace_indices(axes, _observation_axes()))
    posterior_trace_proc = _subspace_trace(covariance, active_indices=_subspace_indices(axes, _process_axes()))
    if config.enable_risk_gate:
        gamma = _gamma_from_posterior(
            covariance,
            support_distance_obs=support_distance_obs,
            support_distance_proc=support_distance_proc,
            axes=axes,
            config=config,
        )
        gamma_obs = float(gamma / (1.0 + max(trust_score, 0.0)))
        gamma_proc = float(gamma)
    else:
        gamma = 1.0
        gamma_obs = 1.0
        gamma_proc = 1.0
    return LatentPosterior(
        mean=np.asarray(mean, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
        posterior_trace=posterior_trace,
        posterior_trace_obs=posterior_trace_obs,
        posterior_trace_proc=posterior_trace_proc,
        support_distance=support_distance,
        support_distance_obs=support_distance_obs,
        support_distance_proc=support_distance_proc,
        gamma=gamma,
        gamma_obs=gamma_obs,
        gamma_proc=gamma_proc,
        abstain=False,
        dominant_obs_axis=_dominant_axis_in_subspace(axes, mean, covariance, _observation_axes()),
        dominant_proc_axis=_dominant_axis_in_subspace(axes, mean, covariance, _process_axes()),
        num_updates=num_updates,
    )


def _build_roam_episode_bootstrap(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    episode_grid,
    window_size_minutes: int,
    evidence_bundle: EvidenceBundle,
    task_adapter: RoamTaskAdapter,
) -> _EpisodeBootstrap:
    adapter = task_adapter
    supervised_indices = adapter.supervised_indices(episode_grid, window_size=window_size_minutes)
    if supervised_indices.size == 0:
        raise ValueError("episode_grid did not yield any supervised target indices")
    first_target_index = int(supervised_indices[0])
    start_index = first_target_index - window_size_minutes + 1
    if start_index < 0:
        raise ValueError("first supervised target does not provide a full prior window")
    initial_window = np.asarray(
        episode_grid.feature_matrix[start_index : first_target_index + 1],
        dtype=np.float64,
    )
    initial_diagnostics = adapter.build_diagnostics(model, reference_stats, initial_window)
    prior_input = PriorEngineInput(
        evidence=evidence_bundle.for_scenario(episode_grid.scenario_name),
        diagnostics_summary=initial_diagnostics.to_summary(),
        diagnostics_payload=adapter.build_prior_diagnostics_payload(initial_diagnostics),
        task_context=adapter.prior_task_context,
        known_context=adapter.build_prior_known_context(
            episode_grid,
            target_index=first_target_index,
            feature_window=initial_window,
            diagnostics=initial_diagnostics,
        ),
        axis_schema=_prior_axes(),
    )
    return _EpisodeBootstrap(
        supervised_indices=np.asarray(supervised_indices, dtype=np.int64),
        prior_input=prior_input,
        initial_diagnostics=initial_diagnostics,
    )


def _aggregate_feature_buffer(feature_buffer: list[np.ndarray]) -> np.ndarray:
    return np.asarray(np.mean(np.vstack(feature_buffer), axis=0), dtype=np.float64)


def _apply_buffered_subspace_update(
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    *,
    feature_buffer: list[np.ndarray],
    latent_observation_fn,
    active_indices: tuple[int, ...],
    observation_variance: float,
    variance_floor: float,
    update_space: str,
) -> tuple[np.ndarray, np.ndarray, bool]:
    if not feature_buffer:
        return posterior_mean, posterior_covariance, False
    latent_observation = latent_observation_fn(_aggregate_feature_buffer(feature_buffer))
    if update_space == "full":
        next_mean, next_covariance = _update_posterior_from_latent_observation(
            posterior_mean,
            posterior_covariance,
            latent_observation,
            observation_variance=observation_variance,
            variance_floor=variance_floor,
        )
    else:
        next_mean, next_covariance = _update_posterior_from_subspace_observation(
            posterior_mean,
            posterior_covariance,
            latent_observation,
            active_indices=active_indices,
            observation_variance=observation_variance,
            variance_floor=variance_floor,
        )
    feature_buffer.clear()
    return next_mean, next_covariance, True


def build_roam_episode_prior_input(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    episode_grid,
    window_size_minutes: int,
    evidence_bundle: EvidenceBundle,
    task_adapter: RoamTaskAdapter,
) -> tuple[PriorEngineInput, ModelDiagnostics]:
    bootstrap = _build_roam_episode_bootstrap(
        model=model,
        reference_stats=reference_stats,
        episode_grid=episode_grid,
        window_size_minutes=window_size_minutes,
        evidence_bundle=evidence_bundle,
        task_adapter=task_adapter,
    )
    return bootstrap.prior_input, bootstrap.initial_diagnostics


def run_roam_episode(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    adapter_model: LatentAdapterModel,
    episode_grid,
    episode_index: int,
    split_name: str,
    window_size_minutes: int,
    evidence_bundle: EvidenceBundle,
    prior_engine: SemanticPriorEngine | None,
    config: RoamLatentConfig,
    task_adapter: RoamTaskAdapter,
    cache_dir: Path | None = None,
    prior_result: PriorEngineResult | None = None,
) -> RoamEpisodeResult:
    config.validate()
    adapter = task_adapter
    bootstrap = _build_roam_episode_bootstrap(
        model=model,
        reference_stats=reference_stats,
        episode_grid=episode_grid,
        window_size_minutes=window_size_minutes,
        evidence_bundle=evidence_bundle,
        task_adapter=adapter,
    )
    supervised_indices = bootstrap.supervised_indices
    prior_input = bootstrap.prior_input
    initial_diagnostics = bootstrap.initial_diagnostics
    evidence = prior_input.evidence
    if prior_result is None:
        if prior_engine is None:
            raise ValueError("prior_engine is required when prior_result is not provided")
        prior_result = prior_engine.infer(prior_input, cache_dir=cache_dir)
    latent_prior = build_latent_prior(
        prior_result,
        config=config,
    )
    prior_trust_score = float(latent_prior.reliability["trust_score"])
    diagnostic_trust_score = adapter_model.reliability_score(adapter.observation_feature_vector(initial_diagnostics))
    trust_score = float(
        np.clip(
            1.0 - (1.0 - prior_trust_score) * (1.0 - diagnostic_trust_score),
            0.0,
            1.0,
        )
    )
    if not config.online.enable_trust_gate:
        trust_score = 0.0
    reliability = {
        "trust_score": trust_score,
        "prior_trust_score": prior_trust_score,
        "diagnostic_trust_score": diagnostic_trust_score,
        "semantic": latent_prior.reliability["semantic"],
    }

    posterior_mean = np.asarray(latent_prior.mean, dtype=np.float64)
    posterior_covariance = np.asarray(latent_prior.covariance, dtype=np.float64)
    update_count = 0
    observation_feature_buffer: list[np.ndarray] = []
    load_feature_buffer: list[np.ndarray] = []
    dynamics_feature_buffer: list[np.ndarray] = []
    readout_feature_buffer: list[np.ndarray] = []
    delayed_samples: list[tuple[np.ndarray, float, float, ModelDiagnostics, np.ndarray, np.ndarray]] = []
    obs_indices = _subspace_indices(config.axes, _observation_axes())
    proc_indices = _subspace_indices(config.axes, _process_axes())
    load_indices = (config.axes.index(SemanticAxis.LOAD),)
    dynamics_indices = (config.axes.index(SemanticAxis.DYNAMICS),)
    readout_indices = (config.axes.index(SemanticAxis.READOUT),)
    observation_variance_eff = config.observation_evidence.observation_variance * (1.0 + trust_score)
    label_variance_floor = min(
        config.observation_evidence.variance_floor,
        config.process_evidence.variance_floor,
    )
    sample_records: list[LatentPredictionRecord] = []

    def _append_process_feature_buffers(
        *,
        basis_vector: np.ndarray,
        residual: float,
        diagnostics: ModelDiagnostics,
        feature_window: np.ndarray,
        summary_residual: np.ndarray,
    ) -> None:
        residual_after_observation = float(
            residual
            - _subspace_correction(
                basis_vector,
                posterior_mean,
                active_indices=obs_indices,
            )
        )
        residual_after_dynamics = float(
            residual_after_observation
            - _subspace_correction(
                basis_vector,
                posterior_mean,
                active_indices=dynamics_indices,
            )
        )
        residual_after_readout = float(
            residual_after_dynamics
            - _subspace_correction(
                basis_vector,
                posterior_mean,
                active_indices=readout_indices,
            )
        )
        load_feature_buffer.append(
            adapter.load_feature_vector_from_context(
                adapter.load_context_vector(diagnostics, feature_window, reference_stats),
                residual=residual_after_readout,
            )
        )
        dynamics_feature_buffer.append(adapter.dynamics_feature_vector(summary_residual))
        readout_feature_buffer.append(
            adapter.readout_feature_vector_from_context(
                adapter.readout_context_vector(diagnostics, feature_window, reference_stats),
                residual=residual_after_dynamics,
            )
        )

    def _release_delayed_sample() -> None:
        nonlocal posterior_mean
        nonlocal posterior_covariance
        nonlocal update_count
        if len(delayed_samples) < config.online.effective_label_delay_steps:
            return
        basis_vector, baseline_prediction, target_value, diagnostics, feature_window, summary_residual = delayed_samples.pop(0)
        residual = float(target_value - baseline_prediction)
        posterior_mean, posterior_covariance = _update_posterior_once(
            posterior_mean,
            posterior_covariance,
            basis_vector,
            residual,
            observation_noise_variance=config.online.label_observation_noise_variance,
            variance_floor=label_variance_floor,
        )
        update_count += 1
        if config.process_evidence.enabled:
            _append_process_feature_buffers(
                basis_vector=basis_vector,
                residual=residual,
                diagnostics=diagnostics,
                feature_window=feature_window,
                summary_residual=summary_residual,
            )

    for event_offset, target_minute in enumerate(supervised_indices.tolist()):
        _release_delayed_sample()
        start_minute = target_minute - window_size_minutes + 1
        feature_window = np.asarray(
            episode_grid.feature_matrix[start_minute : target_minute + 1],
            dtype=np.float64,
        )
        diagnostics = adapter.build_diagnostics(model, reference_stats, feature_window)
        hidden_state = _baseline_hidden_state(model, feature_window)
        observation_feature_vector = adapter.observation_feature_vector(diagnostics)
        summary_residual = adapter.summary_feature_vector(diagnostics) - adapter_model.nominal_summary(hidden_state)
        if config.observation_evidence.enabled:
            observation_feature_buffer.append(observation_feature_vector)
        should_apply_observation_update = (
            config.observation_evidence.enabled
            and event_offset > 0
            and event_offset % config.observation_evidence.effective_update_interval_steps == 0
            and observation_feature_buffer
        )
        should_apply_process_update = (
            config.process_evidence.enabled
            and event_offset > 0
            and event_offset % config.process_evidence.effective_update_interval_steps == 0
            and (load_feature_buffer or dynamics_feature_buffer or readout_feature_buffer)
        )
        if should_apply_observation_update:
            posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
                posterior_mean,
                posterior_covariance,
                feature_buffer=observation_feature_buffer,
                latent_observation_fn=adapter_model.observation_latent_observation,
                active_indices=obs_indices,
                observation_variance=observation_variance_eff,
                variance_floor=config.observation_evidence.variance_floor,
                update_space=config.update_space,
            )
            update_count += int(did_update)
        if should_apply_process_update and dynamics_feature_buffer:
            posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
                posterior_mean,
                posterior_covariance,
                feature_buffer=dynamics_feature_buffer,
                latent_observation_fn=adapter_model.dynamics_latent_observation,
                active_indices=dynamics_indices,
                observation_variance=config.process_evidence.observation_variance,
                variance_floor=config.process_evidence.variance_floor,
                update_space=config.update_space,
            )
            update_count += int(did_update)
        if should_apply_process_update and readout_feature_buffer:
            posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
                posterior_mean,
                posterior_covariance,
                feature_buffer=readout_feature_buffer,
                latent_observation_fn=adapter_model.readout_latent_observation,
                active_indices=readout_indices,
                observation_variance=config.process_evidence.observation_variance,
                variance_floor=config.process_evidence.variance_floor,
                update_space=config.update_space,
            )
            update_count += int(did_update)
        if should_apply_process_update and load_feature_buffer:
            posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
                posterior_mean,
                posterior_covariance,
                feature_buffer=load_feature_buffer,
                latent_observation_fn=adapter_model.load_latent_observation,
                active_indices=load_indices,
                observation_variance=config.process_evidence.observation_variance,
                variance_floor=config.process_evidence.variance_floor,
                update_space=config.update_space,
            )
            update_count += int(did_update)
        basis_vector = adapter_model.axis_coefficients(hidden_state)
        support_distance_obs = _subspace_support_distance(
            posterior_mean,
            adapter_model.support_latents,
            active_indices=obs_indices,
        )
        support_distance_proc = _subspace_support_distance(
            posterior_mean,
            adapter_model.support_latents,
            active_indices=proc_indices,
        )
        support_distance = support_distance_obs + support_distance_proc
        posterior = _build_latent_posterior(
            config.axes,
            posterior_mean,
            posterior_covariance,
            support_distance=support_distance,
            support_distance_obs=support_distance_obs,
            support_distance_proc=support_distance_proc,
            num_updates=update_count,
            trust_score=trust_score,
            config=config.support_geometry,
        )
        obs_basis_vector = np.zeros_like(basis_vector, dtype=np.float64)
        proc_basis_vector = np.zeros_like(basis_vector, dtype=np.float64)
        if obs_indices:
            obs_basis_vector[list(obs_indices)] = basis_vector[list(obs_indices)]
        if proc_indices:
            proc_basis_vector[list(proc_indices)] = basis_vector[list(proc_indices)]
        corr_obs = float(np.dot(obs_basis_vector, posterior.mean))
        corr_proc = float(np.dot(proc_basis_vector, posterior.mean))
        effect_obs = _subspace_effect(posterior.mean, posterior.covariance, active_indices=obs_indices)
        effect_proc = _subspace_effect(posterior.mean, posterior.covariance, active_indices=proc_indices)
        adapted_prediction = (
            diagnostics.prediction
            + posterior.gamma_obs * effect_obs * corr_obs
            + posterior.gamma_proc * effect_proc * corr_proc
        )
        sample_records.append(
            LatentPredictionRecord(
                split_name=split_name,
                scenario_name=episode_grid.scenario_name,
                scenario_zh_name=episode_grid.scenario_zh_name,
                domain_bucket=episode_grid.domain_bucket,
                episode_index=episode_index,
                target_minute=adapter.target_position(episode_grid, target_index=target_minute),
                target_time_h=adapter.target_time_value(episode_grid, target_index=target_minute),
                baseline_prediction=diagnostics.prediction,
                adapted_prediction=float(adapted_prediction),
                target_value=adapter.target_value(episode_grid, target_index=target_minute),
                posterior_mean=np.asarray(posterior.mean, dtype=np.float64),
                posterior_trace=posterior.posterior_trace,
                posterior_trace_obs=posterior.posterior_trace_obs,
                posterior_trace_proc=posterior.posterior_trace_proc,
                support_distance=posterior.support_distance,
                support_distance_obs=posterior.support_distance_obs,
                support_distance_proc=posterior.support_distance_proc,
                gamma=posterior.gamma,
                gamma_obs=posterior.gamma_obs,
                gamma_proc=posterior.gamma_proc,
                trust_score=trust_score,
                abstain=posterior.abstain,
                dominant_obs_axis=posterior.dominant_obs_axis,
                dominant_proc_axis=posterior.dominant_proc_axis,
            )
        )
        delayed_samples.append(
            (
                np.asarray(basis_vector, dtype=np.float64),
                float(diagnostics.prediction),
                adapter.target_value(episode_grid, target_index=target_minute),
                diagnostics,
                np.asarray(feature_window, dtype=np.float64),
                np.asarray(summary_residual, dtype=np.float64),
            )
        )

    while delayed_samples:
        basis_vector, baseline_prediction, target_value, diagnostics, feature_window, summary_residual = delayed_samples.pop(0)
        residual = float(target_value - baseline_prediction)
        posterior_mean, posterior_covariance = _update_posterior_once(
            posterior_mean,
            posterior_covariance,
            basis_vector,
            residual,
            observation_noise_variance=config.online.label_observation_noise_variance,
            variance_floor=label_variance_floor,
        )
        update_count += 1
        if config.process_evidence.enabled:
            _append_process_feature_buffers(
                basis_vector=basis_vector,
                residual=residual,
                diagnostics=diagnostics,
                feature_window=feature_window,
                summary_residual=summary_residual,
            )

    if config.observation_evidence.enabled:
        posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
            posterior_mean,
            posterior_covariance,
            feature_buffer=observation_feature_buffer,
            latent_observation_fn=adapter_model.observation_latent_observation,
            active_indices=obs_indices,
            observation_variance=observation_variance_eff,
            variance_floor=config.observation_evidence.variance_floor,
            update_space=config.update_space,
        )
        update_count += int(did_update)
    if config.process_evidence.enabled:
        posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
            posterior_mean,
            posterior_covariance,
            feature_buffer=dynamics_feature_buffer,
            latent_observation_fn=adapter_model.dynamics_latent_observation,
            active_indices=dynamics_indices,
            observation_variance=config.process_evidence.observation_variance,
            variance_floor=config.process_evidence.variance_floor,
            update_space=config.update_space,
        )
        update_count += int(did_update)
        posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
            posterior_mean,
            posterior_covariance,
            feature_buffer=load_feature_buffer,
            latent_observation_fn=adapter_model.load_latent_observation,
            active_indices=load_indices,
            observation_variance=config.process_evidence.observation_variance,
            variance_floor=config.process_evidence.variance_floor,
            update_space=config.update_space,
        )
        update_count += int(did_update)
        posterior_mean, posterior_covariance, did_update = _apply_buffered_subspace_update(
            posterior_mean,
            posterior_covariance,
            feature_buffer=readout_feature_buffer,
            latent_observation_fn=adapter_model.readout_latent_observation,
            active_indices=readout_indices,
            observation_variance=config.process_evidence.observation_variance,
            variance_floor=config.process_evidence.variance_floor,
            update_space=config.update_space,
        )
        update_count += int(did_update)

    final_support_distance_obs = _subspace_support_distance(
        posterior_mean,
        adapter_model.support_latents,
        active_indices=obs_indices,
    )
    final_support_distance_proc = _subspace_support_distance(
        posterior_mean,
        adapter_model.support_latents,
        active_indices=proc_indices,
    )
    final_support_distance = final_support_distance_obs + final_support_distance_proc
    final_posterior = _build_latent_posterior(
        config.axes,
        posterior_mean,
        posterior_covariance,
        support_distance=final_support_distance,
        support_distance_obs=final_support_distance_obs,
        support_distance_proc=final_support_distance_proc,
        num_updates=update_count,
        trust_score=trust_score,
        config=config.support_geometry,
    )
    gamma_values = np.asarray([record.gamma for record in sample_records], dtype=np.float64)
    trace_values = np.asarray([record.posterior_trace for record in sample_records], dtype=np.float64)
    trace_obs_values = np.asarray([record.posterior_trace_obs for record in sample_records], dtype=np.float64)
    trace_proc_values = np.asarray([record.posterior_trace_proc for record in sample_records], dtype=np.float64)
    distance_values = np.asarray([record.support_distance for record in sample_records], dtype=np.float64)
    distance_obs_values = np.asarray([record.support_distance_obs for record in sample_records], dtype=np.float64)
    distance_proc_values = np.asarray([record.support_distance_proc for record in sample_records], dtype=np.float64)
    trace = LatentEpisodeTrace(
        split_name=split_name,
        scenario_name=episode_grid.scenario_name,
        scenario_zh_name=episode_grid.scenario_zh_name,
        domain_bucket=episode_grid.domain_bucket,
        episode_index=episode_index,
        evidence=evidence.as_dict(),
        initial_diagnostics=initial_diagnostics.as_dict(),
        raw_prior_response=latent_prior.raw_response_text,
        prior=latent_prior.as_dict(),
        reliability=reliability,
        final_posterior=final_posterior.as_dict(),
        dominant_prior_axis=_dominant_axis(config.axes, latent_prior.mean, latent_prior.covariance),
        dominant_posterior_axis=_dominant_axis(config.axes, final_posterior.mean, final_posterior.covariance),
        dominant_obs_axis=final_posterior.dominant_obs_axis,
        dominant_proc_axis=final_posterior.dominant_proc_axis,
        avg_gamma=float(np.mean(gamma_values)) if gamma_values.size else 0.0,
        avg_posterior_trace=float(np.mean(trace_values)) if trace_values.size else 0.0,
        avg_posterior_trace_obs=float(np.mean(trace_obs_values)) if trace_obs_values.size else 0.0,
        avg_posterior_trace_proc=float(np.mean(trace_proc_values)) if trace_proc_values.size else 0.0,
        avg_support_distance=float(np.mean(distance_values)) if distance_values.size else 0.0,
        avg_support_distance_obs=float(np.mean(distance_obs_values)) if distance_obs_values.size else 0.0,
        avg_support_distance_proc=float(np.mean(distance_proc_values)) if distance_proc_values.size else 0.0,
        abstain_count=int(sum(record.abstain for record in sample_records)),
        num_samples=len(sample_records),
        error=latent_prior.error,
    )
    return RoamEpisodeResult(
        sample_records=tuple(sample_records),
        decision_trace=trace,
    )


def collect_roam_episode_outputs(
    *,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    adapter_model: LatentAdapterModel,
    episode_grids: list,
    split_name: str,
    window_size_minutes: int,
    evidence_bundle: EvidenceBundle,
    prior_engine: SemanticPriorEngine | None,
    config: RoamLatentConfig,
    task_adapter: RoamTaskAdapter,
    cache_dir: Path | None = None,
    prior_results: dict[int, PriorEngineResult] | None = None,
) -> tuple[tuple[LatentPredictionRecord, ...], tuple[LatentEpisodeTrace, ...]]:
    all_records: list[LatentPredictionRecord] = []
    all_traces: list[LatentEpisodeTrace] = []
    for episode_index, episode_grid in enumerate(episode_grids):
        episode_prior_result = None
        if prior_results is not None:
            if episode_index not in prior_results:
                raise ValueError(f"missing precomputed prior_result for episode_index={episode_index}")
            episode_prior_result = prior_results[episode_index]
        episode_result = run_roam_episode(
            model=model,
            reference_stats=reference_stats,
            adapter_model=adapter_model,
            episode_grid=episode_grid,
            episode_index=episode_index,
            split_name=split_name,
            window_size_minutes=window_size_minutes,
            evidence_bundle=evidence_bundle,
            prior_engine=prior_engine,
            config=config,
            cache_dir=cache_dir,
            prior_result=episode_prior_result,
            task_adapter=task_adapter,
        )
        all_records.extend(episode_result.sample_records)
        all_traces.append(episode_result.decision_trace)
    return tuple(all_records), tuple(all_traces)


def build_default_evidence_bundle() -> EvidenceBundle:
    return EvidenceBundle(global_evidence=OpenEvidence(), scenario_overrides={})
