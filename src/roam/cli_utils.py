from __future__ import annotations

import json
from pathlib import Path

from roam.experiment_io import load_simple_yaml_mapping, require_mapping, resolve_reference_path
from roam.methods.informer import InformerRegressorConfig, TorchInformerRegressor
from roam.methods.gru import GRURegressorConfig, TorchGRURegressor
from roam.methods.lstm import LSTMRegressorConfig, TorchLSTMRegressor
from roam.methods.mamba import MambaRegressorConfig, TorchMambaRegressor
from roam.methods.sequence_regressor import (
    SequenceRegressorConfig,
    SequenceRegressorProtocol,
    SUPPORTED_SEQUENCE_MODEL_TYPES,
)
from roam.methods.svr import SVRRegressor, SVRRegressorConfig
from roam.methods.transformer import TransformerRegressorConfig, TorchTransformerRegressor
from roam.methods.xgboost import XGBoostRegressor, XGBoostRegressorConfig
from roam.methods.roam import (
    LatentAdapterConfig,
    ObservationEvidenceConfig,
    OnlinePosteriorConfig,
    ProcessEvidenceConfig,
    RoamLatentConfig,
    SupportGeometryConfig,
)
from roam.prior_engine import (
    DeterministicPriorEngine,
    EvidenceBundle,
    OpenAICompatiblePriorEngine,
    OpenEvidence,
    PriorEngineConfig,
    SemanticAxis,
    SemanticPriorEngine,
)
from roam.task_adapter import RoamTaskDefaults

SEQUENCE_MODEL_CONFIG_CLASSES = {
    "svr": SVRRegressorConfig,
    "xgboost": XGBoostRegressorConfig,
    "gru": GRURegressorConfig,
    "lstm": LSTMRegressorConfig,
    "transformer": TransformerRegressorConfig,
    "informer": InformerRegressorConfig,
    "mamba": MambaRegressorConfig,
}

SEQUENCE_MODEL_FACTORIES = {
    "svr": SVRRegressor,
    "xgboost": XGBoostRegressor,
    "gru": TorchGRURegressor,
    "lstm": TorchLSTMRegressor,
    "transformer": TorchTransformerRegressor,
    "informer": TorchInformerRegressor,
    "mamba": TorchMambaRegressor,
}


def select_device(device_name: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested in method config but unavailable.")
    return device_name


def build_sequence_regressor_config(
    method_config: dict[str, object],
    *,
    model_key: str,
) -> SequenceRegressorConfig:
    model_mapping = require_mapping(method_config.get(model_key), label=model_key)
    training_mapping = require_mapping(method_config.get("training"), label="training")
    model_type = str(model_mapping.get("type", "gru")).lower()
    if model_type not in SUPPORTED_SEQUENCE_MODEL_TYPES:
        raise ValueError(
            f"{model_key}.type must be one of {SUPPORTED_SEQUENCE_MODEL_TYPES}, got {model_type!r}"
        )
    resolved_name = str(model_mapping.get("name", model_type))
    config_class = SEQUENCE_MODEL_CONFIG_CLASSES[model_type]
    common_kwargs: dict[str, object] = {
        "name": resolved_name,
        "batch_size": int(training_mapping.get("batch_size", config_class().batch_size)),
        "max_epochs": int(training_mapping.get("max_epochs", config_class().max_epochs)),
        "learning_rate": float(training_mapping.get("learning_rate", config_class().learning_rate)),
        "weight_decay": float(training_mapping.get("weight_decay", config_class().weight_decay)),
        "device": str(training_mapping.get("device", "auto")),
        "random_seed": int(training_mapping.get("random_seed", config_class().random_seed)),
        "gradient_clip_norm": (
            None
            if training_mapping.get("gradient_clip_norm") is None
            else float(training_mapping["gradient_clip_norm"])
        ),
    }
    if model_type in {"gru", "lstm"}:
        return config_class(
            **common_kwargs,
            hidden_size=int(model_mapping.get("hidden_size", config_class().hidden_size)),
            num_layers=int(model_mapping.get("num_layers", config_class().num_layers)),
            dropout=float(model_mapping.get("dropout", config_class().dropout)),
        )
    if model_type in {"transformer", "informer"}:
        return config_class(
            **common_kwargs,
            hidden_size=int(model_mapping.get("hidden_size", config_class().hidden_size)),
            num_layers=int(model_mapping.get("num_layers", config_class().num_layers)),
            dropout=float(model_mapping.get("dropout", config_class().dropout)),
            num_attention_heads=int(
                model_mapping.get("num_attention_heads", config_class().num_attention_heads)
            ),
            feedforward_size=int(
                model_mapping.get("feedforward_size", config_class().feedforward_size)
            ),
            **(
                {}
                if model_type != "informer"
                else {
                    "distil": bool(model_mapping.get("distil", config_class().distil)),
                    "sampling_factor": int(
                        model_mapping.get("sampling_factor", config_class().sampling_factor)
                    ),
                }
            ),
        )
    if model_type == "mamba":
        return config_class(
            **common_kwargs,
            hidden_size=int(model_mapping.get("hidden_size", config_class().hidden_size)),
            num_layers=int(model_mapping.get("num_layers", config_class().num_layers)),
            dropout=float(model_mapping.get("dropout", config_class().dropout)),
            state_size=int(model_mapping.get("state_size", config_class().state_size)),
            conv_kernel_size=int(
                model_mapping.get("conv_kernel_size", config_class().conv_kernel_size)
            ),
            expand_factor=int(model_mapping.get("expand_factor", config_class().expand_factor)),
        )
    if model_type == "svr":
        return config_class(
            **common_kwargs,
            pca_hidden_size=int(model_mapping.get("pca_hidden_size", config_class().pca_hidden_size)),
            kernel=str(model_mapping.get("kernel", config_class().kernel)),
            c=float(model_mapping.get("c", config_class().c)),
            epsilon=float(model_mapping.get("epsilon", config_class().epsilon)),
            gamma=str(model_mapping.get("gamma", config_class().gamma)),
        )
    if model_type == "xgboost":
        return config_class(
            **common_kwargs,
            pca_hidden_size=int(model_mapping.get("pca_hidden_size", config_class().pca_hidden_size)),
            n_estimators=int(model_mapping.get("n_estimators", config_class().n_estimators)),
            max_depth=int(model_mapping.get("max_depth", config_class().max_depth)),
            subsample=float(model_mapping.get("subsample", config_class().subsample)),
            colsample_bytree=float(
                model_mapping.get("colsample_bytree", config_class().colsample_bytree)
            ),
            min_child_weight=float(
                model_mapping.get("min_child_weight", config_class().min_child_weight)
            ),
            reg_lambda=float(model_mapping.get("reg_lambda", config_class().reg_lambda)),
        )
    raise ValueError(f"Unsupported model_type {model_type!r}")


def instantiate_sequence_regressor(config: SequenceRegressorConfig) -> SequenceRegressorProtocol:
    if config.model_type not in SUPPORTED_SEQUENCE_MODEL_TYPES:
        raise ValueError(
            f"Unsupported sequence model_type={config.model_type!r}. "
            f"Expected one of {SUPPORTED_SEQUENCE_MODEL_TYPES}."
    )
    config_class = SEQUENCE_MODEL_CONFIG_CLASSES[config.model_type]
    model_class = SEQUENCE_MODEL_FACTORIES[config.model_type]
    return model_class(config_class(**config.as_dict()))


def build_sequence_regressor_config_from_payload(payload: dict[str, object]) -> SequenceRegressorConfig:
    model_type = str(payload["model_type"])
    if model_type not in SEQUENCE_MODEL_CONFIG_CLASSES:
        raise ValueError(f"Unsupported model_type in serialized payload: {model_type!r}")
    return SEQUENCE_MODEL_CONFIG_CLASSES[model_type](**payload)


def load_experiment_and_method_configs(
    *,
    config_reference: str,
    root: Path,
) -> tuple[Path, dict[str, object], Path, dict[str, object]]:
    experiment_config_path = resolve_reference_path(
        config_reference,
        config_path=root / config_reference,
        root=root,
    )
    if not experiment_config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment_config_path}")
    experiment_config = load_simple_yaml_mapping(experiment_config_path)
    method_config_path = resolve_reference_path(
        str(experiment_config["method"]),
        config_path=experiment_config_path,
        root=root,
    )
    if not method_config_path.exists():
        raise FileNotFoundError(f"Method config not found: {method_config_path}")
    method_config = load_simple_yaml_mapping(method_config_path)
    return experiment_config_path, experiment_config, method_config_path, method_config


def instantiate_sequence_regressor_from_method_config(
    method_config: dict[str, object],
    *,
    model_key: str,
) -> SequenceRegressorProtocol:
    model_config = build_sequence_regressor_config(method_config, model_key=model_key)
    device_name = select_device(model_config.device)
    return instantiate_sequence_regressor(model_config.with_device(device_name))


def write_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_axes(raw_axes: object, *, default_axes: tuple[SemanticAxis, ...]) -> tuple[SemanticAxis, ...]:
    if raw_axes is None:
        return default_axes
    if not isinstance(raw_axes, (list, tuple)):
        raise ValueError("adaptation.latent_state.axes must be a list")
    return tuple(SemanticAxis(str(axis_name)) for axis_name in raw_axes)


def optional_int(raw_value: object) -> int | None:
    if raw_value is None:
        return None
    return int(raw_value)


def _resolve_config_value(
    mapping: dict[str, object],
    key: str,
    *,
    task_default: object,
    fallback: object,
) -> object:
    if key in mapping:
        return mapping[key]
    if task_default is not None:
        return task_default
    return fallback


def _resolve_steps_and_minutes(
    mapping: dict[str, object],
    *,
    steps_key: str,
    minutes_key: str,
    task_default_steps: int | None,
    task_default_minutes: int | None,
    default_steps: int,
) -> tuple[int, int | None]:
    steps = int(
        _resolve_config_value(
            mapping,
            steps_key,
            task_default=task_default_steps,
            fallback=default_steps,
        )
    )
    minutes = _resolve_config_value(
        mapping,
        minutes_key,
        task_default=task_default_minutes,
        fallback=None,
    )
    return steps, optional_int(minutes)


def resolve_update_interval(
    mapping: dict[str, object],
    *,
    task_default_steps: int | None = None,
    task_default_minutes: int | None = None,
    default_steps: int,
) -> tuple[int, int | None]:
    return _resolve_steps_and_minutes(
        mapping,
        steps_key="update_interval_steps",
        minutes_key="update_interval_minutes",
        task_default_steps=task_default_steps,
        task_default_minutes=task_default_minutes,
        default_steps=default_steps,
    )


def resolve_label_delay(
    mapping: dict[str, object],
    *,
    task_default_steps: int | None = None,
    task_default_minutes: int | None = None,
    default_steps: int,
) -> tuple[int, int | None]:
    return _resolve_steps_and_minutes(
        mapping,
        steps_key="label_delay_steps",
        minutes_key="label_delay_minutes",
        task_default_steps=task_default_steps,
        task_default_minutes=task_default_minutes,
        default_steps=default_steps,
    )


def build_roam_latent_config(
    method_config: dict[str, object],
    *,
    task_defaults: RoamTaskDefaults | None = None,
) -> RoamLatentConfig:
    adaptation_mapping = require_mapping(method_config.get("adaptation"), label="adaptation")
    latent_state_mapping = require_mapping(adaptation_mapping.get("latent_state"), label="adaptation.latent_state")
    adapter_mapping = require_mapping(adaptation_mapping.get("adapter"), label="adaptation.adapter")
    observation_mapping = require_mapping(
        adaptation_mapping.get("observation_evidence", {}),
        label="adaptation.observation_evidence",
    )
    process_mapping = require_mapping(
        adaptation_mapping.get("process_evidence", {}),
        label="adaptation.process_evidence",
    )
    online_mapping = require_mapping(adaptation_mapping.get("online"), label="adaptation.online")
    support_mapping = require_mapping(
        adaptation_mapping.get("support_geometry", {}),
        label="adaptation.support_geometry",
    )
    observation_update_interval_steps, observation_update_interval_minutes = resolve_update_interval(
        observation_mapping,
        task_default_steps=(
            None if task_defaults is None else task_defaults.observation_update_interval_steps
        ),
        task_default_minutes=(
            None if task_defaults is None else task_defaults.observation_update_interval_minutes
        ),
        default_steps=ObservationEvidenceConfig().update_interval_steps,
    )
    process_update_interval_steps, process_update_interval_minutes = resolve_update_interval(
        process_mapping,
        task_default_steps=(
            None if task_defaults is None else task_defaults.process_update_interval_steps
        ),
        task_default_minutes=(
            None if task_defaults is None else task_defaults.process_update_interval_minutes
        ),
        default_steps=ProcessEvidenceConfig().update_interval_steps,
    )
    label_delay_steps, label_delay_minutes = resolve_label_delay(
        online_mapping,
        task_default_steps=(None if task_defaults is None else task_defaults.label_delay_steps),
        task_default_minutes=(None if task_defaults is None else task_defaults.label_delay_minutes),
        default_steps=OnlinePosteriorConfig().label_delay_steps,
    )
    return RoamLatentConfig(
        name=str(method_config.get("name", "roam_v1")),
        axes=parse_axes(latent_state_mapping.get("axes"), default_axes=RoamLatentConfig().axes),
        update_space=str(latent_state_mapping.get("update_space", RoamLatentConfig().update_space)),
        prior_variance_min=float(latent_state_mapping.get("prior_variance_min", RoamLatentConfig().prior_variance_min)),
        prior_variance_max=float(latent_state_mapping.get("prior_variance_max", RoamLatentConfig().prior_variance_max)),
        adapter=LatentAdapterConfig(
            ridge_lambda=float(adapter_mapping.get("ridge_lambda", LatentAdapterConfig().ridge_lambda)),
            latent_l2=float(adapter_mapping.get("latent_l2", LatentAdapterConfig().latent_l2)),
            joint_fit_iterations=int(
                adapter_mapping.get("joint_fit_iterations", LatentAdapterConfig().joint_fit_iterations)
            ),
            synthetic_anchor_weight=float(
                adapter_mapping.get("synthetic_anchor_weight", LatentAdapterConfig().synthetic_anchor_weight)
            ),
            real_residual_weight=float(
                adapter_mapping.get("real_residual_weight", LatentAdapterConfig().real_residual_weight)
            ),
            observation_anchor_weight=float(
                adapter_mapping.get("observation_anchor_weight", LatentAdapterConfig().observation_anchor_weight)
            ),
            observation_real_weight=float(
                adapter_mapping.get("observation_real_weight", LatentAdapterConfig().observation_real_weight)
            ),
            process_anchor_weight=float(
                adapter_mapping.get("process_anchor_weight", LatentAdapterConfig().process_anchor_weight)
            ),
            process_real_weight=float(
                adapter_mapping.get("process_real_weight", LatentAdapterConfig().process_real_weight)
            ),
            nominal_summary_weight=float(
                adapter_mapping.get("nominal_summary_weight", LatentAdapterConfig().nominal_summary_weight)
            ),
            observation_bias_magnitude=float(
                _resolve_config_value(
                    adapter_mapping,
                    "observation_bias_magnitude",
                    task_default=(None if task_defaults is None else task_defaults.observation_bias_magnitude),
                    fallback=LatentAdapterConfig().observation_bias_magnitude,
                )
            ),
            observation_scale_delta=float(
                _resolve_config_value(
                    adapter_mapping,
                    "observation_scale_delta",
                    task_default=(None if task_defaults is None else task_defaults.observation_scale_delta),
                    fallback=LatentAdapterConfig().observation_scale_delta,
                )
            ),
            dynamics_slowdown_factor=float(
                adapter_mapping.get("dynamics_slowdown_factor", LatentAdapterConfig().dynamics_slowdown_factor)
            ),
            dynamics_speedup_factor=float(
                adapter_mapping.get("dynamics_speedup_factor", LatentAdapterConfig().dynamics_speedup_factor)
            ),
        ),
        observation_evidence=ObservationEvidenceConfig(
            enabled=bool(observation_mapping.get("enabled", ObservationEvidenceConfig().enabled)),
            observation_variance=float(
                observation_mapping.get(
                    "observation_variance",
                    ObservationEvidenceConfig().observation_variance,
                )
            ),
            update_interval_steps=observation_update_interval_steps,
            update_interval_minutes=observation_update_interval_minutes,
            variance_floor=float(
                observation_mapping.get("variance_floor", ObservationEvidenceConfig().variance_floor)
            ),
        ),
        process_evidence=ProcessEvidenceConfig(
            enabled=bool(process_mapping.get("enabled", ProcessEvidenceConfig().enabled)),
            observation_variance=float(
                process_mapping.get(
                    "observation_variance",
                    ProcessEvidenceConfig().observation_variance,
                )
            ),
            update_interval_steps=process_update_interval_steps,
            update_interval_minutes=process_update_interval_minutes,
            variance_floor=float(process_mapping.get("variance_floor", ProcessEvidenceConfig().variance_floor)),
        ),
        online=OnlinePosteriorConfig(
            label_delay_steps=label_delay_steps,
            label_delay_minutes=label_delay_minutes,
            label_observation_noise_variance=float(
                online_mapping.get(
                    "label_observation_noise_variance",
                    OnlinePosteriorConfig().label_observation_noise_variance,
                )
            ),
            enable_trust_gate=bool(
                online_mapping.get("enable_trust_gate", OnlinePosteriorConfig().enable_trust_gate)
            ),
        ),
        support_geometry=SupportGeometryConfig(
            enable_risk_gate=bool(
                support_mapping.get("enable_risk_gate", SupportGeometryConfig().enable_risk_gate)
            ),
            gamma_uncertainty_scale_obs=float(
                support_mapping.get(
                    "gamma_uncertainty_scale_obs",
                    SupportGeometryConfig().gamma_uncertainty_scale_obs,
                )
            ),
            gamma_uncertainty_scale_proc=float(
                support_mapping.get(
                    "gamma_uncertainty_scale_proc",
                    SupportGeometryConfig().gamma_uncertainty_scale_proc,
                )
            ),
            gamma_support_scale_obs=float(
                support_mapping.get(
                    "gamma_support_scale_obs",
                    SupportGeometryConfig().gamma_support_scale_obs,
                )
            ),
            gamma_support_scale_proc=float(
                support_mapping.get(
                    "gamma_support_scale_proc",
                    SupportGeometryConfig().gamma_support_scale_proc,
                )
            ),
        ),
    )


def prefer_local_yaml_override(config_path: Path) -> Path:
    if config_path.suffix not in {".yaml", ".yml"}:
        return config_path
    local_path = config_path.with_name(f"{config_path.stem}.local{config_path.suffix}")
    if local_path.exists():
        return local_path
    return config_path


def load_prior_engine_mapping(
    method_config: dict[str, object],
    *,
    method_config_path: Path,
    root: Path,
) -> dict[str, object]:
    raw_prior_config = method_config.get("prior_engine")
    if isinstance(raw_prior_config, str):
        prior_config_path = resolve_reference_path(raw_prior_config, config_path=method_config_path, root=root)
        effective_prior_config_path = prefer_local_yaml_override(prior_config_path)
        loaded_mapping = load_simple_yaml_mapping(effective_prior_config_path)
        return require_mapping(
            loaded_mapping,
            label=f"prior_engine@{effective_prior_config_path}",
        )
    return require_mapping(raw_prior_config, label="prior_engine")


def build_prior_engine_config(
    method_config: dict[str, object],
    *,
    method_config_path: Path,
    root: Path,
) -> PriorEngineConfig:
    prior_mapping = load_prior_engine_mapping(
        method_config,
        method_config_path=method_config_path,
        root=root,
    )
    return PriorEngineConfig(
        engine_type=str(prior_mapping.get("type", "openai_compatible")),
        api_url=str(prior_mapping.get("api_url", "")),
        api_key=str(prior_mapping.get("api_key", "")),
        model=str(prior_mapping.get("model", "")),
        timeout_seconds=float(prior_mapping.get("timeout_seconds", PriorEngineConfig().timeout_seconds)),
        temperature=float(prior_mapping.get("temperature", PriorEngineConfig().temperature)),
        max_tokens=(
            None
            if prior_mapping.get("max_tokens", PriorEngineConfig().max_tokens) is None
            else int(prior_mapping.get("max_tokens", PriorEngineConfig().max_tokens))
        ),
        max_attempts=int(prior_mapping.get("max_attempts", PriorEngineConfig().max_attempts)),
        retry_backoff_seconds=float(
            prior_mapping.get("retry_backoff_seconds", PriorEngineConfig().retry_backoff_seconds)
        ),
        enable_cache=bool(prior_mapping.get("enable_cache", PriorEngineConfig().enable_cache)),
        fail_on_invalid_response=bool(
            prior_mapping.get("fail_on_invalid_response", PriorEngineConfig().fail_on_invalid_response)
        ),
        response_format_json_object=bool(
            prior_mapping.get(
                "response_format_json_object",
                PriorEngineConfig().response_format_json_object,
            )
        ),
    )


def parse_open_evidence(mapping: dict[str, object]) -> OpenEvidence:
    return OpenEvidence(
        operator_note=str(mapping.get("operator_note", "")),
        maintenance_note=str(mapping.get("maintenance_note", "")),
        upstream_context=str(mapping.get("upstream_context", "")),
        external_context=str(mapping.get("external_context", "")),
    )


def build_evidence_bundle(experiment_config: dict[str, object]) -> EvidenceBundle:
    evidence_mapping = experiment_config.get("evidence")
    if evidence_mapping is None:
        return EvidenceBundle(global_evidence=OpenEvidence(), scenario_overrides={})
    evidence_config = require_mapping(evidence_mapping, label="evidence")
    scenario_overrides_mapping = evidence_config.get("scenario_overrides", {})
    scenario_overrides: dict[str, OpenEvidence] = {}
    if scenario_overrides_mapping:
        scenario_overrides_raw = require_mapping(scenario_overrides_mapping, label="scenario_overrides")
        for scenario_name, override_value in scenario_overrides_raw.items():
            override_mapping = require_mapping(override_value, label=f"scenario_overrides.{scenario_name}")
            scenario_overrides[str(scenario_name)] = parse_open_evidence(override_mapping)

    global_mapping = dict(evidence_config)
    global_mapping.pop("scenario_overrides", None)
    return EvidenceBundle(
        global_evidence=parse_open_evidence(global_mapping),
        scenario_overrides=scenario_overrides,
    )


def instantiate_prior_engine(config: PriorEngineConfig) -> SemanticPriorEngine:
    if config.engine_type == "openai_compatible":
        return OpenAICompatiblePriorEngine(config)
    return DeterministicPriorEngine(config)
