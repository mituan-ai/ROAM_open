"""Shared regression protocol and torch-based backbone implementations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol
import pickle

import numpy as np

from roam.sequence import SequenceRegressionDataset

SUPPORTED_SEQUENCE_MODEL_TYPES = (
    "svr",
    "xgboost",
    "gru",
    "lstm",
    "transformer",
    "informer",
    "mamba",
)

TORCH_BACKBONE_TYPES = (
    "gru",
    "lstm",
    "transformer",
    "informer",
    "mamba",
)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError(
            "Torch backbones require PyTorch. 请先安装 torch，再运行时序专用模型实验。"
        ) from exc
    return torch, nn


@dataclass(frozen=True)
class SequenceRegressorConfig:
    model_type: str = "gru"
    name: str = "gru"
    batch_size: int = 32
    max_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    random_seed: int = 42
    gradient_clip_norm: float | None = 1.0

    def validate(self) -> None:
        if self.model_type not in SUPPORTED_SEQUENCE_MODEL_TYPES:
            raise ValueError(
                f"Unsupported sequence model_type={self.model_type!r}. "
                f"Expected one of {SUPPORTED_SEQUENCE_MODEL_TYPES}."
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be at least 1")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when provided")

    def with_device(self, device: str) -> SequenceRegressorConfig:
        return replace(self, device=device)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequenceReferenceStats:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    prediction_mean: np.ndarray
    prediction_std: np.ndarray
    hidden_mean: np.ndarray
    hidden_std: np.ndarray
    feature_sensitivity_baseline: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
            "prediction_mean": self.prediction_mean.tolist(),
            "prediction_std": self.prediction_std.tolist(),
            "hidden_mean": self.hidden_mean.tolist(),
            "hidden_std": self.hidden_std.tolist(),
            "feature_sensitivity_baseline": self.feature_sensitivity_baseline.tolist(),
        }


class SequenceRegressorProtocol(Protocol):
    config: SequenceRegressorConfig

    def fit(self, dataset: SequenceRegressionDataset) -> dict[str, float]:
        ...

    def predict(self, features: np.ndarray) -> np.ndarray:
        ...

    def encode_hidden(self, features: np.ndarray) -> np.ndarray:
        ...

    def predict_from_hidden(self, hidden_state: np.ndarray) -> np.ndarray:
        ...

    def evaluate(self, dataset: SequenceRegressionDataset) -> dict[str, float]:
        ...

    def export_reference_stats(self, dataset: SequenceRegressionDataset) -> SequenceReferenceStats:
        ...

    def save_artifact(self, output_dir: Path) -> dict[str, object]:
        ...

    def load_artifact(self, output_dir: Path) -> dict[str, object]:
        ...


def fit_hidden_readout(
    hidden_state: np.ndarray,
    normalized_targets: np.ndarray,
    *,
    ridge_lambda: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_matrix = np.asarray(hidden_state, dtype=np.float64)
    target_matrix = np.asarray(normalized_targets, dtype=np.float64)
    if hidden_matrix.ndim != 2:
        raise ValueError("hidden_state must have shape (num_samples, hidden_dim)")
    if target_matrix.ndim != 2:
        raise ValueError("normalized_targets must have shape (num_samples, output_dim)")
    design_matrix = np.column_stack(
        (
            hidden_matrix,
            np.ones(hidden_matrix.shape[0], dtype=np.float64),
        )
    )
    regularization = ridge_lambda * np.eye(design_matrix.shape[1], dtype=np.float64)
    regularization[-1, -1] = 0.0
    weights = np.linalg.solve(
        design_matrix.T @ design_matrix + regularization,
        design_matrix.T @ target_matrix,
    )
    return (
        np.asarray(weights[:-1, :], dtype=np.float64),
        np.asarray(weights[-1, :], dtype=np.float64),
    )


class BaseSequenceRegressor:
    def __init__(self, config: SequenceRegressorConfig | None = None):
        self.config = config or SequenceRegressorConfig()
        self.config.validate()
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._target_mean: np.ndarray | None = None
        self._target_std: np.ndarray | None = None

    def evaluate(self, dataset: SequenceRegressionDataset) -> dict[str, float]:
        predictions = self.predict(dataset.features)
        errors = predictions - dataset.targets
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(np.square(errors))))
        return {"mae": mae, "rmse": rmse}

    def export_reference_stats(self, dataset: SequenceRegressionDataset) -> SequenceReferenceStats:
        dataset.validate()
        predictions = self.predict(dataset.features)
        hidden_state = self.encode_hidden(dataset.features)
        feature_sensitivity_baseline = self._compute_feature_sensitivity_baseline(dataset.features, predictions)
        return SequenceReferenceStats(
            feature_mean=np.asarray(dataset.features.mean(axis=(0, 1)), dtype=np.float64),
            feature_std=np.asarray(np.clip(dataset.features.std(axis=(0, 1)), 1e-6, None), dtype=np.float64),
            target_mean=np.asarray(dataset.targets.mean(axis=0), dtype=np.float64),
            target_std=np.asarray(np.clip(dataset.targets.std(axis=0), 1e-6, None), dtype=np.float64),
            prediction_mean=np.asarray(predictions.mean(axis=0), dtype=np.float64),
            prediction_std=np.asarray(np.clip(predictions.std(axis=0), 1e-6, None), dtype=np.float64),
            hidden_mean=np.asarray(hidden_state.mean(axis=0), dtype=np.float64),
            hidden_std=np.asarray(np.clip(hidden_state.std(axis=0), 1e-6, None), dtype=np.float64),
            feature_sensitivity_baseline=np.asarray(feature_sensitivity_baseline, dtype=np.float64),
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        if self._feature_mean is None or self._feature_std is None:
            raise RuntimeError("Feature normalization statistics are unavailable before fit")
        return (features - self._feature_mean) / self._feature_std

    def _denormalize_targets(self, targets: np.ndarray) -> np.ndarray:
        if self._target_mean is None or self._target_std is None:
            raise RuntimeError("Target normalization statistics are unavailable before fit")
        return targets * self._target_std + self._target_mean

    def _normalize_targets(self, targets: np.ndarray) -> np.ndarray:
        if self._target_mean is None or self._target_std is None:
            raise RuntimeError("Target normalization statistics are unavailable before fit")
        return (targets - self._target_mean) / self._target_std

    def _prepare_normalization(self, dataset: SequenceRegressionDataset) -> tuple[np.ndarray, np.ndarray]:
        features = dataset.features.astype(np.float32, copy=False)
        targets = dataset.targets.astype(np.float32, copy=False)
        self._feature_mean = features.mean(axis=(0, 1), keepdims=True)
        self._feature_std = np.clip(features.std(axis=(0, 1), keepdims=True), 1e-6, None)
        self._target_mean = targets.mean(axis=0, keepdims=True)
        self._target_std = np.clip(targets.std(axis=0, keepdims=True), 1e-6, None)

        binary_feature_mask = np.all((features == 0.0) | (features == 1.0), axis=(0, 1))
        if np.any(binary_feature_mask):
            self._feature_mean[..., binary_feature_mask] = 0.0
            self._feature_std[..., binary_feature_mask] = 1.0

        normalized_features = (features - self._feature_mean) / self._feature_std
        normalized_targets = (targets - self._target_mean) / self._target_std
        return normalized_features, normalized_targets

    def _compute_feature_sensitivity_baseline(
        self,
        features: np.ndarray,
        predictions: np.ndarray,
    ) -> np.ndarray:
        feature_means = np.asarray(features.mean(axis=(0, 1)), dtype=np.float64)
        sensitivities = np.zeros(features.shape[2], dtype=np.float64)
        for feature_index in range(features.shape[2]):
            masked_features = np.array(features, copy=True, dtype=np.float64)
            masked_features[:, :, feature_index] = feature_means[feature_index]
            masked_predictions = self.predict(masked_features)
            sensitivities[feature_index] = float(np.mean(np.abs(masked_predictions - predictions)))
        return sensitivities

    def _serialization_payload(self) -> dict[str, object]:
        return {
            "config": self.config.as_dict(),
            "feature_mean": None if self._feature_mean is None else np.asarray(self._feature_mean, dtype=np.float64),
            "feature_std": None if self._feature_std is None else np.asarray(self._feature_std, dtype=np.float64),
            "target_mean": None if self._target_mean is None else np.asarray(self._target_mean, dtype=np.float64),
            "target_std": None if self._target_std is None else np.asarray(self._target_std, dtype=np.float64),
        }

    def _restore_serialization_payload(self, payload: dict[str, object]) -> None:
        feature_mean = payload.get("feature_mean")
        feature_std = payload.get("feature_std")
        target_mean = payload.get("target_mean")
        target_std = payload.get("target_std")
        self._feature_mean = None if feature_mean is None else np.asarray(feature_mean, dtype=np.float64)
        self._feature_std = None if feature_std is None else np.asarray(feature_std, dtype=np.float64)
        self._target_mean = None if target_mean is None else np.asarray(target_mean, dtype=np.float64)
        self._target_std = None if target_std is None else np.asarray(target_std, dtype=np.float64)


class TorchSequenceRegressor(BaseSequenceRegressor):
    def __init__(self, config: SequenceRegressorConfig | None = None):
        super().__init__(config)
        self._model: Any | None = None
        self._torch: Any | None = None
        self._input_size: int | None = None
        self._output_dim: int | None = None
        self._window_size: int | None = None

    def fit(self, dataset: SequenceRegressionDataset) -> dict[str, float]:
        dataset.validate()
        self.config.validate()
        torch, nn = _require_torch()
        self._torch = torch
        torch.manual_seed(self.config.random_seed)

        normalized_features, normalized_targets = self._prepare_normalization(dataset)
        train_inputs = torch.as_tensor(normalized_features, dtype=torch.float32)
        train_targets = torch.as_tensor(normalized_targets, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(self.config.batch_size, len(train_dataset)),
            shuffle=True,
        )

        self._input_size = dataset.feature_dim
        self._output_dim = dataset.targets.shape[1]
        self._window_size = dataset.window_size
        self._model = _build_sequence_model(
            nn,
            torch,
            config=self.config,
            input_size=dataset.feature_dim,
            hidden_size=int(getattr(self.config, "hidden_size", 8)),
            num_layers=int(getattr(self.config, "num_layers", 1)),
            dropout=float(getattr(self.config, "dropout", 0.0)),
            output_dim=dataset.targets.shape[1],
            window_size=dataset.window_size,
        ).to(self.config.device)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.MSELoss()

        last_loss = float("nan")
        self._model.train()
        for _ in range(self.config.max_epochs):
            epoch_losses: list[float] = []
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.config.device)
                batch_targets = batch_targets.to(self.config.device)
                optimizer.zero_grad()
                predictions = self._model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)
                loss.backward()
                if self.config.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self.config.gradient_clip_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))
            last_loss = float(np.mean(epoch_losses))

        metrics = {"train_loss": last_loss}
        metrics.update(self.evaluate(dataset))
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise RuntimeError("Model has not been fitted yet")
        normalized_features = self._normalize_features(np.asarray(features, dtype=np.float32))
        self._model.eval()
        with self._torch.no_grad():
            inputs = self._torch.as_tensor(normalized_features, dtype=self._torch.float32, device=self.config.device)
            outputs = self._model(inputs).detach().cpu().numpy()
        return self._denormalize_targets(outputs)

    def encode_hidden(self, features: np.ndarray) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise RuntimeError("Model has not been fitted yet")
        normalized_features = self._normalize_features(np.asarray(features, dtype=np.float32))
        self._model.eval()
        with self._torch.no_grad():
            inputs = self._torch.as_tensor(normalized_features, dtype=self._torch.float32, device=self.config.device)
            hidden_state = self._model.encode(inputs).detach().cpu().numpy()
        return np.asarray(hidden_state, dtype=np.float32)

    def predict_from_hidden(self, hidden_state: np.ndarray) -> np.ndarray:
        if self._model is None or self._torch is None:
            raise RuntimeError("Model has not been fitted yet")
        hidden_state_array = np.asarray(hidden_state, dtype=np.float32)
        if hidden_state_array.ndim == 1:
            hidden_state_array = hidden_state_array[None, :]
        self._model.eval()
        with self._torch.no_grad():
            inputs = self._torch.as_tensor(hidden_state_array, dtype=self._torch.float32, device=self.config.device)
            outputs = self._model.forward_from_hidden(inputs).detach().cpu().numpy()
        return self._denormalize_targets(outputs)

    def save_artifact(self, output_dir: Path) -> dict[str, object]:
        if self._model is None or self._torch is None:
            raise RuntimeError("Model has not been fitted yet")
        payload = self._serialization_payload()
        payload.update(
            {
                "input_size": self._input_size,
                "output_dim": self._output_dim,
                "window_size": self._window_size,
                "state_dict": self._model.state_dict(),
            }
        )
        artifact_path = output_dir / "model.pt"
        self._torch.save(payload, artifact_path)
        return {"artifact_format": "torch_state_dict", "artifact_path": str(artifact_path.resolve())}

    def load_artifact(self, output_dir: Path) -> dict[str, object]:
        torch, nn = _require_torch()
        self._torch = torch
        payload = torch.load(output_dir / "model.pt", map_location=self.config.device, weights_only=False)
        self._restore_serialization_payload(payload)
        self._input_size = int(payload["input_size"])
        self._output_dim = int(payload["output_dim"])
        self._window_size = int(payload["window_size"])
        self._model = _build_sequence_model(
            nn,
            torch,
            config=self.config,
            input_size=self._input_size,
            hidden_size=int(getattr(self.config, "hidden_size", 8)),
            num_layers=int(getattr(self.config, "num_layers", 1)),
            dropout=float(getattr(self.config, "dropout", 0.0)),
            output_dim=self._output_dim,
            window_size=self._window_size,
        ).to(self.config.device)
        self._model.load_state_dict(payload["state_dict"])
        self._model.eval()
        return {"artifact_format": "torch_state_dict", "artifact_path": str((output_dir / 'model.pt').resolve())}


def _build_sequence_model(
    nn: Any,
    torch: Any,
    *,
    config: SequenceRegressorConfig,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    output_dim: int,
    window_size: int,
) -> Any:
    if config.model_type not in TORCH_BACKBONE_TYPES:
        raise ValueError(
            f"Unsupported torch backbone model_type={config.model_type!r}. "
            f"Expected one of {TORCH_BACKBONE_TYPES}."
        )
    sequence_dropout = dropout if num_layers > 1 else 0.0

    class _RecurrentNetwork(nn.Module):
        def __init__(self, sequence_cls: Any) -> None:
            super().__init__()
            self.recurrent = sequence_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=sequence_dropout,
            )
            self.head = nn.Linear(hidden_size, output_dim)

        def encode(self, inputs: Any) -> Any:
            _, hidden = self.recurrent(inputs)
            hidden_state = hidden[0] if isinstance(hidden, tuple) else hidden
            return hidden_state[-1]

        def forward_from_hidden(self, hidden_state: Any) -> Any:
            return self.head(hidden_state)

        def forward(self, inputs: Any) -> Any:
            return self.forward_from_hidden(self.encode(inputs))

    class _TransformerNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            num_attention_heads = int(getattr(config, "num_attention_heads", 4))
            feedforward_size = int(getattr(config, "feedforward_size", hidden_size * 4))
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.position_embedding = nn.Parameter(torch.zeros(1, window_size, hidden_size))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_attention_heads,
                dim_feedforward=feedforward_size,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, output_dim)

        def encode(self, inputs: Any) -> Any:
            projected = self.input_projection(inputs) + self.position_embedding[:, : inputs.shape[1], :]
            encoded = self.encoder(projected)
            return self.norm(encoded[:, -1, :])

        def forward_from_hidden(self, hidden_state: Any) -> Any:
            return self.head(hidden_state)

        def forward(self, inputs: Any) -> Any:
            return self.forward_from_hidden(self.encode(inputs))

    class _InformerNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            try:
                from transformers import InformerConfig, InformerModel
            except ImportError as exc:
                raise ImportError(
                    "Informer backbone requires transformers. 请先安装 transformers 再运行 Informer 实验。"
                ) from exc
            num_attention_heads = int(getattr(config, "num_attention_heads", 4))
            feedforward_size = int(getattr(config, "feedforward_size", hidden_size * 4))
            lags_sequence = [1]
            context_length = max(window_size - max(lags_sequence), 1)
            self.time_feature_size = 1
            informer_config = InformerConfig(
                input_size=input_size,
                prediction_length=1,
                context_length=context_length,
                lags_sequence=lags_sequence,
                d_model=hidden_size,
                encoder_layers=num_layers,
                decoder_layers=1,
                encoder_attention_heads=num_attention_heads,
                decoder_attention_heads=num_attention_heads,
                encoder_ffn_dim=feedforward_size,
                decoder_ffn_dim=feedforward_size,
                dropout=dropout,
                attention_dropout=dropout,
                activation_dropout=dropout,
                num_time_features=self.time_feature_size,
                num_dynamic_real_features=0,
                num_static_real_features=0,
                num_static_categorical_features=0,
                distil=bool(getattr(config, "distil", True)),
                sampling_factor=int(getattr(config, "sampling_factor", 5)),
            )
            self.model = InformerModel(informer_config)
            self.head = nn.Linear(hidden_size, output_dim)

        def encode(self, inputs: Any) -> Any:
            batch_size, sequence_length, _ = inputs.shape
            past_time_features = torch.zeros(
                batch_size,
                sequence_length,
                self.time_feature_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            future_time_features = torch.zeros(
                batch_size,
                1,
                self.time_feature_size,
                dtype=inputs.dtype,
                device=inputs.device,
            )
            past_observed_mask = torch.ones_like(inputs)
            outputs = self.model(
                past_values=inputs,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features,
            )
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                hidden_state = outputs[0]
            return hidden_state[:, -1, :]

        def forward_from_hidden(self, hidden_state: Any) -> Any:
            return self.head(hidden_state)

        def forward(self, inputs: Any) -> Any:
            return self.forward_from_hidden(self.encode(inputs))

    class _MambaNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            try:
                from transformers import MambaConfig, MambaModel
            except ImportError as exc:
                raise ImportError(
                    "Mamba backbone requires transformers with official MambaModel support. "
                    "请先安装 transformers 再运行 Mamba 实验。"
                ) from exc
            expand_factor = int(getattr(config, "expand_factor", 2))
            self.input_projection = nn.Linear(input_size, hidden_size)
            self.model = MambaModel(
                MambaConfig(
                    hidden_size=hidden_size,
                    state_size=int(getattr(config, "state_size", 16)),
                    num_hidden_layers=num_layers,
                    conv_kernel=int(getattr(config, "conv_kernel_size", 4)),
                    expand=expand_factor,
                    intermediate_size=hidden_size * expand_factor,
                )
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Linear(hidden_size, output_dim)

        def encode(self, inputs: Any) -> Any:
            projected = self.input_projection(inputs)
            outputs = self.model(inputs_embeds=projected)
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                hidden_state = outputs[0]
            return self.norm(hidden_state[:, -1, :])

        def forward_from_hidden(self, hidden_state: Any) -> Any:
            return self.head(hidden_state)

        def forward(self, inputs: Any) -> Any:
            return self.forward_from_hidden(self.encode(inputs))

    if config.model_type == "gru":
        return _RecurrentNetwork(nn.GRU)
    if config.model_type == "lstm":
        return _RecurrentNetwork(nn.LSTM)
    if config.model_type == "transformer":
        return _TransformerNetwork()
    if config.model_type == "informer":
        return _InformerNetwork()
    if config.model_type == "mamba":
        return _MambaNetwork()
    raise ValueError(f"Unsupported torch backbone model_type={config.model_type!r}")


def save_pickle_artifact(output_path: Path, payload: dict[str, object]) -> None:
    with output_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)


def load_pickle_artifact(output_path: Path) -> dict[str, object]:
    with output_path.open("rb") as file_obj:
        return pickle.load(file_obj)
