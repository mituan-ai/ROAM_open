from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roam.methods.sequence_regressor import (
    BaseSequenceRegressor,
    SequenceRegressorConfig,
    fit_hidden_readout,
    load_pickle_artifact,
    save_pickle_artifact,
)
from roam.sequence import SequenceRegressionDataset


def _require_sklearn() -> tuple[Any, Any]:
    try:
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
    except ImportError as exc:
        raise ImportError(
            "Classical backbones require scikit-learn. 请先安装 scikit-learn 再运行 SVR/XGBoost 实验。"
        ) from exc
    return PCA, Ridge


@dataclass(frozen=True)
class ClassicalWindowRegressorConfig(SequenceRegressorConfig):
    pca_hidden_size: int = 16

    def validate(self) -> None:
        super().validate()
        if self.pca_hidden_size < 1:
            raise ValueError("pca_hidden_size must be at least 1")


class ClassicalWindowRegressor(BaseSequenceRegressor):
    def __init__(self, config: ClassicalWindowRegressorConfig):
        super().__init__(config)
        self.config = config
        self._estimator: Any | None = None
        self._pca: Any | None = None
        self._hidden_readout_weight: np.ndarray | None = None
        self._hidden_readout_bias: np.ndarray | None = None

    def fit(self, dataset: SequenceRegressionDataset) -> dict[str, float]:
        dataset.validate()
        self.config.validate()
        normalized_features, normalized_targets = self._prepare_normalization(dataset)
        flattened_features = self._flatten_features(normalized_features)
        if normalized_targets.shape[1] != 1:
            raise ValueError("ClassicalWindowRegressor currently supports scalar targets only")

        estimator = self._build_estimator()
        estimator.fit(flattened_features, normalized_targets[:, 0])
        self._estimator = estimator

        PCA, _ = _require_sklearn()
        hidden_size = min(self.config.pca_hidden_size, flattened_features.shape[1], flattened_features.shape[0])
        self._pca = PCA(n_components=hidden_size, random_state=self.config.random_seed)
        hidden_state = np.asarray(self._pca.fit_transform(flattened_features), dtype=np.float64)
        readout_weight, readout_bias = fit_hidden_readout(hidden_state, normalized_targets)
        self._hidden_readout_weight = readout_weight
        self._hidden_readout_bias = readout_bias

        metrics = {"train_loss": 0.0}
        metrics.update(self.evaluate(dataset))
        return metrics

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted yet")
        normalized_features = self._normalize_features(np.asarray(features, dtype=np.float32))
        flattened_features = self._flatten_features(normalized_features)
        predictions = np.asarray(self._estimator.predict(flattened_features), dtype=np.float64)[:, None]
        return self._denormalize_targets(predictions)

    def encode_hidden(self, features: np.ndarray) -> np.ndarray:
        if self._pca is None:
            raise RuntimeError("Model has not been fitted yet")
        normalized_features = self._normalize_features(np.asarray(features, dtype=np.float32))
        flattened_features = self._flatten_features(normalized_features)
        return np.asarray(self._pca.transform(flattened_features), dtype=np.float64)

    def predict_from_hidden(self, hidden_state: np.ndarray) -> np.ndarray:
        if self._hidden_readout_weight is None or self._hidden_readout_bias is None:
            raise RuntimeError("Model has not been fitted yet")
        hidden_array = np.asarray(hidden_state, dtype=np.float64)
        if hidden_array.ndim == 1:
            hidden_array = hidden_array[None, :]
        normalized_predictions = hidden_array @ self._hidden_readout_weight + self._hidden_readout_bias
        return self._denormalize_targets(normalized_predictions)

    def save_artifact(self, output_dir: Path) -> dict[str, object]:
        if self._estimator is None or self._pca is None:
            raise RuntimeError("Model has not been fitted yet")
        artifact_path = output_dir / "model.pkl"
        save_pickle_artifact(
            artifact_path,
            {
                "serialization": self._serialization_payload(),
                "estimator": self._estimator,
                "pca": self._pca,
                "hidden_readout_weight": self._hidden_readout_weight,
                "hidden_readout_bias": self._hidden_readout_bias,
            },
        )
        return {"artifact_format": "pickle", "artifact_path": str(artifact_path.resolve())}

    def load_artifact(self, output_dir: Path) -> dict[str, object]:
        payload = load_pickle_artifact(output_dir / "model.pkl")
        self._restore_serialization_payload(payload["serialization"])
        self._estimator = payload["estimator"]
        self._pca = payload["pca"]
        self._hidden_readout_weight = np.asarray(payload["hidden_readout_weight"], dtype=np.float64)
        self._hidden_readout_bias = np.asarray(payload["hidden_readout_bias"], dtype=np.float64)
        return {"artifact_format": "pickle", "artifact_path": str((output_dir / 'model.pkl').resolve())}

    def _flatten_features(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(features, dtype=np.float64).reshape(features.shape[0], -1)

    def _build_estimator(self) -> Any:
        raise NotImplementedError
