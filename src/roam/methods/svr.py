from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roam.methods.classical import ClassicalWindowRegressor, ClassicalWindowRegressorConfig


@dataclass(frozen=True)
class SVRRegressorConfig(ClassicalWindowRegressorConfig):
    model_type: str = "svr"
    name: str = "svr"
    kernel: str = "rbf"
    c: float = 1.0
    epsilon: float = 0.1
    gamma: str = "scale"

    def validate(self) -> None:
        super().validate()
        if self.c <= 0.0:
            raise ValueError("c must be positive")
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")


class SVRRegressor(ClassicalWindowRegressor):
    def __init__(self, config: SVRRegressorConfig | None = None):
        super().__init__(config or SVRRegressorConfig())

    def _build_estimator(self) -> Any:
        try:
            from sklearn.svm import SVR
        except ImportError as exc:
            raise ImportError(
                "SVR backbone requires scikit-learn. 请先安装 scikit-learn 再运行 SVR 实验。"
            ) from exc
        return SVR(
            kernel=self.config.kernel,
            C=self.config.c,
            epsilon=self.config.epsilon,
            gamma=self.config.gamma,
        )
