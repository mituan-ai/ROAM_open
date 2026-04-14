from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from roam.methods.classical import ClassicalWindowRegressor, ClassicalWindowRegressorConfig


@dataclass(frozen=True)
class XGBoostRegressorConfig(ClassicalWindowRegressorConfig):
    model_type: str = "xgboost"
    name: str = "xgboost"
    n_estimators: int = 256
    max_depth: int = 4
    subsample: float = 1.0
    colsample_bytree: float = 1.0
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0

    def validate(self) -> None:
        super().validate()
        if self.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if not (0.0 < self.subsample <= 1.0):
            raise ValueError("subsample must be in (0, 1]")
        if not (0.0 < self.colsample_bytree <= 1.0):
            raise ValueError("colsample_bytree must be in (0, 1]")
        if self.min_child_weight <= 0.0:
            raise ValueError("min_child_weight must be positive")
        if self.reg_lambda < 0.0:
            raise ValueError("reg_lambda must be non-negative")


class XGBoostRegressor(ClassicalWindowRegressor):
    def __init__(self, config: XGBoostRegressorConfig | None = None):
        super().__init__(config or XGBoostRegressorConfig())

    def _build_estimator(self) -> Any:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost backbone requires xgboost. 请先安装 xgboost 再运行 XGBoost 实验。"
            ) from exc
        return XGBRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_seed,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
        )
