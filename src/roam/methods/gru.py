from __future__ import annotations

from dataclasses import dataclass

from roam.methods.sequence_regressor import (
    SequenceReferenceStats,
    SequenceRegressorConfig,
    TorchSequenceRegressor,
)


@dataclass(frozen=True)
class GRURegressorConfig(SequenceRegressorConfig):
    model_type: str = "gru"
    name: str = "gru"
    hidden_size: int = 8
    num_layers: int = 1
    dropout: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")


GRUReferenceStats = SequenceReferenceStats


class TorchGRURegressor(TorchSequenceRegressor):
    def __init__(self, config: GRURegressorConfig | None = None):
        super().__init__(config or GRURegressorConfig())
