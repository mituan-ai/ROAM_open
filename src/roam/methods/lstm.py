from __future__ import annotations

from dataclasses import dataclass

from roam.methods.sequence_regressor import (
    SequenceReferenceStats,
    SequenceRegressorConfig,
    TorchSequenceRegressor,
)


@dataclass(frozen=True)
class LSTMRegressorConfig(SequenceRegressorConfig):
    model_type: str = "lstm"
    name: str = "lstm"
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


LSTMReferenceStats = SequenceReferenceStats


class TorchLSTMRegressor(TorchSequenceRegressor):
    def __init__(self, config: LSTMRegressorConfig | None = None):
        super().__init__(config or LSTMRegressorConfig())
