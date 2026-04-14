from __future__ import annotations

from dataclasses import dataclass

from roam.methods.sequence_regressor import SequenceRegressorConfig, TorchSequenceRegressor


@dataclass(frozen=True)
class TransformerRegressorConfig(SequenceRegressorConfig):
    model_type: str = "transformer"
    name: str = "transformer"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 4
    feedforward_size: int = 128

    def validate(self) -> None:
        super().validate()
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1")
        if self.feedforward_size < self.hidden_size:
            raise ValueError("feedforward_size must be >= hidden_size")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")


class TorchTransformerRegressor(TorchSequenceRegressor):
    def __init__(self, config: TransformerRegressorConfig | None = None):
        super().__init__(config or TransformerRegressorConfig())
