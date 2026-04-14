from __future__ import annotations

from dataclasses import dataclass

from roam.methods.sequence_regressor import SequenceRegressorConfig, TorchSequenceRegressor


@dataclass(frozen=True)
class InformerRegressorConfig(SequenceRegressorConfig):
    model_type: str = "informer"
    name: str = "informer"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    num_attention_heads: int = 4
    feedforward_size: int = 128
    distil: bool = True
    sampling_factor: int = 5

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
        if self.sampling_factor < 1:
            raise ValueError("sampling_factor must be at least 1")


class TorchInformerRegressor(TorchSequenceRegressor):
    def __init__(self, config: InformerRegressorConfig | None = None):
        super().__init__(config or InformerRegressorConfig())
