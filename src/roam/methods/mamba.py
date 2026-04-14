from __future__ import annotations

from dataclasses import dataclass

from roam.methods.sequence_regressor import SequenceRegressorConfig, TorchSequenceRegressor


@dataclass(frozen=True)
class MambaRegressorConfig(SequenceRegressorConfig):
    model_type: str = "mamba"
    name: str = "mamba"
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.0
    state_size: int = 16
    conv_kernel_size: int = 4
    expand_factor: int = 2

    def validate(self) -> None:
        super().validate()
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.state_size < 1:
            raise ValueError("state_size must be at least 1")
        if self.conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be at least 1")
        if self.expand_factor < 1:
            raise ValueError("expand_factor must be at least 1")


class TorchMambaRegressor(TorchSequenceRegressor):
    def __init__(self, config: MambaRegressorConfig | None = None):
        super().__init__(config or MambaRegressorConfig())
