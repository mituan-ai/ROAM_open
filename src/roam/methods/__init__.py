"""Method implementations retained in the current ROAM repo."""

from roam.methods.informer import InformerRegressorConfig, TorchInformerRegressor
from roam.methods.gru import GRURegressorConfig, TorchGRURegressor
from roam.methods.lstm import LSTMRegressorConfig, TorchLSTMRegressor
from roam.methods.mamba import MambaRegressorConfig, TorchMambaRegressor
from roam.methods.sequence_regressor import (
    SequenceReferenceStats,
    SequenceRegressorConfig,
    SequenceRegressorProtocol,
    TorchSequenceRegressor,
)
from roam.methods.svr import SVRRegressorConfig, SVRRegressor
from roam.methods.transformer import TransformerRegressorConfig, TorchTransformerRegressor
from roam.methods.xgboost import XGBoostRegressorConfig, XGBoostRegressor
from roam.methods.roam import (
    LatentAdapterConfig,
    LatentAdapterModel,
    LatentEpisodeTrace,
    LatentPosterior,
    LatentPredictionRecord,
    LatentPrior,
    OnlinePosteriorConfig,
    RoamLatentConfig,
)

__all__ = [
    "GRURegressorConfig",
    "InformerRegressorConfig",
    "LSTMRegressorConfig",
    "LatentAdapterConfig",
    "LatentAdapterModel",
    "LatentEpisodeTrace",
    "LatentPosterior",
    "LatentPredictionRecord",
    "LatentPrior",
    "MambaRegressorConfig",
    "OnlinePosteriorConfig",
    "SVRRegressor",
    "SVRRegressorConfig",
    "RoamLatentConfig",
    "SequenceReferenceStats",
    "SequenceRegressorConfig",
    "SequenceRegressorProtocol",
    "TorchInformerRegressor",
    "TorchLSTMRegressor",
    "TorchMambaRegressor",
    "TorchSequenceRegressor",
    "TorchTransformerRegressor",
    "TorchGRURegressor",
    "TransformerRegressorConfig",
    "XGBoostRegressor",
    "XGBoostRegressorConfig",
]
