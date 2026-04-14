from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SequenceRegressionDataset:
    """A minimal, validated container for sequence-to-vector regression datasets.

    Shape conventions:
    - features: (num_samples, window_size, feature_dim)
    - targets: (num_samples, output_dim)
    """

    features: np.ndarray
    targets: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    sample_contexts: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float64)
        self.targets = np.asarray(self.targets, dtype=np.float64)
        self.validate()

    def validate(self) -> None:
        if self.features.ndim != 3:
            raise ValueError("features must have shape (num_samples, window_size, feature_dim)")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape (num_samples, output_dim)")
        if self.features.shape[0] != self.targets.shape[0]:
            raise ValueError("features and targets must have the same number of samples")
        if self.features.shape[2] != len(self.feature_names):
            raise ValueError("feature_names must match the last feature dimension")
        if self.sample_contexts and len(self.sample_contexts) != self.features.shape[0]:
            raise ValueError("sample_contexts length must match the number of samples")
        if not np.all(np.isfinite(self.features)):
            raise ValueError("features must contain only finite values")
        if not np.all(np.isfinite(self.targets)):
            raise ValueError("targets must contain only finite values")

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def window_size(self) -> int:
        return int(self.features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[2])

