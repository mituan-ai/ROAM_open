from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from roam.methods.sequence_regressor import (
    SequenceReferenceStats,
    SequenceRegressorProtocol,
)


ARTIFACT_METADATA_FILENAME = "artifact_metadata.json"
REFERENCE_STATS_FILENAME = "reference_stats.json"


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def compute_protocol_hash(payload: dict[str, object]) -> str:
    normalized = json.dumps(_to_jsonable(payload), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_reference_stats(output_path: Path, reference_stats: SequenceReferenceStats) -> None:
    output_path.write_text(
        json.dumps(reference_stats.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_reference_stats(input_path: Path) -> SequenceReferenceStats:
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return SequenceReferenceStats(
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
        target_mean=np.asarray(payload["target_mean"], dtype=np.float64),
        target_std=np.asarray(payload["target_std"], dtype=np.float64),
        prediction_mean=np.asarray(payload["prediction_mean"], dtype=np.float64),
        prediction_std=np.asarray(payload["prediction_std"], dtype=np.float64),
        hidden_mean=np.asarray(payload["hidden_mean"], dtype=np.float64),
        hidden_std=np.asarray(payload["hidden_std"], dtype=np.float64),
        feature_sensitivity_baseline=np.asarray(payload["feature_sensitivity_baseline"], dtype=np.float64),
    )


@dataclass(frozen=True)
class BaseModelArtifact:
    artifact_id: str
    artifact_dir: Path
    protocol_hash: str
    environment: str
    model_type: str
    method_name: str
    train_device: str
    metadata: dict[str, object]
    reference_stats: SequenceReferenceStats
    model: SequenceRegressorProtocol


def save_base_model_artifact(
    *,
    artifact_dir: Path,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    metadata: dict[str, object],
) -> dict[str, object]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_payload = model.save_artifact(artifact_dir)
    write_reference_stats(artifact_dir / REFERENCE_STATS_FILENAME, reference_stats)
    payload = {
        **metadata,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_artifact": model_payload,
    }
    (artifact_dir / ARTIFACT_METADATA_FILENAME).write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def load_base_model_artifact(artifact_dir: Path) -> BaseModelArtifact:
    metadata = json.loads((artifact_dir / ARTIFACT_METADATA_FILENAME).read_text(encoding="utf-8"))
    reference_stats = load_reference_stats(artifact_dir / REFERENCE_STATS_FILENAME)
    from roam.cli_utils import build_sequence_regressor_config_from_payload, instantiate_sequence_regressor

    model_config = build_sequence_regressor_config_from_payload(
        dict(metadata["model_config"])
    )
    model = instantiate_sequence_regressor(model_config)
    model.load_artifact(artifact_dir)
    return BaseModelArtifact(
        artifact_id=str(metadata["artifact_id"]),
        artifact_dir=artifact_dir.resolve(),
        protocol_hash=str(metadata["protocol_hash"]),
        environment=str(metadata["environment"]),
        model_type=str(metadata["model_type"]),
        method_name=str(metadata["method_name"]),
        train_device=str(metadata.get("train_device", model.config.device)),
        metadata=metadata,
        reference_stats=reference_stats,
        model=model,
    )
