from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path


def parse_yaml_scalar(raw_value: str) -> object:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if raw_value.startswith("[") or raw_value.startswith("{") or raw_value.startswith(("'", '"')):
        try:
            return ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            if raw_value.startswith("[") and raw_value.endswith("]"):
                inner = raw_value[1:-1].strip()
                if not inner:
                    return []
                return [segment.strip().strip("'\"") for segment in inner.split(",")]
            raise
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        return raw_value


def _parse_simple_yaml_mapping(config_path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped_line = raw_line.split("#", 1)[0].rstrip()
        if not stripped_line:
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2 != 0:
            raise ValueError(f"Unsupported indentation in {config_path}:{line_number}")

        content = stripped_line.lstrip(" ")
        key, separator, value_text = content.partition(":")
        if not separator:
            raise ValueError(f"Invalid YAML entry in {config_path}:{line_number}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        current_mapping = stack[-1][1]

        normalized_key = key.strip()
        normalized_value = value_text.strip()
        if not normalized_value:
            next_mapping: dict[str, object] = {}
            current_mapping[normalized_key] = next_mapping
            stack.append((indent, next_mapping))
            continue

        current_mapping[normalized_key] = parse_yaml_scalar(normalized_value)

    return root


def load_simple_yaml_mapping(config_path: Path) -> dict[str, object]:
    return _parse_simple_yaml_mapping(config_path.resolve())


def require_mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def resolve_reference_path(reference: str, *, config_path: Path, root: Path) -> Path:
    raw_path = Path(reference)
    if raw_path.is_absolute():
        return raw_path
    local_candidate = (config_path.parent / raw_path).resolve()
    if local_candidate.exists():
        return local_candidate
    root_candidate = (root / raw_path).resolve()
    if root_candidate.exists():
        return root_candidate
    return local_candidate


def normalize_output_name(raw_name: str) -> str:
    normalized = Path(raw_name).name
    if normalized in {"", ".", ".."}:
        raise ValueError(f"Invalid output name: {raw_name}")
    return normalized


def build_timestamped_output_root(output_root: Path) -> Path:
    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M")
    candidate = output_root / timestamp_label
    collision_index = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp_label}_{collision_index:02d}"
        collision_index += 1
    return candidate


def build_output_dir(
    experiment_config: dict[str, object],
    *,
    method_name: str,
    root: Path,
) -> Path:
    experiment_name = normalize_output_name(str(experiment_config["name"]))
    environment_name = normalize_output_name(str(experiment_config["environment"]))
    output_root = Path(str(experiment_config.get("output_root", "outputs")))
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    return build_timestamped_output_root(output_root) / environment_name / method_name / experiment_name
