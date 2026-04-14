from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol

try:
    from openai import OpenAI, OpenAIError
except ImportError as exc:
    OpenAI = None
    OpenAIError = None
    _OPENAI_IMPORT_ERROR = exc
    _OPENAI_EXCEPTIONS: tuple[type[BaseException], ...] = ()
else:
    _OPENAI_IMPORT_ERROR = None
    _OPENAI_EXCEPTIONS = (OpenAIError,)

from roam.task_adapter import PriorKnownContext, PriorTaskContext


PROMPT_VERSION = "roam_structured_latent_prompt_20260407_zh_v5"
SUPPORTED_PRIOR_ENGINE_TYPES = ("openai_compatible", "disabled", "flat")
FLAT_PRIOR_GLOBAL_UNCERTAINTY = 0.5
FLAT_PRIOR_CONFIDENCE = 0.5


class PriorEngineError(RuntimeError):
    """Raised when the live prior-engine path cannot produce a valid response."""


class SemanticAxis(StrEnum):
    BIAS = "bias"
    SCALE = "scale"
    TRUST = "trust"
    LOAD = "load"
    DYNAMICS = "dynamics"
    READOUT = "readout"


AXIS_DESCRIPTIONS: dict[SemanticAxis, str] = {
    SemanticAxis.BIAS: "可见输入上的固定偏置或零点漂移，例如传感器整体偏高/偏低。",
    SemanticAxis.SCALE: "可见输入上的比例、量纲或缩放失真，例如流量计整体放大。",
    SemanticAxis.TRUST: "可见观测的可信度下降；数值越大表示越不可信，当前 benchmark 默认不写负方向。",
    SemanticAxis.LOAD: "当前输入里没有直接暴露出来的 hidden 负荷/上下文变化。",
    SemanticAxis.DYNAMICS: "执行器、排料、压实、脱水等过程响应快慢的变化；正方向更快，负方向更慢。",
    SemanticAxis.READOUT: "在相同可见输入下，输出映射、基线或增益发生漂移。",
}

def _all_axes() -> tuple[SemanticAxis, ...]:
    return tuple(SemanticAxis)


@dataclass(frozen=True)
class OpenEvidence:
    operator_note: str = ""
    maintenance_note: str = ""
    upstream_context: str = ""
    external_context: str = ""

    def merged(self, override: OpenEvidence | None) -> OpenEvidence:
        if override is None:
            return self
        return OpenEvidence(
            operator_note=_merge_text(self.operator_note, override.operator_note),
            maintenance_note=_merge_text(self.maintenance_note, override.maintenance_note),
            upstream_context=_merge_text(self.upstream_context, override.upstream_context),
            external_context=_merge_text(self.external_context, override.external_context),
        )

    def to_prompt_text(self) -> str:
        sections = []
        if self.operator_note:
            sections.append(f"【操作记录】\n{self.operator_note.strip()}")
        if self.maintenance_note:
            sections.append(f"【检修记录】\n{self.maintenance_note.strip()}")
        if self.upstream_context:
            sections.append(f"【上游信息】\n{self.upstream_context.strip()}")
        if self.external_context:
            sections.append(f"【外部补充】\n{self.external_context.strip()}")
        return "\n\n".join(sections) if sections else "【无文本证据】\n未提供额外现场文本证据。"

    def as_dict(self) -> dict[str, str]:
        return {
            "operator_note": self.operator_note,
            "maintenance_note": self.maintenance_note,
            "upstream_context": self.upstream_context,
            "external_context": self.external_context,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    global_evidence: OpenEvidence = field(default_factory=OpenEvidence)
    scenario_overrides: dict[str, OpenEvidence] = field(default_factory=dict)

    def for_scenario(self, scenario_name: str) -> OpenEvidence:
        return self.global_evidence.merged(self.scenario_overrides.get(scenario_name))


@dataclass(frozen=True)
class AxisSemantic:
    axis: SemanticAxis
    direction: float
    strength: float
    confidence: float
    rationale: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "direction": self.direction,
            "strength": self.strength,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PriorEngineInput:
    evidence: OpenEvidence
    diagnostics_summary: str
    diagnostics_payload: dict[str, object]
    task_context: PriorTaskContext
    known_context: PriorKnownContext
    axis_schema: tuple[SemanticAxis, ...] = field(default_factory=_all_axes)

    def cache_key(self, *, model_name: str, prompt_version: str) -> str:
        serialized = json.dumps(
            {
                "model": model_name,
                "prompt_version": prompt_version,
                "axis_schema": [axis.value for axis in self.axis_schema],
                "evidence": self.evidence.as_dict(),
                "diagnostics_summary": self.diagnostics_summary,
                "diagnostics_payload": self.diagnostics_payload,
                "task_context": {
                    "task_goal": self.task_context.task_goal,
                    "environment_boundary_lines": list(self.task_context.environment_boundary_lines),
                    "hard_rule_lines": list(self.task_context.hard_rule_lines),
                },
                "known_context": self.known_context.as_dict(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PriorEngineConfig:
    engine_type: str = "openai_compatible"
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 180.0
    temperature: float = 0.0
    max_tokens: int | None = None
    max_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    enable_cache: bool = False
    fail_on_invalid_response: bool = True
    response_format_json_object: bool = True
    prompt_version: str = PROMPT_VERSION

    def validate(self) -> None:
        if self.engine_type not in SUPPORTED_PRIOR_ENGINE_TYPES:
            raise ValueError(f"Unsupported prior engine type: {self.engine_type}")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive when provided")
        if self.retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must be non-negative")


@dataclass(frozen=True)
class PriorEngineResult:
    axis_semantics: tuple[AxisSemantic, ...]
    global_uncertainty: float
    summary: str
    raw_response_text: str
    api_reliable: bool
    error: str | None = None
    cache_hit: bool = False

    def axis_mapping(self) -> dict[str, dict[str, object]]:
        return {
            axis_semantic.axis.value: axis_semantic.as_dict()
            for axis_semantic in self.axis_semantics
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "axes": self.axis_mapping(),
            "global_uncertainty": self.global_uncertainty,
            "summary": self.summary,
            "raw_response_text": self.raw_response_text,
            "api_reliable": self.api_reliable,
            "error": self.error,
            "cache_hit": self.cache_hit,
        }


class SemanticPriorEngine(Protocol):
    def infer(
        self,
        prior_input: PriorEngineInput,
        *,
        cache_dir: Path | None,
    ) -> PriorEngineResult:
        ...


def _neutral_prior_payload(
    *,
    axis_schema: tuple[SemanticAxis, ...],
    confidence: float,
    global_uncertainty: float,
    summary: str,
) -> str:
    payload_axes = {
        axis.value: {
            "direction": 0.0,
            "strength": 0.0,
            "confidence": confidence,
            "rationale": summary,
        }
        for axis in axis_schema
    }
    return json.dumps(
        {
            "axes": payload_axes,
            "global_uncertainty": global_uncertainty,
            "summary": summary,
        },
        ensure_ascii=False,
    )


class DeterministicPriorEngine:
    def __init__(self, config: PriorEngineConfig):
        self.config = config
        self.config.validate()

    def infer(
        self,
        prior_input: PriorEngineInput,
        *,
        cache_dir: Path | None,
    ) -> PriorEngineResult:
        del cache_dir
        if self.config.engine_type == "disabled":
            raw_response_text = _neutral_prior_payload(
                axis_schema=prior_input.axis_schema,
                confidence=0.0,
                global_uncertainty=1.0,
                summary="disabled_prior",
            )
        elif self.config.engine_type == "flat":
            raw_response_text = _neutral_prior_payload(
                axis_schema=prior_input.axis_schema,
                confidence=FLAT_PRIOR_CONFIDENCE,
                global_uncertainty=FLAT_PRIOR_GLOBAL_UNCERTAINTY,
                summary="flat_prior",
            )
        else:
            raise ValueError(f"DeterministicPriorEngine does not support {self.config.engine_type!r}")
        return parse_prior_engine_response(
            raw_text=raw_response_text,
            axis_schema=prior_input.axis_schema,
            api_reliable=True,
        )


def _merge_text(base_text: str, extra_text: str) -> str:
    segments = [segment.strip() for segment in (base_text, extra_text) if segment and segment.strip()]
    return "\n".join(segments)

def _clamp(value: float, *, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_prior_engine_response(
    raw_text: str,
    *,
    axis_schema: tuple[SemanticAxis, ...],
    api_reliable: bool,
    error: str | None = None,
    cache_hit: bool = False,
) -> PriorEngineResult:
    raw_text = raw_text.strip()
    if not raw_text:
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary="",
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or "empty_response",
            cache_hit=cache_hit,
        )

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary="",
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or "invalid_json",
            cache_hit=cache_hit,
        )

    if not isinstance(payload, dict):
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary="",
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or "invalid_payload",
            cache_hit=cache_hit,
        )

    raw_axes = payload.get("axes")
    if not isinstance(raw_axes, dict):
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary=str(payload.get("summary", "")),
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or "missing_axes",
            cache_hit=cache_hit,
        )

    expected_axis_names = {axis.value for axis in axis_schema}
    provided_axis_names = {str(axis_name) for axis_name in raw_axes}
    if provided_axis_names != expected_axis_names:
        mismatch = "unknown_axes" if provided_axis_names - expected_axis_names else "missing_axes"
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary=str(payload.get("summary", "")),
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or mismatch,
            cache_hit=cache_hit,
        )

    parsed_axes: list[AxisSemantic] = []
    for axis in axis_schema:
        axis_payload = raw_axes.get(axis.value)
        if not isinstance(axis_payload, dict):
            return PriorEngineResult(
                axis_semantics=(),
                global_uncertainty=1.0,
                summary=str(payload.get("summary", "")),
                raw_response_text=raw_text,
                api_reliable=False,
                error=error or f"invalid_axis_payload:{axis.value}",
                cache_hit=cache_hit,
            )
        direction = axis_payload.get("direction")
        strength = axis_payload.get("strength")
        confidence = axis_payload.get("confidence")
        if not isinstance(direction, (int, float)) or not -1.0 <= float(direction) <= 1.0:
            return PriorEngineResult(
                axis_semantics=(),
                global_uncertainty=1.0,
                summary=str(payload.get("summary", "")),
                raw_response_text=raw_text,
                api_reliable=False,
                error=error or f"invalid_direction:{axis.value}",
                cache_hit=cache_hit,
            )
        if not isinstance(strength, (int, float)) or not 0.0 <= float(strength) <= 1.0:
            return PriorEngineResult(
                axis_semantics=(),
                global_uncertainty=1.0,
                summary=str(payload.get("summary", "")),
                raw_response_text=raw_text,
                api_reliable=False,
                error=error or f"invalid_strength:{axis.value}",
                cache_hit=cache_hit,
            )
        if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
            return PriorEngineResult(
                axis_semantics=(),
                global_uncertainty=1.0,
                summary=str(payload.get("summary", "")),
                raw_response_text=raw_text,
                api_reliable=False,
                error=error or f"invalid_confidence:{axis.value}",
                cache_hit=cache_hit,
            )
        parsed_axes.append(
            AxisSemantic(
                axis=axis,
                direction=float(direction),
                strength=float(strength),
                confidence=float(confidence),
                rationale=str(axis_payload.get("rationale", "")),
            )
        )

    global_uncertainty = payload.get("global_uncertainty")
    if not isinstance(global_uncertainty, (int, float)) or not 0.0 <= float(global_uncertainty) <= 1.0:
        return PriorEngineResult(
            axis_semantics=(),
            global_uncertainty=1.0,
            summary=str(payload.get("summary", "")),
            raw_response_text=raw_text,
            api_reliable=False,
            error=error or "invalid_global_uncertainty",
            cache_hit=cache_hit,
        )

    return PriorEngineResult(
        axis_semantics=tuple(parsed_axes),
        global_uncertainty=float(global_uncertainty),
        summary=str(payload.get("summary", "")),
        raw_response_text=raw_text,
        api_reliable=api_reliable and error is None,
        error=error,
        cache_hit=cache_hit,
    )


def build_prior_engine_messages(prior_input: PriorEngineInput) -> list[dict[str, str]]:
    axis_lines = "\n".join(
        f"- {axis.value}: {AXIS_DESCRIPTIONS[axis]}"
        for axis in prior_input.axis_schema
    )
    context = prior_input.task_context
    known_context = prior_input.known_context
    environment_boundary_text = "\n".join(
        f"- {line}"
        for line in context.environment_boundary_lines
    )
    known_context_text = (
        "\n".join(f"- {line}" for line in known_context.summary_lines)
        if known_context.summary_lines
        else "- 未提供额外的已知任务/批次上下文。"
    )
    known_context_payload_text = json.dumps(
        known_context.payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    hard_rule_text = " ".join(
        (
            "若证据不足、证据互相冲突，或变化已经被当前可见输入直接表达，bias、scale、trust、load、dynamics、readout 都应尽量接近 0，并提高 global_uncertainty；",
            *context.hard_rule_lines,
        )
    )
    response_axes = ",\n".join(
        (
            f'    "{axis.value}": '
            '{"direction": 0.0, "strength": 0.0, "confidence": 0.5, "rationale": "一句中文依据"}'
        )
        for axis in prior_input.axis_schema
    )
    system_prompt = (
        "你是 ROAM 的语义先验引擎，只负责根据现场证据和诊断推断固定 latent 环境轴的语义先验。"
        "不要输出控制策略、动作建议、探针、参数更新、训练实现说明或任何 JSON 之外的内容。"
        "输出约束：必须只返回严格 JSON；axes 下每个轴都必须出现且只出现一次；"
        "direction 必须在 [-1, 1]；strength、confidence、global_uncertainty 必须在 [0, 1]。"
        "判别优先级：已知任务/批次上下文与外部证据优先，诊断值只作辅助；证据弱时应减小 strength，并提高 global_uncertainty。"
        "硬规则：若证据明确写的是“观测变化”，不要把主结论写成真实过程变化；"
        "trust 轴默认不写负方向；bias、scale、load、dynamics、readout 可以根据证据使用正负方向，拿不准就写 0；"
        f"{hard_rule_text}"
    )
    user_prompt = (
        "任务目标：\n"
        f"{context.task_goal}\n\n"
        "轴语义表：\n"
        f"{axis_lines}\n\n"
        "当前环境边界：\n"
        f"{environment_boundary_text}\n\n"
        "已知任务/批次上下文：\n"
        "下面这些信息默认视为场景起点已知、时间上合法的外生上下文，不是未来信息。\n"
        f"{known_context_text}\n\n"
        "已知任务/批次上下文 JSON：\n"
        f"{known_context_payload_text}\n\n"
        "外部证据：\n"
        f"{prior_input.evidence.to_prompt_text()}\n\n"
        "诊断摘要：\n"
        f"{prior_input.diagnostics_summary}\n\n"
        "诊断明细 JSON：\n"
        f"{json.dumps(prior_input.diagnostics_payload, ensure_ascii=False, sort_keys=True, indent=2)}\n\n"
        "输出格式：\n"
        "只返回下面这个 JSON 结构，不要加 markdown 代码块、解释文字或额外字段。\n"
        "{\n"
        '  "axes": {\n'
        f"{response_axes}\n"
        "  },\n"
        '  "global_uncertainty": 0.25,\n'
        '  "summary": "一句中文短句总结"\n'
        "}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _resolve_openai_base_url(api_url: str) -> str:
    normalized = api_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]
    return normalized


def _extract_message_content(message: object) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_value = getattr(item, "text", "")
                if isinstance(text_value, str):
                    text_parts.append(text_value)
                    continue
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part)
    raise ValueError("Unsupported OpenAI SDK message content format")


class OpenAICompatiblePriorEngine:
    def __init__(self, config: PriorEngineConfig):
        self.config = config
        self.config.validate()
        self._client = None

    def infer(
        self,
        prior_input: PriorEngineInput,
        *,
        cache_dir: Path | None,
    ) -> PriorEngineResult:
        cache_path = None
        if self.config.enable_cache and cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = prior_input.cache_key(
                model_name=self.config.model,
                prompt_version=self.config.prompt_version,
            )
            cache_path = cache_dir / f"{cache_key}.json"
            if cache_path.exists():
                cached_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_result = parse_prior_engine_response(
                    str(cached_payload.get("raw_response_text", "")),
                    axis_schema=prior_input.axis_schema,
                    api_reliable=bool(cached_payload.get("api_reliable", True)),
                    error=cached_payload.get("error"),
                    cache_hit=True,
                )
                self._raise_if_invalid(cached_result)
                return cached_result

        if not self.config.api_url or not self.config.model:
            return self._handle_failure(
                PriorEngineResult(
                    axis_semantics=(),
                    global_uncertainty=1.0,
                    summary="",
                    raw_response_text="",
                    api_reliable=False,
                    error="missing_api_configuration",
                )
            )

        api_key = self.config.api_key
        if not api_key:
            return self._handle_failure(
                PriorEngineResult(
                    axis_semantics=(),
                    global_uncertainty=1.0,
                    summary="",
                    raw_response_text="",
                    api_reliable=False,
                    error="missing_api_key",
                )
            )

        request_payload = {
            "model": self.config.model,
            "messages": build_prior_engine_messages(prior_input),
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            request_payload["max_tokens"] = self.config.max_tokens
        if self.config.response_format_json_object:
            request_payload["response_format"] = {"type": "json_object"}

        last_error: str | None = None
        for attempt in range(self.config.max_attempts):
            try:
                raw_response_text = self._send_request(api_key=api_key, payload=request_payload)
                parsed = parse_prior_engine_response(
                    raw_response_text,
                    axis_schema=prior_input.axis_schema,
                    api_reliable=True,
                )
                self._raise_if_invalid(parsed)
                if cache_path is not None:
                    cache_path.write_text(
                        json.dumps(
                            {
                                "raw_response_text": raw_response_text,
                                "api_reliable": parsed.api_reliable,
                                "error": parsed.error,
                            },
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                return parsed
            except _OPENAI_EXCEPTIONS + (TimeoutError, ValueError, PriorEngineError) as exc:
                last_error = str(exc)
                if attempt + 1 < self.config.max_attempts:
                    time.sleep(self.config.retry_backoff_seconds)

        return self._handle_failure(
            PriorEngineResult(
                axis_semantics=(),
                global_uncertainty=1.0,
                summary="",
                raw_response_text="",
                api_reliable=False,
                error=last_error or "request_failed",
            )
        )

    def _raise_if_invalid(self, result: PriorEngineResult) -> None:
        if self.config.fail_on_invalid_response and result.error is not None:
            raw_preview = result.raw_response_text[:600]
            raise PriorEngineError(
                f"Prior engine returned an invalid response: {result.error}. "
                f"raw_response_preview={raw_preview!r}"
            )

    def _handle_failure(self, result: PriorEngineResult) -> PriorEngineResult:
        if self.config.fail_on_invalid_response:
            raise PriorEngineError(f"Live prior-engine call failed: {result.error}")
        return result

    def _send_request(self, *, api_key: str, payload: dict[str, object]) -> str:
        if OpenAI is None:
            raise PriorEngineError(
                "openai SDK is not installed. Run `uv pip install openai` before using the live prior engine."
            ) from _OPENAI_IMPORT_ERROR
        if self._client is None:
            self._client = OpenAI(
                api_key=api_key,
                base_url=_resolve_openai_base_url(self.config.api_url),
                max_retries=0,
                timeout=self.config.timeout_seconds,
            )
        response = self._client.chat.completions.create(**payload)
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("OpenAI SDK response did not contain choices")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("OpenAI SDK response did not contain a message")
        return _extract_message_content(message)
