from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

from roam.methods.sequence_regressor import SequenceReferenceStats, SequenceRegressorProtocol
from roam.methods.roam import build_roam_episode_prior_input
from roam.prior_engine import (
    EvidenceBundle,
    OpenAICompatiblePriorEngine,
    PriorEngineConfig,
    PriorEngineResult,
    SemanticPriorEngine,
)
from roam.task_adapter import RoamTaskAdapter


def _instantiate_prior_engine(config: PriorEngineConfig) -> OpenAICompatiblePriorEngine:
    return OpenAICompatiblePriorEngine(config)


def prefetch_eval_prior_results(
    *,
    prior_engine_config: PriorEngineConfig,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    eval_episode_grids_by_scenario: dict[str, list],
    window_size: int,
    evidence_bundle: EvidenceBundle,
    task_adapter: RoamTaskAdapter,
    cache_dir: Path | None,
    prior_engine_factory: Callable[[PriorEngineConfig], SemanticPriorEngine] = _instantiate_prior_engine,
) -> tuple[dict[tuple[str, int], PriorEngineResult], int]:
    episode_specs: list[tuple[str, int, object]] = []
    for scenario_name, episode_grids in eval_episode_grids_by_scenario.items():
        for episode_index, episode_grid in enumerate(episode_grids):
            episode_specs.append((scenario_name, episode_index, episode_grid))
    if not episode_specs:
        return {}, 0

    thread_local = threading.local()

    def _thread_engine() -> SemanticPriorEngine:
        engine = getattr(thread_local, "prior_engine", None)
        if engine is None:
            engine = prior_engine_factory(prior_engine_config)
            setattr(thread_local, "prior_engine", engine)
        return engine

    def _infer_episode_prior(spec: tuple[str, int, object]) -> tuple[tuple[str, int], PriorEngineResult]:
        scenario_name, episode_index, episode_grid = spec
        prior_input, _ = build_roam_episode_prior_input(
            model=model,
            reference_stats=reference_stats,
            episode_grid=episode_grid,
            window_size_minutes=window_size,
            evidence_bundle=evidence_bundle,
            task_adapter=task_adapter,
        )
        prior_result = _thread_engine().infer(prior_input, cache_dir=cache_dir)
        return (scenario_name, episode_index), prior_result

    max_workers = len(episode_specs)
    prior_results: dict[tuple[str, int], PriorEngineResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_infer_episode_prior, spec): (spec[0], spec[1])
            for spec in episode_specs
        }
        for future in as_completed(future_map):
            episode_key, prior_result = future.result()
            prior_results[episode_key] = prior_result
    return prior_results, max_workers
