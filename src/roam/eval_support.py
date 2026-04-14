from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from roam.methods.sequence_regressor import SequenceReferenceStats, SequenceRegressorProtocol
from roam.methods.roam import (
    LatentAdapterModel,
    LatentEpisodeTrace,
    LatentPredictionRecord,
    RoamLatentConfig,
    collect_roam_episode_outputs,
)
from roam.prior_engine import EvidenceBundle, PriorEngineConfig, PriorEngineResult, SemanticPriorEngine
from roam.prior_prefetch import prefetch_eval_prior_results
from roam.task_adapter import RoamTaskAdapter


@dataclass(frozen=True)
class RoamEvalOutputs:
    records: tuple[LatentPredictionRecord, ...]
    traces: tuple[LatentEpisodeTrace, ...]
    records_by_scenario: dict[str, tuple[LatentPredictionRecord, ...]]
    prior_prefetch_workers: int


def write_decision_trace_jsonl(output_path: Path, traces: list[LatentEpisodeTrace] | tuple[LatentEpisodeTrace, ...]) -> None:
    with output_path.open("w", encoding="utf-8") as file_obj:
        for trace in traces:
            file_obj.write(json.dumps(trace.as_dict(), ensure_ascii=False) + "\n")


def summarize_records(
    records: list[LatentPredictionRecord] | tuple[LatentPredictionRecord, ...],
) -> dict[str, object]:
    if not records:
        return {
            "num_samples": 0,
            "baseline_mae": 0.0,
            "baseline_rmse": 0.0,
            "adapted_mae": 0.0,
            "adapted_rmse": 0.0,
            "target_mean": 0.0,
            "target_std": 0.0,
            "avg_gamma": 0.0,
            "avg_gamma_obs": 0.0,
            "avg_gamma_proc": 0.0,
            "avg_trust_score": 0.0,
            "avg_posterior_trace": 0.0,
            "avg_posterior_trace_obs": 0.0,
            "avg_posterior_trace_proc": 0.0,
            "avg_support_distance": 0.0,
            "avg_support_distance_obs": 0.0,
            "avg_support_distance_proc": 0.0,
            "abstain_count": 0,
        }
    targets = np.asarray([record.target_value for record in records], dtype=np.float64)
    baseline_predictions = np.asarray([record.baseline_prediction for record in records], dtype=np.float64)
    adapted_predictions = np.asarray([record.adapted_prediction for record in records], dtype=np.float64)
    baseline_errors = baseline_predictions - targets
    adapted_errors = adapted_predictions - targets
    gamma_values = np.asarray([record.gamma for record in records], dtype=np.float64)
    gamma_obs_values = np.asarray([record.gamma_obs for record in records], dtype=np.float64)
    gamma_proc_values = np.asarray([record.gamma_proc for record in records], dtype=np.float64)
    trust_score_values = np.asarray([record.trust_score for record in records], dtype=np.float64)
    trace_values = np.asarray([record.posterior_trace for record in records], dtype=np.float64)
    trace_obs_values = np.asarray([record.posterior_trace_obs for record in records], dtype=np.float64)
    trace_proc_values = np.asarray([record.posterior_trace_proc for record in records], dtype=np.float64)
    distance_values = np.asarray([record.support_distance for record in records], dtype=np.float64)
    distance_obs_values = np.asarray([record.support_distance_obs for record in records], dtype=np.float64)
    distance_proc_values = np.asarray([record.support_distance_proc for record in records], dtype=np.float64)
    return {
        "num_samples": int(targets.size),
        "baseline_mae": float(np.mean(np.abs(baseline_errors))),
        "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline_errors)))),
        "adapted_mae": float(np.mean(np.abs(adapted_errors))),
        "adapted_rmse": float(np.sqrt(np.mean(np.square(adapted_errors)))),
        "target_mean": float(np.mean(targets)),
        "target_std": float(np.std(targets)),
        "avg_gamma": float(np.mean(gamma_values)) if gamma_values.size else 0.0,
        "avg_gamma_obs": float(np.mean(gamma_obs_values)) if gamma_obs_values.size else 0.0,
        "avg_gamma_proc": float(np.mean(gamma_proc_values)) if gamma_proc_values.size else 0.0,
        "avg_trust_score": float(np.mean(trust_score_values)) if trust_score_values.size else 0.0,
        "avg_posterior_trace": float(np.mean(trace_values)) if trace_values.size else 0.0,
        "avg_posterior_trace_obs": float(np.mean(trace_obs_values)) if trace_obs_values.size else 0.0,
        "avg_posterior_trace_proc": float(np.mean(trace_proc_values)) if trace_proc_values.size else 0.0,
        "avg_support_distance": float(np.mean(distance_values)) if distance_values.size else 0.0,
        "avg_support_distance_obs": float(np.mean(distance_obs_values)) if distance_obs_values.size else 0.0,
        "avg_support_distance_proc": float(np.mean(distance_proc_values)) if distance_proc_values.size else 0.0,
        "abstain_count": int(sum(record.abstain for record in records)),
    }


def summarize_traces(
    traces: list[LatentEpisodeTrace] | tuple[LatentEpisodeTrace, ...],
) -> dict[str, object]:
    if not traces:
        return {
            "num_episodes": 0,
            "avg_gamma": 0.0,
            "avg_gamma_obs": 0.0,
            "avg_gamma_proc": 0.0,
            "avg_trust_score": 0.0,
            "avg_posterior_trace": 0.0,
            "avg_posterior_trace_obs": 0.0,
            "avg_posterior_trace_proc": 0.0,
            "avg_support_distance": 0.0,
            "avg_support_distance_obs": 0.0,
            "avg_support_distance_proc": 0.0,
            "abstain_count": 0,
            "dominant_posterior_axis_counts": {},
            "dominant_obs_axis_counts": {},
            "dominant_proc_axis_counts": {},
        }
    gamma_values = np.asarray([trace.avg_gamma for trace in traces], dtype=np.float64)
    gamma_obs_values = np.asarray(
        [trace.final_posterior.get("gamma_obs", trace.avg_gamma) for trace in traces],
        dtype=np.float64,
    )
    gamma_proc_values = np.asarray(
        [trace.final_posterior.get("gamma_proc", trace.avg_gamma) for trace in traces],
        dtype=np.float64,
    )
    trust_score_values = np.asarray(
        [trace.reliability.get("trust_score", 0.0) for trace in traces],
        dtype=np.float64,
    )
    trace_values = np.asarray([trace.avg_posterior_trace for trace in traces], dtype=np.float64)
    trace_obs_values = np.asarray([trace.avg_posterior_trace_obs for trace in traces], dtype=np.float64)
    trace_proc_values = np.asarray([trace.avg_posterior_trace_proc for trace in traces], dtype=np.float64)
    distance_values = np.asarray([trace.avg_support_distance for trace in traces], dtype=np.float64)
    distance_obs_values = np.asarray([trace.avg_support_distance_obs for trace in traces], dtype=np.float64)
    distance_proc_values = np.asarray([trace.avg_support_distance_proc for trace in traces], dtype=np.float64)
    dominant_axis_counts: dict[str, int] = {}
    dominant_obs_counts: dict[str, int] = {}
    dominant_proc_counts: dict[str, int] = {}
    for trace in traces:
        dominant_axis_counts[trace.dominant_posterior_axis] = dominant_axis_counts.get(trace.dominant_posterior_axis, 0) + 1
        dominant_obs_counts[trace.dominant_obs_axis] = dominant_obs_counts.get(trace.dominant_obs_axis, 0) + 1
        dominant_proc_counts[trace.dominant_proc_axis] = dominant_proc_counts.get(trace.dominant_proc_axis, 0) + 1
    return {
        "num_episodes": len(traces),
        "avg_gamma": float(np.mean(gamma_values)) if gamma_values.size else 0.0,
        "avg_gamma_obs": float(np.mean(gamma_obs_values)) if gamma_obs_values.size else 0.0,
        "avg_gamma_proc": float(np.mean(gamma_proc_values)) if gamma_proc_values.size else 0.0,
        "avg_trust_score": float(np.mean(trust_score_values)) if trust_score_values.size else 0.0,
        "avg_posterior_trace": float(np.mean(trace_values)) if trace_values.size else 0.0,
        "avg_posterior_trace_obs": float(np.mean(trace_obs_values)) if trace_obs_values.size else 0.0,
        "avg_posterior_trace_proc": float(np.mean(trace_proc_values)) if trace_proc_values.size else 0.0,
        "avg_support_distance": float(np.mean(distance_values)) if distance_values.size else 0.0,
        "avg_support_distance_obs": float(np.mean(distance_obs_values)) if distance_obs_values.size else 0.0,
        "avg_support_distance_proc": float(np.mean(distance_proc_values)) if distance_proc_values.size else 0.0,
        "abstain_count": int(sum(trace.abstain_count for trace in traces)),
        "dominant_posterior_axis_counts": dominant_axis_counts,
        "dominant_obs_axis_counts": dominant_obs_counts,
        "dominant_proc_axis_counts": dominant_proc_counts,
    }


def summarize_eval_by_scenario(
    eval_records_by_scenario: dict[str, list[LatentPredictionRecord] | tuple[LatentPredictionRecord, ...]],
) -> dict[str, dict[str, object]]:
    return {
        scenario_name: summarize_records(records)
        for scenario_name, records in eval_records_by_scenario.items()
    }


def summarize_eval_by_group(
    eval_records: list[LatentPredictionRecord] | tuple[LatentPredictionRecord, ...],
) -> dict[str, dict[str, object]]:
    group_records: dict[str, list[LatentPredictionRecord]] = {}
    for record in eval_records:
        group_records.setdefault(record.domain_bucket, []).append(record)
    return {
        domain_bucket: summarize_records(records)
        for domain_bucket, records in group_records.items()
    }


def evaluate_roam_scenarios(
    *,
    prior_engine_config: PriorEngineConfig,
    prior_engine_factory,
    model: SequenceRegressorProtocol,
    reference_stats: SequenceReferenceStats,
    adapter_model: LatentAdapterModel,
    eval_episode_grids_by_scenario: dict[str, list],
    split_name: str,
    window_size: int,
    evidence_bundle: EvidenceBundle,
    config: RoamLatentConfig,
    task_adapter: RoamTaskAdapter,
    cache_dir: Path | None,
) -> RoamEvalOutputs:
    precomputed_prior_results, prior_prefetch_workers = prefetch_eval_prior_results(
        prior_engine_config=prior_engine_config,
        prior_engine_factory=prior_engine_factory,
        model=model,
        reference_stats=reference_stats,
        eval_episode_grids_by_scenario=eval_episode_grids_by_scenario,
        window_size=window_size,
        evidence_bundle=evidence_bundle,
        cache_dir=cache_dir,
        task_adapter=task_adapter,
    )
    eval_records: list[LatentPredictionRecord] = []
    eval_traces: list[LatentEpisodeTrace] = []
    eval_records_by_scenario: dict[str, tuple[LatentPredictionRecord, ...]] = {}
    for scenario_name, scenario_episode_grids in eval_episode_grids_by_scenario.items():
        scenario_prior_results: dict[int, PriorEngineResult] = {
            episode_index: precomputed_prior_results[(scenario_name, episode_index)]
            for episode_index in range(len(scenario_episode_grids))
        }
        scenario_records, scenario_traces = collect_roam_episode_outputs(
            model=model,
            reference_stats=reference_stats,
            adapter_model=adapter_model,
            episode_grids=scenario_episode_grids,
            split_name=split_name,
            window_size_minutes=window_size,
            evidence_bundle=evidence_bundle,
            prior_engine=None,
            config=config,
            cache_dir=cache_dir,
            prior_results=scenario_prior_results,
            task_adapter=task_adapter,
        )
        eval_records.extend(scenario_records)
        eval_traces.extend(scenario_traces)
        eval_records_by_scenario[scenario_name] = scenario_records
    return RoamEvalOutputs(
        records=tuple(eval_records),
        traces=tuple(eval_traces),
        records_by_scenario=eval_records_by_scenario,
        prior_prefetch_workers=prior_prefetch_workers,
    )
