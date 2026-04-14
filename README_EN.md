<p align="center">
  <h1 align="center">ROAM</h1>
  <p align="center">
    <strong>Open-Ended Scenario Reasoning for Specialist Model Adaptation</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.14-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
  <p align="center">
    <a href="README.md">中文</a> | English
  </p>
</p>

---

> **ROAM** freezes a pre-trained specialist backbone and attaches a low-capacity latent corrector that is adapted at deployment time — using structured semantic priors from open-ended scenario evidence, not direct LLM predictions.

## Overview

| Component | Description |
|:--|:--|
| **Semantic Prior** | Converts open-ended scenario evidence (operator notes, maintenance logs, upstream context) into structured axis-level priors via an LLM-based reasoning engine |
| **Latent Corrector** | A low-dimensional latent state (`bias`, `scale`, `load`, `dynamics`, `readout`) fitted by ridge regression on top of a frozen backbone |
| **Online Posterior** | Updates the latent posterior using delayed label residuals, observation anomalies, and process diagnostics with trust gating |
| **Task Adapter** | A generic `RoamTaskAdapter` contract for plugging ROAM into arbitrary sequential regression environments |

### Supported Baselines

| Category | Methods |
|:--|:--|
| Deep Sequence | `GRU` · `LSTM` · `Transformer` · `Informer` · `Mamba` |
| Classical ML | `SVR` · `XGBoost` · `Ridge` · `Lasso` · `ElasticNet` |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Open Evidence                         │
│  (operator notes, maintenance, upstream, external)       │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │  Prior Engine  │  LLM-based semantic reasoning
              │  (structured)  │  → axis-level confidence
              └───────┬────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│              Frozen Specialist Backbone                   │
│         (GRU / LSTM / Transformer / ...)                 │
│                      +                                   │
│         Low-Capacity Latent Corrector                    │
│    [bias, scale, load, dynamics, readout]                │
└──────────────────────┬───────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │    Online      │  delayed labels + observation
              │   Posterior    │  anomalies + process diagnostics
              └───────┬────────┘
                      ▼
               Corrected Output
```

## Project Structure

```
ROAM_open/
├── configs/
│   ├── llm.yaml              # LLM provider template
│   ├── model_gru.yaml        # GRU baseline config
│   ├── model_lstm.yaml       # LSTM baseline config
│   └── model_roam.yaml       # ROAM latent adaptation config
├── src/roam/
│   ├── methods/
│   │   ├── roam.py            # Core: latent adaptation engine
│   │   ├── sequence_regressor.py  # Base protocol & torch wrapper
│   │   ├── gru.py             # GRU regressor
│   │   ├── lstm.py            # LSTM regressor
│   │   ├── transformer.py     # Transformer regressor
│   │   ├── informer.py        # Informer regressor
│   │   ├── mamba.py           # Mamba regressor
│   │   ├── svr.py             # SVR regressor
│   │   ├── xgboost.py         # XGBoost regressor
│   │   └── classical.py       # Ridge / Lasso / ElasticNet
│   ├── prior_engine.py        # Semantic prior engine protocol
│   ├── prior_prefetch.py      # Prior result caching
│   ├── sequence.py            # SequenceRegressionDataset
│   ├── task_adapter.py        # RoamTaskAdapter interface
│   ├── cli_utils.py           # CLI parsing utilities
│   ├── eval_support.py        # Evaluation & trace output
│   ├── experiment_io.py       # YAML config parsing
│   └── base_artifact.py       # Artifact serialization
├── pyproject.toml
└── uv.lock
```

## Getting Started

### Installation

```bash
# install dependencies
uv sync

# with dev tools (pytest, matplotlib)
uv sync --extra dev
```

### Integration

**Step 1** — Prepare a frozen sequential regressor satisfying `SequenceRegressorProtocol`.

**Step 2** — Implement a `RoamTaskAdapter` for your task:

```python
from roam.task_adapter import RoamTaskAdapter, PriorKnownContext, PriorTaskContext
```

**Step 3** — Build a `RoamLatentConfig` from the provided template:

```python
from roam.methods.roam import RoamLatentConfig, fit_latent_adapter
```

**Step 4** — Feed open evidence into the prior engine and run online posterior updates:

```python
from roam.prior_engine import PriorEngineConfig, OpenEvidence, EvidenceBundle
```

## Key Interfaces

```python
# Core adaptation
from roam.methods.roam import (
    RoamLatentConfig,
    fit_latent_adapter,
    build_roam_episode_prior_input,
)

# Task integration
from roam.task_adapter import RoamTaskAdapter, PriorKnownContext, PriorTaskContext

# Prior engine
from roam.prior_engine import (
    PriorEngineConfig,
    OpenEvidence,
    EvidenceBundle,
    SemanticAxis,
)
```

## Configuration

### ROAM Latent Config (`configs/model_roam.yaml`)

Key parameters:

| Parameter | Default | Description |
|:--|:--|:--|
| `base_model` | `gru` | Frozen backbone type |
| `latent_axes` | `[bias, scale, load, dynamics, readout]` | Latent state dimensions |
| `update_space` | `subspace` | Adaptation space |
| `adapter.l2_lambda` | `1.0` | Ridge regularization strength |
| `posterior.label_delay` | `1` | Steps before label becomes available |

### LLM Config (`configs/llm.yaml`)

> Copy to `configs/llm.local.yaml` for local use — this file is git-ignored.

```yaml
type: openai_compatible
api_base: https://your-endpoint
api_key: your-key
model: your-model
```

## Scope

This repository ships the **method framework** — the core adaptation engine, task adapter interface, prior engine protocol, and baseline implementations. Benchmark-specific environments, experiment scripts, and paper artifacts are intentionally excluded.

If your goal is to reuse the core idea — *semantic prior + structured posterior + low-capacity correction* — everything you need is here.

## Recommended Reading Order

1. [`src/roam/methods/roam.py`](src/roam/methods/roam.py) — latent adaptation core
2. [`src/roam/task_adapter.py`](src/roam/task_adapter.py) — task integration contract
3. [`src/roam/prior_engine.py`](src/roam/prior_engine.py) — semantic prior protocol
4. [`configs/model_roam.yaml`](configs/model_roam.yaml) — default configuration

## License

This project is released under the [MIT License](LICENSE).
