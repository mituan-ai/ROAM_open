<p align="center">
  <h1 align="center">ROAM</h1>
  <p align="center">
    <strong>基于开放场景推理的专家模型在线适配框架</strong>
  </p>
  <p align="center">
    <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-%3E%3D3.14-blue?logo=python&logoColor=white" alt="Python"></a>
    <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <a href="#license"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  </p>
  <p align="center">
    中文 | <a href="README_EN.md">English</a>
  </p>
</p>

---

> **ROAM** 冻结预训练的专家主干模型，在部署时仅适配一个低维纠偏 latent——通过开放场景证据构造结构化语义先验，而非让 LLM 直接输出预测。

## 概览

| 组件 | 说明 |
|:--|:--|
| **语义先验** | 将开放场景证据（操作员记录、维护日志、上游上下文）通过 LLM 推理引擎转化为结构化的轴级先验 |
| **Latent 纠偏器** | 在冻结主干之上，用 ridge 回归拟合低维 latent 状态（`bias`、`scale`、`load`、`dynamics`、`readout`） |
| **在线 Posterior** | 利用延迟标签残差、观测异常和过程诊断证据，配合信任门控做在线后验更新 |
| **任务适配器** | 通用 `RoamTaskAdapter` 接口，可接入任意序列回归环境 |

### 支持的 Baseline

| 类别 | 方法 |
|:--|:--|
| 深度序列模型 | `GRU` · `LSTM` · `Transformer` · `Informer` · `Mamba` |
| 经典机器学习 | `SVR` · `XGBoost` · `Ridge` · `Lasso` · `ElasticNet` |

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                     开放证据                              │
│  （操作员记录、维护日志、上游信号、外部上下文）               │
└──────────────────────┬──────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │   先验引擎      │  LLM 语义推理
              │  (structured)  │  → 轴级置信度
              └───────┬────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│              冻结的专家主干模型                             │
│         (GRU / LSTM / Transformer / ...)                 │
│                      +                                   │
│            低维 Latent 纠偏器                              │
│    [bias, scale, load, dynamics, readout]                │
└──────────────────────┬───────────────────────────────────┘
                       ▼
              ┌────────────────┐
              │   在线后验      │  延迟标签 + 观测异常
              │   更新模块      │  + 过程诊断证据
              └───────┬────────┘
                      ▼
                纠偏后的输出
```

## 项目结构

```
ROAM_open/
├── configs/
│   ├── llm.yaml              # LLM 提供商配置模板
│   ├── model_gru.yaml        # GRU baseline 配置
│   ├── model_lstm.yaml       # LSTM baseline 配置
│   └── model_roam.yaml       # ROAM latent 适配配置
├── src/roam/
│   ├── methods/
│   │   ├── roam.py            # 核心：latent 适配引擎
│   │   ├── sequence_regressor.py  # 基础协议 & torch 封装
│   │   ├── gru.py             # GRU 回归器
│   │   ├── lstm.py            # LSTM 回归器
│   │   ├── transformer.py     # Transformer 回归器
│   │   ├── informer.py        # Informer 回归器
│   │   ├── mamba.py           # Mamba 回归器
│   │   ├── svr.py             # SVR 回归器
│   │   ├── xgboost.py         # XGBoost 回归器
│   │   └── classical.py       # Ridge / Lasso / ElasticNet
│   ├── prior_engine.py        # 语义先验引擎协议
│   ├── prior_prefetch.py      # 先验结果缓存
│   ├── sequence.py            # SequenceRegressionDataset
│   ├── task_adapter.py        # RoamTaskAdapter 接口
│   ├── cli_utils.py           # CLI 解析工具
│   ├── eval_support.py        # 评估与 trace 输出
│   ├── experiment_io.py       # YAML 配置解析
│   └── base_artifact.py       # Artifact 序列化
├── pyproject.toml
└── uv.lock
```

## 快速开始

### 安装

```bash
# 安装依赖
uv sync

# 包含开发工具（pytest, matplotlib）
uv sync --extra dev
```

### 接入流程

**Step 1** — 准备一个冻结的序列回归主干，满足 `SequenceRegressorProtocol` 接口。

**Step 2** — 为你的任务实现 `RoamTaskAdapter`：

```python
from roam.task_adapter import RoamTaskAdapter, PriorKnownContext, PriorTaskContext
```

**Step 3** — 从配置模板构造 `RoamLatentConfig`：

```python
from roam.methods.roam import RoamLatentConfig, fit_latent_adapter
```

**Step 4** — 将开放证据送入先验引擎，执行在线后验更新：

```python
from roam.prior_engine import PriorEngineConfig, OpenEvidence, EvidenceBundle
```

## 核心接口

```python
# 核心适配
from roam.methods.roam import (
    RoamLatentConfig,
    fit_latent_adapter,
    build_roam_episode_prior_input,
)

# 任务接入
from roam.task_adapter import RoamTaskAdapter, PriorKnownContext, PriorTaskContext

# 先验引擎
from roam.prior_engine import (
    PriorEngineConfig,
    OpenEvidence,
    EvidenceBundle,
    SemanticAxis,
)
```

## 配置说明

### ROAM Latent 配置（`configs/model_roam.yaml`）

核心参数：

| 参数 | 默认值 | 说明 |
|:--|:--|:--|
| `base_model` | `gru` | 冻结主干类型 |
| `latent_axes` | `[bias, scale, load, dynamics, readout]` | Latent 状态维度 |
| `update_space` | `subspace` | 适配空间 |
| `adapter.l2_lambda` | `1.0` | Ridge 正则化强度 |
| `posterior.label_delay` | `1` | 标签可用前的延迟步数 |

### LLM 配置（`configs/llm.yaml`）

> 本地使用时复制为 `configs/llm.local.yaml`——该文件已加入 `.gitignore`，不会被跟踪。

```yaml
type: openai_compatible
api_base: https://your-endpoint
api_key: your-key
model: your-model
```

## 定位

本仓库提供**方法框架**——核心适配引擎、任务适配接口、先验引擎协议和 baseline 实现。Benchmark 环境、实验脚本和论文资产不在此仓库范围内。

如果你的目标是复用 ROAM 的核心思路——*语义先验 + 结构化后验 + 低维纠偏*——这里已经包含了所有必要组件。

## 推荐阅读顺序

1. [`src/roam/methods/roam.py`](src/roam/methods/roam.py) — latent 适配核心
2. [`src/roam/task_adapter.py`](src/roam/task_adapter.py) — 任务接入接口
3. [`src/roam/prior_engine.py`](src/roam/prior_engine.py) — 语义先验协议
4. [`configs/model_roam.yaml`](configs/model_roam.yaml) — 默认配置

## License

本项目采用 [MIT License](LICENSE) 开源发布。
