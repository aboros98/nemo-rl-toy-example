# NeMo RL — Complete Reference

Everything you need to know to run RL training with NeMo RL: algorithms, logging,
tips & tricks, features, and gotchas. Written for someone who's done GRPO with TRL
and wants to know what NVIDIA's stack offers.

---

## Table of Contents

1. [Available Algorithms](#available-algorithms)
2. [Training Backends (NOT DeepSpeed)](#training-backends)
3. [Logging & Monitoring](#logging--monitoring)
4. [What Metrics Get Logged](#what-metrics-get-logged)
5. [Tips & Tricks](#tips--tricks)
6. [Features Available Now](#features-available-now)
7. [Roadmap (v0.6)](#roadmap-v06)
8. [DeepScaleR Recipe — Staged GRPO Training](#deepscaler-recipe)
9. [Full Logger YAML Config](#full-logger-yaml-config)
10. [Troubleshooting](#troubleshooting)

---

## Available Algorithms

NeMo RL supports more than just GRPO. Here's the **exact** current state from the
GitHub README `main` branch (v0.5.0+, Feb 2026):

### Algorithm Support Matrix

| Algorithm | DTensor | DTensor + LoRA | Megatron | Megatron + LoRA | Single Node | Multi-Node |
|-----------|:-------:|:--------------:|:--------:|:---------------:|:-----------:|:----------:|
| **GRPO** | ✅ | ✅ | ✅ | 🔜 v0.6 | ✅ | ✅ |
| **DAPO** | ✅ | ✅ | ✅ | 🔜 v0.6 | ✅ | ✅ |
| **GSPO** | ✅ | ✅ | ✅ | 🔜 v0.6 | ✅ | ✅ |
| **DPO** | ✅ | ✅ | ✅ | 🔜 v0.6 | ✅ | ✅ |
| **SFT** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **On-policy Distillation** | ✅ | — | — | — | ✅ | ✅ |
| **Reward Modeling (RM)** | ✅ | — | ✅ | — | ✅ | ✅ |

**Key takeaway: LoRA for RL (GRPO/DPO) works on DTensor only. Megatron LoRA is
coming in v0.6. SFT LoRA works on both backends already.**

### What Each Algorithm Does

| Algorithm | What it does |
|-----------|-------------|
| **GRPO** | Group Relative Policy Optimization. K completions per prompt, group-relative advantages. Token-level importance sampling. The workhorse. |
| **DAPO** | GRPO + Clip-Higher + dynamic sampling + token-level loss + overlong shaping. Strictly better for code/reasoning. Added Oct 2025. |
| **GSPO** | Group Sequence Policy Optimization. GRPO variant from the **Qwen team** (Alibaba). Difference: importance sampling is per-**sequence** instead of per-**token**. Set `importance_sampling_level: "sequence"`. Better credit assignment at the response level. |
| **DPO** | Direct Preference Optimization. Learns from preference pairs, no reward model needed. |
| **On-policy Distillation** | Student generates on-policy, aligns logits to teacher via KL. Near-teacher quality at lower cost. Added Sep 2025. |
| **SFT** | Supervised fine-tuning. LoRA supported on both DTensor and Megatron. Baseline method. |
| **RM** | Reward Modeling. Train reward models for RLHF pipelines. |

### GRPO vs GSPO vs DAPO — When to Use What

| | GRPO | GSPO | DAPO |
|---|------|------|------|
| **Importance sampling** | Per-token | Per-sequence | Per-token |
| **Clipping** | Symmetric | Symmetric | Asymmetric (Clip-Higher) |
| **Dynamic sampling** | Optional | Optional | Built-in |
| **Length bias** | Yes (mitigated by `token_level_loss`) | No (sequence-level) | Mitigated by overlong shaping |
| **Origin** | DeepSeek | Qwen (Alibaba) | DeepSeek/community |
| **Best for** | General | Tasks where response-level quality > token quality | Code, reasoning, long outputs |
| **Config** | Default | `importance_sampling_level: "sequence"` | `ratio_clip_c`, `reward_shaping.enabled` |

### What about PPO?

NeMo RL doesn't have PPO. GRPO is PPO's successor for LLM RL — it removes the critic
network and uses group-relative advantages instead. Same clipped surrogate objective,
simpler infrastructure.

---

## Training Backends

NeMo RL has exactly **two** training backends. No DeepSpeed.

Source: [Training Backends doc](https://docs.nvidia.com/nemo/rl/nightly/design-docs/training-backends.html)

### The Two Backends

| Backend | Best for | Parallelism | Config key |
|---------|----------|-------------|------------|
| **DTensor (FSDP2)** | Models <30B, single/multi-node | FSDP2, TP, SP, CP, PP | `policy.dtensor_cfg.enabled: true` |
| **Megatron-Core** | Models >30B, max performance | TP, PP, SP, CP, EP, FSDP (6D) | `policy.megatron_cfg.enabled: true` |

Backend is auto-selected from your YAML. If both are enabled, **Megatron takes priority**.

### DeepSpeed ZeRO3: NOT supported

**DeepSpeed is not a backend in NeMo RL.** There is no DeepSpeed code in the repo,
no config keys for it, and no roadmap mention. Some web sources incorrectly claim
support — the [actual training backends doc](https://docs.nvidia.com/nemo/rl/nightly/design-docs/training-backends.html)
and the [README](https://github.com/NVIDIA-NeMo/RL/blob/main/README.md) list only
DTensor and Megatron.

**FSDP2 ≈ ZeRO3 functionally**: both shard parameters, gradients, and optimizer states
across GPUs. FSDP2 uses PyTorch-native DTensor sharding; ZeRO3 uses DeepSpeed's own
partitioning. FSDP2 also supports `cpu_offload: true`.

If you specifically need DeepSpeed ZeRO3, use **TRL + DeepSpeed** instead of NeMo RL.

### DTensor Config Example

```yaml
policy:
  dtensor_cfg:
    enabled: true
    cpu_offload: false
    activation_checkpointing: true
    tensor_parallel_size: 1
    sequence_parallel: false
    context_parallel_size: 1
    lora_cfg:
      enabled: true
      dim: 16
      alpha: 32
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:64"
```

### Megatron Config Example

```yaml
policy:
  megatron_cfg:
    enabled: true
    activation_checkpointing: true
    tensor_model_parallel_size: 8
    pipeline_model_parallel_size: 2
    sequence_parallel: true
    context_parallel_size: 1
```

Requires `git submodule update --init --recursive` and a one-time HF→Megatron checkpoint conversion.

---

## Logging & Monitoring

NeMo RL has a centralized `Logger` that wraps **five** backends. All enabled backends
receive the same metrics through a single `log_metrics()` call.

### Supported Backends

| Backend | Type | Best for |
|---------|------|----------|
| **TensorBoard** | Local file-based | Quick local visualization, CI pipelines |
| **Weights & Biases (W&B)** | Cloud | Team collaboration, rich dashboards, artifact tracking |
| **MLflow** | Local or remote server | Experiment management, model versioning, deployment |
| **SwanLab** | Cloud (Android/iOS/Web) | Mobile monitoring, multi-user collaboration |
| **ClearML** | Cloud or self-hosted | Full MLOps pipeline |

### Architecture

```
Training Loop
    │
    ▼
Logger (wrapper)  ──▶  TensorboardLogger  ──▶ ./logs/tensorboard/
    │                   WandbLogger        ──▶ wandb cloud
    │                   MLflowLogger       ──▶ mlflow tracking server
    │                   SwanlabLogger      ──▶ swanlab cloud
    │
    └── logger.log_metrics({"loss": 0.12, "grad_norm": 1.5, ...}, step=42)
```

The logger runs on the single **controller** process (the Ray driver that coordinates
training). It gathers distributed metrics with reductions (mean, max across ranks)
before logging.

### Logger Interface

```python
class LoggerInterface(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        """Log a dictionary of metrics."""

    @abstractmethod
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log dictionary of hyperparameters."""
```

Usage in your training loop:
```python
logger = Logger(cfg=logger_config)

# Log anything you want — all enabled backends receive it
logger.log_metrics({
    "loss": 0.123,
    "gradient_norm": 1.18,
    "reward_mean": 0.72,
    "reward_std": 0.15,
    "reward_min": 0.1,
    "reward_max": 0.95,
    "kl_divergence": 0.003,
    "policy_entropy": 2.1,
    "syntax_valid_pct": 0.85,
}, step=current_step)
```

---

## What Metrics Get Logged

NeMo RL's GRPO training loop logs these metrics per step:

### Core Training Metrics

| Metric | Description | What to watch for |
|--------|-------------|-------------------|
| `loss` | Clipped surrogate loss | Should decrease, but RL loss is noisy — trend matters, not individual steps |
| `gradient_norm` | L2 norm of gradients (before clipping) | Spikes = instability. Sustained >10 = lower LR or increase `max_grad_norm` |
| `learning_rate` | Current LR from scheduler | Sanity check |

### Reward Metrics

| Metric | Description | What to watch for |
|--------|-------------|-------------------|
| `reward_mean` | Average reward across all completions in the step | Should trend upward. Plateau = reward hacking or data exhaustion |
| `reward_std` | Reward standard deviation | Should decrease over time as model gets more consistent |
| `reward_min` / `reward_max` | Extremes | `reward_max` approaching 1.0 = model solving some prompts perfectly |

### GRPO-Specific Metrics

| Metric | Description | What to watch for |
|--------|-------------|-------------------|
| `advantages_mean` | Mean normalized advantage | Should hover near 0 (by construction) |
| `advantages_std` | Advantage spread | If near 0 → all rewards same → no learning signal |
| `kl_divergence` | KL(π_θ ‖ π_ref) | Rising fast = model drifting from reference. Add/increase KL penalty |
| `policy_ratio_mean` | Mean importance sampling ratio π_new/π_old | Should stay near 1.0. Far from 1.0 = stale reference |
| `clip_fraction` | Fraction of tokens clipped by surrogate | High (>0.3) = updates too aggressive, reduce LR |

### GPU Metrics (optional)

If `monitor_gpus: true`:

| Metric | Description |
|--------|-------------|
| `gpu_memory_used` | VRAM usage per GPU |
| `gpu_utilization` | GPU compute utilization % |

### Custom Metrics (your reward components)

In our CQL pipeline, we also log:
```python
logger.log_metrics({
    "syntax_valid_pct": 0.85,      # % of completions that are valid CQL
    "execution_success_pct": 0.72,  # % that "executed" successfully
    "structure_similarity_mean": 0.72,  # Average pipeline function Jaccard
}, step=step)
```

**Tip**: Log your individual reward components, not just the combined score.
This is the only way to diagnose which component is limiting training.

---

## Tips & Tricks

### Memory Management

**OOM after a few iterations (not immediately)?** That's memory fragmentation, not
model size. FlashAttention2-compatible models are less prone to this.

```bash
# Option 1: Environment variable (applies to all Ray actors)
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64 uv run python examples/run_grpo.py ...

# Option 2: In YAML config (persistent, per-run)
policy:
  dtensor_cfg:
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:64"
```

If reserved-but-unused memory is large, try:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Submodule Errors

```
ModuleNotFoundError: No module named 'megatron'
```

Fix:
```bash
git submodule update --init --recursive
NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...
```

### Colocated vs Dedicated Generation

**Colocated** (`generation.colocated.enabled: true`):
- Training and vLLM generation share the same GPU
- Offloads optimizer to CPU during generation
- Best for single-GPU or memory-constrained setups
- Our dummy config uses this

**Dedicated** (`generation.colocated.enabled: false`):
- Separate GPU(s) for generation
- Higher throughput but needs more hardware
- Use for production multi-node runs

### Staged Training (DeepScaleR Pattern)

Don't train at full context length from step 1. Gradually increase:

1. **Stage 1**: 8K context → train until reward stabilizes
2. **Stage 2**: 16K context → load stage 1 checkpoint, train more
3. **Stage 3**: 24K context → load stage 2 checkpoint, train to convergence

```bash
# Stage 1
uv run examples/run_grpo.py --config=grpo-8K.yaml

# Stage 2 (load previous checkpoint)
uv run examples/run_grpo.py --config=grpo-16K.yaml \
  policy.model_name=/path/to/8K/ckpt/hf

# Stage 3
uv run examples/run_grpo.py --config=grpo-24K.yaml \
  policy.model_name=/path/to/16K/ckpt/hf
```

For CQL this is less relevant (queries are short), but important for reasoning tasks.

### Sequence Packing

Enabled by default in NeMo RL for both DTensor and Megatron Core. Packs multiple
short sequences into one long sequence to maximize GPU utilization. Huge throughput
gains for datasets with variable-length responses (like CQL where queries range
from 10 to 100 tokens).

### Reward Shaping Dos and Don'ts

**DO:**
- Give partial credit. All-or-nothing binary rewards → zero-variance groups → no gradient.
- Log reward components separately. You can't diagnose what you can't see.
- Use dynamic sampling (`use_dynamic_sampling: true`) to skip zero-variance groups.
- Start with KL=0, add β=0.01 once model is producing reasonable output.

**DON'T:**
- Use huge reward ranges. If rewards span [-100, 100], normalization noise dominates.
- Penalize heavily for exploration. Harsh penalties early on → model collapses to
  safe-but-useless outputs (e.g., always generating `*`).
- Trust the reward curve alone. Periodically inspect actual model outputs.

### Checkpoint and Evaluate Frequently

Convert and evaluate checkpoints on target benchmarks (e.g., CQL validity rate on
held-out test set) rather than relying solely on reward curves. Reward hacking can
produce rising reward curves with degrading actual performance.

---

## Features Available Now

Everything in NeMo RL as of the latest release:

| Feature | Details |
|---------|---------|
| **GRPO / GSPO / DAPO** | Full algorithm suite with all variants |
| **SFT with LoRA** | On both DTensor and Megatron Core backends |
| **DPO** | Direct preference optimization |
| **On-policy Distillation** | Student generates on-policy, aligns to teacher via KL |
| **Multi-turn RL** | Multi-step tool use, games, dialogue training |
| **DTensor (FSDP2)** | PyTorch-native TP/SP/CP for models <10B |
| **Megatron Core** | TP/PP/SP/CP/EP/FSDP for 70B+ models |
| **Sequence Packing** | Both backends, massive throughput gains |
| **vLLM Generation** | Colocated or dedicated, optimized batched inference |
| **FP8 Training** | End-to-end low-precision on Megatron Core |
| **FP8 vLLM Generation** | Low-precision inference |
| **VLM Support** | SFT and GRPO on vision-language models |
| **Async RL** | Asynchronous rollouts + replay buffers for off-policy GRPO |
| **HuggingFace Integration** | Direct HF model loading (DTensor), checkpoint conversion (Megatron) |
| **NeMo Gym Integration** | HTTP-based reward environments |
| **Megatron Inference** | Day-0 model support without weight conversion |
| **GB200 / ARM64** | Container support for latest NVIDIA hardware |
| **MoE Models** | Optimized weight transfer for DeepSeekV3, Qwen3 MoE |
| **Ray Orchestration** | Distributed training coordination |
| **Worker Isolation** | Process isolation between RL actors (no global state leaks) |
| **GPU Metric Logging** | Automatic VRAM and utilization tracking |

---

## Roadmap (v0.6)

What's coming next (from the GitHub README):

| Feature | Description |
|---------|-------------|
| **LoRA for RL on Megatron** | LoRA support for GRPO and DPO on Megatron Core backend (DTensor LoRA already works) |
| **GDPO** | New RL algorithm variant |
| **Muon Optimizer** | Emerging optimizer for SFT and RL |
| **SGLang Inference** | Alternative inference backend |
| **Multi-teacher Distillation** | Distill from multiple teachers simultaneously |
| **Cross-tokenizer Distillation** | Distill across different tokenizers |
| **Speculative Decoding** | Faster rollouts via speculative inference |
| **Fault Tolerance** | Auto-scaling and resiliency |
| **Improved MoE Performance** | Better Megatron Core training + generation for large MoE models |
| **New Models** | Qwen3-Next, Nemotron-Super |

---

## DeepScaleR Recipe

The reference recipe for GRPO training at scale, based on the DeepScaleR paper.
Uses DeepSeek-R1-Distill-Qwen-1.5B on math reasoning.

### Hardware Requirements

| Stage | Context Length | Hardware |
|-------|--------------|----------|
| Stage 1 | 8K tokens | 1× 8xH100 node |
| Stage 2 | 16K tokens | 2× 8xH100 nodes |
| Stage 3 | 24K tokens | 2× 8xH100 nodes |

### Commands

```bash
# Stage 1: Short context
uv run examples/run_grpo.py \
  --config=examples/configs/recipes/llm/grpo-deepscaler-1.5b-8K.yaml

# Stage 2: Medium context (load stage 1 checkpoint)
uv run examples/run_grpo.py \
  --config=examples/configs/recipes/llm/grpo-deepscaler-1.5b-16K.yaml \
  policy.model_name=/path/to/8K/ckpt/hf

# Stage 3: Long context (load stage 2 checkpoint)
uv run examples/run_grpo.py \
  --config=examples/configs/recipes/llm/grpo-deepscaler-1.5b-24K.yaml \
  policy.model_name=/path/to/16K/ckpt/hf
```

### Key Lessons from DeepScaleR

1. **Stage training by context length** — don't start at max length
2. **Evaluate frequently** — convert checkpoints to HF format, run AIME24 benchmark
3. **Reward must be verifiable** — math uses symbolic equivalence, we use CQL structure+fields matching
4. **Group size matters** — K=16 or K=32 generations per prompt gives stable advantages
5. **Dynamic sampling is essential** — binary rewards create many zero-variance groups

---

## Full Logger YAML Config

Copy-paste this into your NeMo RL config. Enable what you need.

```yaml
logger:
  # --- Core settings ---
  log_dir: './logs'                    # Root directory for all logs

  # --- Backend toggles ---
  wandb_enabled: false                 # Weights & Biases (cloud)
  tensorboard_enabled: true            # TensorBoard (local)
  mlflow_enabled: false                # MLflow (local or server)
  swanlab_enabled: false               # SwanLab (cloud/mobile)

  # --- Weights & Biases ---
  wandb:
    project: "cql-rlvr"               # W&B project name
    name: "grpo-cql-run-1"            # Run name
    # entity: "my-team"               # Optional: W&B team/org

  # --- TensorBoard ---
  tensorboard:
    log_dir: './logs/tensorboard'      # TensorBoard log directory

  # --- MLflow ---
  mlflow:
    experiment_name: "cql-rlvr"        # Experiment name in MLflow UI
    run_name: "grpo-run-1"            # Run name
    tracking_uri: null                 # null = local, or "http://server:5000"

  # --- SwanLab ---
  swanlab:
    project: "cql-rlvr"
    name: "grpo-cql-run-1"

  # --- Validation sample printing ---
  num_val_samples_to_print: 5         # Print N validation samples during training
                                       # Great for eyeballing model outputs

  # --- GPU monitoring ---
  monitor_gpus: true                   # Log GPU memory and utilization
  gpu_monitoring:
    collection_interval: 10            # Seconds between GPU metric collections
    flush_interval: 10                 # Seconds between flushing to loggers
```

### Viewing Logs

```bash
# TensorBoard (local)
tensorboard --logdir=./logs/tensorboard --port=6006
# Open http://localhost:6006

# MLflow (local)
mlflow ui --host 0.0.0.0 --port=5000
# Open http://localhost:5000

# W&B — logs stream to wandb.ai automatically

# SwanLab — logs stream to swanlab.cn automatically
```

### Our Dummy Pipeline Logger

Our `cql_nemo_rl_nemotron30b.yaml` already has logger config. When transitioning to real
NeMo RL, just enable the backends you want:

```yaml
# In configs/cql_nemo_rl_nemotron30b.yaml, add:
logger:
  tensorboard_enabled: true
  wandb_enabled: true
  wandb:
    project: "cql-rlvr"
    name: "production-run"
  monitor_gpus: true
```

---

## Troubleshooting

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM after a few steps | Memory fragmentation | `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` |
| OOM immediately | Model too large for GPU | Enable activation checkpointing, reduce batch size, or use LoRA |
| `ModuleNotFoundError: megatron` | Missing submodules | `git submodule update --init --recursive` |
| Reward stays flat | Zero-variance groups (all same reward) | Enable `use_dynamic_sampling: true` |
| Reward oscillates wildly | Learning rate too high | Reduce LR, increase `max_grad_norm` clipping |
| Model outputs nonsense | KL drift too large | Add `reference_policy_kl_penalty: 0.01` |
| Model outputs only short responses | Length bias in loss | Use `token_level_loss: true` (Dr. GRPO normalization) |
| Model outputs excessively long responses | No length penalty | Enable `reward_shaping.enabled: true` with overlong penalty |
| Training hangs on multi-node | Ray worker networking | Check Ray dashboard, ensure ports are open |
| vLLM generation errors | Version mismatch | Ensure vLLM version matches NeMo RL requirements |
| Gradient norm spikes | Reward outliers or bad data | Inspect outlier prompts, consider clipping rewards |
| Slow generation | Short sequences not packed | Verify sequence packing is enabled (default: true) |

### Debugging Checklist

1. **Start with dry-run**: Validate configs and data loading first
2. **Run 1 step**: Check that rewards are computed and loss is not NaN
3. **Run 10 steps**: Check reward trend — should be slightly positive or stable
4. **Check GPU memory**: `nvidia-smi` during training, ensure not at 100%
5. **Inspect model outputs**: Read actual generated CQL, not just metrics
6. **Log everything**: Enable `monitor_gpus`, log all reward components
7. **Use Ray dashboard**: `ray dashboard` for distributed debugging

---

## References

- [NeMo RL Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html)
- [NeMo RL GitHub](https://github.com/NVIDIA-NeMo/RL)
- [Logger Design Doc](https://docs.nvidia.com/nemo/rl/latest/design-docs/logger.html)
- [Logger API Reference](https://docs.nvidia.com/nemo/rl/latest/apidocs/nemo_rl/nemo_rl.utils.logger.html)
- [Tips and Tricks](https://docs.nvidia.com/nemo/rl/nightly/about/tips-and-tricks.html)
- [Features & Roadmap](https://docs.nvidia.com/nemo/rl/nightly/about/features.html)
- [Algorithms Index](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/index.html)
- [GRPO Guide](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)
- [DAPO Guide](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)
- [DeepScaleR Recipe](https://docs.nvidia.com/nemo/rl/nightly/guides/grpo-deepscaler.html)
- [On-policy Distillation](https://docs.nvidia.com/nemo/rl/latest/about/algorithms/on-policy-distillation.html)
- [NVIDIA Blog: DeepScaleR + NeMo RL](https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-reproducing-a-deepscaler-recipe-using-grpo/)

---

*See also: [Training Parameters Deep-Dive](nemo_gym_rl_guide.md#training-parameters-deep-dive) |
[Reward Design & Strategy](rewards_and_strategy.md) |
[SLURM Multi-Node](slurm_multinode.md)*
