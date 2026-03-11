# NeMo RL — Every Config Parameter Explained

Complete reference for every configuration parameter in NeMo RL's GRPO pipeline.
Sourced directly from the nightly API docs, the ProRLv2 guide, and real example YAMLs.
Written for someone who's done GRPO with TRL and wants a 1:1 mapping.

---

## Table of Contents

1. [Top-Level Config Structure (MasterConfig)](#1-top-level-config-structure)
2. [GRPO Section](#2-grpo-section)
   - [Core Training Loop](#21-core-training-loop)
   - [Dynamic Sampling (DAPO)](#22-dynamic-sampling)
   - [Advantage Estimator](#23-advantage-estimator)
   - [Reward Shaping](#24-reward-shaping)
   - [Reward Scaling](#25-reward-scaling)
   - [Overlong Filtering](#26-overlong-filtering)
   - [Async GRPO](#27-async-grpo)
   - [Validation](#28-validation)
   - [Miscellaneous GRPO](#29-miscellaneous)
3. [Loss Function (ClippedPGLossConfig)](#3-loss-function)
   - [Clipping](#31-clipping)
   - [KL Penalty](#32-kl-penalty)
   - [Token vs Sequence Level](#33-token-vs-sequence-level)
   - [Importance Sampling](#34-importance-sampling)
   - [Algorithm Variants via loss_fn](#35-algorithm-variants-via-loss_fn)
4. [Policy Section](#4-policy-section)
   - [Model & Tokenizer](#41-model--tokenizer)
   - [Batch Sizes](#42-batch-sizes)
   - [Gradient Clipping](#43-gradient-clipping)
   - [Optimizer & Scheduler (DTensor)](#44-optimizer--scheduler-dtensor)
   - [Sequence Length & Padding](#45-sequence-length--padding)
   - [Dynamic Batching](#46-dynamic-batching)
   - [Sequence Packing](#47-sequence-packing)
5. [DTensor Config (FSDP2)](#5-dtensor-config)
   - [LoRA on DTensor](#51-lora-on-dtensor)
6. [Megatron Config](#6-megatron-config)
   - [Parallelism](#61-parallelism)
   - [Optimizer (Megatron)](#62-optimizer-megatron)
   - [Scheduler (Megatron)](#63-scheduler-megatron)
   - [DDP Config](#64-ddp-config)
   - [LoRA on Megatron (PEFT)](#65-lora-on-megatron)
7. [Generation Config (vLLM)](#7-generation-config)
8. [Data Config](#8-data-config)
9. [Logger Config](#9-logger-config)
10. [Cluster Config](#10-cluster-config)
11. [Checkpointing Config](#11-checkpointing-config)
12. [Metrics NeMo RL Logs Automatically](#12-automatic-metrics)
13. [Complete Annotated YAML — GRPO 1B Example](#13-complete-annotated-yaml)
14. [Quick Reference: TRL → NeMo RL Parameter Mapping](#14-trl-mapping)

---

## 1. Top-Level Config Structure

Every NeMo RL GRPO run has exactly this structure (`MasterConfig`):

```yaml
policy:     # PolicyConfig — model, optimizer, parallelism, generation
loss_fn:    # ClippedPGLossConfig — clipping, KL, importance sampling
grpo:       # GRPOConfig — training loop, sampling, rewards, advantages
env:        # dict — environment(s) for reward computation
data:       # DataConfig — datasets, preprocessing
logger:     # GRPOLoggerConfig — TensorBoard, W&B, MLflow
cluster:    # ClusterConfig — Ray cluster resources
checkpointing:  # CheckpointingConfig — save/resume
```

> **Key difference from TRL:** `loss_fn` is a **top-level** section, NOT nested under `grpo`.
> This is the most common misconfiguration.

---

## 2. GRPO Section

### 2.1 Core Training Loop

```yaml
grpo:
  num_prompts_per_step: 32       # Prompts per RL step (≈ "batch_size" in TRL)
  num_generations_per_prompt: 16  # G in GRPO — rollouts per prompt
  max_num_steps: 200             # Total RL steps (like max_steps in TRL)
  max_num_epochs: 1              # Max epochs over the dataset
  max_rollout_turns: 1           # For multi-turn RL (future), keep at 1
  seed: 42
```

| Parameter | TRL Equivalent | What It Does |
|-----------|---------------|--------------|
| `num_prompts_per_step` | `batch_size` | How many unique prompts per RL step |
| `num_generations_per_prompt` | `num_generations` | How many completions to sample per prompt (the "G" in GRPO) |
| `max_num_steps` | `max_steps` | Stop after this many gradient updates |
| `max_num_epochs` | `num_train_epochs` | Stop after this many passes over the dataset |
| `max_rollout_turns` | _(N/A)_ | For multi-turn RL; keep 1 for standard RLVR |
| `seed` | `seed` | Random seed for reproducibility |

**Effective global batch size** = `num_prompts_per_step × num_generations_per_prompt`

For our CQL pipeline: 4 prompts × 4 generations = 16 samples per step.

### 2.2 Dynamic Sampling

From the DAPO paper. Filters out prompt groups where all generations got the same reward
(std=0), because those provide zero gradient signal.

```yaml
grpo:
  use_dynamic_sampling: true     # Enable DAPO dynamic sampling
  batch_multiplier: 1.5          # Generate 1.5× prompts to compensate for filtering
  dynamic_sampling_max_gen_batches: 5  # Max generation rounds before giving up
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `use_dynamic_sampling` | `false` | Skip prompt groups with zero reward variance |
| `batch_multiplier` | `1.0` | Over-generate by this factor to fill the batch after filtering |
| `dynamic_sampling_max_gen_batches` | `3` | Max generation rounds to accumulate enough diverse prompts |

**How it works:**
1. Generate `num_prompts_per_step × batch_multiplier` prompts
2. Compute rewards for all generations
3. Filter: keep only prompt groups where `std(rewards) > 0`
4. If not enough prompts, generate another batch (up to `max_gen_batches`)
5. If still not enough, raise an error

**When to use:** Always for verifiable-reward tasks (math, code, CQL).
If 80% of your prompts have unanimous rewards, you're wasting 80% of compute without it.

**TRL equivalent:** None — TRL doesn't have this. You'd have to implement custom filtering.

### 2.3 Advantage Estimator

Controls how advantages (A_t) are computed from rewards.

```yaml
grpo:
  # Standard GRPO (default)
  normalize_rewards: true
  use_leave_one_out_baseline: false

  # OR: Reinforce++ (ProRLv2 recipe)
  adv_estimator:
    name: "reinforce_plus_plus"      # or "grpo" (default)
    normalize_rewards: true
    use_leave_one_out_baseline: false
    minus_baseline: true             # Subtract group mean before global normalization
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `normalize_rewards` | `true` | Normalize advantages within each prompt group: `a_i = r_i - mean(r)` |
| `use_leave_one_out_baseline` | `false` | Each sample's baseline = mean of all OTHER samples in group (reduces bias) |
| `adv_estimator.name` | `"grpo"` | Which estimator: `"grpo"` or `"reinforce_plus_plus"` |
| `adv_estimator.minus_baseline` | `true` | (Reinforce++ only) Subtract per-group mean before global normalization |

**Standard GRPO advantage:**
```
A_i = (r_i - mean(r_group)) / std(r_group)
```

**Leave-one-out baseline:**
```
A_i = (r_i - mean(r_group \ {r_i})) / std(r_group)
```
Each sample doesn't influence its own baseline → lower bias.

**Reinforce++ (recommended for long runs):**
1. Per group: `a_i = r_i - mean(r_group)`  (local baseline)
2. Global: `A = (a - mean(a_all)) / std(a_all)`  (global normalization)

This decoupled local+global normalization is more stable for prolonged training.

**TRL equivalent:** `use_leave_one_out_baseline` maps to TRL's `use_rloo_baseline`.

### 2.4 Reward Shaping

Post-processes rewards before advantage computation. Currently supports:

#### DAPO Overlong Reward Shaping
Penalizes responses that approach max length, creating a soft boundary:

```yaml
grpo:
  reward_shaping:
    enabled: true
    overlong_buffer_length: 4096    # Buffer zone before max length
    overlong_buffer_penalty: 1.0    # Max penalty (subtracted from reward)
    max_response_length: 20480      # Where penalty is maximum
    stop_properly_penalty_coef: null  # MUST be null to use overlong shaping
```

#### "Stop Properly" Penalty (Truncation Penalty)
Scales rewards for truncated responses (hit max length without EOS):

```yaml
grpo:
  reward_shaping:
    enabled: true
    stop_properly_penalty_coef: 0.0   # 0.0 = zero reward for truncated
                                       # 1.0 = no penalty (keep original)
                                       # 0.5 = half the original reward
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `enabled` | `false` | Enable reward shaping |
| `overlong_buffer_length` | — | Start penalizing this many tokens before max length |
| `overlong_buffer_penalty` | — | Maximum penalty value |
| `max_response_length` | — | Full penalty kicks in at this length |
| `stop_properly_penalty_coef` | `null` | Scale factor for truncated responses' rewards |

> **Important:** `stop_properly_penalty_coef` and DAPO overlong shaping are **mutually exclusive**.
> If `stop_properly_penalty_coef` is set (not null), the overlong shaping is skipped entirely.

**For CQL:** Our queries are short (<512 tokens). Use `stop_properly_penalty_coef: 0.0`
to zero out rewards for truncated responses (they're almost certainly garbage).

### 2.5 Reward Scaling

Linear mapping of rewards from one range to another, with clamping:

```yaml
grpo:
  reward_scaling:
    enabled: true
    source_min: 0.0    # Clamp original rewards to [source_min, source_max]
    source_max: 1.0
    target_min: 0.0    # Map to [target_min, target_max]
    target_max: 1.0
```

**Formula:** `scaled = target_min + (clamped_reward - source_min) / (source_max - source_min) * (target_max - target_min)`

**When to use:** When your reward function returns values in an unexpected range
(e.g., 0-100) and you want to normalize to [0, 1] before advantage computation.

**For CQL:** Our rewards are already in [0, 1], so `enabled: false`.

### 2.6 Overlong Filtering

```yaml
grpo:
  overlong_filtering: false   # default
```

When `true`, samples that hit `max_total_sequence_length` without EOS are **excluded from
the loss** (their `sample_mask` is set to 0). They still contribute to reward baselines
but produce zero gradient. Different from reward shaping — this is hard masking.

**When to use:** Long-form reasoning (math proofs, code generation) where truncation
means the answer is incomplete and the gradient would be misleading.

### 2.7 Async GRPO

Asynchronous training with a replay buffer. Overlap generation with training:

```yaml
grpo:
  async_grpo:
    enabled: false                # Enable async mode
    max_trajectory_age_steps: 2   # Max age of trajectories in replay buffer
    in_flight_weight_updates: false
    recompute_kv_cache_after_weight_updates: false
```

**When to use:** Multi-node setups where generation is the bottleneck.
Lets training proceed on slightly stale trajectories while generation continues.

**For CQL dummy run:** Keep disabled. Only useful at scale.

### 2.8 Validation

```yaml
grpo:
  val_period: 10          # Validate every N steps
  val_batch_size: 16      # Prompts per validation batch
  val_at_start: true      # Validate before training starts
  val_at_end: true        # Validate after training ends
  max_val_samples: 100    # Cap on total validation samples
```

### 2.9 Miscellaneous

```yaml
grpo:
  skip_reference_policy_logprobs_calculation: false  # Skip ref logprobs (saves compute if KL=0)
  calculate_advantages_on_gpu: false  # Compute advantages on GPU (faster for large batches)
  seq_logprob_error_threshold: null   # Mask sequences with logprob error above this threshold
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `skip_reference_policy_logprobs_calculation` | `false` | Skip reference policy logprob computation. Set `true` if `reference_policy_kl_penalty: 0.0` |
| `calculate_advantages_on_gpu` | `false` | Move advantage computation to GPU (faster but uses VRAM) |
| `seq_logprob_error_threshold` | `null` | If set, mask sequences where `abs(gen_logprobs - train_logprobs) > threshold` |

---

## 3. Loss Function

The `ClippedPGLossConfig` implements PPO/GRPO/GSPO/DAPO/REINFORCE++ all in one.
Which algorithm you get depends on which flags you set.

### 3.1 Clipping

```yaml
loss_fn:
  ratio_clip_min: 0.2     # ε for lower bound: clip at 1 - 0.2 = 0.8
  ratio_clip_max: 0.2     # ε for upper bound: clip at 1 + 0.2 = 1.2
  ratio_clip_c: null       # Dual-clip parameter (null = disabled)
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `ratio_clip_min` | `0.2` | Lower epsilon: ratio clipped to `[1 - ratio_clip_min, ...]` |
| `ratio_clip_max` | `0.2` | Upper epsilon: ratio clipped to `[..., 1 + ratio_clip_max]` |
| `ratio_clip_c` | `null` | Dual-clip c value. When A_t < 0, adds `max(..., c * A_t)`. Must be >1, usually 3.0 |

> **IMPORTANT naming convention:** `ratio_clip_min/max` are **epsilon values**, NOT the actual
> clip bounds. The bounds are `[1 - ratio_clip_min, 1 + ratio_clip_max]`.

**Symmetric clipping** (standard GRPO): `ratio_clip_min = ratio_clip_max = 0.2`
→ Clip to [0.8, 1.2]

**Asymmetric clipping** (DAPO "Clip-Higher"): `ratio_clip_min = 0.2, ratio_clip_max = 0.28`
→ Clip to [0.8, 1.28] — allows more policy expansion than contraction

**Dual-clipping** (Ye et al. 2019): Set `ratio_clip_c: 3.0`
→ For negative advantages, adds floor at `c * A_t` to prevent excessive updates

**TRL equivalent:** `cliprange` and `cliprange_value`. TRL uses symmetric only.

### 3.2 KL Penalty

```yaml
loss_fn:
  reference_policy_kl_penalty: 0.001   # β — KL coefficient
  reference_policy_kl_type: "k3"       # KL divergence type
  kl_input_clamp_value: null            # Clamp KL inputs (null = no clamping)
  kl_output_clamp_value: null           # Clamp KL outputs (null = no clamping)
  use_on_policy_kl_approximation: false # Use importance-weighted on-policy KL
  use_kl_in_reward: false               # Add KL term to reward instead of loss
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `reference_policy_kl_penalty` | `0.001` | β in `L - β * KL(π_θ ∥ π_ref)`. Set to 0.0 to disable KL entirely |
| `reference_policy_kl_type` | `"k3"` | KL approximation: `"k1"` (simple), `"k2"` (absolute), `"k3"` (Schulman — unbiased, always positive) |
| `kl_input_clamp_value` | `null` | Clamp log-ratio inputs before KL computation (stability) |
| `kl_output_clamp_value` | `null` | Clamp KL divergence output (prevents extreme values) |
| `use_on_policy_kl_approximation` | `false` | Weight KL by importance ratio π_θ/π_old for on-policy estimate |
| `use_kl_in_reward` | `false` | Subtract KL from reward instead of adding to loss (Reinforce++ style) |

**KL types explained:**
- `"k1"` (forward): `log(π_θ/π_ref)` — simple log-ratio, can be negative
- `"k2"` (absolute): `|log(π_θ/π_ref)|` — always positive, penalizes both directions
- `"k3"` (Schulman): `(π_ref/π_θ) - log(π_ref/π_θ) - 1` — unbiased, always ≥ 0, recommended

**For CQL:** Start with `reference_policy_kl_penalty: 0.0` (pure GRPO, no KL constraint).
This is standard for verifiable-reward tasks. Add KL if the model drifts too far.

**TRL equivalent:** `beta` (KL coefficient), `kl_estimator` (k1/k2/k3).

### 3.3 Token vs Sequence Level

```yaml
loss_fn:
  token_level_loss: true   # Apply loss per-token (recommended for variable-length)
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `token_level_loss` | `true` | `true`: Loss computed per-token under masking. `false`: Loss aggregated per-sequence first |

**Token-level (recommended):**
```
L = (1/T) Σ_t clip(r_t) * A
```
Each token gets its own gradient proportional to its importance ratio.

**Sequence-level:**
```
L = clip(R) * A   where R = Σ_t log(π_θ(t)/π_old(t))
```
One gradient signal for the entire sequence.

**When to use token-level:** Almost always. Especially for variable-length outputs,
CoT reasoning, and any case where responses vary significantly in length.
This is the **Dr. GRPO fix** — removes length bias.

**TRL equivalent:** `loss_type="grpo"` is token-level by default in TRL.

### 3.4 Importance Sampling

Corrects for training-generation backend mismatch (especially MoE models):

```yaml
loss_fn:
  use_importance_sampling_correction: false    # Enable IS correction
  truncated_importance_sampling_ratio: null     # Upper bound for TIS
  truncated_importance_sampling_ratio_min: null # Lower bound (ICE-POP)
  truncated_importance_sampling_type: null      # "tis", "icepop", or "seq-mask-tis"
  sequence_level_importance_ratios: false       # GSPO: sequence-level geometric mean
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `use_importance_sampling_correction` | `false` | Multiply loss by `π_train(t)/π_gen(t)` to correct for backend mismatch |
| `truncated_importance_sampling_ratio` | `null` | Upper bound for IS weights (prevents extreme corrections) |
| `truncated_importance_sampling_ratio_min` | `null` | Lower bound for ICE-POP filtering |
| `truncated_importance_sampling_type` | `null` | `"tis"`: clamp weights. `"icepop"`: zero weights outside bounds. `"seq-mask-tis"`: sequence-level geometric-mean mask |
| `sequence_level_importance_ratios` | `false` | Use geometric mean of token ratios (GSPO). Required for GSPO |

#### TIS variants explained:

**TIS (Truncated Importance Sampling):**
```python
weights = clamp(π_train / π_gen, max=truncated_ratio)
```
Simple clamping. Safe but can accumulate bias.

**ICE-POP:**
```python
weights = where((π_train/π_gen >= min) & (π_train/π_gen <= max), π_train/π_gen, 0)
```
Zero out outlier tokens entirely. More aggressive but cleaner gradients.
Typical bounds: `min=0.5, max=5.0`.

**seq-mask-tis:**
```python
geo_mean = exp(mean(log(π_train / π_gen)))  # per sequence
mask = (geo_mean >= min) & (geo_mean <= max)  # mask entire sequences
weights = raw_token_ratios * mask  # no per-token clamping for retained sequences
```
Tighter bounds: `min=0.999, max=1.002`.

**When to use:** Only when your generation backend (vLLM) and training backend (DTensor/Megatron)
produce different logprobs for the same tokens. Common with MoE models.
For standard dense models with the same precision, this is usually unnecessary.

### 3.5 Algorithm Variants via loss_fn

The `ClippedPGLossFn` implements multiple algorithms. Here's how to configure each:

```yaml
# Standard GRPO
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2
  token_level_loss: true
  reference_policy_kl_penalty: 0.001

# DAPO (Clip-Higher + dynamic sampling in grpo section)
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28      # Asymmetric: allows more exploration
  token_level_loss: true
  reference_policy_kl_penalty: 0.0  # No KL
  ratio_clip_c: 5.0          # Dual-clip

# GSPO (sequence-level importance sampling)
loss_fn:
  sequence_level_importance_ratios: true
  token_level_loss: false     # GSPO uses sequence-level loss
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2

# REINFORCE/RLOO (no PPO ratio)
loss_fn:
  disable_ppo_ratio: true     # Removes ratio term entirely
  # ratio_clip_min/max are ignored

# Truly On-Policy (force ratio = 1.0)
loss_fn:
  force_on_policy_ratio: true  # Sets r(θ) = 1.0 always
  # Only works with 1 update per rollout

# Dr. GRPO (just remove length normalization — this is implicit when token_level_loss: true
# and you DON'T divide by sequence length. NeMo RL's token_level_loss already does this.)
```

| Flag | Algorithm | Effect |
|------|-----------|--------|
| Default | GRPO | Standard clipped PG with group baseline |
| `ratio_clip_max > ratio_clip_min` | DAPO | Asymmetric "Clip-Higher" |
| `ratio_clip_c: 3.0` | Dual-Clip | Extra floor for negative advantages |
| `sequence_level_importance_ratios: true` | GSPO | Geometric mean of token ratios |
| `disable_ppo_ratio: true` | REINFORCE/RLOO | No importance ratio |
| `force_on_policy_ratio: true` | On-Policy PG | Ratio forced to 1.0 |

---

## 4. Policy Section

### 4.1 Model & Tokenizer

```yaml
policy:
  model_name: "Qwen/Qwen2.5-1.5B"   # HuggingFace model ID or local path
  precision: "bf16-mixed"              # "bf16-mixed", "fp16", "fp32"

  tokenizer:
    name: "Qwen/Qwen2.5-1.5B"        # Usually same as model_name
    chat_template: null                # Override chat template (path to jinja2 file)
    chat_template_kwargs: null         # Extra kwargs for chat template

  hf_config_overrides: {}              # Override any HF config value (e.g., attn_implementation)

  automodel_kwargs:                    # Passed to AutoModelForCausalLM.from_pretrained
    use_liger_kernel: false            # Liger kernel for efficient attention
    force_hf: false                    # Force HuggingFace implementation (no custom backends)
```

### 4.2 Batch Sizes

```yaml
policy:
  train_global_batch_size: 16     # Total samples per training step across all GPUs
  train_micro_batch_size: 4       # Samples per GPU per forward pass
  logprob_batch_size: 8           # Batch size for logprob computation (can be larger)
  logprob_chunk_size: null        # Chunk logprob computation (saves memory)
  generation_batch_size: 16       # Batch size for vLLM generation
```

| Parameter | What It Does |
|-----------|--------------|
| `train_global_batch_size` | Total samples processed per gradient step. Must equal `num_prompts_per_step × num_generations_per_prompt` |
| `train_micro_batch_size` | Samples per GPU per forward pass. Gradient accumulation steps = `global / (micro × num_gpus)` |
| `logprob_batch_size` | Batch size for reference policy logprob computation. Can be larger than `train_micro_batch_size` since no gradients |
| `logprob_chunk_size` | Split logprob computation into chunks of this size (trade speed for memory) |
| `generation_batch_size` | How many prompts to send to vLLM at once |

**Gradient accumulation** is implicit: `grad_accum = global / (micro × world_size)`.
No explicit `gradient_accumulation_steps` parameter like TRL — it's computed automatically.

### 4.3 Gradient Clipping

```yaml
policy:
  max_grad_norm: 1.0    # Max L2 norm for gradient clipping (null = no clipping)
```

**TRL equivalent:** `max_grad_norm`. Same behavior.

This clips the **global** gradient norm across all parameters. Standard practice:
- `1.0` for most training
- `0.5` for unstable early training
- `null` to disable (not recommended for RL)

### 4.4 Optimizer & Scheduler (DTensor backend)

```yaml
policy:
  optimizer:
    name: "torch.optim.AdamW"           # Full class path required
    kwargs:
      lr: 5.0e-6
      weight_decay: 0.01
      betas: [0.9, 0.999]
      eps: 1.0e-8
      foreach: false                     # Required for DTensor
      fused: false                       # Required for DTensor
```

NeMo RL uses PyTorch-native optimizers and `SequentialLR` for scheduling on the DTensor backend.
The scheduler config accepts **any `torch.optim.lr_scheduler` class** by full name.

#### Scheduler Formats

**Format 1: Sequential (list of schedulers + milestones)** — use `SequentialLR` under the hood:

```yaml
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs:
        start_factor: 0.1
        end_factor: 1.0
        total_iters: 20
    - name: "torch.optim.lr_scheduler.ConstantLR"
      kwargs:
        factor: 1.0
        total_iters: 10000000000
    - milestones: [20]           # Switch from LinearLR → ConstantLR at step 20
```

**Format 2: Single scheduler (dict):**

```yaml
  scheduler:
    name: "torch.optim.lr_scheduler.CosineAnnealingLR"
    kwargs:
      T_max: 500
      eta_min: 1.0e-7
```

#### Which Scheduler for GRPO/RLVR?

Based on literature (DeepSeek-R1, arXiv 2503.06639, Unsloth RLVR guide):

| Schedule | When to use | Notes |
|----------|------------|-------|
| **Warmup → Constant** | Default for GRPO | Most stable, used by DeepSeek-R1, recommended |
| **Warmup → Cosine** | Long runs (500+ steps) | Smoother convergence, slight risk of underfitting |
| **Constant (no warmup)** | Quick experiments | Simple, works if LR is tuned well |
| **OneCycleLR** | Not recommended | KL spikes, unstable for RL |

**Recommended: Warmup → Constant** (default in our configs):
```yaml
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs: { start_factor: 0.1, end_factor: 1.0, total_iters: 20 }
    - name: "torch.optim.lr_scheduler.ConstantLR"
      kwargs: { factor: 1.0, total_iters: 10000000000 }
    - milestones: [20]
```

**Alternative: Warmup → Cosine Decay** (for long runs):
```yaml
  scheduler:
    - name: "torch.optim.lr_scheduler.LinearLR"
      kwargs: { start_factor: 0.1, end_factor: 1.0, total_iters: 20 }
    - name: "torch.optim.lr_scheduler.CosineAnnealingLR"
      kwargs: { T_max: 480, eta_min: 1.0e-7 }
    - milestones: [20]
```

**All available schedulers** (any `torch.optim.lr_scheduler` class works):
- `LinearLR` — linear ramp
- `ConstantLR` — constant multiplier
- `CosineAnnealingLR` — cosine decay to `eta_min`
- `CosineAnnealingWarmRestarts` — cosine with warm restarts (`T_0`, `T_mult`)
- `ExponentialLR` — exponential decay by `gamma`
- `StepLR` — step decay every `step_size` steps
- `MultiStepLR` — decay at specific `milestones`
- `PolynomialLR` — polynomial decay
- `OneCycleLR` — 1cycle super-convergence

**Typical RL learning rates:**
- Full fine-tune: `1e-6` to `5e-7`
- LoRA: `5e-6` to `5e-5` (we use `5e-6` for GRPO, `2e-5` for SFT)

### 4.5 Sequence Length & Padding

```yaml
policy:
  max_total_sequence_length: 2048    # Max total length (prompt + response)
  make_sequence_length_divisible_by: 8  # Pad to multiple of this (GPU efficiency)
  refit_buffer_size_gb: 1.0          # Buffer for weight refitting to vLLM
```

### 4.6 Dynamic Batching

Reduces padding waste by grouping similar-length sequences:

```yaml
policy:
  dynamic_batching:
    enabled: true
    train_mb_tokens: 8192         # Target tokens per micro-batch
    logprob_mb_tokens: 16384      # Target tokens per logprob micro-batch
    sequence_length_round: 64     # Round sequence lengths to this multiple
```

| Parameter | What It Does |
|-----------|--------------|
| `train_mb_tokens` | Target token count per micro-batch (replaces `train_micro_batch_size` effectively) |
| `logprob_mb_tokens` | Same but for logprob computation |
| `sequence_length_round` | Round padded length to multiples of this (GPU memory alignment) |

**When to use:** When response lengths vary significantly. For CQL (short outputs),
less critical but still helpful.

### 4.7 Sequence Packing

Packs multiple short sequences into one "super-sequence" to eliminate padding:

```yaml
policy:
  sequence_packing:
    enabled: true
    train_mb_tokens: 8192
    logprob_mb_tokens: 16384
    algorithm: "first_fit_decreasing"  # Packing algorithm
```

| Parameter | What It Does |
|-----------|--------------|
| `train_mb_tokens` | Target tokens per packed super-sequence |
| `algorithm` | `"first_fit_decreasing"` — standard bin-packing |

**Dynamic batching vs sequence packing:** Use one or the other, not both.
- Dynamic batching: simpler, groups similar lengths, less efficient
- Sequence packing: more complex, eliminates all padding, more efficient for mixed lengths

---

## 5. DTensor Config (FSDP2)

PyTorch-native distributed training. Functionally equivalent to DeepSpeed ZeRO-3
but integrated with PyTorch's DTensor/FSDP2.

```yaml
policy:
  dtensor_cfg:
    enabled: true
    cpu_offload: false              # Offload params/grads to CPU (saves GPU, slower)
    sequence_parallel: false        # Enable sequence parallelism
    activation_checkpointing: false # Recompute activations in backward (saves memory)
    tensor_parallel_size: 1         # TP degree (split attention heads across GPUs)
    context_parallel_size: 1        # CP degree (split sequence across GPUs)
    custom_parallel_plan: null      # Custom DTensor parallel plan
    clear_cache_every_n_steps: null # Clear CUDA cache periodically (prevents fragmentation)
    _v2: false                      # Use v2 DTensor worker (experimental)
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `cpu_offload` | `false` | Offload optimizer states and gradients to CPU. Like ZeRO-3 offload |
| `sequence_parallel` | `false` | Split the sequence dimension across TP workers |
| `activation_checkpointing` | `false` | Trade compute for memory: recompute activations during backward |
| `tensor_parallel_size` | `1` | Number of GPUs for tensor parallelism. Keep 1 for single GPU |
| `context_parallel_size` | `1` | Number of GPUs for context (sequence) parallelism |
| `clear_cache_every_n_steps` | `null` | Call `torch.cuda.empty_cache()` every N steps |

**Single GPU recipe:**
```yaml
dtensor_cfg:
  enabled: true
  cpu_offload: false
  activation_checkpointing: true  # Enable if OOM
  tensor_parallel_size: 1
  context_parallel_size: 1
```

**Multi-GPU recipe (4×A100 80GB):**
```yaml
dtensor_cfg:
  enabled: true
  cpu_offload: false
  activation_checkpointing: false
  tensor_parallel_size: 2
  sequence_parallel: true
  context_parallel_size: 1
```

### 5.1 LoRA on DTensor

```yaml
policy:
  dtensor_cfg:
    enabled: true
    lora_cfg:
      enabled: true
      target_modules: []               # Empty = auto-detect linear layers
      exclude_modules: ["lm_head"]     # Don't apply LoRA to these
      match_all_linear: true           # Apply to all Linear layers
      dim: 16                          # LoRA rank (r)
      alpha: 32                        # LoRA alpha (scaling = alpha/dim)
      dropout: 0.0                     # LoRA dropout (0 for RL, dropout hurts)
      dropout_position: "pre"          # "pre" or "post" — where to apply dropout
      lora_A_init: "kaiming_uniform"   # Initialization for A matrix
      use_triton: false                # Use Triton kernels for LoRA
```

| Parameter | Typical Value | What It Does |
|-----------|---------------|--------------|
| `dim` | `16` | LoRA rank. Higher = more capacity, more memory. 8-64 typical |
| `alpha` | `32` | Scaling factor. Effective scale = `alpha / dim`. Usually `2 × dim` |
| `dropout` | `0.0` | Set to 0 for RL — dropout adds noise that hurts policy gradient |
| `target_modules` | `[]` | List of module names. Empty + `match_all_linear: true` = all linear layers |
| `exclude_modules` | `["lm_head"]` | Don't LoRA the output head |
| `lora_A_init` | `"kaiming_uniform"` | Standard Kaiming init for A, B is zero-initialized |
| `use_triton` | `false` | Use Triton-fused LoRA kernels (experimental, may be faster) |

> **Note:** NeMo RL LoRA uses a **merge-weight** approach during generation.
> LoRA weights are merged into base weights before vLLM generation, then un-merged for training.
> This introduces a small training-inference mismatch but is much faster.

**DoRA:** Not supported in NeMo RL. Would need custom implementation.

---

## 6. Megatron Config

NVIDIA's Megatron-Core backend. For models 70B+ with full 6D parallelism.

```yaml
policy:
  megatron_cfg:
    enabled: true
    activation_checkpointing: true
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 2
    context_parallel_size: 1
    sequence_parallel: true
    pipeline_dtype: "bfloat16"
    apply_rope_fusion: true
    bias_activation_fusion: true
    empty_unused_memory_level: 2
    # MoE-specific
    freeze_moe_router: false
    moe_per_layer_logging: false
    moe_enable_deepep: false
    moe_token_dispatcher_type: "alltoall"
    moe_shared_expert_overlap: false
    expert_tensor_parallel_size: 1
    expert_model_parallel_size: 1
```

### 6.1 Parallelism

| Parameter | What It Does |
|-----------|--------------|
| `tensor_model_parallel_size` | Split attention heads/FFN across this many GPUs |
| `pipeline_model_parallel_size` | Split layers across this many pipeline stages |
| `context_parallel_size` | Split sequence length across GPUs |
| `sequence_parallel` | Enable SP alongside TP (reduces activation memory) |
| `num_layers_in_first_pipeline_stage` | Override layer count in first PP stage |
| `num_layers_in_last_pipeline_stage` | Override layer count in last PP stage |

**Total GPUs needed** = `TP × PP × CP × DP` (DP is implicit from remaining GPUs).

### 6.2 Optimizer (Megatron)

```yaml
policy:
  megatron_cfg:
    optimizer:
      optimizer: "adam"
      lr: 1.0e-6
      min_lr: 1.0e-7
      weight_decay: 0.01
      bf16: true
      fp16: false
      params_dtype: "bfloat16"
      adam_beta1: 0.9
      adam_beta2: 0.95
      adam_eps: 1.0e-8
      sgd_momentum: 0.0
      use_distributed_optimizer: true
      use_precision_aware_optimizer: false
      clip_grad: 1.0
      optimizer_cpu_offload: false
      optimizer_offload_fraction: 0.0
```

| Parameter | What It Does |
|-----------|--------------|
| `use_distributed_optimizer` | Shard optimizer states across DP ranks (like ZeRO-1) |
| `use_precision_aware_optimizer` | Use mixed-precision optimizer for memory savings |
| `clip_grad` | Max gradient norm (Megatron's equivalent of `max_grad_norm`) |
| `optimizer_cpu_offload` | Offload optimizer states to CPU |
| `optimizer_offload_fraction` | What fraction to offload (0.0-1.0) |

### 6.3 Scheduler (Megatron)

```yaml
policy:
  megatron_cfg:
    scheduler:
      lr_decay_style: "cosine"        # "cosine", "linear", "constant"
      lr_decay_iters: null            # Steps for LR decay (null = use max_num_steps)
      lr_warmup_iters: 10
      lr_warmup_init: 0.0
      start_weight_decay: 0.01
      end_weight_decay: 0.01
      weight_decay_incr_style: "constant"
```

### 6.4 DDP Config

```yaml
policy:
  megatron_cfg:
    distributed_data_parallel_config:
      grad_reduce_in_fp32: true            # Reduce gradients in FP32 (precision)
      overlap_grad_reduce: true            # Overlap gradient reduction with backward
      overlap_param_gather: true           # Overlap parameter gathering with forward
      use_custom_fsdp: false               # Use Megatron's custom FSDP implementation
      data_parallel_sharding_strategy: "full_shard"  # "full_shard" or "no_shard"
```

### 6.5 LoRA on Megatron (PEFT)

> **Status: Coming in v0.6.** Not yet available for RL algorithms (GRPO/DPO).
> SFT LoRA on Megatron is supported.

```yaml
policy:
  megatron_cfg:
    peft:
      enabled: true
      target_modules: []
      exclude_modules: []
      dim: 16
      alpha: 32
      dropout: 0.0
      dropout_position: "pre"
      lora_A_init_method: "kaiming_uniform"
      lora_B_init_method: "zero"
      a2a_experimental: false
      lora_dtype: null
```

---

## 7. Generation Config (vLLM)

Controls how responses are generated during rollouts:

```yaml
policy:
  generation:
    backend: "vllm"              # Generation backend ("vllm" or "megatron")
    max_new_tokens: 512          # Max tokens to generate per response
    temperature: 1.0             # Sampling temperature (higher = more diverse)
    top_p: 1.0                   # Nucleus sampling cutoff
    top_k: null                  # Top-k sampling (null = disabled)
    stop_token_ids: null         # Stop generation at these token IDs
    stop_strings: null           # Stop generation at these strings
    model_name: null             # Override model for generation (usually same as policy)

    colocated:
      enabled: true              # Run vLLM on same GPUs as training
      resources:
        gpus_per_node: null      # null = use all available
        num_nodes: null
```

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `backend` | `"vllm"` | Generation engine. vLLM for fast inference |
| `max_new_tokens` | `512` | Maximum response length in tokens |
| `temperature` | `1.0` | Sampling temperature. Higher = more diverse rollouts |
| `top_p` | `1.0` | Nucleus sampling. 1.0 = no filtering. 0.9 = top 90% probability mass |
| `top_k` | `null` | Top-k sampling. `null` = disabled |
| `colocated.enabled` | `true` | Colocate vLLM with training on same GPUs (memory sharing) |

**For RL:** Use `temperature: 1.0` to ensure diverse rollouts. Never use `temperature: 0`
(greedy) during training — you need exploration for policy gradient to work.

---

## 8. Data Config

```yaml
data:
  max_input_seq_length: 1024     # Max prompt length (truncate longer)
  shuffle: true                   # Shuffle training data
  add_bos: false                  # Add BOS token
  add_eos: false                  # Add EOS token
  add_generation_prompt: true     # Add the assistant turn prefix
  add_system_prompt: true         # Include system prompt
  num_workers: 4                  # DataLoader workers
  use_multiple_dataloader: false  # Enable multi-dataset dataloading
  num_prompts_per_dataloader: 16  # Prompts from each dataloader when multi-dataset

  train:
    dataset_name: "ResponseDataset"  # Or a built-in name
    data_path: "/path/to/train.jsonl"
    input_key: "input"
    output_key: "output"
    env_name: "math"
    processor: "math_hf_data_processor"
    prompt_file: null              # Custom prompt template file
    system_prompt_file: null       # Custom system prompt file
    split_validation_size: 0.05   # Use 5% as validation
    seed: 42

  validation:
    data_path: "/path/to/val.jsonl"

  default:
    dataset_name: "ResponseDataset"
    input_key: "input"
    output_key: "output"
    processor: "math_hf_data_processor"
    env_name: "math"
```

**JSONL format:**
```json
{"input": "Write a CQL query to...", "output": "#event_simpleName=ProcessRollup2 | ..."}
```

---

## 9. Logger Config

```yaml
logger:
  log_dir: "logs/"
  wandb_enabled: false
  wandb:
    project: "cql-rlvr"
    name: "grpo-run-1"
    entity: null
  tensorboard_enabled: true
  tensorboard:
    log_dir: "logs/tb"
  mlflow_enabled: false
  mlflow:
    tracking_uri: "logs/mlflow"
    experiment_name: "cql"
  num_val_samples_to_print: 5    # Print this many val samples to console/log
```

---

## 10. Cluster Config

```yaml
cluster:
  gpus_per_node: 1
  num_nodes: 1
  monitor_gpus: true
  gpu_monitoring:
    collection_interval: 10     # Seconds between GPU stats collection
```

---

## 11. Checkpointing Config

```yaml
checkpointing:
  checkpoint_dir: "results/checkpoints"
  save_period: 50                # Save every N steps
  max_to_keep: 3                 # Keep only last N checkpoints
  resume: true                   # Resume from latest checkpoint if exists
```

---

## 12. Automatic Metrics

NeMo RL logs these metrics automatically at every training step:

### Core Training Metrics
| Metric | What It Means |
|--------|---------------|
| `loss` | Training loss value |
| `gradient_norm` | L2 norm of gradients (watch for spikes = instability) |
| `learning_rate` | Current LR (should follow your schedule) |
| `approx_entropy` | Approximate policy entropy (tracks diversity collapse) |
| `kl_divergence` | KL(π_θ ∥ π_ref) — how far policy has drifted |

### Reward Metrics
| Metric | What It Means |
|--------|---------------|
| `reward/mean` | Mean reward across batch |
| `reward/std` | Reward standard deviation |
| `reward/max`, `reward/min` | Reward range |

### GRPO-Specific Metrics
| Metric | What It Means |
|--------|---------------|
| `advantages/mean`, `advantages/std` | Advantage statistics |
| `ratio/mean` | Mean importance ratio π_θ/π_old (should be ~1.0) |
| `ratio/clip_fraction` | Fraction of ratios that were clipped (too high = too aggressive) |

### Backend Mismatch Metrics
| Metric | What It Means |
|--------|---------------|
| `token_mult_prob_error` | Average multiplicative probability error between backends |
| `gen_kl_error` | KL(P_gen ∥ P_policy) — generation vs training mismatch |
| `policy_kl_error` | KL(P_policy ∥ P_gen) — reverse direction |
| `js_divergence_error` | Jensen-Shannon divergence between backends |
| `sampling_importance_ratio` | Mean IS ratio between train and gen backends (should be ~1.0) |
| `is_oob_ratio` | Fraction of tokens/sequences filtered by ICE-POP/seq-mask-tis |

### Dynamic Sampling Metrics (when enabled)
| Metric | What It Means |
|--------|---------------|
| `dynamic_sampling_num_gen_batches` | How many generation rounds were needed |
| `dynamic_sampling_filtered_ratio` | Fraction of prompts filtered (zero variance) |

### Validation Metrics
| Metric | What It Means |
|--------|---------------|
| `val/reward_mean` | Mean reward on validation set |
| `val/accuracy` | Task accuracy on validation |

**Watch for:**
- `gradient_norm` spikes → lower LR or increase `max_grad_norm`
- `approx_entropy` dropping fast → policy is collapsing, reduce LR
- `token_mult_prob_error` trending up → weight refit bug
- `ratio/clip_fraction` > 0.3 → clipping too much, lower LR or increase epsilon
- `kl_divergence` growing → policy drifting, increase `reference_policy_kl_penalty`

---

## 13. Complete Annotated YAML — GRPO 1B Example

Based on `grpo_math_1B.yaml` from the NeMo RL repo, annotated for CQL:

```yaml
# ============================================================
# POLICY — Model, optimizer, distributed training, generation
# ============================================================
policy:
  model_name: "Qwen/Qwen2.5-1.5B"
  precision: "bf16-mixed"

  tokenizer:
    name: "Qwen/Qwen2.5-1.5B"

  # Batch sizes
  train_global_batch_size: 64    # num_prompts × num_generations = 8 × 8
  train_micro_batch_size: 8      # Per-GPU batch → grad_accum = 64/8 = 8 steps
  generation_batch_size: 32
  logprob_batch_size: 16

  # Sequence lengths
  max_total_sequence_length: 2048
  make_sequence_length_divisible_by: 8

  # Gradient clipping
  max_grad_norm: 1.0

  # Optimizer (DTensor backend)
  optimizer:
    name: "adamw"
    kwargs:
      lr: 1.0e-6
      weight_decay: 0.01

  scheduler:
    - name: "LinearLR"
      kwargs: { start_factor: 0.1, total_iters: 10 }
    - milestones: [10]
    - name: "CosineAnnealingLR"
      kwargs: { T_max: 190, eta_min: 1.0e-7 }

  # DTensor (FSDP2) backend
  dtensor_cfg:
    enabled: true
    cpu_offload: false
    activation_checkpointing: false
    tensor_parallel_size: 1
    context_parallel_size: 1
    sequence_parallel: false

  # Dynamic batching
  dynamic_batching:
    enabled: false

  # Generation (vLLM)
  generation:
    backend: "vllm"
    max_new_tokens: 512
    temperature: 1.0
    top_p: 1.0
    colocated:
      enabled: true

# ============================================================
# LOSS — Clipping, KL, token-level
# ============================================================
loss_fn:
  ratio_clip_min: 0.2           # Clip to [0.8, 1.28] (DAPO asymmetric)
  ratio_clip_max: 0.28
  ratio_clip_c: null            # No dual-clip
  reference_policy_kl_penalty: 0.0   # No KL penalty for RLVR
  reference_policy_kl_type: "k3"
  token_level_loss: true
  use_on_policy_kl_approximation: false
  use_importance_sampling_correction: false

# ============================================================
# GRPO — Training loop, sampling, rewards
# ============================================================
grpo:
  num_prompts_per_step: 8
  num_generations_per_prompt: 8
  max_num_steps: 200
  max_num_epochs: 1
  seed: 42

  # Advantage estimation
  normalize_rewards: true
  use_leave_one_out_baseline: false

  # Dynamic sampling (DAPO)
  use_dynamic_sampling: true
  batch_multiplier: 1.5
  dynamic_sampling_max_gen_batches: 5

  # Reward shaping
  reward_shaping:
    enabled: true
    stop_properly_penalty_coef: 0.0

  # Reward scaling
  reward_scaling:
    enabled: false

  # Validation
  val_period: 20
  val_batch_size: 16
  val_at_start: true
  val_at_end: true
  max_val_samples: 100

  # Misc
  overlong_filtering: false
  skip_reference_policy_logprobs_calculation: true  # No KL → skip this

# ============================================================
# DATA
# ============================================================
data:
  max_input_seq_length: 1024
  shuffle: true
  train:
    data_path: "data/train.jsonl"
    input_key: "input"
    output_key: "output"
    env_name: "cql"
  default:
    dataset_name: "ResponseDataset"
    processor: "math_hf_data_processor"

# ============================================================
# ENV — CQL reward environment
# ============================================================
env:
  cql:
    type: "nemo_gym"              # External NeMo Gym reward server
    url: "http://localhost:8000"

# ============================================================
# LOGGER
# ============================================================
logger:
  log_dir: "logs/"
  tensorboard_enabled: true
  wandb_enabled: false
  num_val_samples_to_print: 5

# ============================================================
# CLUSTER
# ============================================================
cluster:
  gpus_per_node: 1
  num_nodes: 1

# ============================================================
# CHECKPOINTING
# ============================================================
checkpointing:
  checkpoint_dir: "results/checkpoints"
  save_period: 50
  max_to_keep: 3
  resume: true
```

---

## 14. Quick Reference: TRL → NeMo RL Parameter Mapping

| TRL Parameter | NeMo RL Equivalent | Notes |
|---------------|-------------------|-------|
| `learning_rate` | `policy.optimizer.kwargs.lr` | Same meaning |
| `batch_size` | `grpo.num_prompts_per_step` | Per-step unique prompts |
| `mini_batch_size` | `policy.train_micro_batch_size` | Per-GPU forward pass batch |
| `num_generations` | `grpo.num_generations_per_prompt` | Rollouts per prompt (G) |
| `max_steps` | `grpo.max_num_steps` | Total RL steps |
| `max_grad_norm` | `policy.max_grad_norm` | L2 gradient clipping |
| `cliprange` | `loss_fn.ratio_clip_min/max` | NeMo uses epsilon notation |
| `beta` (KL) | `loss_fn.reference_policy_kl_penalty` | β coefficient |
| `kl_estimator` | `loss_fn.reference_policy_kl_type` | k1/k2/k3 |
| `loss_type="grpo"` | `loss_fn.token_level_loss: true` | Token-level = Dr. GRPO |
| `use_rloo_baseline` | `grpo.use_leave_one_out_baseline` | Leave-one-out baseline |
| `temperature` | `policy.generation.temperature` | Sampling temperature |
| `max_new_tokens` | `policy.generation.max_new_tokens` | Max response length |
| `gradient_accumulation_steps` | _(implicit)_ | `global_batch / (micro × GPUs)` |
| `lora_r` | `policy.dtensor_cfg.lora_cfg.dim` | LoRA rank |
| `lora_alpha` | `policy.dtensor_cfg.lora_cfg.alpha` | LoRA scaling |
| _(no equivalent)_ | `grpo.use_dynamic_sampling` | TRL doesn't have this |
| _(no equivalent)_ | `grpo.reward_shaping` | TRL doesn't have this |
| _(no equivalent)_ | `loss_fn.ratio_clip_c` | TRL doesn't support dual-clip |
| _(no equivalent)_ | `loss_fn.sequence_level_importance_ratios` | GSPO (TRL doesn't have) |
| _(no equivalent)_ | `loss_fn.use_importance_sampling_correction` | Backend mismatch correction |

---

## Sources

- [NeMo RL GRPO API Docs (nightly)](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.algorithms.grpo.html)
- [NeMo RL Loss Functions API (nightly)](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.algorithms.loss.loss_functions.html)
- [NeMo RL Policy API (nightly)](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.models.policy.html)
- [NeMo RL Reward Functions API (nightly)](https://docs.nvidia.com/nemo/rl/nightly/apidocs/nemo_rl/nemo_rl.algorithms.reward_functions.html)
- [GRPO Walkthrough](https://docs.nvidia.com/nemo/rl/nightly/guides/grpo.html)
- [ProRLv2 Guide](https://docs.nvidia.com/nemo/rl/nightly/guides/prorlv2.html)
- [DAPO Guide](https://docs.nvidia.com/nemo/rl/nightly/guides/dapo.html)
- [grpo_math_1B.yaml](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/grpo_math_1B.yaml)
- [prorlv2.yaml](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/prorlv2.yaml)
