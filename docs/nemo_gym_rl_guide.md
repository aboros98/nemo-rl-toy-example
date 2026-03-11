# Understanding NeMo Gym & NeMo RL — A Practical Guide

If you've done GRPO with TRL on GSM8K, you already know 80% of what's happening here.
This doc explains the NVIDIA-specific pieces, all the training knobs you'd want to tune,
how to build a Gym environment from scratch, and how our dummy pipeline maps to the real thing.

---

## Table of Contents

1. [The Two Frameworks](#the-two-frameworks)
2. [How They Connect](#how-they-connect)
3. [Training Parameters Deep-Dive](#training-parameters-deep-dive)
4. [Algorithm Variants: GRPO vs DAPO vs Dr. GRPO](#algorithm-variants)
5. [Distributed Training: DTensor vs Megatron vs DeepSpeed](#distributed-training)
6. [Building a NeMo Gym Environment From Scratch](#building-a-nemo-gym-environment)
7. [Our Dummy Pipeline → Real NeMo RL](#transitioning-to-real-nemo-rl)
8. [NeMo RL Config Explained](#nemo-rl-config-explained)
9. [Key Differences from TRL](#key-differences-from-trl)
10. [References](#references)

---

## The Two Frameworks

```
┌──────────────┐         HTTP (verify)        ┌──────────────┐
│   NeMo RL    │ ──────────────────────────▶  │   NeMo Gym   │
│              │                               │              │
│  GRPO loop   │  ◀──────────────────────────  │  Reward env  │
│  Policy model│         { reward: 0.7 }       │  Resource    │
│  vLLM gen    │                               │  servers     │
│  Loss/optim  │                               │              │
└──────────────┘                               └──────────────┘
     ▲                                               ▲
     │                                               │
 cql_nemo_rl_nemotron30b.yaml                cql_environment.py
```

**NeMo RL** = the training framework. It does what TRL does: loads the model, generates completions,
computes advantages, updates weights. Handles distributed training, LoRA, vLLM generation,
sequence packing, checkpointing. Think of it as "TRL but for NVIDIA's scale."

**NeMo Gym** = the reward environment. It does what your `reward_fn` callback does in TRL.
But instead of a Python function passed to the trainer, it's a **FastAPI server** that NeMo RL
calls over HTTP. This separation means you can run reward servers on different hardware,
swap reward logic without rebuilding training code, and scale them independently.

## How They Connect

In TRL, you pass a reward function directly:
```python
# TRL style
trainer = GRPOTrainer(reward_funcs=[my_reward_fn], ...)
```

In NeMo RL + NeMo Gym, the reward is an HTTP call:
```
NeMo RL generates completions
    → POST /verify { responses_create_params: ..., response: ... }
    ← { reward: 0.7, ...original_request_fields... }
```

The key NeMo Gym base classes (from `nemo_gym/base_resources_server.py`):

```python
class BaseRunRequest(BaseModel):
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming

class BaseVerifyRequest(BaseRunRequest):
    response: NeMoGymResponse  # The model's generated response

class BaseVerifyResponse(BaseVerifyRequest):
    reward: float  # Your computed reward
```

---

## Training Parameters Deep-Dive

These are the knobs that matter for GRPO training. All configurable in `cql_nemo_rl_nemotron30b.yaml`.

### Gradient Clipping

```yaml
policy:
  max_grad_norm: 1.0   # L2 norm clipping. Standard value.
```

Same as `torch.nn.utils.clip_grad_norm_()`. Prevents exploding gradients — especially
important in RL where reward variance can cause large gradient spikes. Start at 1.0,
lower to 0.5 if you see training instability.

### Gradient Accumulation

NeMo RL handles this implicitly through `train_global_batch_size` and `train_micro_batch_size`:

```yaml
policy:
  train_global_batch_size: 16   # Total batch = num_prompts × num_generations
  train_micro_batch_size: 4     # Per-GPU micro-batch
```

Effective gradient accumulation steps = `global_batch / (micro_batch × num_GPUs)`.
With 1 GPU: 16/4 = 4 accumulation steps. With 4 GPUs: 16/(4×4) = 1 (no accumulation needed).

Unlike TRL where you set `gradient_accumulation_steps` directly, NeMo RL infers it from
batch sizes. This is more natural for multi-GPU scaling since you don't need to recalculate
accumulation steps when adding GPUs.

### Activation Checkpointing (Gradient Checkpointing)

Same thing, two names. Recomputes activations during backward instead of storing them.
Saves ~30-40% GPU memory at a ~15% speed cost.

```yaml
# DTensor/FSDP2 backend (models <10B):
policy:
  dtensor_cfg:
    activation_checkpointing: true    # Enable for memory savings

# Megatron-Core backend (models >10B):
policy:
  megatron_cfg:
    activation_checkpointing: true
```

Our dummy config has it `false` since we're not training a real model. For production
with Nemotron-Mini-4B + LoRA on a single A100 40GB, you'll want it `true`.

### KL Penalty (Reference Policy Divergence)

```yaml
loss_fn:
  reference_policy_kl_penalty: 0.0    # β: weight of KL term in loss
  reference_policy_kl_type: "k3"      # KL approximation: k1, k2, or k3
  use_kl_in_reward: false             # If true, KL is subtracted from reward instead of added to loss
  use_on_policy_kl_approximation: false
```

The full loss is: `L = -advantage × clipped_ratio + β × KL(π_θ ‖ π_ref)`

- **β = 0.0** (our config): No KL penalty. Model can drift freely from the reference.
  Good for early exploration. This is what DeepSeek-R1 used initially.
- **β = 0.01–0.05**: Mild constraint. Prevents reward hacking.
  Add this in production once the model is producing good CQL.
- **β = 0.1+**: Strong constraint. Model stays very close to reference.
  Use if you see mode collapse or output degradation.

**`use_kl_in_reward: true`** (Reinforce++ style): Instead of a loss term, KL is subtracted
directly from the reward signal. This makes KL interact with advantage normalization.
More aggressive than the loss term approach.

**`reference_policy_kl_type`**: The k3 approximation
(`KL ≈ (r-1) - log(r)` where `r = π_θ/π_ref`) is the default and most stable.

**TRL equivalent:**
```python
GRPOConfig(beta=0.01, ...)  # TRL calls it "beta", NeMo RL calls it "reference_policy_kl_penalty"
```

### Policy Ratio Clipping

```yaml
loss_fn:
  ratio_clip_min: 0.2     # ε_low → lower bound = 1 - 0.2 = 0.8
  ratio_clip_max: 0.28    # ε_high → upper bound = 1 + 0.28 = 1.28
  ratio_clip_c: null       # DAPO's token-level ratio clamping (see below)
```

**These are epsilon values, not absolute bounds.** The importance sampling ratio
`r = π_new/π_old` is clipped to `[1 - ε_low, 1 + ε_high]`.

- **Symmetric** (standard PPO): `ratio_clip_min: 0.2, ratio_clip_max: 0.2` → [0.8, 1.2]
- **Asymmetric / Clip-Higher** (our config): `0.2, 0.28` → [0.8, 1.28]
  Wider upper clip lets the model update more aggressively on high-advantage tokens.
  Used in DAPO and empirically better for code generation.

### Reward Normalization and Dynamic Sampling

```yaml
grpo:
  normalize_rewards: true                  # Subtract mean, divide by std within group
  use_leave_one_out_baseline: true         # LOO: advantage_i = r_i - mean(r_{j≠i})
  use_dynamic_sampling: true               # Filter zero-variance groups
  dynamic_sampling_max_gen_batches: 5      # Max retries to find high-variance groups
```

**Leave-one-out baseline**: Instead of `advantage_i = r_i - mean(all)`, uses
`advantage_i = r_i - mean(others)`. Reduces variance in the advantage estimate.
Same idea as the GSM8K GRPO you've done.

**Dynamic sampling**: If all K generations for a prompt get the same reward (all correct
or all wrong), the normalized advantages are all zero — no gradient. Dynamic sampling
re-generates with new prompts to avoid wasting compute. Mostly matters with binary rewards;
with our continuous reward, zero-variance groups are rare.

### Overlong Reward Shaping

```yaml
grpo:
  reward_shaping:
    enabled: false                          # We disable for CQL (short queries)
    overlong_buffer_length: 128             # Tokens before max where penalty starts
    overlong_buffer_penalty: 1.0            # Penalty magnitude
    max_response_length: 512                # Max allowed response length
    stop_properly_penalty_coef: null        # Penalty if model doesn't produce stop token
```

When enabled, responses approaching `max_response_length` get a linearly increasing penalty
starting at `max_response_length - overlong_buffer_length`. This discourages the model
from generating excessively long outputs.

**For CQL, we keep this off** — CQL queries are short (10-50 tokens). But if you were training
a chain-of-thought reasoning model, this would be important.

### Reward Scaling

```yaml
grpo:
  reward_scaling:
    enabled: false
    source_min: 0.0
    source_max: 1.0
    target_min: 0.0
    target_max: 1.0
```

Linear rescaling of rewards before normalization. Useful when combining reward signals
from different sources that operate on different scales. Our rewards are already [0, 1]
so we don't need this.

---

## Algorithm Variants

NeMo RL supports three GRPO-family algorithms. Here's what they fix and when to use them.

### Vanilla GRPO (DeepSeek-R1)

The original: generate K completions per prompt, compute advantages within the group,
clip the policy ratio symmetrically, update.

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.2          # Symmetric clipping
  token_level_loss: true
  reference_policy_kl_penalty: 0.01
grpo:
  use_dynamic_sampling: false
  reward_shaping:
    enabled: false
```

**Known problems:**
1. **Length bias**: Same advantage applied to every token. Longer responses accumulate
   more gradient just from having more tokens, even if reward is the same.
   The loss is `sum_t(advantage × log π(token_t))` — longer = bigger sum.
2. **Difficulty bias**: Hard prompts where all generations fail get zero variance and
   waste compute.
3. **Conservative clipping**: Symmetric clip limits how fast the model can learn from
   high-advantage actions.

### DAPO (Decoupled Clip and Dynamic Sampling)

Fixes problems 2 and 3 from vanilla GRPO. NeMo RL supports DAPO natively.

```yaml
loss_fn:
  ratio_clip_min: 0.2
  ratio_clip_max: 0.28         # Clip-Higher: asymmetric, wider upper clip
  ratio_clip_c: 0.2            # Token-level clipping (DAPO-specific)
  token_level_loss: true
  reference_policy_kl_penalty: 0.0   # DAPO typically uses no KL
grpo:
  use_dynamic_sampling: true   # Filter zero-variance groups
  reward_shaping:
    enabled: true              # Overlong penalty
    overlong_buffer_length: 128
    overlong_buffer_penalty: 1.0
```

**DAPO's four innovations:**
1. **Clip-Higher** (`ratio_clip_max > ratio_clip_min`): Lets high-advantage tokens
   update more aggressively. Maintains exploration diversity.
2. **Dynamic sampling**: Re-samples when all generations have same reward.
3. **Token-level clipping** (`ratio_clip_c`): Clamps individual token ratios
   before computing the sequence-level objective. Prevents single tokens from
   dominating the update.
4. **Overlong reward shaping**: Smooth penalty near max length to stop runaway generation.

**Our config uses DAPO-style settings** (Clip-Higher + dynamic sampling), minus the
overlong shaping since CQL queries are short.

Reference: https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html

### Dr. GRPO ("GRPO Done Right")

Fixes problem 1 — the length bias. Simple but powerful.

**The core change: length-normalized loss.**

```
# Vanilla GRPO (length-biased):
Loss = sum_t [ advantage_t × log π(a_t|...) ]      # longer response → bigger gradient

# Dr. GRPO (length-normalized):
Loss = (1/L) × sum_t [ advantage_t × log π(a_t|...) ]   # divide by response length L
```

This single change ensures the gradient magnitude doesn't depend on response length.
The model optimizes for response *quality*, not verbosity.

**In NeMo RL**, `token_level_loss: true` already computes per-token loss. The length
normalization is effectively achieved by averaging over tokens rather than summing.
To explicitly enable Dr. GRPO-style behavior, this is already the default when
`token_level_loss: true` — the loss is a mean over tokens, not a sum.

**In TRL**, this maps to:
```python
GRPOConfig(loss_type="dr_grpo", ...)  # or loss_agg_mode="token-mean" in some versions
```

### λ-GRPO (Bleeding Edge)

Latest research (arXiv:2510.06870). Instead of fixed length normalization, learns a
per-token weighting function λ that adaptively determines how much each token contributes
to the gradient. Unifies GRPO, Dr. GRPO, and DAPO as special cases. Not yet in NeMo RL
but worth watching.

### Which to use for CQL?

**Start with our current config (DAPO-style + token_level_loss):**
- Clip-Higher (0.2/0.28) for better exploration
- Dynamic sampling to avoid wasted compute
- Token-level loss (which implicitly length-normalizes via averaging)
- KL = 0 initially, add 0.01 when moving to production

If the model starts generating verbose CQL (unlikely since CQL is naturally concise),
enable overlong shaping. If the model plateau, try full DAPO with `ratio_clip_c`.

---

## Distributed Training

NeMo RL does NOT use DeepSpeed. It has two native backends.

### DTensor / FSDP2 (PyTorch native) — models <10B

This is what we use. PyTorch's Fully Sharded Data Parallel via DTensor.

```yaml
policy:
  dtensor_cfg:
    enabled: true
    cpu_offload: false          # Offload params to CPU (slower but saves VRAM)
    sequence_parallel: false    # Parallelize along sequence dimension
    activation_checkpointing: false
    tensor_parallel_size: 1     # Split model across GPUs (1 = no split)
    context_parallel_size: 1    # Split context/KV cache across GPUs
```

**Multi-GPU scaling with DTensor:**
```yaml
# 2 GPUs on 1 node:
cluster:
  gpus_per_node: 2
  num_nodes: 1
policy:
  dtensor_cfg:
    tensor_parallel_size: 2     # Split model across 2 GPUs

# 4 GPUs on 1 node (Nemotron-Mini-4B):
cluster:
  gpus_per_node: 4
  num_nodes: 1
policy:
  dtensor_cfg:
    tensor_parallel_size: 1     # Model fits on 1 GPU with LoRA
    # FSDP handles data parallelism automatically across 4 GPUs
  train_global_batch_size: 64   # 4× larger batch
```

**DTensor does FSDP2 (not ZeRO3), but the concept is the same:**
model parameters, gradients, and optimizer states are sharded across GPUs.
The main difference: DTensor uses PyTorch-native tensor sharding plans, ZeRO3 uses
DeepSpeed's partitioning. In practice, FSDP2 is more tightly integrated with
PyTorch and better for NeMo RL's architecture.

### Megatron-Core — models >10B

For when you need serious scale. 6D parallelism: tensor, pipeline, sequence, context,
expert, and data parallel. Used for 70B+ models.

```yaml
policy:
  megatron_cfg:
    enabled: true
    activation_checkpointing: true
    tensor_model_parallel_size: 8
    pipeline_model_parallel_size: 2
    context_parallel_size: 1
    sequence_parallel: true
```

You won't need this for Nemotron-Mini-4B, but if you scale to Nemotron-70B,
this is the path. Reference config: `grpo_math_1B_megatron.yaml` in the NeMo RL repo.

### What about DeepSpeed ZeRO3?

**DeepSpeed is NOT supported in NeMo RL. Period.** The official training backends doc
lists exactly two backends: DTensor (FSDP2) and Megatron-Core. There is zero DeepSpeed
code in the repo, no config keys for it, and no roadmap mention. Some web sources
incorrectly claim DeepSpeed support — the actual source code and docs prove otherwise.

FSDP2 does the same thing as ZeRO3 (shards params + grads + optimizer states across
GPUs), just via PyTorch-native DTensor instead of DeepSpeed's partitioning. DTensor
also supports CPU offloading (`cpu_offload: true`).

**If you need DeepSpeed ZeRO3 specifically**, use TRL + DeepSpeed instead of NeMo RL.
For NeMo RL, use DTensor (FSDP2) — it's the functional equivalent.

---

## Building a NeMo Gym Environment

Here's how to build a reward environment from scratch — what you'd do if starting this
CQL project without the dummy pipeline.

### Step 1: Scaffold with the CLI

```bash
git clone https://github.com/NVIDIA-NeMo/Gym.git && cd Gym
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev

# Scaffold a new resource server:
ng_init_resources_server +entrypoint=resources_servers/cql_verifier
```

This creates:
```
resources_servers/cql_verifier/
├── app.py              # Your FastAPI server — implement verify() here
├── configs/
│   └── cql_verifier.yaml
├── data/
│   └── example.jsonl   # Example tasks (5+)
├── tests/
│   └── test_app.py     # Tests for your endpoints
└── requirements.txt
```

### Step 2: Implement the verify endpoint

```python
# app.py
from nemo_gym.base_resources_server import (
    SimpleResourcesServer, BaseVerifyRequest, BaseVerifyResponse,
    BaseResourcesServerConfig,
)

class CQLVerifierConfig(BaseResourcesServerConfig):
    pass

class CQLVerifier(SimpleResourcesServer):
    config: CQLVerifierConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Extract the generated CQL from the model's response
        generated_cql = body.response.choices[0].message.content

        # Extract golden CQL from the prompt metadata
        # (you attach this when creating prompts)
        golden_cql = body.responses_create_params.metadata.get("golden_cql", "")

        # Compute reward — this is YOUR domain logic
        result = compute_reward(generated_cql, golden_cql)

        # Return: all original request fields + reward
        return BaseVerifyResponse(**body.model_dump(), reward=result["reward"])
```

The `compute_reward()` function is **exactly our existing code** from
`cql_rewards.py` — modular weighted sum of format, ngram, execution.

### Step 3: Configure the environment

```yaml
# configs/cql_verifier.yaml
cql_verifier_resources_server:
  resources_servers:
    cql_verifier:
      entrypoint: app.py
      domain: coding
```

### Step 4: Add test data and validate

```bash
# Add example tasks to data/example.jsonl
# Run tests
pytest tests/

# Generate example rollouts (uses an OpenAI-compatible API for generation)
ng_collect_rollouts \
  +agent_name=cql_simple_agent \
  +input_jsonl_fpath=data/example.jsonl \
  +output_jsonl_fpath=data/example_rollouts.jsonl
```

### Step 5: Connect to NeMo RL

Point your NeMo RL config to this Gym environment. NeMo Gym starts the server
automatically, NeMo RL calls `/verify` on each generation.

**That's it.** The separation is clean: NeMo Gym handles reward logic,
NeMo RL handles training. Your reward functions are the domain expertise.

---

## Transitioning to Real NeMo RL

### What our dummy pipeline mocks vs what's real

| Component | Our Dummy | Real NeMo RL + Gym |
|-----------|----------|-------------------|
| **Model generation** | `random.random()` picks templates | vLLM generates real tokens |
| **Reward server** | Plain FastAPI with `VerifyRequest` | Subclass `SimpleResourcesServer` with `BaseVerifyRequest` |
| **Training loop** | `DummyTrainingLoop` | NeMo RL GRPO engine (Ray-orchestrated) |
| **Config loading** | `yaml.safe_load()` | OmegaConf (supports `${variable}` interpolation) |
| **Reward functions** | ✅ **Real** — `compute_reward()` is production code | Same, called from `verify()` |
| **CQL validator** | ✅ **Real** — no mocks | Same |
| **CQL tokenizer** | ✅ **Real** — no mocks | Same |
| **Execution check** | 🔶 **Mocked** (80% random) | Real LogScale sandbox |

### The transition diff

```python
# Before (dummy):
@app.post("/verify")
async def verify(request: VerifyRequest):
    golden_cql = request.metadata.get("golden_cql", "")
    generated_cql = request.response

# After (real NeMo Gym):
from nemo_gym.base_resources_server import SimpleResourcesServer, BaseVerifyRequest, BaseVerifyResponse

class CQLServer(SimpleResourcesServer):
    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        generated_cql = body.response.choices[0].message.content
        golden_cql = body.responses_create_params.metadata.get("golden_cql", "")
```

`compute_reward()`, `reward_syntax()`, `reward_execution()`, `reward_ngram()` stay
**exactly the same**.

### Launch with real NeMo RL

```bash
# Instead of: python scripts/train_grpo.py
uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml
```

Our `cql_nemo_rl_nemotron30b.yaml` is verified against the real NVIDIA recipe.

---

## NeMo RL Config Explained

Section-by-section reference for `cql_nemo_rl_nemotron30b.yaml`.

### `grpo:` — Algorithm parameters

```yaml
grpo:
  num_prompts_per_step: 4      # Prompts per training step
  num_generations_per_prompt: 4 # Completions per prompt (the "group" in GRPO)
  normalize_rewards: true       # z-score within group
  use_leave_one_out_baseline: true  # LOO: advantage_i = r_i - mean(r_{j≠i})
  use_dynamic_sampling: true    # Skip zero-variance groups
```

TRL equivalent: `GRPOConfig(num_generations=4, ...)`

### `loss_fn:` — Loss with clipping and KL

```yaml
loss_fn:
  reference_policy_kl_penalty: 0.0   # β in: L = -adv × clip(ratio) + β × KL
  ratio_clip_min: 0.2                # ε_low → bound at 1-0.2 = 0.8
  ratio_clip_max: 0.28               # ε_high → bound at 1+0.28 = 1.28
  token_level_loss: true              # Per-token loss (Dr. GRPO-style normalization)
```

### `policy:` — Model, optimizer, generation

```yaml
policy:
  model_name: "nvidia/Nemotron-Mini-4B-Instruct"
  max_grad_norm: 1.0           # Gradient clipping
  dtensor_cfg:
    activation_checkpointing: false  # Set true to save memory
    lora_cfg:
      enabled: true
      dim: 16                  # LoRA rank
      alpha: 32                # Scale = alpha/dim = 2.0
  generation:
    backend: "vllm"
    colocated:
      enabled: true            # Share GPU between training and generation
```

### `env:` — Environment connection

```yaml
env:
  cql:
    num_workers: 1
    reward_server_url: "http://localhost:8080"
```

In real NeMo RL this points to NeMo Gym. The `math` environment uses
`math_verify_impl: "hf_math_verify"` instead.

---

## Key Differences from TRL

| Aspect | TRL | NeMo RL |
|--------|-----|---------|
| Reward | Python function | HTTP server (NeMo Gym) |
| Generation | HF generate or vLLM | vLLM (colocated or dedicated) |
| Distributed | DeepSpeed ZeRO | DTensor (FSDP2) or Megatron-Core |
| Config | Python dataclass | YAML + OmegaConf |
| Clipping | Symmetric only | Symmetric or Clip-Higher |
| Length normalization | `loss_type="dr_grpo"` | `token_level_loss: true` |
| KL penalty | `beta=0.01` | `reference_policy_kl_penalty: 0.01` |
| Multi-GPU | DeepSpeed stages | Native tensor/pipeline/context parallelism |
| Grad clipping | `max_grad_norm` in TrainingArguments | `policy.max_grad_norm` |
| Activation checkpointing | `gradient_checkpointing=True` | `dtensor_cfg.activation_checkpointing: true` |
| Grad accumulation | `gradient_accumulation_steps: N` | Implicit from `global_batch / (micro_batch × GPUs)` |
| Scaling | ~8B practical max | 70B+ with Megatron-Core |

---

## References

**NeMo RL:**
- [GRPO in-depth walkthrough](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)
- [DAPO guide](https://docs.nvidia.com/nemo/rl/latest/guides/dapo.html)
- [Loss functions API](https://docs.nvidia.com/nemo/rl/latest/apidocs/nemo_rl/nemo_rl.algorithms.loss_functions.html)
- [GitHub: configs + examples](https://github.com/NVIDIA-NeMo/RL/tree/main/examples)
- [Features and roadmap](https://docs.nvidia.com/nemo/rl/latest/about/features.html)

**NeMo Gym:**
- [Creating a resource server](https://docs.nvidia.com/nemo/gym/0.1.0/tutorials/creating-resource-server.html)
- [Training environment setup](https://docs.nvidia.com/nemo/gym/latest/environment-tutorials/creating-training-environment.html)
- [GitHub](https://github.com/NVIDIA-NeMo/Gym)

**Algorithm papers:**
- [GRPO / DAPO / Dr. GRPO deep-dive (HuggingFace forum)](https://discuss.huggingface.co/t/offering-a-technical-deep-dive-on-grpo-dapo-dr-grpo-algorithms/154480)
- [λ-GRPO: learnable token preferences (arXiv:2510.06870)](https://arxiv.org/abs/2510.06870)
- [MMR-GRPO: diversity-aware reward reweighting (arXiv:2601.09085)](https://arxiv.org/abs/2601.09085)

**Our project:**
- [Reward design & strategy](rewards_and_strategy.md)
- [SLURM multi-node deployment](slurm_multinode.md)
