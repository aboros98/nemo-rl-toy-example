# CQL RLVR — Reinforcement Learning from Verifiable Rewards for NL-to-CQL

Train a model to generate CrowdStrike LogScale Query Language (CQL) from natural language,
using NVIDIA NeMo RL for GRPO/SFT training and NeMo Gym for reward evaluation.

**Target:** Nemotron-3-Nano-30B-A3B on 1 node × 8 B200 GPUs.

---

## Installation

NeMo RL **cannot be installed with pip**. Two options:

### Option A — NGC Container (recommended for training)

```bash
# Pull the official container (latest: v0.5.0)
docker pull nvcr.io/nvidia/nemo-rl:v0.5.0

# Run with GPUs
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $(pwd):/workspace/cql_rlvr \
  nvcr.io/nvidia/nemo-rl:v0.5.0

# Inside container, all dependencies are pre-installed.
# Run scripts with `uv run python ...`
```

### Option B — Source install (for development)

```bash
# 1. Install uv (Python environment manager used by NeMo RL)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone NeMo RL with submodules
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# 3. Create virtual environment (let uv read .python-version)
uv venv

# 4. Scripts are run with uv run (NOT python or pip)
uv run python examples/run_grpo.py
```

### This project's Python dependencies (for local dry-run/tests only)

```bash
pip install -r requirements.txt
python3 -m pytest utils/ -v
```

---

## Quick Start

### 1. Validate locally (no GPU needed)

```bash
# Fetch training data (3 public sources → 442 train / 55 val / 56 test)
python3 scripts/fetch_data.py

# Run unit tests
python3 -m pytest utils/ -v

# Dry-run configs (validates structure, data paths, LoRA settings)
python3 scripts/run_grpo_cql.py --dry-run
python3 scripts/run_sft_cql.py --config configs/sft_cql_config.yaml --dry-run
```

### 2. Train with NeMo RL (inside container or source install)

```bash
# SFT warmup (recommended before GRPO)
uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml

# GRPO reinforcement learning
uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml

# Override any parameter via Hydra-style CLI
uv run python scripts/run_grpo_cql.py ++grpo.max_num_steps=50 ++policy.optimizer.kwargs.lr=1e-5
```

### 3. Train on SLURM cluster

```bash
# One command — handles container, Ray, and NeMo RL
bash scripts/slurm/submit_sft.sh              # SFT warmup
bash scripts/slurm/submit_grpo.sh             # GRPO training
bash scripts/slurm/submit_grpo.sh --steps 100 # Override steps
```

See [scripts/slurm/README.md](scripts/slurm/README.md) for SLURM details.

---

## How the Scripts Work

Our scripts (`run_grpo_cql.py`, `run_sft_cql.py`) are **thin wrappers** around
the official NeMo RL examples. They follow the exact same API pattern:

```
Official:  examples/run_grpo.py    →  Our:  scripts/run_grpo_cql.py
Official:  examples/run_sft.py     →  Our:  scripts/run_sft_cql.py
Official:  examples/configs/*.yaml →  Our:  configs/*.yaml
Official:  ray.sub                 →  Our:  scripts/slurm/ray_1node.sub
```

**What we add on top of the official pattern:**
1. `--dry-run` flag — validates config and data without NeMo RL installed
2. `register_cql_processor()` — our custom data processor for system/user/assistant roles
3. CQL-specific configs — reward server, LoRA tuned for Mamba2, schema-aware prompts

**The training flow is identical to official:**
```
register_omegaconf_resolvers() → load_config() → parse_hydra_overrides()
→ OmegaConf.to_container() → init_ray() → get_tokenizer()
→ setup_response_data() → setup() → grpo_train()
```

---

## Project Structure

```
cql_rlvr/
├── scripts/
│   ├── run_grpo_cql.py              # GRPO training (matches official run_grpo.py)
│   ├── run_sft_cql.py               # SFT training (matches official run_sft.py)
│   ├── train_grpo.py                # Dummy local training loop (no NeMo RL needed)
│   ├── fetch_data.py                # Data pipeline (3 public sources)
│   ├── run_dummy_pipeline.sh        # End-to-end local validation
│   ├── launch_nemotron30b.sh        # Production launcher
│   └── slurm/
│       ├── ray_1node.sub            # SLURM Ray launcher (simplified from official ray.sub)
│       ├── submit_grpo.sh           # One-command GRPO submission
│       ├── submit_sft.sh            # One-command SFT submission
│       └── README.md                # Beginner SLURM guide
├── configs/
│   ├── cql_nemo_rl_nemotron30b.yaml      # GRPO LoRA (Nemotron-30B, 8×B200)
│   ├── cql_nemo_rl_nemotron30b_full.yaml  # GRPO full fine-tuning (no LoRA)
│   ├── sft_cql_config.yaml               # SFT warmup (LoRA)
│   ├── sft_cql_full_config.yaml          # SFT warmup (full fine-tuning)
│   ├── cql_nemo_rl_config.yaml           # Dummy validation (single GPU, small model)
│   └── cql_gym_config.yaml               # NeMo Gym environment config
├── resources/
│   ├── cql_resource_server.py       # Reward server (syntax + execution + ngram)
│   └── cql_system_prompt.txt        # Few-shot CQL expert prompt
├── utils/
│   ├── cql_tokenizer.py             # Semantic CQL tokenizer
│   ├── cql_validator.py             # CQL syntax validator
│   ├── cql_data_processor.py        # Custom NeMo RL data processor
│   └── test_*.py                    # Unit tests (61 tests)
├── data/                            # train.jsonl, val.jsonl, test.jsonl
├── logs/                            # Training logs, TensorBoard
├── docs/                            # Full documentation
└── requirements.txt
```

---

## Configs — Verified Against Official NVIDIA Recipes

Our production config is based on the **official NVIDIA recipe**:
[`grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml)

| Setting | Official Recipe | Our Config | Notes |
|---------|----------------|------------|-------|
| `num_prompts_per_step` | 2 | 2 | ✓ Match |
| `num_generations_per_prompt` | 8 | 8 | ✓ Match |
| `train_global_batch_size` | 16 | 16 | ✓ Match |
| `LoRA dim` | 128 | 128 | ✓ Match |
| `LoRA alpha` | 512 | 512 | ✓ Match |
| `exclude_modules` | `['*out_proj*']` | `['*out_proj*']` | ✓ Critical for Mamba2 |
| `match_all_linear` | false | false | ✓ Match |
| `use_triton` | false | false | ✓ Required for Mamba2 |
| `sequence_packing` | false | false | ✓ Not validated for Mamba2 |
| `vllm TP` | 4 | 4 | ✓ Match |
| `gpu_memory_utilization` | 0.7 | 0.7 | ✓ Match |
| `num_nodes` | 2 (80GB GPUs) | 1 (192GB B200s) | Adapted for B200 |

---

## Reward Function

Three components, combined with a **hard invariant** (invalid < valid):

| Component | Weight | Description |
|-----------|--------|-------------|
| Syntax | 0.4 | CQL validator — binary valid/invalid |
| Execution | 0.3 | Mock 80% success (production: LogScale sandbox) |
| N-gram | 0.3 | Bigram similarity to golden query |

Max reward: 1.0. Invalid syntax → hard penalty (guaranteed < any valid query).

---

## LoRA vs Full Fine-Tuning

Both modes are supported. Choose based on your goals:

| | LoRA | Full Fine-Tuning |
|---|---|---|
| **GRPO config** | `cql_nemo_rl_nemotron30b.yaml` | `cql_nemo_rl_nemotron30b_full.yaml` |
| **SFT config** | `sft_cql_config.yaml` | `sft_cql_full_config.yaml` |
| **Trainable params** | ~1% of model (LoRA adapters) | 100% (all 30B params) |
| **Memory per GPU** | ~15 GB | ~45 GB |
| **B200 headroom** | ~177 GB free | ~147 GB free |
| **Training speed** | Faster (fewer params to update) | Slower (~2-3×) |
| **Quality ceiling** | Good for domain adaptation | Higher (can reshape all weights) |
| **Recommended for** | First runs, iteration, constrained budget | Final production model |

### Run LoRA

```bash
# SFT warmup with LoRA
uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml

# GRPO with LoRA
uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml
```

### Run Full Fine-Tuning

```bash
# SFT warmup — full fine-tuning
uv run python scripts/run_sft_cql.py --config configs/sft_cql_full_config.yaml

# GRPO — full fine-tuning
uv run python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b_full.yaml
```

### SLURM — just pass the config

```bash
# LoRA GRPO (default)
bash scripts/slurm/submit_grpo.sh

# Full fine-tuning GRPO
bash scripts/slurm/submit_grpo.sh --config configs/cql_nemo_rl_nemotron30b_full.yaml

# Full fine-tuning SFT
bash scripts/slurm/submit_sft.sh --config configs/sft_cql_full_config.yaml
```

### Dry-run any config (no GPU needed)

```bash
python3 scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b_full.yaml --dry-run
python3 scripts/run_sft_cql.py --config configs/sft_cql_full_config.yaml --dry-run
```

---

## Training Pipeline

**Recommended order:** SFT warmup → GRPO reinforcement learning

```bash
# Step 1: SFT warmup (~200 steps, teaches basic CQL structure)
uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml

# Step 2: GRPO training (500+ steps, reward-driven optimization)
# Point to the SFT checkpoint as the starting model:
uv run python scripts/run_grpo_cql.py \
  --config configs/cql_nemo_rl_nemotron30b.yaml \
  ++policy.model_name=results/sft_cql/best_checkpoint
```

---

## Docs

- **[NeMo RL Parameters](docs/nemo_rl_parameters.md)** — complete config reference, scheduler options
- **[GRPO vs GSPO vs DAPO](docs/grpo_algorithms.md)** — algorithm deep-dive
- **[NeMo Gym & RL Guide](docs/nemo_gym_rl_guide.md)** — architecture, reward servers
- **[Multi-Node SLURM](docs/slurm_multinode.md)** — beginner SLURM guide
- **[Rewards & Strategy](docs/rewards_and_strategy.md)** — reward design, weight math
