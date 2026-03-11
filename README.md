# CQL RLVR — NL-to-CQL with NeMo RL

Train Nemotron-3-Nano-30B-A3B to generate CrowdStrike LogScale Query Language (CQL) from natural language using GRPO reinforcement learning.

**Setup:** 1 node × 8 B200 GPUs, NVIDIA NeMo RL, NGC container.

---

## How-To: From Zero to Training

### Step 0 — Local setup (Mac/laptop, no GPU)

```bash
git clone git@github.com:aboros98/nemo-rl-toy-example.git cql_rlvr
cd cql_rlvr

# Install local dependencies (validator, tokenizer tests)
pip install -r requirements.txt

# Fetch training data (3 public sources → 442 train / 55 val / 56 test)
python3 scripts/fetch_data.py

# Run unit tests (45 tests — validator + tokenizer)
python3 -m pytest utils/test_cql_validator.py utils/test_cql_tokenizer.py -v

# Test reward logic locally — no NeMo RL needed
python3 scripts/test_rewards_local.py
python3 scripts/test_rewards_local.py --n 50  # more examples

# Dry-run all configs (validates YAML structure)
python3 scripts/run_grpo_cql.py --dry-run
python3 scripts/run_sft_cql.py --dry-run
python3 scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b_full.yaml --dry-run
python3 scripts/run_sft_cql.py --config configs/sft_cql_full_config.yaml --dry-run
```

### Step 1 — Get NeMo RL on your GPU node

NeMo RL **cannot be installed with pip**. Use the NGC container:

```bash
# Pull the container
docker pull nvcr.io/nvidia/nemo-rl:v0.5.0

# Interactive shell (for debugging)
docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $(pwd):/workspace/cql_rlvr \
  nvcr.io/nvidia/nemo-rl:v0.5.0

# Inside the container, everything is ready:
cd /workspace/cql_rlvr
uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml
```

Alternative — source install for development:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl && uv venv
# Then copy this project into the nemo-rl directory or adjust sys.path
```

### Step 2 — SFT warmup (recommended first)

Supervised fine-tuning teaches the model basic CQL structure before RL.

```bash
# LoRA (fast, ~1% params, good for iteration)
sbatch scripts/slurm/sft.sh

# Full fine-tuning (all params, higher ceiling)
CONFIG=configs/sft_cql_full_config.yaml sbatch scripts/slurm/sft.sh

# Override anything
OVERRIDES="++sft.max_num_steps=100 ++policy.optimizer.kwargs.lr=5e-5" sbatch scripts/slurm/sft.sh
```

### Step 3 — GRPO training

Reinforcement learning using your reward function to optimize generation quality.

```bash
# LoRA GRPO (default production config)
sbatch scripts/slurm/grpo.sh

# Full fine-tuning GRPO
CONFIG=configs/cql_nemo_rl_nemotron30b_full.yaml sbatch scripts/slurm/grpo.sh

# Point to SFT checkpoint as starting model
OVERRIDES="++policy.model_name=/path/to/sft_checkpoint" sbatch scripts/slurm/grpo.sh

# Quick test run
OVERRIDES="++grpo.max_num_steps=10" sbatch scripts/slurm/grpo.sh
```

### Step 4 — Monitor with TensorBoard

```bash
tensorboard --logdir logs/ --port 6006
```

Key metrics to watch:
- `accuracy` — fraction of syntactically valid CQL (should increase)
- `reward/mean` — average reward (should increase)
- `generation_lengths` — watch for length hacking (shouldn't explode)
- `fraction_of_samples_properly_ended` — should stay high (>0.9)

---

## What Each Script Does

```
scripts/
├── run_grpo_cql.py          # GRPO training — the main RL script
├── run_sft_cql.py           # SFT training — supervised warmup
├── test_rewards_local.py    # Test rewards on Mac (no GPU/NeMo needed)
├── fetch_data.py            # Download + clean training data
└── slurm/
    ├── grpo.sh              # sbatch this → runs GRPO on SLURM
    └── sft.sh               # sbatch this → runs SFT on SLURM
```

**Training scripts** (~90 lines each) are thin wrappers around official NeMo RL:
1. Register our custom CQL data processor + reward environment
2. Load YAML config + CLI overrides
3. Call `grpo_train()` or `sft_train()` — identical to official examples

**SLURM scripts** (~55 lines each) are self-contained — one file does everything:
1. `#SBATCH` headers allocate 1 node × 8 GPUs
2. Launch the NGC container with your code mounted
3. Start Ray head (NeMo RL's orchestration layer)
4. Run the training script inside the container

Override via environment variables:
```bash
CONFIG=configs/sft_cql_full_config.yaml sbatch scripts/slurm/sft.sh
OVERRIDES="++grpo.max_num_steps=50" sbatch scripts/slurm/grpo.sh
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.6.0 sbatch scripts/slurm/grpo.sh
```

---

## How the Reward Works

During GRPO, for each prompt NeMo RL generates N rollouts (default: 8), then asks the **environment** to score them.

The reward flow:
```
Prompt → vLLM generates 8 CQL queries → CQLEnvironment.step() scores each → GRPO computes advantages
```

Our environment (`environments/cql_environment.py`, ~80 lines):
- Extracts the assistant's response from each conversation
- Runs it through `utils/cql_validator.validate()`
- Returns **1.0** for valid CQL, **0.0** for invalid
- `global_post_process_and_metrics()` masks rewards for incomplete sequences and logs accuracy to TensorBoard

**To change the reward** — edit `environments/cql_environment.py` or create a new environment. See [docs/reward_environments.md](docs/reward_environments.md) for the full guide.

**To test reward changes locally** — edit `compute_reward()` in `scripts/test_rewards_local.py` and run:
```bash
python3 scripts/test_rewards_local.py
```

---

## Configs

| Config | Training | Mode | Use for |
|--------|----------|------|---------|
| `cql_nemo_rl_nemotron30b.yaml` | GRPO | LoRA | **Default production** — start here |
| `cql_nemo_rl_nemotron30b_full.yaml` | GRPO | Full FT | Higher ceiling, more memory |
| `sft_cql_config.yaml` | SFT | LoRA | SFT warmup before GRPO |
| `sft_cql_full_config.yaml` | SFT | Full FT | SFT warmup, full params |
| `cql_nemo_rl_config.yaml` | GRPO | LoRA | Dummy validation (small model) |

All configs are verified against the [official NVIDIA recipe](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml). Key Mamba2 constraints enforced:
- `lora_cfg.exclude_modules: ['*out_proj*']` — mandatory (SSM has zero gradient)
- `sequence_packing.enabled: false` — not validated for Mamba2
- `use_triton: false` — required for Mamba2

---

## LoRA vs Full Fine-Tuning

| | LoRA | Full Fine-Tuning |
|---|---|---|
| **Trainable params** | ~1% (adapters only) | 100% (all 30B) |
| **Memory / GPU** | ~15 GB | ~45 GB |
| **Speed** | Fast | ~2-3× slower |
| **Quality** | Good for domain adaptation | Higher ceiling |
| **Start with** | ✅ This one first | When LoRA plateaus |

---

## Project Structure

```
cql_rlvr/
├── scripts/
│   ├── run_grpo_cql.py          # GRPO training (~90 lines)
│   ├── run_sft_cql.py           # SFT training (~130 lines)
│   ├── test_rewards_local.py    # Local reward testing (Mac)
│   ├── fetch_data.py            # Data pipeline
│   └── slurm/
│       ├── grpo.sh              # SLURM: sbatch → GRPO
│       ├── sft.sh               # SLURM: sbatch → SFT
│       └── README.md            # SLURM beginner guide
├── environments/
│   └── cql_environment.py       # Reward environment (~80 lines)
├── configs/
│   ├── cql_nemo_rl_nemotron30b.yaml       # GRPO LoRA (production)
│   ├── cql_nemo_rl_nemotron30b_full.yaml  # GRPO full FT
│   ├── sft_cql_config.yaml                # SFT LoRA
│   ├── sft_cql_full_config.yaml           # SFT full FT
│   └── cql_nemo_rl_config.yaml            # Dummy validation
├── utils/
│   ├── cql_validator.py         # CQL syntax validator
│   ├── cql_tokenizer.py         # CQL semantic tokenizer
│   ├── cql_data_processor.py    # NeMo RL data processor (system/user/assistant)
│   └── test_*.py                # Unit tests
├── resources/
│   ├── cql_resource_server.py   # FastAPI reward server (alternative to environment)
│   └── cql_system_prompt.txt    # Few-shot system prompt
├── data/                        # train.jsonl, val.jsonl, test.jsonl
├── logs/                        # TensorBoard logs
└── docs/                        # All documentation
```

---

## Docs

| Doc | What it covers |
|-----|---------------|
| [Hyperparameters](docs/hyperparameters.md) | Every tunable param — GRPO, optimizer, LoRA, vLLM, scheduler |
| [Reward Environments](docs/reward_environments.md) | How to build custom rewards, patterns, templates |
| [GRPO Algorithms](docs/grpo_algorithms.md) | GRPO vs GSPO vs DAPO deep-dive |
| [NeMo RL Parameters](docs/nemo_rl_parameters.md) | Full config reference |
| [NeMo Gym & RL Guide](docs/nemo_gym_rl_guide.md) | Architecture, how it all fits together |
| [Model Setup & Logging](docs/model_setup_and_logging.md) | Chat templates, TensorBoard |
| [SLURM Guide](docs/slurm_multinode.md) | Beginner SLURM + multi-node |
| [SFT Packing](docs/sft_packing.md) | Sequence packing for SFT efficiency |
| [Rewards & Strategy](docs/rewards_and_strategy.md) | Reward design, weight math |
