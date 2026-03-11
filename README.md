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

# Run unit tests (69 tests — validator + tokenizer + rewards)
python3 -m pytest utils/ -v

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

NeMo RL **cannot be installed with pip**. Two options — **source install is recommended** for debugging:

**Option A — Source install (recommended for development)**
```bash
# Clone NeMo RL
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl && uv venv && source .venv/bin/activate

# Clone this project alongside (or inside)
git clone git@github.com:aboros98/nemo-rl-toy-example.git cql_rlvr

# Set PYTHONPATH so both projects can import each other
export PYTHONPATH=$(pwd)/cql_rlvr:$PYTHONPATH

# Run training
uv run python cql_rlvr/scripts/run_sft_cql.py --config cql_rlvr/configs/sft_cql_config.yaml
```

Why source: you can `print()` / `breakpoint()` anywhere in NeMo RL internals, reward functions, or data processors. You WILL need this.

**Option B — NGC container (for SLURM / reproducible deployment)**
```bash
docker pull nvcr.io/nvidia/nemo-rl:v0.5.0

docker run --gpus all -it --rm \
  --shm-size=16g \
  -v $(pwd):/workspace/cql_rlvr \
  nvcr.io/nvidia/nemo-rl:v0.5.0

# Inside the container:
cd /workspace/cql_rlvr
uv run python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml
```

Switch to Docker only when you're done debugging and want reproducible SLURM runs.

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
- `mean_reward` — average combined reward (should increase)
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

```
Prompt → vLLM generates 8 responses → CQLEnvironment.step() scores each → GRPO computes advantages
```

### Three reward components (R1-style)

| Component | Weight | What it measures | Range |
|-----------|--------|-----------------|-------|
| **N-gram similarity** | 0.8 | Bigram F1 between generated CQL and reference | 0.0 – 1.0 |
| **Format (think tags)** | 0.2 | Does the model use `<think>...</think>` reasoning? | 0.0 / 0.5 / 1.0 |
| **Execution** | 0.0 | Placeholder — Docker LogScale compilation check | 0.0 always |

**Combined reward** = `0.8 × ngram + 0.2 × format + 0.0 × execution` (configurable in YAML)

**Format reward scoring:**
- `0.0` — no `<think>` or `</think>` tags
- `0.5` — has one tag but not both
- `1.0` — has both `<think>` and `</think>`

**Example rewards:**
| Response | Format | N-gram | Total |
|----------|--------|--------|-------|
| `<think>reasoning</think>\n<perfect CQL>` | 1.0 | 1.0 | **1.0** |
| `<perfect CQL>` (no think tags) | 0.0 | 1.0 | **0.8** |
| `<think>reasoning</think>\n<wrong CQL>` | 1.0 | 0.2 | **0.36** |
| `<think>only thinking, no CQL` | 0.5 | 0.0 | **0.10** |
| empty response | 0.0 | 0.0 | **0.0** |

### Architecture

```
utils/cql_rewards.py              ← Pure Python reward logic (no GPU deps)
  ├── compute_format_reward()     ← Think tag scoring
  ├── compute_ngram_reward()      ← Bigram similarity
  ├── compute_execution_reward()  ← Placeholder for Docker LogScale
  └── compute_combined_reward()   ← Weighted sum of all three

environments/cql_environment.py   ← NeMo RL wrapper (imports from cql_rewards.py)
scripts/test_rewards_local.py     ← Local testing (imports from cql_rewards.py)
```

Both the local test script and the real environment import from the **same** `utils/cql_rewards.py` — so local testing always matches production.

### Changing reward weights

In `configs/cql_nemo_rl_nemotron30b.yaml`:
```yaml
env:
  cql:
    num_workers: 8
    reward_weights:
      format: 0.2      # think tag compliance
      ngram: 0.8        # bigram similarity to reference
      execution: 0.0    # set >0 when Docker LogScale is ready
```

Or override from CLI:
```bash
OVERRIDES="++env.cql.reward_weights.format=0.3 ++env.cql.reward_weights.ngram=0.7" sbatch scripts/slurm/grpo.sh
```

### Testing rewards locally (Mac, no GPU)

```bash
# Default weights
python3 scripts/test_rewards_local.py

# Custom weights
python3 scripts/test_rewards_local.py --weights '{"format":0.3,"ngram":0.7,"execution":0.0}'

# More examples
python3 scripts/test_rewards_local.py --n 50 --data data/val.jsonl
```

**To change the reward logic** — edit `utils/cql_rewards.py`, then re-run the test script. Both local and production use the same file.

### Adding the execution reward (Docker LogScale)

When you have a Docker container that can compile CQL queries:

1. Edit `compute_execution_reward()` in `utils/cql_rewards.py` — add HTTP call to Docker
2. Test locally: `python3 scripts/test_rewards_local.py --weights '{"format":0.1,"ngram":0.3,"execution":0.6}'`
3. Update config: set `execution: 0.6` (or whatever weight you choose)
4. Push and retrain

---

## Configs

| Config | Training | Mode | Use for |
|--------|----------|------|---------|
| `cql_nemo_rl_nemotron30b.yaml` | GRPO | LoRA | **Default production** — start here |
| `cql_nemo_rl_nemotron30b_full.yaml` | GRPO | Full FT | Higher ceiling, more memory |
| `sft_cql_config.yaml` | SFT | LoRA | SFT warmup before GRPO |
| `sft_cql_full_config.yaml` | SFT | Full FT | SFT warmup, full params |


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
│   └── sft_cql_full_config.yaml           # SFT full FT
├── utils/
│   ├── cql_rewards.py          # Reward functions (format + ngram + execution)
│   ├── cql_validator.py         # CQL syntax validator
│   ├── cql_tokenizer.py         # CQL semantic tokenizer
│   ├── cql_data_processor.py    # NeMo RL data processor (system/user/assistant)
│   └── test_*.py                # Unit tests
├── resources/
│   └── cql_system_prompt.txt    # Few-shot system prompt
├── data/                        # train.jsonl, val.jsonl, test.jsonl
├── logs/                        # TensorBoard logs
└── docs/                        # All documentation
```

---

## Docs

| Doc | What it covers |
|-----|---------------|
| **[Local Setup](docs/local_setup.md)** | **Start here — Mac/laptop dev setup, testing rewards, troubleshooting** |
| [Hyperparameters](docs/hyperparameters.md) | Every tunable param — GRPO, optimizer, LoRA, vLLM, scheduler |
| [Reward Environments](docs/reward_environments.md) | How to build custom rewards, patterns, templates |
| [GRPO Algorithms](docs/grpo_algorithms.md) | GRPO vs GSPO vs DAPO deep-dive |
| [NeMo RL Parameters](docs/nemo_rl_parameters.md) | Full config reference |
| [NeMo Gym & RL Guide](docs/nemo_gym_rl_guide.md) | Architecture, how it all fits together |
| [Model Setup & Logging](docs/model_setup_and_logging.md) | Chat templates, TensorBoard |
| [SLURM Guide](docs/slurm_multinode.md) | Beginner SLURM + multi-node |
| [SFT Packing](docs/sft_packing.md) | Sequence packing for SFT efficiency |
| [Rewards & Strategy](docs/rewards_and_strategy.md) | Reward design, weight math |
