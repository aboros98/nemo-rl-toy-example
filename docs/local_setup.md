# Local Setup Guide — CQL RLVR

Everything you need to develop, test rewards, and validate configs on your Mac before touching a GPU cluster.

---

## Prerequisites

You need:
- **Python 3.11+** (3.13 or 3.14 both work)
- **Git** (for cloning data sources)
- **pip** (comes with Python)

Check:
```bash
python3 --version   # Should show 3.11+
git --version       # Any recent version
```

If you're on macOS with Homebrew:
```bash
brew install python@3.13 git
```

---

## Step 1 — Clone the repo

```bash
git clone git@github.com:aboros98/nemo-rl-toy-example.git cql_rlvr
cd cql_rlvr
```

---

## Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. Every command below assumes the venv is active.

To deactivate later: `deactivate`  
To reactivate: `source .venv/bin/activate`

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `pyyaml`, `pytest`, `fastapi`, `uvicorn`, `requests`, `pydantic`.

These are only for local testing — the GPU cluster uses the NGC container which has everything pre-installed.

Verify:
```bash
python -c "import yaml, pytest, fastapi; print('All imports OK')"
```

---

## Step 4 — Fetch training data

```bash
python scripts/fetch_data.py
```

This clones 3 public repos, extracts NL↔CQL pairs, deduplicates, and splits 80/10/10:

| Source | What it is | Pairs |
|--------|-----------|-------|
| ByteRay-Labs/Query-Hub | CQL queries with descriptions | ~200 |
| CrowdStrike/logscale-community-content | Community CQL queries from markdown | ~150 |
| microsoft/NL2KQL | KQL benchmark (transfers to CQL) | ~200 |

Output:
```
data/train.jsonl   — 442 examples
data/val.jsonl     — 55 examples
data/test.jsonl    — 56 examples
```

Each line is JSON:
```json
{
  "nl_query": "Show me all processes that accessed sensitive files in the last 24h",
  "cql_query": "ProcessRollup2\n| where Timestamp > ago(24h)\n| where TargetFileName has \"sensitive\"",
  "source": "query_hub",
  "tags": ["process", "file"],
  "schema_context": "Event: ProcessRollup2. Fields: Timestamp, TargetFileName, ..."
}
```

**Note:** The data is already committed to the repo, so this step is optional unless you want to re-fetch.

---

## Step 5 — Run unit tests

```bash
python -m pytest utils/ -v
```

Expected: **69 passed**. These test:
- CQL syntax validator (balanced parens, valid pipes, known functions, unclosed strings)
- CQL tokenizer (semantic tokenization, bigram similarity, structural similarity)
- Reward functions (format, ngram, combined, invariants)

To run a specific test file:
```bash
python -m pytest utils/test_reward_invariant.py -v
```

---

## Step 6 — Test reward logic

This is the most important local step — validate your reward function before deploying to the cluster.

```bash
python scripts/test_rewards_local.py
```

This runs the **exact same reward logic** the GPU training will use (`utils/cql_rewards.py`), on real data from `data/train.jsonl`. No NeMo RL, no Ray, no GPU needed.

Output:
```
Reward weights: format=0.2, ngram=0.8, execution=0.0

=== Ground truth queries (no <think> tags) — 10 examples ===
  #   Total    Fmt  Ngram  Query (first 60 chars)
------------------------------------------------------------------------------------------
  1   0.800   0.00   1.00  EmailEvents | | where Timestamp > ago(7d) ...
  2   0.800   0.00   1.00  DeviceNetworkEvents | | where Timestamp ...
Mean: 0.800  (format=0.0 expected since no think tags)

=== Same queries wrapped in <think>...</think> ===
  #   Total    Fmt  Ngram  Think?
------------------------------------------------------------------------------------------
  1   1.000   1.00   1.00  True
Mean: 1.000  (format=1.0 expected with proper tags)

=== Edge cases ===
Empty string                   0.000   0.00   0.00  ...
Think but no CQL after         0.200   1.00   0.00  ...
Perfect with think             1.000   1.00   1.00  ...

=== Incentive check: does <think> get higher reward? ===
  Bare CQL:         reward=0.800 (format=0.0)
  With <think> tags: reward=1.000 (format=1.0)
  Delta: +0.200 ✓ think tags rewarded
```

Options:
```bash
python scripts/test_rewards_local.py --n 50                    # Test 50 examples
python scripts/test_rewards_local.py --data data/val.jsonl     # Test on validation set
python scripts/test_rewards_local.py --weights '{"format":0.3,"ngram":0.7,"execution":0.0}'  # Custom weights
```

### How to iterate on rewards

1. Edit `utils/cql_rewards.py` (the reward functions)
2. Run `python scripts/test_rewards_local.py` to see the effect
3. Check: are responses with `<think>` tags scoring higher?
4. Check: is the reward spread wide enough for GRPO to learn? (If everything scores ~0.8, there's no gradient signal)
5. Both the local test and the real `CQLEnvironment` import from the same file — no copy-paste needed

---

## Step 7 — Dry-run configs

Validates YAML structure and prints key settings — no NeMo RL needed:

```bash
# GRPO configs
python scripts/run_grpo_cql.py --dry-run
python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b_full.yaml --dry-run

# SFT configs
python scripts/run_sft_cql.py --dry-run
python scripts/run_sft_cql.py --config configs/sft_cql_full_config.yaml --dry-run
```

Expected output:
```
[DRY RUN] Config OK: configs/cql_nemo_rl_nemotron30b.yaml
  Model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
  Steps: 500
  GPUs: 8
  Mode: LoRA (rank=128)
```

If you see `Missing: <key>`, the YAML is malformed.

---

## Step 8 — Explore rewards interactively (optional)

The reward playground lets you experiment with different inputs and see how rewards are computed:

```bash
python notebooks/reward_playground.py
```

Runs 5 demos (same query scored 5 ways, tokenizer internals, ngram math, weight sweep, GRPO simulation) then drops into a REPL where you can call `score()`, `compare()`, `sweep_weights()`, etc.

---

## What you CAN do locally (no GPU)

| Task | Command | Works on Mac? |
|------|---------|:---:|
| Run unit tests | `python -m pytest utils/ -v` | ✅ |
| Test reward logic | `python scripts/test_rewards_local.py` | ✅ |
| Dry-run all configs | `python scripts/run_grpo_cql.py --dry-run` | ✅ |
| Fetch/inspect data | `python scripts/fetch_data.py` | ✅ |
| Reward playground | `python notebooks/reward_playground.py` | ✅ |
| Edit rewards, re-test | Edit `cql_rewards.py`, run tests | ✅ |
| Edit configs, dry-run | Edit YAML, `--dry-run` | ✅ |

## What you CANNOT do locally

| Task | Why | Where to do it |
|------|-----|---------------|
| Actual training (SFT/GRPO) | Needs NeMo RL + NVIDIA GPUs | SLURM cluster with NGC container |
| Model inference | 30B model needs ~60GB VRAM | Same |
| Full pipeline test | Needs Ray + vLLM + GPUs | Same |

---

## Project files you'll touch most

```
utils/cql_rewards.py              ← YOUR REWARD FUNCTIONS (edit this)
scripts/test_rewards_local.py     ← Test rewards locally (run this)
configs/cql_nemo_rl_nemotron30b.yaml  ← GRPO hyperparameters + reward weights (tune this)
configs/sft_cql_config.yaml       ← SFT hyperparameters (tune this)
data/train.jsonl                  ← Training data (inspect this)
resources/cql_system_prompt.txt   ← System prompt with <think> examples (edit this)
```

---

## Typical local workflow

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Edit your reward logic
vim utils/cql_rewards.py

# 3. Test it locally
python scripts/test_rewards_local.py

# 4. Try different weights
python scripts/test_rewards_local.py --weights '{"format":0.3,"ngram":0.5,"execution":0.2}'

# 5. Run unit tests to make sure nothing broke
python -m pytest utils/ -v

# 6. Tweak config / reward weights
vim configs/cql_nemo_rl_nemotron30b.yaml

# 7. Validate config
python scripts/run_grpo_cql.py --dry-run

# 8. Commit and push
git add -A && git commit -m "Improve reward function" && git push

# 9. SSH to cluster and submit
ssh cluster
cd cql_rlvr && git pull
sbatch scripts/slurm/grpo.sh
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'`**  
→ You forgot to activate the venv or install deps:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**`python: command not found`**  
→ Use `python3` instead of `python`, or activate the venv (which aliases `python` → `python3`)

**Tests fail with `ModuleNotFoundError`**  
→ Make sure you installed requirements:
```bash
pip install -r requirements.txt
python -m pytest utils/ -v
```

**`fetch_data.py` fails cloning**  
→ Check your internet connection and GitHub access. The data is already in `data/` so this step is optional.

**Dry-run says `Missing: env`**  
→ You're dry-running an SFT config with the GRPO script. SFT configs don't have an `env` section. Use the right script:
```bash
python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml --dry-run      # SFT
python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml --dry-run  # GRPO
```
