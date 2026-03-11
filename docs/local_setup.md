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
python -m pytest utils/test_cql_validator.py utils/test_cql_tokenizer.py -v
```

Expected: **45 passed**. These test:
- CQL syntax validator (balanced parens, valid pipes, known functions, unclosed strings)
- CQL tokenizer (semantic tokenization, bigram similarity, structural similarity)

To run a specific test:
```bash
python -m pytest utils/test_cql_validator.py::test_valid_simple_query -v
```

---

## Step 6 — Test reward logic

This is the most important local step — validate your reward function before deploying to the cluster.

```bash
python scripts/test_rewards_local.py
```

This runs the **exact same reward logic** the GPU training will use, on real data from `data/train.jsonl`. No NeMo RL, no Ray, no GPU needed.

Output:
```
Testing reward on 10 examples from data/train.jsonl

  #   Reward  Valid  BiSim  Query (first 60 chars)
-------------------------------------------------------------------
  1   -0.400  False  1.000  EmailEvents | | where Timestamp > ...
  2    1.000   True  1.000  VMComputer | | where PhysicalMemory...
  ...

=== Broken query tests ===
Empty string                -0.500  False  0.000  Empty query
Just text                    0.400   True  0.000  -
Unclosed paren              -0.500  False  0.000  Unclosed '('...

=== Invariant check: invalid must always score < valid ===
  Valid reward:   1.000
  Invalid reward: -0.500
  Invariant holds: ✓ YES
```

Options:
```bash
python scripts/test_rewards_local.py --n 50           # Test 50 examples
python scripts/test_rewards_local.py --data data/val.jsonl  # Test on validation set
```

### How to iterate on rewards

1. Edit `compute_reward()` in `scripts/test_rewards_local.py`
2. Run `python scripts/test_rewards_local.py`
3. Check: are valid queries always scoring higher than invalid ones?
4. Check: is the reward spread wide enough for GRPO to learn? (If everything scores 0.95-1.0, there's no gradient signal)
5. Once happy, copy the logic to `environments/cql_environment.py`

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

## Step 8 — Test the reward server (optional)

The FastAPI reward server is an alternative reward interface (the default for GRPO is the Ray environment, not this server). Useful for debugging:

```bash
# Start the server
python resources/cql_resource_server.py &

# Test it
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write CQL to find failed logins",
    "response": "#event_simpleName=UserLogon | where LogonType=\"10\" | where status!=\"Success\" | stats count() by UserName",
    "metadata": {"golden_query": "#event_simpleName=UserLogon | where status!=\"Success\""}
  }'

# Stop the server
kill %1
```

---

## What you CAN do locally (no GPU)

| Task | Command | Works on Mac? |
|------|---------|:---:|
| Run unit tests | `python -m pytest utils/test_*.py -v` | ✅ |
| Test reward logic | `python scripts/test_rewards_local.py` | ✅ |
| Dry-run all configs | `python scripts/run_grpo_cql.py --dry-run` | ✅ |
| Fetch/inspect data | `python scripts/fetch_data.py` | ✅ |
| Test reward server | `python resources/cql_resource_server.py` | ✅ |
| Edit rewards, re-test | Edit `cql_environment.py`, run tests | ✅ |
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
environments/cql_environment.py   ← YOUR REWARD FUNCTION (edit this)
scripts/test_rewards_local.py     ← Test rewards locally (run this)
configs/cql_nemo_rl_nemotron30b.yaml  ← GRPO hyperparameters (tune this)
configs/sft_cql_config.yaml       ← SFT hyperparameters (tune this)
data/train.jsonl                  ← Training data (inspect this)
```

---

## Typical local workflow

```bash
# 1. Activate venv
source .venv/bin/activate

# 2. Edit your reward
vim environments/cql_environment.py

# 3. Test it
python scripts/test_rewards_local.py

# 4. Run unit tests to make sure nothing broke
python -m pytest utils/test_cql_validator.py utils/test_cql_tokenizer.py -v

# 5. Tweak config
vim configs/cql_nemo_rl_nemotron30b.yaml

# 6. Validate config
python scripts/run_grpo_cql.py --dry-run

# 7. Commit and push
git add -A && git commit -m "Improve reward function" && git push

# 8. SSH to cluster and submit
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

**Tests fail with `No module named 'fastapi'`**  
→ Only `test_reward_invariant.py` needs FastAPI. Run the core tests instead:
```bash
python -m pytest utils/test_cql_validator.py utils/test_cql_tokenizer.py -v
```

**`fetch_data.py` fails cloning**  
→ Check your internet connection and GitHub access. The data is already in `data/` so this step is optional.

**Dry-run says `Missing: env`**  
→ You're dry-running an SFT config with the GRPO script. SFT configs don't have an `env` section. Use the right script:
```bash
python scripts/run_sft_cql.py --config configs/sft_cql_config.yaml --dry-run      # SFT
python scripts/run_grpo_cql.py --config configs/cql_nemo_rl_nemotron30b.yaml --dry-run  # GRPO
```
