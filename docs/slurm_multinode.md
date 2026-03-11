# Running CQL RLVR on SLURM — Beginner-Friendly Guide

You have RunPod / cloud GPUs now. Eventually you'll want a SLURM cluster (Lambda, CoreWeave,
university HPC, or your own). This doc assumes **zero SLURM experience**.

---

## What is SLURM? (30-second version)

SLURM is a job scheduler for GPU clusters. Think of it like a queue:

1. You write a **script** saying "I need 2 nodes with 8 GPUs each for 4 hours"
2. You **submit** it: `sbatch my_script.sh`
3. SLURM **waits** until those resources are free, then runs your script
4. Your script runs, writes logs, and exits

That's it. The key commands:

| Command | What It Does | Example |
|---------|-------------|---------|
| `sbatch script.sh` | Submit a job to the queue | `sbatch train.sh` |
| `squeue -u $USER` | See your jobs in the queue | Shows PENDING/RUNNING |
| `scancel JOB_ID` | Cancel a job | `scancel 12345` |
| `srun` | Run a command on allocated nodes | `srun nvidia-smi` |
| `scontrol show job JOB_ID` | Job details | Node names, time left |

The `#SBATCH` lines at the top of a script are **not comments** — they're directives
that SLURM reads to know what resources you need.

---

## What You Need

- A SLURM cluster with GPUs (most cloud GPU providers offer this)
- A **shared filesystem** — all nodes must see the same files. This is almost always
  pre-configured on clusters (NFS, Lustre, etc.)
- SSH access to a **login node** (where you submit jobs, NOT where they run)

---

## Step-by-Step: Your First SLURM Job

### 1. Upload your code to shared storage

```bash
# On the login node
cd /shared/  # or wherever the shared filesystem is mounted
git clone <your-repo> cql_rlvr
cd cql_rlvr
```

### 2. Prepare data (one-time, no GPU needed)

```bash
python scripts/fetch_data.py  # Creates data/train.jsonl, data/val.jsonl, data/test.jsonl
```

### 3. Submit training — ONE COMMAND

We provide ready-made SLURM scripts. The architecture is:

```
submit_grpo.sh  →  sets CONTAINER, MOUNTS, COMMAND  →  sbatch ray_1node.sub
                                                            │
                                                            ├── SLURM allocates 1 node × 8 GPUs
                                                            ├── Starts Ray head inside NeMo RL container
                                                            ├── Waits for all GPUs online
                                                            └── Runs run_grpo_cql.py
```

**SFT warmup (optional but recommended):**
```bash
bash scripts/slurm/submit_sft.sh
```

**GRPO training:**
```bash
bash scripts/slurm/submit_grpo.sh
```

**With overrides:**
```bash
bash scripts/slurm/submit_grpo.sh --steps 100 --lr 1e-5
bash scripts/slurm/submit_grpo.sh ++grpo.num_generations_per_prompt=16
```

### 4. Monitor

```bash
# Watch the queue
watch -n5 squeue -u $USER

# Tail training logs (once RUNNING)
tail -f logs/<JOB_ID>/driver.log

# Cancel a job
scancel <JOB_ID>
```

---

## NeMo RL Installation

NeMo RL is distributed primarily via **NGC Docker containers**:

```bash
# Option 1: NGC container (recommended for production)
docker pull nvcr.io/nvidia/nemo-rl:v0.4.0.nemotron_3_nano

# Option 2: Source install (for development / customization)
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
uv sync    # Installs all dependencies via uv package manager
```

Inside the container, scripts are run with `uv run python ...`.
Ray is included for distributed orchestration.

---

## How NeMo RL Actually Launches on SLURM

**Important:** NeMo RL does NOT use `torchrun` like TRL/HuggingFace. It uses **Ray**.

The flow is:
1. SLURM allocates N nodes
2. Your script starts a **Ray cluster** across those nodes
3. NeMo RL creates **Ray actors** for: policy training, vLLM generation, environments
4. Ray handles all communication between actors

This is why you see `--ntasks-per-node=1` — you start ONE process per node, and Ray
spawns workers internally. No `torchrun`, no manual `MASTER_ADDR`/`MASTER_PORT`.

```bash
# The real NeMo RL launch command (from their examples):
uv run examples/run_grpo_math.py --config examples/configs/grpo_math_1B.yaml

# On SLURM, NeMo RL auto-detects the Ray cluster. You just need to:
# 1. Start Ray head on first node
# 2. Start Ray workers on other nodes
# 3. Run the training script on the head node
```

NeMo RL has a built-in SLURM launcher. See the [cluster docs](https://docs.nvidia.com/nemo/rl/nightly/cluster.html).

---

## Config Changes for Multi-Node

The only things that change between single-GPU and multi-node:

```yaml
# configs/cql_nemo_rl_config.yaml

cluster:
  gpus_per_node: 8         # GPUs per machine (RunPod: 1, A100 node: 8)
  num_nodes: 2             # Number of machines

policy:
  # Scale batch size with total GPUs
  train_global_batch_size: 128   # Was 16 on single GPU
  train_micro_batch_size: 4      # Per-GPU, stays the same

  # For big models (>30B), split across GPUs:
  dtensor_cfg:
    tensor_parallel_size: 2      # Split model across 2 GPUs (if needed)

grpo:
  num_prompts_per_step: 32       # More prompts (more GPUs = bigger batches)
  num_generations_per_prompt: 8
```

**Rule:** Double your `num_nodes` → double your `num_prompts_per_step`.
Everything else stays the same.

---

## Scaling Guidelines

| Setup | GPUs | `num_prompts_per_step` | `num_generations` | `global_batch` |
|-------|------|------------------------|-------------------|----------------|
| RunPod 1×A100 | 1 | 4 | 4 | 16 |
| 1 node × 8 GPUs | 8 | 16 | 8 | 128 |
| 2 nodes × 8 GPUs | 16 | 32 | 8 | 256 |
| 8 nodes × 8 GPUs | 64 | 64 | 16 | 1024 |

---

## Reward Server Scaling

For large clusters (>16 GPUs), the reward server may bottleneck. Options:

1. **Multiple Uvicorn workers** (easiest):
   ```bash
   uvicorn resources.cql_resource_server:create_app \
       --factory --host 0.0.0.0 --port 8080 --workers 8
   ```

2. **NeMo Gym native scaling** — when using NeMo Gym SDK, it replicates the environment
   automatically via the `num_workers` config.

---

## Common SLURM Mistakes (and fixes)

| Mistake | Fix |
|---------|-----|
| Job stuck in PENDING forever | Your resource request is too big. Try fewer nodes/GPUs, or shorter `--time` |
| `srun: error: Unable to create step` | Job hasn't started yet. Wait for RUNNING status |
| Scripts not found | Wrong `cd` path. Use absolute paths: `/shared/cql_rlvr/` |
| Permission denied on logs/ | `mkdir -p logs/` before submitting |
| NCCL timeout between nodes | Check: `ibstat` for InfiniBand, or set `NCCL_IB_DISABLE=1` for TCP fallback |
| OOM during generation | Lower `generation_batch_size`; enable `activation_checkpointing: true` |
| Reward server unreachable | Check the hostname file exists; verify port 8080 is open between nodes |
| Job killed by time limit | Increase `--time` or add checkpointing so you can resume |

---

## SLURM Cheat Sheet

```bash
# Submit a job
sbatch my_script.sh

# Submit with overrides (no editing the script)
sbatch --nodes=4 --time=12:00:00 my_script.sh

# See your jobs
squeue -u $USER

# See ALL jobs on the cluster
squeue

# Cancel a job
scancel 12345

# Cancel ALL your jobs
scancel -u $USER

# Check job details
scontrol show job 12345

# See available partitions (queues) and their limits
sinfo

# Interactive shell on a GPU node (for debugging)
srun --nodes=1 --gres=gpu:1 --time=01:00:00 --pty bash

# Check how much quota/priority you have
sacctmgr show user $USER
```

---

## RunPod → SLURM Migration Path

| RunPod | SLURM Equivalent |
|--------|-----------------|
| SSH into pod, run `python train.py` | `sbatch train.sh` (non-interactive) |
| Multiple terminals | `#SBATCH --output` captures stdout |
| Kill the process | `scancel JOB_ID` |
| Check GPU usage with `nvidia-smi` | `srun nvidia-smi` on allocated nodes |
| Upload files | Put files on shared filesystem |
| Environment variables | `export` in the `#SBATCH` script |
| Container (Docker) | Use `#SBATCH --container-image=nvcr.io/...` with Enroot/Pyxis |
