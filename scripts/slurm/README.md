# SLURM Quick Start — CQL RLVR Training

## Prerequisites

1. **NeMo RL container** pulled on your cluster:
   ```bash
   # On the cluster head node or build node:
   docker pull nvcr.io/nvidia/nemo-rl:v0.5.0
   # Or for enroot (common on SLURM clusters):
   enroot import docker://nvcr.io/nvidia/nemo-rl:v0.5.0
   ```

2. **Model weights** downloaded (the container will auto-download, but it's faster to pre-cache):
   ```bash
   huggingface-cli login
   huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
   huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16  # tokenizer
   ```

3. **Data** prepared:
   ```bash
   python scripts/fetch_data.py    # Creates data/train.jsonl, data/val.jsonl, data/test.jsonl
   ```

## Quick Start

### Step 1: SFT Warmup (optional but recommended)

Teaches the model basic CQL syntax before reward-driven training:

```bash
# Submit SFT job — one command!
bash scripts/slurm/submit_sft.sh
```

### Step 2: GRPO Training

The main reinforcement learning step:

```bash
# Submit GRPO job — one command!
bash scripts/slurm/submit_grpo.sh
```

### Common Overrides

```bash
# Change number of training steps:
bash scripts/slurm/submit_grpo.sh --steps 100

# Change learning rate:
bash scripts/slurm/submit_grpo.sh --lr 1e-5

# Use different config (e.g., small model for testing):
bash scripts/slurm/submit_grpo.sh --config configs/cql_nemo_rl_config.yaml

# Multiple overrides (Hydra-style):
bash scripts/slurm/submit_grpo.sh ++grpo.num_generations_per_prompt=16 ++loss_fn.reference_policy_kl_penalty=0.001

# Use different container:
CONTAINER=nvcr.io/nvidia/nemo-rl:v0.5.0 bash scripts/slurm/submit_grpo.sh
```

## Monitoring Your Job

```bash
# Check job status
squeue -u $USER

# See all your jobs
squeue -u $USER -o "%.8i %.9P %.40j %.8T %.10M %.6D %R"

# Follow the SLURM output log
tail -f logs/slurm-<JOB_ID>.out

# Follow the actual training output (once job starts)
tail -f logs/<JOB_ID>/driver.log

# Cancel a job
scancel <JOB_ID>
```

## What the Scripts Do

```
submit_grpo.sh / submit_sft.sh
    │
    ├── Sets CONTAINER, MOUNTS, COMMAND environment variables
    │
    └── sbatch ray_1node.sub
            │
            ├── SLURM allocates 1 node with 8 GPUs
            ├── Starts Ray head node inside the NeMo RL container
            ├── Waits for all GPUs to be online
            └── Runs your training script (run_grpo_cql.py or run_sft_cql.py)
```

## Files

| File | Purpose |
|------|---------|
| `ray_1node.sub` | SLURM job script — launches Ray + container (1 node) |
| `submit_grpo.sh` | One-command wrapper for GRPO training |
| `submit_sft.sh` | One-command wrapper for SFT training |

## Cluster Configuration

Edit the `#SBATCH` directives in `ray_1node.sub` for your cluster:

```bash
#SBATCH --account=YOUR_ACCOUNT      # Your SLURM account
#SBATCH --partition=YOUR_PARTITION    # GPU partition name
#SBATCH --time=4:00:00              # Wall time limit
#SBATCH --gpus-per-node=8           # GPUs per node
```

## Troubleshooting

**Job stays in PENDING state:**
- Check `squeue` — might be waiting for resources
- Run `sinfo` to see available partitions and nodes

**Container not found:**
- Make sure the container is pulled: `enroot list` or check your container registry

**Out of memory:**
- Reduce batch size: `++policy.train_micro_batch_size=1`
- Increase `gpu_memory_utilization`: `++policy.generation.vllm_cfg.gpu_memory_utilization=0.8`

**Ray fails to start:**
- Check `logs/<JOB_ID>/ray-head.log` for errors
- Ensure ports are not in use (rare on fresh SLURM allocation)

## NeMo RL Installation Options

NeMo RL is distributed primarily via **NGC Docker containers**:

```bash
# Option 1: NGC container (recommended)
docker pull nvcr.io/nvidia/nemo-rl:v0.5.0

# Option 2: Source install (for development)
git clone https://github.com/NVIDIA-NeMo/RL.git
cd RL
uv sync    # Installs all dependencies
```

Inside the container, scripts are run with `uv run python ...` because NeMo RL uses
`uv` as its package manager.
