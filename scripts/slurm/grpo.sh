#!/bin/bash
# GRPO training on 1 node × 8 GPUs via SLURM.
#
# Usage:
#   sbatch scripts/slurm/grpo.sh
#   sbatch scripts/slurm/grpo.sh  # then override via env vars:
#   CONFIG=configs/cql_nemo_rl_config.yaml sbatch scripts/slurm/grpo.sh

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=cql-grpo
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --partition=YOUR_PARTITION

set -eoux pipefail

CONTAINER=${CONTAINER:-"nvcr.io/nvidia/nemo-rl:v0.5.0"}
MOUNTS=${MOUNTS:-"$(pwd):/workspace/cql_rlvr"}
CONFIG=${CONFIG:-"configs/cql_nemo_rl_nemotron30b.yaml"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
OVERRIDES=${OVERRIDES:-""}

export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export PYTHONPATH=/workspace/cql_rlvr:${PYTHONPATH:-}

LOG_DIR="logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"

head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
head_node_ip=$(getent hosts "$head_node" 2>/dev/null | awk '{print $1}' | head -1 || echo "$head_node")

SRUN_ARGS="--no-container-mount-home --container-mounts=$MOUNTS --container-image=$CONTAINER"

# Start Ray head
srun $SRUN_ARGS --container-name=ray-head --nodes=1 --ntasks=1 -w "$head_node" \
    -o "$LOG_DIR/ray-head.log" \
    bash -c "ray start --head --disable-usage-stats \
        --resources='{\"worker_units\": $GPUS_PER_NODE}' \
        --node-ip-address=$head_node_ip --port=54514 \
        --dashboard-port=8265 --include-dashboard=True --block" &

echo "Waiting for Ray..."
sleep 30

# Run GRPO training
srun --overlap --container-name=ray-head --nodes=1 --ntasks=1 -w "$head_node" \
    -o "$LOG_DIR/driver.log" \
    bash -c "uv run python /workspace/cql_rlvr/scripts/run_grpo_cql.py \
        --config /workspace/cql_rlvr/$CONFIG $OVERRIDES"

echo "Done. Logs: $LOG_DIR/"
