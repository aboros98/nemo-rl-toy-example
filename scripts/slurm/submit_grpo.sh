#!/bin/bash
# ===========================================================================
# submit_grpo.sh — One-command GRPO training submission
#
# Usage:
#   bash scripts/slurm/submit_grpo.sh                    # default production
#   bash scripts/slurm/submit_grpo.sh --steps 50         # override steps
#   bash scripts/slurm/submit_grpo.sh --config configs/cql_nemo_rl_config.yaml  # different config
# ===========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
CONFIG="configs/cql_nemo_rl_nemotron30b.yaml"
EXTRA_OVERRIDES=""
CONTAINER="${CONTAINER:-nvcr.io/nvidia/nemo-rl:v0.5.0}"
GPUS="${GPUS_PER_NODE:-8}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --steps) EXTRA_OVERRIDES="$EXTRA_OVERRIDES ++grpo.max_num_steps=$2"; shift 2 ;;
        --lr) EXTRA_OVERRIDES="$EXTRA_OVERRIDES ++policy.optimizer.kwargs.lr=$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        *) EXTRA_OVERRIDES="$EXTRA_OVERRIDES $1"; shift ;;
    esac
done

# Ensure logs directory exists
mkdir -p logs

echo "╔══════════════════════════════════════════╗"
echo "║  CQL RLVR — GRPO Training Submission    ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Config:    $CONFIG"
echo "║  Container: $CONTAINER"
echo "║  GPUs:      $GPUS"
echo "║  Overrides: $EXTRA_OVERRIDES"
echo "╚══════════════════════════════════════════╝"

# Build the training command
# Paths inside the container (mounted at /workspace/cql_rlvr)
TRAIN_CMD="uv run python /workspace/cql_rlvr/scripts/run_grpo_cql.py"
if [[ "$CONFIG" = /* ]]; then
    TRAIN_CMD+=" --config $CONFIG"
else
    TRAIN_CMD+=" --config /workspace/cql_rlvr/$CONFIG"
fi
TRAIN_CMD+=" $EXTRA_OVERRIDES"

# Submit to SLURM
CONTAINER="$CONTAINER" \
MOUNTS="$PROJECT_ROOT:/workspace/cql_rlvr" \
COMMAND="$TRAIN_CMD" \
GPUS_PER_NODE="$GPUS" \
sbatch "$PROJECT_ROOT/scripts/slurm/ray_1node.sub"

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER                   # Check job status"
echo "  tail -f logs/slurm-*.out          # Follow SLURM output"
echo "  tail -f logs/*/driver.log         # Follow training output"
