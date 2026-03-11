#!/usr/bin/env bash
# =============================================================================
# Production Launch — Nemotron-30B-A3B GRPO on 1×8 B200
#
# Usage:
#   bash scripts/launch_nemotron30b.sh              # Full training
#   bash scripts/launch_nemotron30b.sh --dry-run    # Validate only
#   bash scripts/launch_nemotron30b.sh --steps 50   # Override steps
#
# Prerequisites:
#   - 8× B200 GPUs (192GB each)
#   - NeMo RL installed (pip install nemo-rl or Docker nvcr.io/nvidia/nemo-rl)
#   - Model weights downloaded: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
#   - Data prepared: data/train.jsonl, data/val.jsonl
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

CONFIG="configs/cql_nemo_rl_nemotron30b.yaml"
GYM_CONFIG="configs/cql_gym_config.yaml"
SERVER_PID=""
DRY_RUN=false
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --steps) EXTRA_ARGS="$EXTRA_ARGS grpo.max_num_steps=$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

cleanup() {
    echo -e "\n${YELLOW}[CLEANUP]${NC} Shutting down..."
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  Stopping resource server (PID: $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    echo -e "${GREEN}[CLEANUP]${NC} Done."
}
trap cleanup EXIT

# ============================================================================
# Pre-flight checks
# ============================================================================
echo -e "${GREEN}[CHECK]${NC} Validating environment..."

# Check GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [[ "$GPU_COUNT" -lt 8 ]]; then
    echo -e "${YELLOW}[WARN]${NC} Found $GPU_COUNT GPUs (config expects 8). Adjust cluster.gpus_per_node."
fi
echo "  GPUs detected: $GPU_COUNT"

# Check GPU memory
if command -v nvidia-smi &>/dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "  GPU memory: ${GPU_MEM}MB per GPU"
    if [[ "$GPU_MEM" -lt 80000 ]]; then
        echo -e "${YELLOW}[WARN]${NC} GPU memory < 80GB. Nemotron-30B may need larger GPUs."
        echo "  Consider: lower gpu_memory_utilization, reduce batch size, or use smaller model."
    fi
fi

# Check config exists
if [[ ! -f "$CONFIG" ]]; then
    echo -e "${RED}[ERROR]${NC} Config not found: $CONFIG"
    exit 1
fi
echo "  Config: $CONFIG"

# Check data
if [[ ! -f "data/train.jsonl" ]]; then
    echo -e "${RED}[ERROR]${NC} Training data not found. Run: python3 scripts/fetch_data.py"
    exit 1
fi
TRAIN_COUNT=$(wc -l < "data/train.jsonl" | tr -d ' ')
echo "  Training examples: $TRAIN_COUNT"

# Check NeMo RL
if python3 -c "import nemo_rl" 2>/dev/null; then
    NEMO_RL_VERSION=$(python3 -c "import nemo_rl; print(nemo_rl.__version__)" 2>/dev/null || echo "unknown")
    echo "  NeMo RL: $NEMO_RL_VERSION"
else
    echo -e "${YELLOW}[WARN]${NC} NeMo RL not installed. Install: pip install nemo-rl"
    if [[ "$DRY_RUN" == false ]]; then
        echo -e "${RED}[ERROR]${NC} Cannot run production training without NeMo RL."
        echo "  For dummy validation, use: bash scripts/run_dummy_pipeline.sh"
        exit 1
    fi
fi

echo -e "${GREEN}[OK]${NC} Environment checks passed"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo -e "${GREEN}[DRY RUN]${NC} Config and environment validated. No training performed."
    echo ""
    echo "  To start training:"
    echo "    bash scripts/launch_nemotron30b.sh"
    echo ""
    echo "  To run with custom steps:"
    echo "    bash scripts/launch_nemotron30b.sh --steps 100"
    exit 0
fi

# ============================================================================
# Start reward server
# ============================================================================
echo ""
echo -e "${GREEN}[LAUNCH]${NC} Starting CQL reward server..."

python3 -c "
import uvicorn, sys
sys.path.insert(0, '$PROJECT_ROOT')
from resources.cql_resource_server import create_app
app = create_app('$PROJECT_ROOT/data/train.jsonl')
uvicorn.run(app, host='0.0.0.0', port=8080, log_level='warning', workers=4)
" &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 30); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Reward server ready (PID: $SERVER_PID, 4 workers)"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo -e "${RED}[ERROR]${NC} Reward server failed to start"
        exit 1
    fi
    sleep 1
done

# ============================================================================
# Launch NeMo RL GRPO training
# ============================================================================
echo ""
echo -e "${GREEN}[LAUNCH]${NC} Starting GRPO training with Nemotron-30B-A3B..."
echo "  Config: $CONFIG"
echo "  Extra args: $EXTRA_ARGS"
echo ""

# NeMo RL launch — uses Ray internally for distributed training
uv run python -m nemo_rl.algorithms.grpo \
    --config "$CONFIG" \
    $EXTRA_ARGS

echo ""
echo -e "${GREEN}[DONE]${NC} Training complete!"
echo "  Checkpoints: results/cql_grpo_nemotron30b/"
echo "  Logs: logs/"
echo "  TensorBoard: tb_logs-cql-nemotron30b/"
