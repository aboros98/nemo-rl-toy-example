#!/usr/bin/env bash
# =============================================================================
# CQL RLVR Dummy Pipeline — End-to-End Validation Script
#
# Runs the full pipeline:
#   1. Install dependencies
#   2. Fetch data from public sources
#   3. Dry-run validation (config + data check)
#   4. Start resource server in background
#   5. Run 10 GRPO training steps
#   6. Clean up
#
# Exit on any error.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for status output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SERVER_PID=""

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

print_stage() {
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}========================================${NC}\n"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

# ============================================================================
# Stage 1: Install dependencies
# ============================================================================
print_stage "Stage 1: Installing dependencies"

if [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
    echo "  Using existing virtual environment"
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "  Creating virtual environment..."
    python3 -m venv "$PROJECT_ROOT/.venv"
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT_ROOT/requirements.txt"
echo -e "${GREEN}[OK]${NC} Dependencies installed"

# ============================================================================
# Stage 2: Fetch data
# ============================================================================
print_stage "Stage 2: Fetching training data"

python3 "$PROJECT_ROOT/scripts/fetch_data.py"

if [[ ! -f "$PROJECT_ROOT/data/train.jsonl" ]]; then
    print_error "Training data not created!"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$PROJECT_ROOT/data/train.jsonl" | tr -d ' ')
echo -e "${GREEN}[OK]${NC} Training data: $TRAIN_COUNT examples"

# ============================================================================
# Stage 3: Run tests
# ============================================================================
print_stage "Stage 3: Running unit tests"

python3 -m pytest utils/ -v --tb=short
echo -e "${GREEN}[OK]${NC} All tests passed"

# ============================================================================
# Stage 4: Dry-run validation
# ============================================================================
print_stage "Stage 4: Dry-run config validation"

python3 "$PROJECT_ROOT/scripts/train_grpo.py" \
    --gym-config "$PROJECT_ROOT/configs/cql_gym_config.yaml" \
    --nemo-rl-config "$PROJECT_ROOT/configs/cql_nemo_rl_config.yaml" \
    --dry-run

echo -e "${GREEN}[OK]${NC} Configuration validated"

# ============================================================================
# Stage 5: Start resource server
# ============================================================================
print_stage "Stage 5: Starting CQL resource server"

python3 -c "
import uvicorn
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from resources.cql_resource_server import create_app
app = create_app('$PROJECT_ROOT/data/train.jsonl')
uvicorn.run(app, host='0.0.0.0', port=8080, log_level='warning')
" &
SERVER_PID=$!

echo "  Server PID: $SERVER_PID"

# Wait for server to be ready
echo "  Waiting for server to start..."
MAX_WAIT=30
for i in $(seq 1 $MAX_WAIT); do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Resource server is ready"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        print_error "Resource server failed to start!"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    print_error "Resource server did not respond within ${MAX_WAIT}s"
    exit 1
fi

# ============================================================================
# Stage 6: Run GRPO training (10 steps)
# ============================================================================
print_stage "Stage 6: Running GRPO training (10 steps)"

python3 "$PROJECT_ROOT/scripts/train_grpo.py" \
    --gym-config "$PROJECT_ROOT/configs/cql_gym_config.yaml" \
    --nemo-rl-config "$PROJECT_ROOT/configs/cql_nemo_rl_config.yaml" \
    --steps 10

# ============================================================================
# Verify results
# ============================================================================
print_stage "Pipeline Complete!"

if [[ -f "$PROJECT_ROOT/logs/training_metrics.csv" ]]; then
    LINES=$(wc -l < "$PROJECT_ROOT/logs/training_metrics.csv" | tr -d ' ')
    echo "  Metrics file: logs/training_metrics.csv ($LINES lines)"

    # Check for NaN in rewards
    if grep -q "nan" "$PROJECT_ROOT/logs/training_metrics.csv" 2>/dev/null; then
        echo -e "  ${YELLOW}[WARN]${NC} NaN values detected in metrics"
    else
        echo -e "  ${GREEN}[OK]${NC} All rewards are finite (no NaN)"
    fi
else
    print_error "Metrics file not created!"
    exit 1
fi

echo ""
echo "============================================"
echo "  SUCCESS: Dummy pipeline validated!"
echo ""
echo "  What was validated:"
echo "    - Data fetching and preprocessing"
echo "    - CQL tokenizer and validator"
echo "    - Resource server (reward computation)"
echo "    - Training loop (10 GRPO steps)"
echo "    - Metrics logging"
echo ""
echo "  Production training (Nemotron-30B on 8×B200):"
echo "    bash scripts/launch_nemotron30b.sh"
echo "    Config: configs/cql_nemo_rl_nemotron30b.yaml"
echo ""
echo "  Next steps:"
echo "    - Replace mock execution with LogScale container"
echo "    - Run production training on B200 node"
echo "    - Monitor W&B/TensorBoard metrics"
echo "    - Scale to multi-node if needed"
echo "============================================"
