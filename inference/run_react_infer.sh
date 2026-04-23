#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a
source "$ENV_FILE"
set +a

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY not configured in .env file"
    exit 1
fi

echo "OPENROUTER_MODEL: $OPENROUTER_MODEL"
echo "Starting inference..."

cd "$( dirname -- "${BASH_SOURCE[0]}" )"

# Prepare redundancy check arguments
REDUNDANCY_ARGS=""
if [ "$REDUNDANCY_ENABLED" = "True" ] || [ "$REDUNDANCY_ENABLED" = "true" ]; then
    REDUNDANCY_ARGS="--enable_redundancy_check --redundancy_strategy ${REDUNDANCY_STRATEGY:-rephase} --redundancy_scope ${REDUNDANCY_SCOPE:-single_turn} --redundancy_similarity_threshold ${REDUNDANCY_SIMILARITY_THRESHOLD:-0.8} --redundancy_max_retries ${REDUNDANCY_MAX_RETRIES:-2}"
fi

python -u run_multi_react.py --dataset "$DATASET" --data_file "$DATA_FILE" --output "$OUTPUT_PATH" --max_workers $MAX_WORKERS --temperature $TEMPERATURE --presence_penalty $PRESENCE_PENALTY --total_splits ${WORLD_SIZE:-1} --worker_split $((${RANK:-0} + 1)) --roll_out_count $ROLLOUT_COUNT $REDUNDANCY_ARGS
