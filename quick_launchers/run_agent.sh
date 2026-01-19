#!/usr/bin/env bash
set -euo pipefail

# Configuration
export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"
MODEL="${MODEL:-gpt-4o-mini}"
KNOWLEDGE_MODEL="${KNOWLEDGE_MODEL:-gpt-4o}"
WORKSPACE="./execution_agent_workspace"
MAX_RETRIES=2

# Check for metadata file argument
if [ $# -lt 1 ]; then
    echo "Usage: $0 <metadata-file.json> [additional-args...]"
    exit 1
fi

METADATA_FILE="$1"
shift  # Remove first argument, pass rest to the agent

python -m execution_agent.main \
    --experiment-file "$METADATA_FILE" \
    --model "$MODEL" \
    --knowledge-model "$KNOWLEDGE_MODEL" \
    --workspace-root "$WORKSPACE" \
    --max-retries "$MAX_RETRIES" \
    "$@"