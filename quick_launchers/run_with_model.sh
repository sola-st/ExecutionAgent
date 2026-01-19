#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"

# Model presets
case "${1:-}" in
    "fast")
        MODEL="gpt-4o-mini"
        KNOWLEDGE_MODEL="gpt-4o-mini"
        ;;
    "balanced")
        MODEL="gpt-4o-mini"
        KNOWLEDGE_MODEL="gpt-4o"
        ;;
    "quality")
        MODEL="gpt-4o"
        KNOWLEDGE_MODEL="gpt-4o"
        ;;
    "claude")
        MODEL="claude-sonnet-4-20250514"
        KNOWLEDGE_MODEL="claude-sonnet-4-20250514"
        ;;
    *)
        echo "Usage: $0 <fast|balanced|quality|claude> <metadata-file.json>"
        exit 1
        ;;
esac

METADATA_FILE="${2:-}"
if [ -z "$METADATA_FILE" ]; then
    echo "Usage: $0 <fast|balanced|quality|claude> <metadata-file.json>"
    exit 1
fi

echo "Running with MODEL=$MODEL, KNOWLEDGE_MODEL=$KNOWLEDGE_MODEL"

python -m execution_agent.main \
    --experiment-file "$METADATA_FILE" \
    --model "$MODEL" \
    --knowledge-model "$KNOWLEDGE_MODEL" \
    --workspace-root "./execution_agent_workspace" \
    --max-retries 2