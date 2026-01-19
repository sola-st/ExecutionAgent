#!/usr/bin/env bash
set -euo pipefail

export OPENAI_API_KEY="${OPENAI_API_KEY:-your-api-key-here}"

# Directory containing pre-created metadata files
METADATA_DIR="./metadata_files"
WORKSPACE="./execution_agent_workspace"
MODEL="gpt-4o-mini"
LOG_FILE="batch_run_$(date +%Y%m%d_%H%M%S).log"

echo "Starting batch run at $(date)" | tee "$LOG_FILE"

# Run each metadata file
for meta_file in "$METADATA_DIR"/*.json; do
    if [ -f "$meta_file" ]; then
        project_name=$(basename "$meta_file" .json)
        echo ""
        echo "========================================" | tee -a "$LOG_FILE"
        echo "Running: $project_name" | tee -a "$LOG_FILE"
        echo "========================================" | tee -a "$LOG_FILE"

        python -m execution_agent.main \
            --experiment-file "$meta_file" \
            --model "$MODEL" \
            --workspace-root "$WORKSPACE" \
            --max-retries 2 \
            2>&1 | tee -a "$LOG_FILE"

        exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            echo "SUCCESS: $project_name" | tee -a "$LOG_FILE"
        else
            echo "FAILED: $project_name (exit code: $exit_code)" | tee -a "$LOG_FILE"
        fi
    fi
done

echo ""
echo "Batch run completed at $(date)" | tee -a "$LOG_FILE"