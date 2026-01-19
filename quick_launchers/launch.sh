#!/bin/bash
#
# Execution Agent Launcher - Shell Wrapper
#
# Quick shortcuts for common operations:
#   ./launch.sh list                    # List all projects
#   ./launch.sh list python             # List Python projects
#   ./launch.sh run scipy               # Run single project
#   ./launch.sh run all                 # Run all projects
#   ./launch.sh run python              # Run all Python projects
#   ./launch.sh run scipy,pandas        # Run multiple projects
#
# For full options, use: python launcher.py --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
MODEL="${AGENT_MODEL:-gpt-4o-mini}"
STEP_LIMIT="${AGENT_STEP_LIMIT:-40}"
MAX_RETRIES="${AGENT_MAX_RETRIES:-2}"
PARALLEL="${AGENT_PARALLEL:-1}"

show_help() {
    cat << 'EOF'
Execution Agent Launcher
========================

Usage: ./launch.sh <command> [options] [arguments]

Commands:
  list [language]           List available projects
  run <selection>           Run agent on selected projects
  meta <selection>          Create metadata files only
  help                      Show this help message

Selection options:
  all                       All 50 projects
  python                    All Python projects (14)
  java                      All Java projects (10)
  javascript                All JavaScript projects (12)
  c                         All C projects (10)
  c++                       All C++ projects (8)
  <project-name>            Single project by name
  <name1>,<name2>,...       Multiple projects (comma-separated)

Run options (via environment variables):
  AGENT_MODEL               Model to use (default: gpt-4o-mini)
  AGENT_STEP_LIMIT          Step limit per attempt (default: 40)
  AGENT_MAX_RETRIES         Max retries after budget exhaustion (default: 2)
  AGENT_PARALLEL            Number of parallel runs (default: 1)

Examples:
  ./launch.sh list                              # List all projects
  ./launch.sh list python                       # List Python projects only
  ./launch.sh run scipy                         # Run scipy project
  ./launch.sh run python                        # Run all Python projects
  ./launch.sh run scipy,pandas,numpy            # Run multiple projects
  AGENT_MODEL=gpt-4o ./launch.sh run scipy      # Use specific model
  AGENT_PARALLEL=4 ./launch.sh run all          # Run 4 in parallel

For more options: python launcher.py --help
EOF
}

case "${1:-}" in
    list)
        shift
        if [ -n "${1:-}" ]; then
            python launcher.py --list --language "$1"
        else
            python launcher.py --list
        fi
        ;;

    run)
        shift
        if [ -z "${1:-}" ]; then
            echo "Error: No selection provided"
            echo "Usage: ./launch.sh run <selection>"
            echo "Example: ./launch.sh run scipy"
            exit 1
        fi
        python launcher.py \
            --run "$1" \
            --model "$MODEL" \
            --step-limit "$STEP_LIMIT" \
            --max-retries "$MAX_RETRIES" \
            --parallel "$PARALLEL" \
            "${@:2}"
        ;;

    meta)
        shift
        if [ -z "${1:-}" ]; then
            echo "Error: No selection provided"
            echo "Usage: ./launch.sh meta <selection>"
            exit 1
        fi
        python launcher.py --create-meta "$1"
        ;;

    help|--help|-h)
        show_help
        ;;

    "")
        show_help
        ;;

    *)
        echo "Unknown command: $1"
        echo "Use './launch.sh help' for usage information"
        exit 1
        ;;
esac
