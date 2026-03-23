#!/usr/bin/env bash
#
# Run end-to-end tests for the Skincare Advisory Agent.
#
# Usage:
#   ./tests/run_e2e.sh              # run all e2e tests
#   ./tests/run_e2e.sh -k "Chat"    # run only chat tests
#
# Requires:
#   - Ollama running locally with a model (default: llama3.2:latest)
#   - test-venv with dependencies installed
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="$PROJECT_DIR/test-venv/bin/python"

cd "$PROJECT_DIR"

echo "=== Skincare Agent E2E Tests ==="
echo "Project:  $PROJECT_DIR"
echo "Python:   $PYTHON"
echo ""

# Verify Ollama is reachable
if ! curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running at http://localhost:11434"
    echo "Start Ollama first: ollama serve"
    exit 1
fi
echo "Ollama: OK"

# Verify the model is available
MODEL="${SKINCARE_LLM_MODEL:-llama3.2:latest}"
if ! curl -sf http://localhost:11434/api/tags | python3 -c "
import sys, json
tags = json.load(sys.stdin)
names = [m['name'] for m in tags.get('models', [])]
if '$MODEL' not in names:
    print(f'ERROR: Model $MODEL not found. Available: {names}')
    sys.exit(1)
" 2>/dev/null; then
    echo "WARNING: Could not verify model '$MODEL' is pulled."
fi
echo "Model:  $MODEL"
echo ""

# Run pytest
exec "$PYTHON" -m pytest tests/test_e2e.py -v -s --tb=short "$@"
