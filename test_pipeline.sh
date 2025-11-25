#!/usr/bin/env bash
set -euo pipefail

# Simple root-level test entrypoint that calls orchestrator.sh
# Usage: ./test_pipeline.sh [input]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORCH="$ROOT_DIR/orchestrator.sh"

if [ ! -x "$ORCH" ]; then
  # If not executable, try to run with bash
  log() { printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"; }
  log "Note: making $ORCH executable for convenience"
  chmod +x "$ORCH" || true
fi

INPUT=${1:-}

if [ -n "$INPUT" ]; then
  exec "$ORCH" full "$INPUT"
else
  echo "No input provided. Running full pipeline without ingest step (build + smoke checks)."
  exec "$ORCH" full
fi
