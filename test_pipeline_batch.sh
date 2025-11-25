#!/usr/bin/env bash
# Quick wrapper for running the pipeline in optimized BATCH MODE
# This trades a bit more cost per token for MUCH better latency and quality

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default batch mode settings optimized for rate limits with smart throttling
# gpt-4o-mini: 200K TPM (tokens per minute) - default model
# gpt-4o: 30K TPM - specify with --model gpt-4o
# Built-in rate limiter prevents exceeding limits automatically
BATCH_SIZE=15              # Process 15 symbols at once
MAX_BATCH_TOKENS=25000     # Max tokens per batch request
BATCH_WORKERS=3            # Parallel workers (reduced for rate limit safety)
INCLUDE_SEMANTIC=true      # Include semantic context from embeddings

log() { printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"; }

INPUT=${1:-}

if [ -z "$INPUT" ]; then
    log "Usage: ./test_pipeline_batch.sh <github-url-or-path> [additional-args]"
    log ""
    log "This script runs the pipeline in BATCH MODE for optimal performance:"
    log "  - Processes multiple symbols per LLM request (faster, less latency)"
    log "  - Uses gpt-4o with larger context windows"
    log "  - Includes semantic relationships from embeddings"
    log "  - Stays under rate limits with intelligent batching"
    log ""
    log "Examples:"
    log "  ./test_pipeline_batch.sh https://github.com/owner/repo"
    log "  ./test_pipeline_batch.sh https://github.com/owner/repo --batch-size 20"
    log "  ./test_pipeline_batch.sh /path/to/local/repo --model gpt-4o-mini"
    exit 1
fi

shift || true
EXTRA_ARGS=("$@")

# Build the batch mode arguments
BATCH_ARGS=(
    "--llm-batch-mode"
    "--llm-batch-size" "$BATCH_SIZE"
    "--llm-batch-workers" "$BATCH_WORKERS"
    "--max-batch-tokens" "$MAX_BATCH_TOKENS"
)

if [ "$INCLUDE_SEMANTIC" = true ]; then
    BATCH_ARGS+=("--include-semantic-context")
fi

log "================================================"
log "  🚀 Running Pipeline in PARALLEL BATCH MODE"
log "================================================"
log "Input:             $INPUT"
log "Batch Size:        $BATCH_SIZE symbols per request"
log "Batch Workers:     $BATCH_WORKERS parallel workers"
log "Max Batch Tokens:  $MAX_BATCH_TOKENS tokens"
log "Semantic Context:  $INCLUDE_SEMANTIC"
log "Extra Args:        ${EXTRA_ARGS[*]:-none}"
log "================================================"
log ""

# Run the main test script with batch mode flags
exec "$ROOT_DIR/test_pipeline.sh" "$INPUT" "${BATCH_ARGS[@]}" "${EXTRA_ARGS[@]}"
