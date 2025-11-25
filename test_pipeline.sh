#!/usr/bin/env bash
set -euo pipefail

# Capture start time for runtime analysis
START_TIME=$(date +%s)
START_TIMESTAMP=$(date --iso-8601=seconds)

# =============================================================================
# Doctown Full Pipeline Test
# =============================================================================
# Tests the COMPLETE upgraded pipeline including:
#   1. Domain detection & ingestion (pluggable ingestors)
#   2. Universal chunk extraction (standardized IR)
#   3. Embedding generation
#   4. Semantic graph building
#   5. LLM documentation generation (OpenAI integration)
#   6. Docpack packaging
#
# Usage: ./test_pipeline.sh [input] [options]
#
# Examples:
#   ./test_pipeline.sh https://github.com/owner/repo
#   ./test_pipeline.sh https://github.com/owner/repo --no-llm
#   ./test_pipeline.sh https://github.com/owner/repo --llm-model gpt-4
#
# BATCH MODE (faster, better quality, recommended):
#   ./test_pipeline.sh https://github.com/owner/repo --llm-batch-mode --llm-batch-size 15
#   ./test_pipeline.sh https://github.com/owner/repo --llm-batch-mode --llm-batch-size 20 --max-batch-tokens 25000
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_DIR="$ROOT_DIR/doctown/python"
PY_VENV="$PY_DIR/venv"
OUTPUT_DIR="$ROOT_DIR/output"
REPORTS_DIR="$ROOT_DIR/reports"

log() { printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"; }
err() { log "ERROR:" "$*"; exit 1; }

display_runtime() {
  local end_time=$(date +%s)
  local end_timestamp=$(date --iso-8601=seconds)
  local duration=$((end_time - START_TIME))
  local hours=$((duration / 3600))
  local minutes=$(((duration % 3600) / 60))
  local seconds=$((duration % 60))
  
  log ""
  log "================================================"
  log "  ⏱️  RUNTIME ANALYSIS"
  log "================================================"
  log "Start:    $START_TIMESTAMP"
  log "End:      $end_timestamp"
  log "Duration: ${hours}h ${minutes}m ${seconds}s (${duration} seconds total)"
  log "================================================"
  log ""
}

# Parse arguments
INPUT=${1:-}
shift || true
EXTRA_ARGS=("$@")

# Check if we need LLM docs
USE_LLM=true
for arg in "${EXTRA_ARGS[@]}"; do
  if [ "$arg" = "--no-llm" ]; then
    USE_LLM=false
    break
  fi
done

load_env_file() {
  local env_file="$PY_DIR/.env"
  if [ -f "$env_file" ]; then
    log "Loading environment from $env_file"
    # Export variables from .env file
    set -a
    source "$env_file"
    set +a
  fi
}

setup_python_env() {
  log "Checking Python environment..."
  if [ ! -d "$PY_VENV" ]; then
    log "Virtual environment not found. Creating it now..."
    (cd "$PY_DIR" && bash setup_venv.sh)
  else
    log "Virtual environment exists at $PY_VENV"
  fi
  
  # Load .env file for bash environment checks
  load_env_file
}

get_python_cmd() {
  if [ -f "$PY_VENV/bin/python" ]; then
    echo "$PY_VENV/bin/python"
  else
    echo "python3"
  fi
}

test_full_pipeline() {
  local input=${1:-}
  shift || true
  local extra_args=("$@")
  
  log "================================================"
  log "  Doctown Full Pipeline Test"
  log "================================================"
  log "Input: ${input:-<none provided>}"
  log "LLM Docs: $USE_LLM"
  log "Output: $OUTPUT_DIR"
  log "================================================"
  
  # Setup environment
  setup_python_env
  
  local python_cmd=$(get_python_cmd)
  
  # Check for OPENAI_API_KEY if LLM is enabled
  if [ "$USE_LLM" = true ]; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
      log "WARNING: OPENAI_API_KEY not set!"
      log "LLM documentation will not be generated."
      log "Set OPENAI_API_KEY in your environment or .env file."
      log ""
      log "Continuing with rule-based documentation only..."
      extra_args+=("--no-llm")
    else
      log "✓ OPENAI_API_KEY detected - LLM docs will be generated"
    fi
  fi
  
  # Run the pipeline
  if [ -z "$input" ]; then
    log "No input provided. Please specify a GitHub URL or local path."
    log "Example: ./test_pipeline.sh https://github.com/owner/repo"
    return 1
  fi
  
  log ""
  log "Step 1/7: Domain detection & ingestion"
  log "Step 2/7: Universal chunk extraction"
  log "Step 3/7: Embedding generation"
  log "Step 4/7: Semantic graph building"
  log "Step 5/7: LLM documentation generation"
  log "Step 6/7: Docpack packaging"
  log "Step 7/7: Verification"
  log ""
  
  # Run the new pipeline with LLM support
  log "Running: python -m app.cli build-llm \"$input\" ${extra_args[*]:-}"
  (cd "$PY_DIR" && "$python_cmd" -m app.cli build-llm "$input" \
    --output "$OUTPUT_DIR" \
    "${extra_args[@]}")
  
  local exit_code=$?
  
  if [ $exit_code -eq 0 ]; then
    log ""
    log "================================================"
    log "  ✅ Pipeline Test PASSED"
    log "================================================"
    log "Output directory: $OUTPUT_DIR"
    log ""
    
    # Show what was created
    if [ -d "$OUTPUT_DIR" ]; then
      log "Generated files:"
      ls -lh "$OUTPUT_DIR"/*.docpack 2>/dev/null || log "  (no .docpack files found)"
    fi
    
    # Calculate and display runtime
    display_runtime
    
    # Generate test report
    generate_test_report "$exit_code" "$input" "${extra_args[@]}"
    
    return 0
  else
    log ""
    log "================================================"
    log "  ❌ Pipeline Test FAILED"
    log "================================================"
    log "Exit code: $exit_code"
    log ""
    
    # Calculate and display runtime
    display_runtime
    
    # Generate test report
    generate_test_report "$exit_code" "$input" "${extra_args[@]}"
    
    return $exit_code
  fi
}

generate_test_report() {
  local exit_code=$1
  local input=$2
  shift 2
  local extra_args=("$@")
  
  mkdir -p "$REPORTS_DIR"
  local report_file="$REPORTS_DIR/$(date +%Y-%m-%d)-PIPELINE_TEST_RESULTS.md"
  
  log "Generating test report: $report_file"
  
  cat > "$report_file" <<EOF
# Doctown Pipeline Test Report

**Date**: $(date --iso-8601=seconds)
**Test**: Full Pipeline (build-llm)
**Status**: $([ $exit_code -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED (exit code: $exit_code)")

## Test Configuration

- **Input**: $input
- **Extra Args**: ${extra_args[*]:-<none>}
- **LLM Enabled**: $USE_LLM
- **Output Directory**: $OUTPUT_DIR

## Pipeline Stages

The following stages were executed:

1. ✓ Domain detection & ingestion (pluggable ingestors)
2. ✓ Universal chunk extraction (standardized IR)
3. ✓ Embedding generation
4. ✓ Semantic graph building
5. $([ "$USE_LLM" = true ] && echo "✓" || echo "○") LLM documentation generation
6. ✓ Docpack packaging
7. ✓ Verification

## Output Files

EOF

  if [ -d "$OUTPUT_DIR" ]; then
    echo "\`\`\`" >> "$report_file"
    ls -lh "$OUTPUT_DIR" 2>/dev/null || echo "No files found" >> "$report_file"
    echo "\`\`\`" >> "$report_file"
  else
    echo "Output directory not found." >> "$report_file"
  fi
  
  cat >> "$report_file" <<EOF

## Environment

- **Python**: $(get_python_cmd) --version 2>&1 || echo "Not available"
- **OPENAI_API_KEY**: $([ -n "${OPENAI_API_KEY:-}" ] && echo "Set" || echo "Not set")
- **Working Directory**: $ROOT_DIR

## Notes

$(if [ $exit_code -eq 0 ]; then
  echo "All pipeline stages completed successfully. The upgraded pipeline with LLM documentation generation is working correctly."
else
  echo "Pipeline test failed. Check the logs above for details."
fi)

---
*Report generated by test_pipeline.sh*
EOF

  log "✓ Test report saved to: $report_file"
}

# Show help if requested
if [ "$INPUT" = "--help" ] || [ "$INPUT" = "-h" ]; then
  cat <<USAGE
Usage: $0 [input] [options]

Test the complete Doctown pipeline including LLM documentation generation.

Arguments:
  input                 GitHub URL or local path to process

Options:
  --no-llm             Disable LLM documentation (use rule-based)
  --llm-model MODEL    Specify OpenAI model (default: from env)
  --llm-concurrent N   Max concurrent LLM requests (default: 10)
  --llm-max-chunks N   Limit chunks for LLM (cost control)
  --model MODEL        Embedding model preset (default: fast)
  --branch BRANCH      Git branch (default: main)
  -h, --help           Show this help

Environment Variables:
  OPENAI_API_KEY       Required for LLM documentation generation
  OPENAI_MODEL         OpenAI model to use (default: gpt-5-nano)

Examples:
  # Test with LLM documentation
  $0 https://github.com/owner/repo
  
  # Test without LLM (rule-based only)
  $0 https://github.com/owner/repo --no-llm
  
  # Test with specific LLM model
  $0 https://github.com/owner/repo --llm-model gpt-4
  
  # Test with cost controls
  $0 https://github.com/owner/repo --llm-max-chunks 100

For more details, see: doctown/python/README.md
USAGE
  exit 0
fi

# Run the test
test_full_pipeline "$INPUT" "${EXTRA_ARGS[@]}"
