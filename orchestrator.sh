#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Doctown Orchestrator
# =============================================================================
# Manages the full Doctown pipeline: Rust ingest + Python embedding + packaging
#
# Usage: ./orchestrator.sh <command> [args]
#
# Commands:
#   build_rust              Build the Rust ingest tool
#   ingest <input>          Run Rust ingest on a GitHub URL or zip path
#   embed                   Test the embedder with sample texts
#   build_docpack <input>   Build a complete .docpack from a repository
#   full [input]            Run the full test pipeline
#   help                    Show this help
#
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTOWN_DIR="$ROOT_DIR/doctown"
RUST_DIR="$DOCTOWN_DIR/rust"
PY_DIR="$DOCTOWN_DIR/python"
PY_APP_DIR="$PY_DIR/app"
PY_VENV="$PY_DIR/venv"
OUTPUT_DIR="$ROOT_DIR/output"

log() { printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"; }
err() { log "ERROR:" "$*"; exit 1; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || err "Required command '$1' not found in PATH"
}

setup_python_env() {
  log "Checking Python environment..."
  if [ ! -d "$PY_VENV" ]; then
    log "Virtual environment not found. Creating it now..."
    check_cmd python3
    (cd "$PY_DIR" && bash setup_venv.sh)
  else
    log "Virtual environment exists at $PY_VENV"
  fi
}

get_python_cmd() {
  if [ -f "$PY_VENV/bin/python" ]; then
    echo "$PY_VENV/bin/python"
  else
    echo "python3"
  fi
}

build_rust() {
  log "Building Rust ingest tool (cargo build)..."
  if [ -f "$RUST_DIR/Cargo.toml" ]; then
    (cd "$RUST_DIR" && cargo build)
    log "Rust build finished"
  else
    err "Could not find $RUST_DIR/Cargo.toml"
  fi
}

ingest_repo() {
  local input=${1:-}
  if [ -z "$input" ]; then
    err "ingest requires an <input> arg (GitHub URL or local zip path)"
  fi
  check_cmd cargo
  log "Running ingest for input: $input"
  (cd "$RUST_DIR" && cargo run --quiet -- "$input" --pretty)
}

run_python() {
  log "Running Python app (attempting to run $PY_APP_DIR/main.py)..."
  local python_cmd=$(get_python_cmd)
  if [ -f "$PY_APP_DIR/main.py" ]; then
    (cd "$PY_APP_DIR" && "$python_cmd" main.py)
  else
    log "No main.py found in $PY_APP_DIR; skipping"
  fi
}

embed_texts() {
  log "Running embedder test (Python)"
  setup_python_env
  local python_cmd=$(get_python_cmd)
  
  # Set environment variables for the Python script
  export EMBEDDING_MODEL_PRESET=fast
  export PY_APP_DIR="$PY_APP_DIR"
  
  # Run embedder test with proper error handling
  "$python_cmd" - <<'PY'
import sys
import os
sys.path.insert(0, os.environ.get('PY_APP_DIR', '.'))

def test_embedder():
    """Test the embedder with sample texts"""
    try:
        from embedder import embed_texts
        
        # Sample texts to embed
        samples = [
            "hello world",
            "testing the embedding pipeline",
            "code documentation search"
        ]
        
        print(f"📝 Testing embedder with {len(samples)} sample texts...")
        print(f"   Model preset: {os.getenv('EMBEDDING_MODEL_PRESET', 'default')}")
        
        # Generate embeddings
        embeddings = embed_texts(samples)
        
        print(f"\n✅ Embeddings generated successfully!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Dtype: {embeddings.dtype}")
        print(f"   Sample vector (first 5 dims): {embeddings[0][:5].tolist()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Embedder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embedder()
    sys.exit(0 if success else 1)
PY
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    log "✓ Embedder test passed"
  else
    log "✗ Embedder test failed (exit code: $exit_code)"
    return $exit_code
  fi
}

build_docpack() {
  local input=${1:-}
  local output_dir=${2:-$OUTPUT_DIR}
  local branch=${3:-main}
  local model=${4:-fast}
  
  if [ -z "$input" ]; then
    err "build_docpack requires an <input> arg (GitHub URL or local zip path)"
  fi
  
  log "Building docpack for: $input"
  log "Output directory: $output_dir"
  
  # Ensure environment is ready
  setup_python_env
  build_rust
  
  local python_cmd=$(get_python_cmd)
  
  # Run the CLI
  (cd "$PY_DIR" && "$python_cmd" -m app.cli build "$input" \
    --output "$output_dir" \
    --branch "$branch" \
    --model "$model")
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    log "✓ Docpack build completed successfully"
    log "Check $output_dir for the .docpack file"
  else
    log "✗ Docpack build failed (exit code: $exit_code)"
    return $exit_code
  fi
}

full_pipeline() {
  local input=${1:-}
  log "Starting full pipeline"
  
  # Setup Python environment first
  setup_python_env
  
  # Build Rust ingest tool
  build_rust
  
  # Run ingest if input provided
  if [ -n "$input" ]; then
    ingest_repo "$input"
  else
    log "No input repo provided; skipping ingest_repo"
  fi
  
  # Run Python app smoke check (optional)
  log "(Optional) Running python app as a smoke check"
  run_python || log "python app step returned non-zero (continuing)"
  
  # Test embedder (critical component)
  log "Testing embedder functionality..."
  if ! embed_texts; then
    log "WARNING: Embedder test failed, but continuing pipeline"
  fi
  
  log "Full pipeline finished"
}

usage() {
  cat <<USAGE
Usage: $0 <command> [args]

Commands:
  build_rust                        Build the Rust ingest tool (cargo build)
  ingest <input>                    Run the Rust ingest on a GitHub URL or zip path
  run_python                        Run the Python app (if main.py exists)
  embed                             Test the embedder with sample texts
  build_docpack <input> [out] [br]  Build a .docpack file from a repository
  full [input]                      Run the full test pipeline
  help                              Show this help

Environment Variables:
  EMBEDDING_MODEL_PRESET     Model preset: fast, balanced, quality, multilingual, code (default: fast)
  EMBEDDING_MODEL            Override with specific HuggingFace model ID
  EMBEDDING_DEVICE           Device: cuda, cpu, or auto (default: auto)

Examples:
  # Build Rust tool only
  $0 build_rust
  
  # Run ingest only (outputs JSON)
  $0 ingest https://github.com/owner/repo
  
  # Test embedder
  $0 embed
  
  # Build a complete .docpack file
  $0 build_docpack https://github.com/owner/repo
  $0 build_docpack https://github.com/owner/repo ./docpacks develop
  
  # Run full test pipeline
  $0 full https://github.com/owner/repo
USAGE
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

cmd=$1; shift || true
case "$cmd" in
  build_rust) build_rust "$@" ;;
  ingest) ingest_repo "$@" ;;
  run_python) run_python "$@" ;;
  embed) embed_texts "$@" ;;
  build_docpack) build_docpack "$@" ;;
  full) full_pipeline "$@" ;;
  help|--help|-h) usage ;;
  *) err "Unknown command: $cmd" ;;
esac
