#!/usr/bin/env bash
set -euo pipefail

# Simple orchestrator for the doctown pipeline.
# Usage: ./orchestrator.sh <command> [args]
# Commands: build_rust | ingest | run_python | embed | full

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCTOWN_DIR="$ROOT_DIR/doctown"
RUST_DIR="$DOCTOWN_DIR/rust"
PY_APP_DIR="$DOCTOWN_DIR/python/app"

log() { printf "%s %s\n" "$(date --iso-8601=seconds)" "$*"; }
err() { log "ERROR:" "$*"; exit 1; }

check_cmd() {
  command -v "$1" >/dev/null 2>&1 || err "Required command '$1' not found in PATH"
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
  check_cmd python3
  if [ -f "$PY_APP_DIR/main.py" ]; then
    (cd "$PY_APP_DIR" && python3 main.py)
  else
    log "No main.py found in $PY_APP_DIR; skipping"
  fi
}

embed_texts() {
  local texts_file=${1:-}
  if [ -n "$texts_file" ] && [ ! -f "$texts_file" ]; then
    err "texts file '$texts_file' not found"
  fi
  check_cmd python3
  log "Running embedder (Python)"
  # Run a small inline snippet that imports embedder.py and calls embed if available.
  python3 - <<PY
import sys
sys.path.insert(0, "$PY_APP_DIR")
try:
    import embedder
    if hasattr(embedder, 'embed'):
        samples = ["hello world", "example text"]
        print('Embedding sample texts:', samples)
        vecs = embedder.embed(samples)
        print('Vectors:', vecs)
    else:
        print('embedder.py loaded but no embed() function found — treat as placeholder')
except Exception as e:
    print('embedder invocation failed (this may be expected if embedder is unimplemented):', e)
PY
}

full_pipeline() {
  local input=${1:-}
  log "Starting full pipeline"
  build_rust
  if [ -n "$input" ]; then
    ingest_repo "$input"
  else
    log "No input repo provided; skipping ingest_repo"
  fi
  # run_python could be a long-running server; keep it optional
  log "(Optional) Running python app as a smoke check"
  run_python || log "python app step returned non-zero (continuing)"
  embed_texts
  log "Full pipeline finished"
}

usage() {
  cat <<USAGE
Usage: $0 <command> [args]

Commands:
  build_rust                 Build the Rust ingest tool (cargo build)
  ingest <input>             Run the Rust ingest on a GitHub URL or zip path
  run_python                 Run the Python app (if main.py exists)
  embed [texts_file]         Run the embedder smoke check (calls embed())
  full [input]               Run the full pipeline: build -> ingest (optional) -> python -> embed
  help                       Show this help

Examples:
  $0 build_rust
  $0 ingest https://github.com/owner/repo
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
  full) full_pipeline "$@" ;;
  help|--help|-h) usage ;;
  *) err "Unknown command: $cmd" ;;
esac
