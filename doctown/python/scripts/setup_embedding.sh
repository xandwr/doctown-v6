#!/bin/bash
# Quick setup script for embedding models
# Makes it stupid easy to get started

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")"

echo "🎯 Doctown Embedding Model Setup"
echo "=================================="
echo ""

# Check if Python exists
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install requirements if needed
if [ ! -f "$PYTHON_DIR/requirements.txt" ]; then
    echo "❌ requirements.txt not found"
    exit 1
fi

echo "📦 Checking Python dependencies..."
python3 -m pip install -q -r "$PYTHON_DIR/requirements.txt"
echo "✅ Dependencies installed"
echo ""

# Show available commands
if [ $# -eq 0 ]; then
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  list              List all available model presets"
    echo "  download [model]  Download a specific model (preset or ID)"
    echo "  test              Run the test script"
    echo "  info              Show current configuration"
    echo ""
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 download fast"
    echo "  $0 download BAAI/bge-large-en-v1.5"
    echo "  $0 test"
    echo ""
    echo "Environment Variables:"
    echo "  EMBEDDING_MODEL         Model ID or preset"
    echo "  EMBEDDING_MODEL_PRESET  Preset name (fast, balanced, quality, etc.)"
    echo "  EMBEDDING_CACHE_DIR     Cache directory"
    echo "  EMBEDDING_DEVICE        Device (cuda, cpu, auto)"
    echo ""
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    list)
        echo "📋 Available Models:"
        echo ""
        python3 "$SCRIPT_DIR/download_model.py" --list
        ;;
    
    download)
        if [ $# -eq 0 ]; then
            echo "❌ Please specify a model to download"
            echo "Usage: $0 download [preset|model-id]"
            echo ""
            echo "Examples:"
            echo "  $0 download fast"
            echo "  $0 download BAAI/bge-large-en-v1.5"
            exit 1
        fi
        
        MODEL=$1
        shift
        
        # Check if it's a preset or full model ID
        if [[ $MODEL =~ / ]]; then
            python3 "$SCRIPT_DIR/download_model.py" --model "$MODEL" "$@"
        else
            python3 "$SCRIPT_DIR/download_model.py" --preset "$MODEL" "$@"
        fi
        ;;
    
    test)
        echo "🧪 Running embedding tests..."
        echo ""
        python3 "$PYTHON_DIR/examples/test_embedder.py" "$@"
        ;;
    
    info)
        echo "📊 Current Configuration:"
        echo ""
        echo "Environment Variables:"
        echo "  EMBEDDING_MODEL:        ${EMBEDDING_MODEL:-<not set>}"
        echo "  EMBEDDING_MODEL_PRESET: ${EMBEDDING_MODEL_PRESET:-<not set>}"
        echo "  EMBEDDING_CACHE_DIR:    ${EMBEDDING_CACHE_DIR:-./models/embeddings (default)}"
        echo "  EMBEDDING_DEVICE:       ${EMBEDDING_DEVICE:-auto (default)}"
        echo "  HF_TOKEN:               ${HF_TOKEN:+<set>}${HF_TOKEN:-<not set>}"
        echo ""
        
        if [ -d "${EMBEDDING_CACHE_DIR:-./models/embeddings}" ]; then
            echo "Cached Models:"
            find "${EMBEDDING_CACHE_DIR:-./models/embeddings}" -maxdepth 2 -type d -name "models--*" 2>/dev/null | \
                sed 's/.*models--/  /' | sed 's/__/\//g' || echo "  (none)"
        fi
        echo ""
        ;;
    
    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Run '$0' without arguments to see usage"
        exit 1
        ;;
esac
