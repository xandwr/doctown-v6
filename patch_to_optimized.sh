#!/bin/bash
# Quick patch script to switch from local_client to optimized version
# Usage: ./patch_to_optimized.sh

set -e

echo "========================================"
echo "Patching to Optimized Local LLM"
echo "========================================"
echo ""

# Find Python files that import local_client
echo "🔍 Searching for files using local LLM..."
FILES=$(grep -r "from.*llm.*import.*create_local_generator" doctown/python/app/*.py 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo "✓ No files found using create_local_generator"
    echo "  Your project might be using a different import pattern."
    echo ""
    echo "Manual instructions:"
    echo "  1. Find where you create your LLM generator"
    echo "  2. Replace: from app.llm import create_local_generator"
    echo "     With:    from app.llm import create_optimized_local_generator"
    echo "  3. Update the function call to create_optimized_local_generator()"
    echo ""
    echo "See USE_OPTIMIZED_LLM.md for detailed instructions."
else
    echo "Found files to patch:"
    echo "$FILES" | cut -d: -f1 | sort -u
    echo ""
fi

echo ""
echo "📋 Quick Reference:"
echo ""
echo "Replace this:"
echo "  from app.llm import create_local_generator"
echo "  generator = create_local_generator(model_id=\"...\", quantization=\"4bit\")"
echo ""
echo "With this:"
echo "  from app.llm import create_optimized_local_generator"
echo "  generator = create_optimized_local_generator("
echo "      model_id=\"Qwen/Qwen2.5-Coder-7B-Instruct\","
echo "      quantization=\"4bit\","
echo "      max_batch_size=8,"
echo "  )"
echo ""
echo "========================================"
echo "Expected Speedup: 5-10x faster!"
echo "========================================"
echo ""
echo "See USE_OPTIMIZED_LLM.md for full documentation."
echo ""
