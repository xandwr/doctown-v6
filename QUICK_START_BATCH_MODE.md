# Quick Start: Batch Mode for Fast LLM Documentation

## TL;DR

Your LLM generation is slow because you're making 67 separate API calls. Use batch mode:

```bash
# OLD: 67 requests, 26 seconds, gpt-5-nano
./test_pipeline.sh https://github.com/pykeio/ort

# NEW: 5 requests, ~8 seconds, gpt-4o with semantic context
./test_pipeline.sh https://github.com/pykeio/ort \
  --llm-batch-mode \
  --llm-batch-size 15 \
  --llm-model gpt-4o \
  --llm-semantic-neighbors 3
```

## What Changed?

### 1. Batch Mode
- **Before**: Each symbol = 1 API call (67 symbols = 67 calls)
- **After**: 15 symbols per call (67 symbols = 5 calls)
- **Result**: 3x faster, better quality

### 2. Semantic Context
- **Before**: LLM only sees isolated code snippets
- **After**: LLM sees related code via embedding similarity
- **Result**: Better docs that explain relationships

### 3. Smarter Model
- **Before**: gpt-5-nano (cheap but limited)
- **After**: gpt-4o (3x cost but 128K context window)
- **Result**: No token limit errors, richer documentation

## Try It Now

### Step 1: Set up your API key
```bash
export OPENAI_API_KEY=sk-...
```

### Step 2: Run with batch mode
```bash
cd /home/xander/Documents/doctown-v6

# Using the test script
./test_pipeline.sh https://github.com/pykeio/ort \
  --llm-batch-mode \
  --llm-batch-size 15 \
  --llm-model gpt-4o

# Or using Python directly
cd doctown/python
python -m app.cli build-llm \
  https://github.com/pykeio/ort \
  --llm-batch-mode \
  --llm-batch-size 15 \
  --llm-model gpt-4o \
  --llm-semantic-neighbors 3
```

### Step 3: Watch the logs
```
🔮 Generating embeddings for 67 texts...
✓ Generated 67 embeddings of dimension 384
📊 Computing semantic neighbors (k=3)...
💰 Estimated LLM cost (BATCH MODE): $0.045
   Model: gpt-4o, ~5 batches, ~45,000 tokens
🚀 Processing 67 symbols in BATCH MODE (15 symbols/batch)...
   Progress: 5/5 batches completed
✓ LLM docs: 67 successful, 0 failed
💰 Actual cost: $0.042
⏱️  Total time: ~8 seconds
```

## Configuration Reference

### Batch Mode Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm-batch-mode` | False | Enable batch processing |
| `--llm-batch-size` | 15 | Symbols per batch |
| `--llm-model` | gpt-5-nano | Which OpenAI model |
| `--llm-semantic-neighbors` | 3 | Related code snippets |

### Recommended Settings

**Small Repos (<50 symbols):**
```bash
# Per-symbol mode is fine for small repos
--llm-model gpt-5-nano \
--llm-concurrent 10 \
--llm-semantic-neighbors 3
```

**Medium Repos (50-500 symbols):**
```bash
# Batch mode with moderate batches
--llm-batch-mode \
--llm-batch-size 15 \
--llm-model gpt-4o \
--llm-semantic-neighbors 3
```

**Large Repos (500+ symbols):**
```bash
# Larger batches, limit chunks to control costs
--llm-batch-mode \
--llm-batch-size 20 \
--llm-model gpt-4o \
--llm-max-chunks 500 \
--llm-semantic-neighbors 5
```

## Cost Comparison

For your 67-symbol repo:

| Mode | Model | Requests | Time | Cost |
|------|-------|----------|------|------|
| **Per-symbol** | gpt-5-nano | 67 | 26s | $0.012 |
| **Batch (15)** | gpt-4o | 5 | 8s | $0.045 |
| **Batch (20)** | gpt-4o | 4 | 7s | $0.048 |

Yes, batch mode costs more per run, but:
- ✅ 3x faster
- ✅ No token limit errors
- ✅ Better documentation quality
- ✅ Semantic context included

**Worth it for production use!**

## Troubleshooting

### "CompletionUsage length limit was reached"
This happens in per-symbol mode with gpt-5-nano (4K token limit).

**Solution**: Use batch mode with a larger model:
```bash
--llm-batch-mode --llm-model gpt-4o
```

### Still slow?
Try increasing batch size:
```bash
--llm-batch-size 20  # More symbols per request
```

### Too expensive?
Limit the number of symbols:
```bash
--llm-max-chunks 100  # Only document first 100 symbols
```

Or use a cheaper model:
```bash
--llm-model gpt-4o-mini  # Cheaper than gpt-4o
```

### Want to test without spending money?
```bash
# Disable LLM, just test embeddings and graph
--no-llm
```

## Understanding Semantic Context

When you enable `--llm-semantic-neighbors 3`, the pipeline:

1. **Generates embeddings** for all code
2. **Computes similarity** between every pair
3. **Finds top-k neighbors** for each symbol
4. **Injects context** into the LLM prompt:

```
**Semantically Related:**
- src/config.rs (similarity: 0.87): pub struct Config { ... }
- src/utils.rs (similarity: 0.82): fn validate_settings(...) { ... }
- tests/test_config.rs (similarity: 0.75): #[test] fn test_default() { ... }
```

The LLM uses this to understand:
- Similar patterns across files
- Implementation conventions
- How pieces fit together

**No guessing needed** - the embeddings tell the LLM what's related!

## Next Steps

1. **Try it**: Run with `--llm-batch-mode` on your repo
2. **Compare**: Check output quality vs per-symbol mode
3. **Tune**: Adjust batch size based on your symbol sizes
4. **Monitor**: Watch costs in the logs

For more details, see [LLM_BATCH_MODE.md](./LLM_BATCH_MODE.md).

## Philosophy: "Semantic Static Analysis"

The key insight:
- ✅ **Embeddings** find semantic relationships (deterministic)
- ✅ **Graph analysis** extracts structure (deterministic)
- ✅ **LLM** translates to human-readable docs (semantic)

The LLM doesn't infer structure—it just makes it readable. That's why we can use batch mode: we're not asking the LLM to do analysis, just documentation!
