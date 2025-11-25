# LLM Per-Symbol Processing Architecture

## Problem Statement

The previous LLM documentation generation approach was creating arbitrarily large prompts by batching multiple symbols together, leading to frequent token limit errors. This was particularly problematic because:

1. **Unpredictable failures**: Large files or complex symbols would hit token limits inconsistently
2. **No granular control**: Couldn't identify which specific symbol caused the failure
3. **Wasted API calls**: Failed batches meant all symbols in that batch were lost
4. **Poor error recovery**: No automatic retry mechanism for oversized content

## Solution: Per-Symbol Parallel Processing

The new architecture treats each chunk (symbol) as an atomic unit and processes them individually with intelligent parallelism:

### Key Changes

#### 1. **Individual Symbol Processing** (`openai_client.py`)
- Each symbol (function, class, etc.) gets its own LLM API call
- Symbols are NEVER combined into giant prompts
- Each call is independent and can succeed/fail without affecting others

#### 2. **Parallel Execution with Concurrency Control** (`pipeline.py`)
```python
# Process all symbols in parallel with max_concurrent limit
result = await generator.generate_batch_async(
    chunks_to_document,
    max_concurrent=self.config.llm_max_concurrent,  # Default: 10
    progress_callback=progress,
)
```

**Benefits:**
- Fast: Up to 10 symbols documented simultaneously
- Safe: Respects rate limits and doesn't overwhelm the API
- Scalable: Easily adjust concurrency based on needs

#### 3. **Automatic Token Limit Retry** (`openai_client.py`)
When a symbol exceeds token limits:
1. Detect the error automatically
2. Truncate the symbol text to 75% of original length
3. Add a `[... content truncated due to length ...]` marker
4. Retry the API call once
5. If still fails, mark as failed with clear error message

```python
if retry_with_truncation and "token limit" in error:
    truncated_text = chunk.text[:int(len(chunk.text) * 0.75)]
    truncated_text += "\n\n[... content truncated due to length ...]"
    # Retry with truncated content
```

#### 4. **Token Distribution Analysis** (`pipeline.py`)
Before processing, analyze the symbol size distribution:
```
Symbol distribution: 5 large (>4K chars), 42 medium (1.2-4K chars), 156 small (<1.2K chars)
Total estimated input: ~45,320 tokens
```

This helps understand the workload and identify potential problematic symbols upfront.

#### 5. **Enhanced Error Reporting**
- Clear logging of which symbols failed and why
- Token usage breakdown per symbol
- Failed symbol paths and error messages
- Cost tracking remains accurate

## Architecture Diagram

```
Input Chunks (Symbols)
        ↓
    [Token Estimation]
        ↓
    [Size Analysis]
        ↓
┌───────────────────────┐
│  Parallel Processing  │
│  (max_concurrent=10)  │
├───────────────────────┤
│  Symbol 1 → LLM API   │ ← Individual call
│  Symbol 2 → LLM API   │ ← Individual call
│  Symbol 3 → LLM API   │ ← Individual call
│      ...              │
│  Symbol N → LLM API   │ ← Individual call
└───────────────────────┘
        ↓
    [Results Aggregation]
        ↓
    [Cost Calculation]
        ↓
    Documentation JSON
```

## Benefits

### 1. **No More Token Limit Failures (mostly)**
- Each symbol processed individually
- Automatic retry with truncation
- Only truly massive symbols (>context window) will fail

### 2. **Maximum Parallelism**
- Small symbols don't wait for large ones
- Efficient use of API concurrency
- Faster overall processing

### 3. **Better Error Recovery**
- One failed symbol doesn't affect others
- Clear identification of problematic symbols
- Automatic truncation recovery

### 4. **Improved Observability**
- Token distribution analysis upfront
- Per-symbol progress tracking
- Detailed failure reporting
- Accurate cost tracking

### 5. **Scalability**
- Easy to adjust concurrency (10, 20, 50 concurrent)
- Works with any number of symbols
- No artificial limits or arbitrary batching

## Configuration

Controlled via `PipelineConfig`:

```python
config = PipelineConfig(
    llm_max_concurrent=10,      # Max parallel API calls (default: 10)
    llm_max_chunks=None,        # Optional: limit total symbols for cost control
    max_completion_tokens=4096, # Max output tokens per symbol
)
```

## Example Output

```
Step 5/7: LLM documentation generation
  Symbol distribution: 3 large (>4K chars), 28 medium (1.2-4K chars), 142 small (<1.2K chars)
  Total estimated input: ~42,156 tokens
  Estimated LLM cost: $0.0234 for 173 symbols
    Model: gpt-5-nano, ~63,234 tokens
  Processing 173 symbols with max 10 concurrent requests...
    Progress: 10/173 symbols documented
    Progress: 20/173 symbols documented
    ...
    Progress: 173/173 symbols documented
  ✓ LLM docs: 171 successful, 2 failed
  Failed symbols (showing first 5):
    - src/lib/large_module.rs: Token limit exceeded after truncation retry
    - src/util/complex.rs: Rate limit (429) - temporary failure
  Token usage: 42,345 input + 28,123 output = 70,468 total
  💰 Actual cost: $0.0198
```

## Implementation Files

1. **`doctown/python/app/llm/openai_client.py`**
   - Added `estimate_tokens()` function
   - Enhanced `generate_for_chunk()` with retry logic
   - Enhanced `generate_for_chunk_async()` with retry logic
   - Updated class documentation

2. **`doctown/python/app/pipeline.py`**
   - Refactored `_generate_llm_docs()` for per-symbol processing
   - Added token distribution analysis
   - Enhanced progress tracking and error reporting
   - Better logging of symbol-level results

## Migration Notes

### Breaking Changes
None - the API remains the same. Existing code continues to work.

### Behavioral Changes
- Symbols are now processed individually instead of in arbitrary batches
- Token limit errors trigger automatic retry with truncation
- Progress logging shows "symbols" instead of "chunks"
- More detailed error messages for failures

### Performance Impact
**Positive:**
- Faster processing due to maximum parallelism
- Better success rate due to automatic retry
- No wasted work from failed batches

**Considerations:**
- Slightly higher API call overhead (one call per symbol)
- This is negligible compared to the benefits

## Future Enhancements

1. **Smarter Truncation**: Could use AST-aware truncation to preserve symbol boundaries
2. **Adaptive Concurrency**: Automatically adjust based on rate limit responses
3. **Symbol Prioritization**: Process important symbols first (public APIs, entry points)
4. **Caching**: Cache documentation for unchanged symbols across runs
5. **Streaming**: Stream results as they complete instead of waiting for all

## Conclusion

The per-symbol processing architecture eliminates the root cause of token limit issues by keeping symbols atomic and independent. Combined with automatic retry and maximum parallelism, this provides a robust, fast, and reliable LLM documentation generation system.
