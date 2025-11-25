# LLM Batch Mode & Semantic Context Enhancement

## The Problem

The original per-symbol LLM processing was experiencing:

1. **High Latency**: 67 symbols took 26+ seconds due to network overhead (one API call per symbol)
2. **Token Limit Errors**: `gpt-5-nano` only has 4096 max completion tokens, causing failures on larger symbols
3. **No Semantic Context**: The LLM wasn't receiving information about code relationships from embeddings/graph analysis

## The Solution

### 1. Semantic Context Injection

**What it does:**
- Computes k-nearest neighbors for each chunk using cosine similarity on embeddings
- Provides the LLM with information about semantically related code
- Enables the LLM to understand implementation patterns and cross-references

**How it works:**
```python
# Embeddings enable semantic analysis
semantic_neighbors = compute_k_nearest_neighbors(embeddings, k=3)

# Context includes related code snippets
context = """
**Semantically Related:**
- src/parser.rs (similarity: 0.87): fn parse_ast(...) { ... }
- src/analyzer.rs (similarity: 0.82): fn analyze_types(...) { ... }
"""
```

This gives the LLM actual implementation details instead of just the isolated symbol.

### 2. Batch Processing Mode

**What it does:**
- Combines multiple symbols into a single LLM request
- Reduces network round-trips dramatically (67 requests → ~5 batches)
- Enables larger, smarter models with better context windows

**Performance Comparison:**

| Mode | Requests | Latency | Model | Context Window |
|------|----------|---------|-------|----------------|
| **Per-Symbol** | 67 | 26s | gpt-5-nano | 4K tokens |
| **Batch (15/batch)** | 5 | ~8s | gpt-4o/gpt-5 | 128K tokens |

**Cost Comparison:**

With batch mode, you use a more expensive model BUT:
- Fewer API calls = Less network overhead
- Prompt caching on larger models = Lower effective cost per token
- Better context = Higher quality docs in one pass

Example for 67 symbols:
- **Per-symbol mode** (gpt-5-nano): $0.012, 67 requests, 26s
- **Batch mode** (gpt-4o): $0.045, 5 requests, ~8s  ← 3x cost, 3x faster, better quality

### 3. Configuration

#### CLI Options

```bash
# Enable batch mode with a larger model
python -m app.cli build-llm \
  https://github.com/user/repo \
  --llm-batch-mode \
  --llm-batch-size 15 \
  --llm-model gpt-4o \
  --llm-semantic-neighbors 3

# Traditional per-symbol mode (default)
python -m app.cli build-llm \
  https://github.com/user/repo \
  --llm-model gpt-5-nano \
  --llm-concurrent 10 \
  --llm-semantic-neighbors 3
```

#### Pipeline Configuration

```python
from app.pipeline import PipelineConfig, DocumentationPipeline

config = PipelineConfig(
    input_source="https://github.com/user/repo",
    
    # Batch mode settings
    llm_batch_mode=True,          # Enable batch processing
    llm_batch_size=15,            # Symbols per batch request
    llm_model="gpt-4o",           # Use a larger model
    
    # Semantic context
    llm_semantic_neighbors=3,     # Include 3 nearest neighbors
    
    # Per-symbol mode settings (when batch_mode=False)
    llm_max_concurrent=10,        # Concurrent requests
)

pipeline = DocumentationPipeline(config)
result = await pipeline.run_async()
```

## When to Use Each Mode

### Per-Symbol Mode (Default)
**Best for:**
- Small codebases (<100 symbols)
- Budget constraints (cheapest option)
- Quick prototyping

**Characteristics:**
- Each symbol processed independently
- High concurrency (10+ parallel requests)
- Cheap models (gpt-5-nano)
- Network bound (latency matters)

### Batch Mode
**Best for:**
- Medium to large codebases (100+ symbols)
- Production use cases
- When latency matters
- When context quality matters

**Characteristics:**
- Multiple symbols per request
- Lower request count
- Smarter models (gpt-4o, gpt-5)
- Better cross-symbol analysis
- Faster total time despite higher per-token cost

## Architecture: "Semantic Static Analysis"

The key insight is that the pipeline performs **semantic static analysis**:

1. **Deterministic Extraction** (Rust ingester)
   - Parse code into AST
   - Extract symbols, types, relationships
   - Create Universal Chunks

2. **Semantic Analysis** (Python embeddings)
   - Generate embeddings for each chunk
   - Compute semantic similarity graph
   - Find related code patterns

3. **Human-Readable Synthesis** (LLM)
   - Convert structured data into prose
   - Explain relationships and patterns
   - Generate examples and usage notes

**The LLM is the last step**: It doesn't infer anything structural—all that is deterministic. It just makes it human-readable with proper context.

## Implementation Details

### Semantic Neighbor Computation

```python
def _compute_semantic_neighbors(
    chunks: list[UniversalChunk],
    embeddings: np.ndarray,
    k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """
    Compute k-nearest neighbors using cosine similarity.
    Returns: {chunk_id: [(neighbor_id, similarity_score), ...]}
    """
    # Normalize for cosine similarity
    normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute similarity matrix
    similarity_matrix = normalized @ normalized.T
    
    # Extract top-k for each chunk
    neighbors_map = {}
    for i, chunk in enumerate(chunks):
        similarities = similarity_matrix[i]
        similarities[i] = -1  # Exclude self
        
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        neighbors = [
            (chunks[j].chunk_id, float(similarities[j]))
            for j in top_k_indices
            if similarities[j] > 0.3  # Minimum threshold
        ]
        neighbors_map[chunk.chunk_id] = neighbors
    
    return neighbors_map
```

### Batch Request Format

```python
async def generate_multi_batch_async(
    chunks: list[UniversalChunk],
    batch_size: int = 15,
    context_map: dict[str, str] = None,
):
    """Process multiple symbols in one request."""
    
    # Build combined prompt
    user_prompt = "Generate documentation for the following symbols:\n"
    
    for idx, chunk in enumerate(batch, 1):
        user_prompt += f"""
{'='*60}
SYMBOL {idx}/{len(batch)}: {chunk.chunk_id}
{'='*60}
**File:** {chunk.path}
**Type:** {chunk.type}

{context_map.get(chunk.chunk_id, '')}

```{chunk.metadata.language}
{chunk.text}
```
"""
    
    # Response format: JSON array with docs for each symbol
    response = await client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=self.max_completion_tokens * len(batch),
        response_format={"type": "json_object"},
    )
    
    # Parse array and map back to chunks
    docs = json.loads(response.choices[0].message.content)
    return docs
```

## Prompt Engineering Updates

The prompts now explicitly instruct the LLM to use semantic context:

```python
CODE_SYSTEM_PROMPT = """
...
5. RELATIONSHIPS: When semantic context is provided, explain how this code 
   relates to similar code

Use provided semantic relationships to understand implementation patterns 
and cross-references.
"""

CODE_USER_TEMPLATE = """
...
{context}

**Analysis Guidelines:**
- If semantic relationships are provided above, use them to understand how 
  this code fits into the broader system
- Identify patterns, conventions, and architectural relationships
- Reference related code when explaining functionality
"""
```

## Monitoring & Debugging

The pipeline logs detailed information:

```
12:50:57 [INFO]   Symbol distribution: 0 large, 0 medium, 67 small
12:50:57 [INFO]   Computing semantic neighbors (k=3)...
12:50:57 [INFO]   Estimated LLM cost (BATCH MODE): $0.0450
12:50:57 [INFO]     Model: gpt-4o, ~5 batches, ~45,000 tokens
12:50:57 [INFO]   Processing 67 symbols in BATCH MODE (15 symbols/batch)...
12:51:05 [INFO]     Progress: 5/5 batches completed
12:51:05 [INFO]   ✓ LLM docs: 67 successful, 0 failed
12:51:05 [INFO]   Token usage: 32,150 input + 12,850 output = 45,000 total
12:51:05 [INFO]   💰 Actual cost: $0.0423
```

## Migration Guide

### From Per-Symbol to Batch Mode

1. **Update your model**: Switch from `gpt-5-nano` to a larger model
   ```bash
   export OPENAI_MODEL=gpt-4o  # or gpt-5
   ```

2. **Enable batch mode**:
   ```bash
   --llm-batch-mode --llm-batch-size 15
   ```

3. **Monitor costs**: First run will show estimates
   ```
   Estimated LLM cost (BATCH MODE): $0.0450
   ```

4. **Adjust batch size** based on your symbol sizes:
   - Small symbols (<500 tokens): batch_size=20
   - Medium symbols (500-2K tokens): batch_size=15
   - Large symbols (2K+ tokens): batch_size=10

### Enabling Semantic Context

Just add:
```bash
--llm-semantic-neighbors 3
```

No other changes needed! The pipeline automatically:
1. Computes embeddings (already done)
2. Finds k-nearest neighbors
3. Injects context into prompts

## Future Enhancements

- [ ] Adaptive batch sizing based on token estimates
- [ ] Parallel batch processing (process multiple batches concurrently)
- [ ] Prompt caching optimization for repeated patterns
- [ ] Cross-batch context sharing (for related files)
- [ ] Support for local LLMs (Ollama, llama.cpp)

## References

- `doctown/python/app/pipeline.py`: Main orchestration + semantic neighbor computation
- `doctown/python/app/llm/openai_client.py`: Batch mode implementation
- `doctown/python/app/llm/prompts.py`: Updated prompts with semantic context
- `doctown/python/app/cli.py`: CLI argument parsing

## Questions?

This represents a fundamental shift in how we use LLMs for documentation:

**OLD**: LLM infers everything from isolated code snippets
**NEW**: Deterministic analysis → LLM translates to human-readable

The embeddings and graph analysis do the heavy lifting. The LLM just makes it pretty. 🎨
