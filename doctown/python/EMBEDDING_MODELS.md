# Embedding Models - Hackable AF 🔧

This system lets you use **any** HuggingFace embedding model at runtime with zero config changes. Just set an env var or pass a model ID.

## Quick Start

```python
from app.embedder import EmbeddingModel, embed_texts

# Option 1: Use a preset
embedder = EmbeddingModel(model_id="fast")

# Option 2: Use any HuggingFace model
embedder = EmbeddingModel(model_id="BAAI/bge-large-en-v1.5")

# Option 3: Use environment variable
# export EMBEDDING_MODEL="intfloat/e5-large-v2"
embedder = EmbeddingModel()

# Generate embeddings
texts = ["Hello world", "Semantic search rocks"]
embeddings = embedder.embed(texts)
```

## Available Presets

Use these shorthand names for quick experimentation:

| Preset | Model | Dimensions | Best For |
|--------|-------|------------|----------|
| `fast` | all-MiniLM-L6-v2 | 384 | Development, testing |
| `balanced` | all-mpnet-base-v2 | 768 | Production, general use |
| `quality` | bge-large-en-v1.5 | 1024 | Maximum accuracy |
| `multilingual` | paraphrase-multilingual-mpnet | 768 | Multi-language code |
| `code` | codebert-base | 768 | Code-specific embeddings |

## Environment Variables

All configuration can be done via env vars:

```bash
# Model selection (preset name or HuggingFace ID)
export EMBEDDING_MODEL="fast"
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Or use preset env var
export EMBEDDING_MODEL_PRESET="quality"

# Cache directory
export EMBEDDING_CACHE_DIR="./models/embeddings"

# Device selection
export EMBEDDING_DEVICE="cuda"  # cuda, cpu, or auto

# HuggingFace token (for private models)
export HF_TOKEN="hf_your_token_here"
```

## CLI Tools

### Download Models Ahead of Time

```bash
# List available presets
python scripts/download_model.py --list

# Download a preset
python scripts/download_model.py --preset fast
python scripts/download_model.py --preset quality

# Download any HuggingFace model
python scripts/download_model.py --model BAAI/bge-large-en-v1.5
python scripts/download_model.py --model intfloat/e5-large-v2

# Custom cache directory
python scripts/download_model.py --preset fast --cache ./my-cache

# Private models
python scripts/download_model.py --model private/model --token hf_xxx
```

### Test Embeddings

```bash
# Run the example script
python examples/test_embedder.py

# With custom model
EMBEDDING_MODEL="quality" python examples/test_embedder.py
```

## Code Examples

### Basic Usage

```python
from app.embedder import EmbeddingModel

# Initialize with any model
embedder = EmbeddingModel(model_id="BAAI/bge-small-en-v1.5")
embedder.load_model()

# Embed some text
texts = ["Hello", "World"]
embeddings = embedder.embed(texts)

print(f"Shape: {embeddings.shape}")
print(f"Device: {embeddings.device}")
```

### Global Singleton (Recommended for Servers)

```python
from app.embedder import get_embedder, embed_texts

# This loads the model once and reuses it
embeddings = embed_texts([
    "Document 1",
    "Document 2",
    "Document 3"
])

# Or get the embedder instance
embedder = get_embedder()
info = embedder.get_model_info()
print(info)
```

### Advanced Configuration

```python
from app.embedder import EmbeddingModel

embedder = EmbeddingModel(
    model_id="BAAI/bge-large-en-v1.5",
    cache_dir="./my-cache",
    device="cuda",
    use_auth_token="hf_xxx"
)

# Download without loading
model_path = embedder.download_model()
print(f"Model at: {model_path}")

# Load when ready
embedder.load_model()

# Generate with custom parameters
embeddings = embedder.embed(
    texts=["test"],
    batch_size=64,
    normalize_embeddings=True,  # L2 normalization
    show_progress_bar=False
)
```

### Switching Models at Runtime

```python
from app.embedder import EmbeddingModel

# Test different models easily
models_to_test = ["fast", "balanced", "quality"]

for model_preset in models_to_test:
    embedder = EmbeddingModel(model_id=model_preset)
    embedder.load_model()
    
    embeddings = embedder.embed(["test"])
    info = embedder.get_model_info()
    
    print(f"{model_preset}: {info['embedding_dim']} dims")
```

## How It Works

1. **Model Resolution**: Checks presets, env vars, and parameters in order
2. **Download**: Uses `snapshot_download()` to fetch the entire model directory
3. **Caching**: Models are cached locally, never re-downloaded
4. **Loading**: Loads with `sentence-transformers` but you can swap in raw transformers
5. **Inference**: Batched encoding with progress bars and device management

## Customization Points

Want to hack it further? Here's what you can modify:

### Use a Different Backend

Replace `SentenceTransformer` with raw `transformers`:

```python
from transformers import AutoModel, AutoTokenizer

# In embedder.py, replace the load_model() method
self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
self.model = AutoModel.from_pretrained(str(model_path))
```

### Add ONNX Support

For faster inference:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Replace model loading
self.model = ORTModelForFeatureExtraction.from_pretrained(
    str(model_path),
    export=True,  # Convert to ONNX on first load
)
```

### Custom Pooling

Override the embed method:

```python
def custom_embed(self, texts):
    # Your custom tokenization and pooling logic
    pass
```

### Add More Presets

Edit `MODEL_PRESETS` in `app/embedder.py`:

```python
MODEL_PRESETS = {
    "fast": "...",
    "your_preset": "your/model/id",
}
```

Or edit `config/embedding_models.json` for documentation.

## Model Recommendations

### For Development
- **fast**: Quick iterations, small downloads

### For Production
- **balanced**: Good quality/speed trade-off
- **quality**: Best results, slower

### For Code Search
- **code**: Trained on code data
- Try: `microsoft/unixcoder-base`, `salesforce/codet5-base`

### For Multilingual
- **multilingual**: 50+ languages
- Try: `intfloat/multilingual-e5-large`

### Latest SOTA Models
- `BAAI/bge-large-en-v1.5` (English)
- `intfloat/e5-large-v2` (English)
- `thenlper/gte-large` (English)

## Architecture

```
User Code
    ↓
EmbeddingModel.__init__()
    ↓
resolve_model_id() → Preset/Env/Param resolution
    ↓
snapshot_download() → Downloads full model directory
    ↓
SentenceTransformer.load() → Loads model into memory
    ↓
embedder.embed() → Generates embeddings
    ↓
torch.Tensor output
```

## Troubleshooting

### Model Download Fails
```bash
# Check HuggingFace status
curl https://status.huggingface.co

# Try with token
export HF_TOKEN="your_token"

# Check disk space
df -h

# Clear cache and retry
rm -rf ./models/embeddings/*
```

### CUDA Out of Memory
```bash
# Use CPU instead
export EMBEDDING_DEVICE="cpu"

# Or use a smaller model
export EMBEDDING_MODEL="fast"

# Or reduce batch size
embedder.embed(texts, batch_size=8)
```

### Model Not Found
```bash
# List available models on HuggingFace
# https://huggingface.co/models?pipeline_tag=sentence-similarity

# Check the exact model ID
python scripts/download_model.py --model your-model-id
```

## Integration with Pipeline

Once you've tested your embedding model:

```python
# In your main pipeline
from app.embedder import get_embedder

def process_documents(docs):
    embedder = get_embedder()
    
    # Generate embeddings for all docs
    embeddings = embedder.embed(docs, batch_size=32)
    
    # Use embeddings for semantic search, clustering, etc.
    return embeddings
```

The embedder will be loaded once per process and reused across all requests.

---

**Bottom line**: Pick any model from HuggingFace, set an env var or pass it as a parameter, and you're done. No code changes needed. Hackable as fuck. 🚀
