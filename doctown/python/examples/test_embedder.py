#!/usr/bin/env python3
"""
Example script showing how to use the embedding system.
Super hackable - just set env vars or pass model IDs directly.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import EmbeddingModel, get_embedder, embed_texts


def example_1_default_model():
    """Use the default model."""
    print("\n" + "="*60)
    print("Example 1: Default Model")
    print("="*60)
    
    embedder = EmbeddingModel()
    embedder.load_model()
    
    texts = [
        "This is a test sentence.",
        "Another example for embedding.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    embeddings = embedder.embed(texts)
    print(f"\n📊 Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"   First embedding (truncated): {embeddings[0][:5].tolist()}")


def example_2_custom_model():
    """Use a different model at runtime."""
    print("\n" + "="*60)
    print("Example 2: Custom Model")
    print("="*60)
    
    # Try a different model - uncomment to test
    # embedder = EmbeddingModel(model_id="BAAI/bge-small-en-v1.5")
    
    # Or use environment variable:
    # export EMBEDDING_MODEL="intfloat/e5-base-v2"
    embedder = EmbeddingModel(model_id="sentence-transformers/all-MiniLM-L6-v2")
    embedder.load_model()
    
    texts = ["Hello, world!"]
    embeddings = embedder.embed(texts)
    
    print(f"\n📊 Model info:")
    for key, value in embedder.get_model_info().items():
        print(f"   {key}: {value}")


def example_3_global_instance():
    """Use the global singleton pattern."""
    print("\n" + "="*60)
    print("Example 3: Global Instance (Singleton Pattern)")
    print("="*60)
    
    # This is the most common usage pattern for a server
    embeddings = embed_texts([
        "Using the global embedder instance",
        "Model is loaded once and reused",
        "Perfect for API endpoints"
    ])
    
    print(f"\n📊 Generated embeddings: {embeddings.shape}")


def example_4_batch_processing():
    """Process larger batches efficiently."""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    embedder = get_embedder()
    
    # Generate some test documents
    texts = [f"This is test document number {i}" for i in range(100)]
    
    embeddings = embedder.embed(texts, batch_size=16)
    print(f"\n📊 Processed {len(texts)} documents")
    print(f"   Embedding shape: {embeddings.shape}")
    print(f"   Device: {embeddings.device}")


if __name__ == "__main__":
    print("\n🚀 Doctown Embedding System Demo")
    print("="*60)
    print("\nHackable configurations:")
    print("  EMBEDDING_MODEL=your-model-id")
    print("  EMBEDDING_CACHE_DIR=./your/cache/path")
    print("  EMBEDDING_DEVICE=cuda|cpu|auto")
    print("  HF_TOKEN=your_token_for_private_models")
    print("\nOr pass arguments directly to EmbeddingModel()")
    
    # Run examples
    example_1_default_model()
    
    # Uncomment to run other examples:
    # example_2_custom_model()
    # example_3_global_instance()
    # example_4_batch_processing()
    
    print("\n✅ Done!\n")
