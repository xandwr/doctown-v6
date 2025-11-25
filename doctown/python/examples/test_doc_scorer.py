#!/usr/bin/env python3
"""
Example: Testing the DocScore system

This demonstrates how DocScore filters chunks based on multi-signal scoring.
"""
import numpy as np
from app.ingestors.base import UniversalChunk, ChunkType, ChunkMetadata, Domain
from app.doc_scorer import DocScorer


def create_sample_chunks():
    """Create sample chunks representing different types of code."""
    chunks = [
        # High-value: Public function with complexity
        UniversalChunk(
            chunk_id="chunk_001",
            domain=Domain.CODE,
            path="src/core.rs",
            type=ChunkType.FUNCTION,
            text="pub fn do_inference(model: &Model, input: &Tensor) -> Result<Tensor> {\n" + "    " * 40,
            metadata=ChunkMetadata(
                language="rust",
                line_start=10,
                line_end=65,
                depth=1,
                visibility="public",
            ),
            start_offset=100,
            end_offset=2000,
        ),
        
        # High-value: Core struct
        UniversalChunk(
            chunk_id="chunk_002",
            domain=Domain.CODE,
            path="src/tensor.rs",
            type=ChunkType.CLASS,
            text="pub struct Tensor {\n    data: Vec<f32>,\n    shape: Vec<usize>,\n}",
            metadata=ChunkMetadata(
                language="rust",
                line_start=5,
                line_end=20,
                depth=1,
                visibility="public",
            ),
            start_offset=50,
            end_offset=250,
        ),
        
        # Low-value: Match arm (internal node)
        UniversalChunk(
            chunk_id="chunk_003",
            domain=Domain.CODE,
            path="src/parser.rs",
            type=ChunkType.TEXT,  # Not a high-level type
            text="Some(x) => x,",
            metadata=ChunkMetadata(
                language="rust",
                line_start=45,
                line_end=45,
                depth=3,  # Nested deep
            ),
            start_offset=1200,
            end_offset=1215,
        ),
        
        # Low-value: Literal
        UniversalChunk(
            chunk_id="chunk_004",
            domain=Domain.CODE,
            path="src/config.rs",
            type=ChunkType.TEXT,
            text='"default_model"',
            metadata=ChunkMetadata(
                language="rust",
                line_start=8,
                line_end=8,
                depth=2,
            ),
            start_offset=200,
            end_offset=217,
        ),
        
        # Medium-value: Helper function
        UniversalChunk(
            chunk_id="chunk_005",
            domain=Domain.CODE,
            path="src/utils.rs",
            type=ChunkType.FUNCTION,
            text="fn format_output(value: f32) -> String {\n    format!(\"{:.2}\", value)\n}",
            metadata=ChunkMetadata(
                language="rust",
                line_start=50,
                line_end=52,
                depth=1,
                visibility="private",
            ),
            start_offset=800,
            end_offset=900,
        ),
    ]
    
    return chunks


def create_sample_embeddings(n_chunks: int, n_dims: int = 384):
    """
    Create sample embeddings that simulate semantic relationships.
    
    Chunks 0 and 1 (core functions) are similar to each other.
    Chunks 2-4 (internal nodes) are similar to each other.
    """
    embeddings = np.random.randn(n_chunks, n_dims).astype(np.float32)
    
    # Make chunks 0 and 1 semantically similar (core components)
    embeddings[1] = embeddings[0] + np.random.randn(n_dims) * 0.1
    
    # Make chunks 2, 3, 4 semantically similar (internal/helper code)
    base = np.random.randn(n_dims)
    embeddings[2] = base + np.random.randn(n_dims) * 0.1
    embeddings[3] = base + np.random.randn(n_dims) * 0.1
    embeddings[4] = base + np.random.randn(n_dims) * 0.15
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings


def main():
    """Demonstrate DocScore system."""
    print("=" * 60)
    print("DocScore System Demo")
    print("=" * 60)
    
    # Create sample data
    chunks = create_sample_chunks()
    embeddings = create_sample_embeddings(len(chunks))
    
    print(f"\n📦 Created {len(chunks)} sample chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  {i}: {chunk.type.value:15} {chunk.path:20} "
              f"depth={chunk.metadata.depth} vis={chunk.metadata.visibility or 'none':7}")
    
    # Initialize scorer
    scorer = DocScorer(
        structural_weight=0.50,
        semantic_weight=0.35,
        complexity_weight=0.15,
        doc_threshold=0.65,
    )
    
    print(f"\n⚙️  DocScore Configuration:")
    print(f"  Weights: structural=0.50, semantic=0.35, complexity=0.15")
    print(f"  Threshold: 0.65")
    
    # Score chunks
    print(f"\n🔍 Scoring chunks...")
    scored_chunks = scorer.score_chunks(chunks, embeddings, graph=None)
    
    # Display results
    print(f"\n📊 Scoring Results:")
    print(f"  {'Chunk':<10} {'Type':<15} {'Struct':>6} {'Sem':>6} {'Comp':>6} {'Final':>6} {'Doc?':>5}")
    print(f"  {'-'*10} {'-'*15} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*5}")
    
    for chunk in scored_chunks:
        breakdown = chunk.metadata.extra.get("doc_score_breakdown", {})
        score = chunk.metadata.extra.get("doc_score", 0.0)
        worthy = chunk.metadata.extra.get("doc_worthy", False)
        
        print(f"  {chunk.chunk_id:<10} {chunk.type.value:<15} "
              f"{breakdown.get('structural', 0):.2f}   "
              f"{breakdown.get('semantic', 0):.2f}   "
              f"{breakdown.get('complexity', 0):.2f}   "
              f"{score:.2f}   "
              f"{'✅' if worthy else '❌':<5}")
    
    # Statistics
    stats = scorer.get_score_statistics(scored_chunks)
    print(f"\n📈 Statistics:")
    print(f"  Mean score: {stats['mean']:.3f}")
    print(f"  Median score: {stats['median']:.3f}")
    print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    print(f"  Doc-worthy: {stats['doc_worthy']}/{stats['count']} "
          f"({stats['doc_worthy_percent']:.1f}%)")
    
    # Filter doc-worthy
    doc_worthy = scorer.filter_doc_worthy(scored_chunks)
    print(f"\n✅ Doc-worthy chunks ({len(doc_worthy)}):")
    for chunk in doc_worthy:
        score = chunk.metadata.extra.get("doc_score", 0.0)
        print(f"  - {chunk.chunk_id}: {chunk.type.value} in {chunk.path} (score={score:.2f})")
    
    print(f"\n💡 Interpretation:")
    print(f"  - Public functions/structs scored high (structural + semantic)")
    print(f"  - Internal nodes (match arms, literals) scored low")
    print(f"  - Private helper function scored medium (filtered out at 0.65)")
    print(f"  - Only {len(doc_worthy)}/{len(chunks)} chunks need LLM documentation")
    print(f"  - Cost savings: {100 * (1 - len(doc_worthy)/len(chunks)):.0f}%")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
