"""
DocScore: Multi-Signal Importance Scoring for Documentation Worthiness

This module implements a sophisticated scoring system that combines three signals
to determine which AST nodes/chunks are "doc-worthy" for LLM documentation:

1. Structural Signal (50%): AST-level importance based on node type and depth
2. Semantic Signal (35%): Embedding-based prominence using graph metrics
3. Complexity Signal (15%): Size and cyclomatic complexity heuristics

The combined DocScore ranges from 0.0 to 1.0, where higher scores indicate
nodes that are more worthy of detailed documentation.

Example:
    scorer = DocScorer(
        structural_weight=0.50,
        semantic_weight=0.35,
        complexity_weight=0.15,
        doc_threshold=0.65
    )
    
    # Score chunks with embeddings and graph data
    scored_chunks = scorer.score_chunks(chunks, embeddings, graph)
    
    # Filter to doc-worthy chunks only
    doc_worthy = [c for c in scored_chunks if c.metadata.extra.get('doc_score', 0) >= 0.65]
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ingestors.base import UniversalChunk, ChunkType

logger = logging.getLogger(__name__)


# Top-level node types that are inherently doc-worthy
HIGH_LEVEL_KINDS = {
    ChunkType.FUNCTION,
    ChunkType.METHOD,
    ChunkType.CLASS,
    ChunkType.MODULE,
    ChunkType.VARIABLE,  # Top-level constants/configs only
}

# Secondary important types
SECONDARY_KINDS = {
    ChunkType.IMPORT,
    ChunkType.COMMENT,  # Doc comments
}


@dataclass
class ScoringWeights:
    """Configurable weights for the three scoring signals."""
    structural: float = 0.50
    semantic: float = 0.35
    complexity: float = 0.15
    
    def __post_init__(self):
        total = self.structural + self.semantic + self.complexity
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class DocScoreBreakdown:
    """Detailed breakdown of score components for debugging/analysis."""
    structural: float
    semantic: float
    complexity: float
    final: float
    
    def to_dict(self) -> dict:
        return {
            "structural": round(self.structural, 4),
            "semantic": round(self.semantic, 4),
            "complexity": round(self.complexity, 4),
            "final": round(self.final, 4),
        }


class DocScorer:
    """
    Multi-signal scorer for determining documentation worthiness.
    
    This class implements the core DocScore algorithm that combines:
    - Structural signals from AST analysis
    - Semantic signals from embedding graphs
    - Complexity heuristics from code metrics
    
    Usage:
        scorer = DocScorer(doc_threshold=0.65)
        scored_chunks = scorer.score_chunks(chunks, embeddings, graph)
    """
    
    def __init__(
        self,
        structural_weight: float = 0.50,
        semantic_weight: float = 0.35,
        complexity_weight: float = 0.15,
        doc_threshold: float = 0.65,
        min_doc_size: int = 100,  # Min chars for size-based doc-worthiness
        complexity_threshold: int = 4,  # Min cyclomatic complexity for auto-doc
    ):
        """
        Initialize the DocScorer.
        
        Args:
            structural_weight: Weight for structural signals (default 0.50)
            semantic_weight: Weight for semantic signals (default 0.35)
            complexity_weight: Weight for complexity signals (default 0.15)
            doc_threshold: Minimum score to be doc-worthy (default 0.65)
            min_doc_size: Minimum chunk size for documentation (default 100 chars)
            complexity_threshold: Minimum complexity for auto-doc (default 4)
        """
        self.weights = ScoringWeights(structural_weight, semantic_weight, complexity_weight)
        self.doc_threshold = doc_threshold
        self.min_doc_size = min_doc_size
        self.complexity_threshold = complexity_threshold
    
    def score_chunks(
        self,
        chunks: list[UniversalChunk],
        embeddings: Optional[np.ndarray] = None,
        graph: Optional[dict] = None,
    ) -> list[UniversalChunk]:
        """
        Score all chunks and add DocScore metadata.
        
        This is the main entry point. It computes all three signals and
        combines them into a final DocScore for each chunk.
        
        Args:
            chunks: List of chunks to score
            embeddings: Optional embedding matrix (for semantic scoring)
            graph: Optional graph structure (for semantic scoring)
        
        Returns:
            Same chunks list with doc_score and breakdown added to metadata.extra
        """
        logger.info(f"Scoring {len(chunks)} chunks for doc-worthiness...")
        
        # Compute semantic scores if embeddings available
        semantic_scores = None
        if embeddings is not None and embeddings.size > 0:
            semantic_scores = self._compute_semantic_scores(chunks, embeddings, graph)
        
        # Score each chunk
        doc_worthy_count = 0
        for i, chunk in enumerate(chunks):
            structural = self._compute_structural_score(chunk)
            semantic = semantic_scores[i] if semantic_scores is not None else 0.0
            complexity = self._compute_complexity_score(chunk)
            
            # Weighted combination
            final_score = (
                self.weights.structural * structural
                + self.weights.semantic * semantic
                + self.weights.complexity * complexity
            )
            
            # Store in metadata
            breakdown = DocScoreBreakdown(structural, semantic, complexity, final_score)
            chunk.metadata.extra["doc_score"] = round(final_score, 4)
            chunk.metadata.extra["doc_score_breakdown"] = breakdown.to_dict()
            chunk.metadata.extra["doc_worthy"] = final_score >= self.doc_threshold
            
            if final_score >= self.doc_threshold:
                doc_worthy_count += 1
        
        logger.info(
            f"  DocScore complete: {doc_worthy_count}/{len(chunks)} "
            f"({100*doc_worthy_count/len(chunks):.1f}%) are doc-worthy"
        )
        
        return chunks
    
    def _compute_structural_score(self, chunk: UniversalChunk) -> float:
        """
        Compute structural importance based on AST node type and depth.
        
        This is the "obvious" signal: top-level declarations are important.
        
        Scoring logic:
        - Top-level functions/methods/classes: 1.0
        - Module-level constructs: 0.9
        - Secondary types (imports, doc comments): 0.3
        - Everything else: 0.0
        
        Depth penalty: Each level of nesting reduces score by 20%
        """
        score = 0.0
        
        # Base score by type
        if chunk.type in HIGH_LEVEL_KINDS:
            score = 1.0
        elif chunk.type == ChunkType.MODULE:
            score = 0.9
        elif chunk.type in SECONDARY_KINDS:
            score = 0.3
        else:
            score = 0.0
        
        # Depth penalty (if available in metadata)
        depth = chunk.metadata.depth
        if depth > 1:
            # Reduce score by 20% per level of nesting
            score *= (0.8 ** (depth - 1))
        
        # Visibility boost (public items are more doc-worthy)
        visibility = chunk.metadata.visibility
        if visibility == "public":
            score = min(1.0, score * 1.2)  # 20% boost for public items
        elif visibility == "private":
            score *= 0.7  # 30% penalty for private items
        
        return min(1.0, score)
    
    def _compute_semantic_score(
        self,
        chunks: list[UniversalChunk],
        embeddings: np.ndarray,
        graph: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute semantic importance using embedding-based metrics.
        
        This uses graph theory to detect important nodes:
        1. PageRank: Nodes referenced by many others
        2. Centrality: Nodes close to cluster centers
        3. Uniqueness: Nodes representing unique concepts
        
        Returns array of scores [0.0, 1.0] for each chunk.
        """
        if embeddings.size == 0:
            return np.zeros(len(chunks))
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        
        # Compute similarity matrix
        similarity_matrix = normalized @ normalized.T
        
        # 1. PageRank score: degree centrality (how many connections)
        # Each similarity > threshold counts as an edge
        threshold = 0.5
        adjacency = (similarity_matrix > threshold).astype(float)
        np.fill_diagonal(adjacency, 0)  # No self-loops
        
        degree = adjacency.sum(axis=1)
        max_degree = degree.max() if degree.max() > 0 else 1
        pagerank_scores = degree / max_degree
        
        # 2. Centroid score: closeness to cluster center
        # Use mean embedding as cluster center
        centroid = embeddings.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        
        centroid_similarities = normalized @ centroid
        centroid_scores = np.maximum(0, centroid_similarities)  # Clip negative values
        
        # 3. Uniqueness score: inverse of average similarity to others
        # Unique nodes have low similarity to most others
        avg_similarity = similarity_matrix.sum(axis=1) / (len(chunks) - 1)
        uniqueness_scores = 1.0 - avg_similarity
        uniqueness_scores = np.maximum(0, uniqueness_scores)
        
        # Combine the three semantic signals
        # PageRank: 40%, Centrality: 40%, Uniqueness: 20%
        semantic_scores = (
            0.40 * pagerank_scores
            + 0.40 * centroid_scores
            + 0.20 * uniqueness_scores
        )
        
        return semantic_scores
    
    def _compute_semantic_scores(
        self,
        chunks: list[UniversalChunk],
        embeddings: np.ndarray,
        graph: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Wrapper that handles graph-based PageRank if graph is available,
        otherwise falls back to embedding-only semantic scoring.
        """
        if graph and "nodes" in graph:
            # Use graph PageRank if available
            return self._compute_graph_semantic_scores(chunks, embeddings, graph)
        else:
            # Fall back to embedding-only scoring
            return self._compute_semantic_score(chunks, embeddings, graph)
    
    def _compute_graph_semantic_scores(
        self,
        chunks: list[UniversalChunk],
        embeddings: np.ndarray,
        graph: dict,
    ) -> np.ndarray:
        """
        Compute semantic scores using graph structure.
        
        Uses the pre-computed PageRank from the graph to boost scores
        for chunks in highly-connected files.
        """
        # Get base semantic scores
        base_scores = self._compute_semantic_score(chunks, embeddings, None)
        
        # Extract PageRank by file
        file_pageranks = {}
        for node in graph.get("nodes", []):
            if node["type"] == "file":
                path = node["path"]
                file_pageranks[path] = node.get("pagerank", 0.0)
        
        # Apply PageRank boost to chunks
        boosted_scores = base_scores.copy()
        for i, chunk in enumerate(chunks):
            pagerank = file_pageranks.get(chunk.path, 0.0)
            # Boost by up to 30% based on PageRank
            boost = 1.0 + (0.3 * pagerank)
            boosted_scores[i] = min(1.0, base_scores[i] * boost)
        
        return boosted_scores
    
    def _compute_complexity_score(self, chunk: UniversalChunk) -> float:
        """
        Compute complexity-based score using size and cyclomatic complexity.
        
        Three factors:
        1. Chunk length: Bigger functions need docs
        2. Cyclomatic complexity: Complex logic needs explanation
        3. Symbol span: Wide-spanning code is important
        
        Returns score [0.0, 1.0]
        """
        score = 0.0
        
        # 1. Size score (40% weight)
        chunk_length = len(chunk.text)
        if chunk_length >= self.min_doc_size:
            # Sigmoid scaling: 100 chars = 0.2, 500 chars = 0.8, 1000+ chars = 1.0
            size_score = min(1.0, chunk_length / 1000.0)
            score += 0.40 * size_score
        
        # 2. Cyclomatic complexity (40% weight)
        complexity = self._estimate_cyclomatic_complexity(chunk.text)
        if complexity >= self.complexity_threshold:
            # Linear scaling: 4 = 0.4, 10 = 1.0
            complexity_score = min(1.0, complexity / 10.0)
            score += 0.40 * complexity_score
        
        # 3. Symbol span (20% weight)
        line_start = chunk.metadata.line_start or 0
        line_end = chunk.metadata.line_end or 0
        if line_start and line_end:
            span = line_end - line_start
            # Sigmoid: 50 lines = 0.3, 200+ lines = 1.0
            span_score = min(1.0, span / 200.0)
            score += 0.20 * span_score
        
        return min(1.0, score)
    
    def _estimate_cyclomatic_complexity(self, code: str) -> int:
        """
        Estimate cyclomatic complexity by counting control flow keywords.
        
        This is a simple heuristic that counts:
        - Conditionals: if, elif, else if, case, when, match
        - Loops: for, while, loop
        - Exception handling: try, catch, except, rescue
        - Logical operators: &&, ||, and, or
        - Ternary operators: ? :
        
        Returns: Estimated complexity (1 = no branches, higher = more complex)
        """
        complexity = 1  # Base complexity
        
        # Keywords to count
        patterns = [
            r'\bif\b', r'\belif\b', r'\belse\s+if\b',
            r'\bmatch\b', r'\bcase\b', r'\bwhen\b',
            r'\bfor\b', r'\bwhile\b', r'\bloop\b',
            r'\btry\b', r'\bcatch\b', r'\bexcept\b', r'\brescue\b',
            r'&&', r'\|\|', r'\band\b', r'\bor\b',
            r'\?[^:]',  # Ternary (? without being part of ?:)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            complexity += len(matches)
        
        return complexity
    
    def filter_doc_worthy(
        self,
        chunks: list[UniversalChunk],
        threshold: Optional[float] = None,
    ) -> list[UniversalChunk]:
        """
        Filter chunks to only doc-worthy ones.
        
        Args:
            chunks: Chunks to filter (must have been scored)
            threshold: Optional override of default doc_threshold
        
        Returns:
            Filtered list of doc-worthy chunks
        """
        threshold = threshold or self.doc_threshold
        
        doc_worthy = [
            c for c in chunks
            if c.metadata.extra.get("doc_score", 0.0) >= threshold
        ]
        
        logger.info(
            f"Filtered to {len(doc_worthy)}/{len(chunks)} doc-worthy chunks "
            f"(threshold={threshold})"
        )
        
        return doc_worthy
    
    def get_score_statistics(self, chunks: list[UniversalChunk]) -> dict:
        """
        Compute statistics about DocScores across chunks.
        
        Returns dict with mean, median, min, max, and percentiles.
        """
        scores = [c.metadata.extra.get("doc_score", 0.0) for c in chunks]
        
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        
        return {
            "count": len(scores),
            "mean": float(np.mean(scores_array)),
            "median": float(np.median(scores_array)),
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "std": float(np.std(scores_array)),
            "percentiles": {
                "p25": float(np.percentile(scores_array, 25)),
                "p50": float(np.percentile(scores_array, 50)),
                "p75": float(np.percentile(scores_array, 75)),
                "p90": float(np.percentile(scores_array, 90)),
                "p95": float(np.percentile(scores_array, 95)),
            },
            "doc_worthy": sum(1 for s in scores if s >= self.doc_threshold),
            "doc_worthy_percent": 100 * sum(1 for s in scores if s >= self.doc_threshold) / len(scores),
        }
