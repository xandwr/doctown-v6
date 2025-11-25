"""
Doctown Python Components.

This package provides:
- EmbeddingModel: Flexible embedding model manager
- DocpackBuilder: Builds .docpack files from repositories
- CLI: Command-line interface for docpack operations

Quick Start:
    from app.docpack_builder import build_docpack
    
    result = build_docpack("https://github.com/owner/repo")
    if result.success:
        print(f"Created: {result.docpack_path}")
"""

from .embedder import EmbeddingModel, embed_texts, get_embedder
from .docpack_builder import DocpackBuilder, DocpackConfig, DocpackResult, build_docpack
from .binary_io import write_embeddings_bin, read_embeddings_bin
from .utils import DocpackError, IngestError, EmbeddingError, PackagingError

__all__ = [
    # Embedder
    "EmbeddingModel",
    "embed_texts",
    "get_embedder",
    # Docpack Builder
    "DocpackBuilder",
    "DocpackConfig", 
    "DocpackResult",
    "build_docpack",
    # Binary I/O
    "write_embeddings_bin",
    "read_embeddings_bin",
    # Errors
    "DocpackError",
    "IngestError",
    "EmbeddingError",
    "PackagingError",
]