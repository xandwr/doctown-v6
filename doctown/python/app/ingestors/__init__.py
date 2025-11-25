"""
Domain Ingestors - Pluggable document normalization engine.

This module implements the "compiler frontend → IR → backend" architecture for documentation.
Everything downstream from ingestors receives the same universal chunk format,
making the pipeline domain-agnostic.

Architecture:
    Raw Input → DomainIngestor → UniversalChunk → Embeddings → Graph → Docs → Docpack
                 (pluggable)      (fixed schema)   (universal)  (universal)
"""
from .base import DomainIngestor, UniversalChunk, RawElement, ChunkMetadata
from .registry import IngestorRegistry, select_ingestor, get_registry
from .code import CodeIngestor
from .generic import GenericTextIngestor

__all__ = [
    # Base classes
    "DomainIngestor",
    "UniversalChunk", 
    "RawElement",
    "ChunkMetadata",
    # Registry
    "IngestorRegistry",
    "select_ingestor",
    "get_registry",
    # Built-in ingestors
    "CodeIngestor",
    "GenericTextIngestor",
]
