"""
Base classes for the domain ingestor system.

This defines the universal chunk schema and the interface all domain ingestors must implement.
The key insight: downstream consumers (embedders, graph builders, doc generators, packagers)
don't care HOW you extracted chunks—only that they conform to UniversalChunk schema.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Literal


class Domain(str, Enum):
    """
    Supported domain types.
    
    Each domain can have its own:
    - Detection heuristics
    - Element extraction logic  
    - Chunk conversion rules
    - LLM prompt templates
    """
    CODE = "code"
    FINANCE = "finance"
    LEGAL = "legal"
    RESEARCH = "research"
    GENERIC = "generic"


class ChunkType(str, Enum):
    """
    Universal chunk types that span all domains.
    
    These are semantic roles, not file formats.
    A "table" chunk might come from CSV, Excel, or markdown.
    A "function" chunk only makes sense for code.
    """
    # Code-specific
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE = "module"
    IMPORT = "import"
    COMMENT = "comment"
    
    # Document-specific
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FIGURE = "figure"
    EQUATION = "equation"
    QUOTE = "quote"
    
    # Legal-specific
    CLAUSE = "clause"
    DEFINITION = "definition"
    SECTION = "section"
    ARTICLE = "article"
    
    # Finance-specific  
    TIMESERIES = "timeseries"
    FORMULA = "formula"
    METRIC = "metric"
    
    # Research-specific
    ABSTRACT = "abstract"
    CITATION = "citation"
    METHODOLOGY = "methodology"
    FINDING = "finding"
    
    # Generic
    TEXT = "text"
    METADATA = "metadata"
    UNKNOWN = "unknown"


@dataclass
class ChunkMetadata:
    """
    Domain-specific metadata that travels with chunks.
    
    This is the "escape hatch" for domain-specific info that doesn't
    fit the universal schema but might be useful for:
    - LLM prompts (e.g., knowing this is a "public function")
    - Search filters (e.g., "show only clauses from Section 4")
    - UI rendering (e.g., syntax highlighting language)
    """
    # Common metadata
    language: Optional[str] = None  # Programming or natural language
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # Hierarchical context
    parent_id: Optional[str] = None  # Parent chunk/element ID
    parent_name: Optional[str] = None  # Human-readable parent name
    depth: int = 0  # Nesting level
    
    # Semantic role
    semantic_role: Optional[str] = None  # e.g., "constructor", "error_handler", "preamble"
    visibility: Optional[str] = None  # e.g., "public", "private", "internal"
    
    # Domain-specific extensions (free-form)
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict, omitting None values."""
        result = {}
        if self.language is not None:
            result["language"] = self.language
        if self.line_start is not None:
            result["line_start"] = self.line_start
        if self.line_end is not None:
            result["line_end"] = self.line_end
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.parent_name is not None:
            result["parent_name"] = self.parent_name
        if self.depth > 0:
            result["depth"] = self.depth
        if self.semantic_role is not None:
            result["semantic_role"] = self.semantic_role
        if self.visibility is not None:
            result["visibility"] = self.visibility
        if self.extra:
            result["extra"] = self.extra
        return result


@dataclass
class UniversalChunk:
    """
    The universal chunk schema - the IR of the documentation pipeline.
    
    Every ingestor MUST emit chunks in this format. Downstream consumers
    (embedders, graph builders, doc generators) only understand this schema.
    
    This is the "secret glue" that makes the pipeline domain-agnostic.
    
    Example:
        {
            "chunk_id": "chunk_abc123",
            "domain": "code",
            "path": "src/lib.rs",
            "type": "function",
            "text": "pub fn calculate_total(items: &[Item]) -> f64 { ... }",
            "metadata": {
                "language": "rust",
                "line_start": 10,
                "line_end": 54,
                "semantic_role": "public_function",
                "parent_name": "impl Calculator"
            }
        }
    """
    # Stable identity
    chunk_id: str
    
    # Domain classification
    domain: Domain
    
    # Source location  
    path: str  # File path or logical location
    
    # Semantic type
    type: ChunkType
    
    # The actual content
    text: str
    
    # Optional rich metadata
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    
    # Position within file (for ordering)
    start_offset: int = 0
    end_offset: int = 0
    
    @classmethod
    def generate_id(cls, path: str, start: int, domain: str = "") -> str:
        """Generate a stable chunk ID from path and position."""
        content = f"{domain}:{path}:{start}"
        hash_hex = hashlib.sha256(content.encode()).hexdigest()[:12]
        return f"chunk_{hash_hex}"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to the JSON-serializable dict format for chunks.json."""
        return {
            "chunk_id": self.chunk_id,
            "domain": self.domain.value if isinstance(self.domain, Domain) else self.domain,
            "path": self.path,
            "type": self.type.value if isinstance(self.type, ChunkType) else self.type,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "start": self.start_offset,
            "end": self.end_offset,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UniversalChunk":
        """Create a UniversalChunk from a dict (e.g., from JSON)."""
        metadata_dict = data.get("metadata", {})
        metadata = ChunkMetadata(
            language=metadata_dict.get("language"),
            line_start=metadata_dict.get("line_start"),
            line_end=metadata_dict.get("line_end"),
            parent_id=metadata_dict.get("parent_id"),
            parent_name=metadata_dict.get("parent_name"),
            depth=metadata_dict.get("depth", 0),
            semantic_role=metadata_dict.get("semantic_role"),
            visibility=metadata_dict.get("visibility"),
            extra=metadata_dict.get("extra", {}),
        )
        
        return cls(
            chunk_id=data["chunk_id"],
            domain=Domain(data.get("domain", "generic")),
            path=data.get("path", data.get("file_path", "")),
            type=ChunkType(data.get("type", "text")),
            text=data["text"],
            metadata=metadata,
            start_offset=data.get("start", 0),
            end_offset=data.get("end", 0),
        )


@dataclass  
class RawElement:
    """
    A raw semantic element extracted by a domain ingestor.
    
    This is an intermediate representation between raw file content
    and UniversalChunks. Ingestors extract these, then convert them to chunks.
    
    Examples:
        - Code: AST node (function, class, etc.)
        - Legal: A clause or section
        - Finance: A table or time series
        - Research: A paragraph or citation
    """
    # Where this came from
    source_path: str
    
    # What kind of element (domain-specific)
    element_type: str
    
    # The raw content
    content: str
    
    # Position in source
    start_offset: int = 0
    end_offset: int = 0
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    
    # Optional structured data (AST node, parsed table, etc.)
    structured_data: Optional[Any] = None
    
    # Domain-specific attributes
    attributes: dict[str, Any] = field(default_factory=dict)


class DomainIngestor(ABC):
    """
    Abstract base class for domain-specific ingestors.
    
    To add support for a new domain:
    1. Subclass DomainIngestor
    2. Implement detect(), extract_elements(), to_chunks()
    3. Optionally override get_prompt_template() for custom LLM prompts
    4. Register with IngestorRegistry
    
    The pipeline flow:
        files → detect() → extract_elements() → to_chunks() → UniversalChunk[]
    """
    
    # Human-readable name for this ingestor
    name: str = "base"
    
    # Which domain this ingestor produces
    domain: Domain = Domain.GENERIC
    
    # Priority for detection (higher = checked first)
    priority: int = 0
    
    @abstractmethod
    def detect(self, files: dict[str, bytes]) -> bool:
        """
        Determine if this ingestor should handle the given files.
        
        Args:
            files: Dict mapping file paths to file contents (as bytes)
        
        Returns:
            True if this ingestor can/should handle these files
        
        Example detection strategies:
            - File extensions: *.rs, *.py, Cargo.toml
            - Magic bytes: PDF header, Excel signature
            - Content keywords: "whereas", "party", "hereinafter"
            - Structural patterns: CSV headers, JSON schema
        """
        pass
    
    @abstractmethod
    def extract_elements(self, files: dict[str, bytes]) -> list[RawElement]:
        """
        Extract semantic units from the files.
        
        This is where domain-specific parsing happens:
            - Code: AST parsing
            - Legal: Clause extraction
            - Finance: Table/timeseries parsing
            - Research: Section/citation extraction
        
        Args:
            files: Dict mapping file paths to file contents (as bytes)
        
        Returns:
            List of RawElements representing semantic units
        """
        pass
    
    @abstractmethod
    def to_chunks(self, elements: list[RawElement]) -> list[UniversalChunk]:
        """
        Convert raw elements into universal chunks.
        
        This normalizes domain-specific elements into the universal schema.
        
        Args:
            elements: List of RawElements from extract_elements()
        
        Returns:
            List of UniversalChunks ready for the pipeline
        """
        pass
    
    def ingest(self, files: dict[str, bytes]) -> list[UniversalChunk]:
        """
        Full ingestion pipeline: extract elements and convert to chunks.
        
        This is the main entry point used by the pipeline.
        """
        elements = self.extract_elements(files)
        return self.to_chunks(elements)
    
    def get_prompt_template(self) -> str:
        """
        Get the LLM prompt template for this domain.
        
        Override to provide domain-specific prompts for documentation generation.
        The template can use these placeholders:
            - {text}: The chunk text
            - {type}: The chunk type
            - {path}: The source path
            - {metadata}: JSON-formatted metadata
            - {context}: Related chunks/context
        
        Returns:
            A prompt template string
        """
        return DEFAULT_PROMPT_TEMPLATE
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for LLM documentation generation.
        
        Override to provide domain-specific system instructions.
        """
        return DEFAULT_SYSTEM_PROMPT


# Default prompts (can be overridden per-domain)
DEFAULT_SYSTEM_PROMPT = """You are a technical documentation assistant. Your task is to generate 
clear, accurate, and helpful documentation for the given content. Be concise but thorough.
Focus on explaining WHAT the content does, WHY it exists, and HOW to use it."""

DEFAULT_PROMPT_TEMPLATE = """Generate documentation for the following content:

**Source:** {path}
**Type:** {type}

```
{text}
```

{context}

Provide:
1. A one-sentence summary
2. A detailed description (2-3 paragraphs)
3. Key points or notes (if applicable)

Respond in JSON format:
{{
    "summary": "...",
    "description": "...",
    "notes": ["..."]
}}"""
