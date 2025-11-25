"""
Generic Text Ingestor - Fallback for unknown content types.

This ingestor handles any text-based content that doesn't match
a specialized domain ingestor. It does intelligent paragraph-aware
chunking and tries to preserve semantic boundaries.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .base import (
    DomainIngestor,
    Domain,
    ChunkType,
    ChunkMetadata,
    UniversalChunk,
    RawElement,
)

logger = logging.getLogger(__name__)

# Text file extensions
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".asciidoc", ".adoc",
    ".tex", ".latex", ".org", ".wiki",
    ".csv", ".tsv", ".log",
    ".xml", ".html", ".htm", ".xhtml",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
}

# Binary extensions to skip
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".ttf", ".otf", ".woff", ".woff2",
    ".pyc", ".pyo", ".class", ".o", ".obj",
}


class GenericTextIngestor(DomainIngestor):
    """
    Fallback ingestor for generic text content.
    
    This handles:
    - Markdown/RST documentation
    - Plain text files
    - Config files (YAML, JSON, TOML)
    - Log files
    - Any other text-based content
    
    Uses paragraph-aware chunking to preserve semantic boundaries.
    """
    
    name = "generic"
    domain = Domain.GENERIC
    priority = 0  # Lowest priority - fallback ingestor
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """
        Initialize the generic text ingestor.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (avoids tiny fragments)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def detect(self, files: dict[str, bytes]) -> bool:
        """
        Always returns True - this is the fallback ingestor.
        
        In practice, this will only be reached if no other ingestor matches.
        """
        # Check if there's at least one text-like file
        for path in files.keys():
            ext = Path(path).suffix.lower()
            if ext not in BINARY_EXTENSIONS:
                return True
        return False
    
    def extract_elements(self, files: dict[str, bytes]) -> list[RawElement]:
        """
        Extract text elements with paragraph-aware chunking.
        """
        elements = []
        
        for file_path, content in files.items():
            ext = Path(file_path).suffix.lower()
            
            # Skip binary files
            if ext in BINARY_EXTENSIONS:
                continue
            
            # Try to decode as text
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = content.decode("latin-1")
                except Exception:
                    logger.debug(f"Skipping binary file: {file_path}")
                    continue
            
            if not text.strip():
                continue
            
            # Determine content type and chunk appropriately
            if ext in {".md", ".markdown"}:
                file_elements = self._chunk_markdown(file_path, text)
            elif ext in {".json", ".yaml", ".yml", ".toml"}:
                file_elements = self._chunk_structured(file_path, text)
            else:
                file_elements = self._chunk_paragraphs(file_path, text)
            
            elements.extend(file_elements)
        
        return elements
    
    def _chunk_paragraphs(self, file_path: str, text: str) -> list[RawElement]:
        """
        Chunk text by paragraphs, respecting semantic boundaries.
        """
        # Split into paragraphs (double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        
        elements = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If single paragraph exceeds chunk size, split it
            if para_length > self.chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    elements.append(RawElement(
                        source_path=file_path,
                        element_type="paragraph",
                        content=chunk_text,
                        start_offset=chunk_start,
                        end_offset=chunk_start + len(chunk_text),
                    ))
                    chunk_start += len(chunk_text) + 2
                    current_chunk = []
                    current_length = 0
                
                # Split large paragraph
                for sub_chunk in self._split_long_text(para):
                    elements.append(RawElement(
                        source_path=file_path,
                        element_type="paragraph",
                        content=sub_chunk,
                        start_offset=chunk_start,
                        end_offset=chunk_start + len(sub_chunk),
                    ))
                    chunk_start += len(sub_chunk)
                continue
            
            # Add to current chunk if it fits
            if current_length + para_length + 2 <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_length + 2
            else:
                # Flush current chunk
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    elements.append(RawElement(
                        source_path=file_path,
                        element_type="paragraph",
                        content=chunk_text,
                        start_offset=chunk_start,
                        end_offset=chunk_start + len(chunk_text),
                    ))
                    chunk_start += len(chunk_text) + 2
                
                current_chunk = [para]
                current_length = para_length
        
        # Flush remaining
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                elements.append(RawElement(
                    source_path=file_path,
                    element_type="paragraph",
                    content=chunk_text,
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(chunk_text),
                ))
        
        return elements
    
    def _chunk_markdown(self, file_path: str, text: str) -> list[RawElement]:
        """
        Chunk markdown by sections (headings).
        """
        elements = []
        
        # Split by headings
        sections = re.split(r'^(#{1,6}\s+.+)$', text, flags=re.MULTILINE)
        
        current_heading = None
        current_content = []
        chunk_start = 0
        
        for part in sections:
            part = part.strip()
            if not part:
                continue
            
            # Check if this is a heading
            if re.match(r'^#{1,6}\s+', part):
                # Flush previous section
                if current_content:
                    section_text = "\n\n".join(filter(None, [current_heading] + current_content))
                    if len(section_text) >= self.min_chunk_size:
                        heading_match = re.match(r'^(#+)', current_heading or "#")
                        heading_level = len(heading_match.group(1)) if heading_match else 0
                        elements.append(RawElement(
                            source_path=file_path,
                            element_type="section",
                            content=section_text,
                            start_offset=chunk_start,
                            end_offset=chunk_start + len(section_text),
                            attributes={
                                "heading": current_heading,
                                "heading_level": heading_level,
                            }
                        ))
                        chunk_start += len(section_text) + 2
                
                current_heading = part
                current_content = []
            else:
                # Regular content - may need to chunk if too large
                if len(part) > self.chunk_size:
                    # Large content block - split it
                    for sub_chunk in self._split_long_text(part):
                        current_content.append(sub_chunk)
                else:
                    current_content.append(part)
        
        # Flush final section
        if current_content or current_heading:
            section_text = "\n\n".join(filter(None, [current_heading] + current_content))
            if len(section_text) >= self.min_chunk_size:
                heading_match = re.match(r'^(#+)', current_heading or "#")
                heading_level = len(heading_match.group(1)) if heading_match else 0
                elements.append(RawElement(
                    source_path=file_path,
                    element_type="section",
                    content=section_text,
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(section_text),
                    attributes={
                        "heading": current_heading,
                        "heading_level": heading_level,
                    }
                ))
        
        return elements if elements else self._chunk_paragraphs(file_path, text)
    
    def _chunk_structured(self, file_path: str, text: str) -> list[RawElement]:
        """
        Chunk structured files (JSON, YAML, TOML) more conservatively.
        """
        # For config files, try to keep them whole if small enough
        if len(text) <= self.chunk_size * 2:
            return [RawElement(
                source_path=file_path,
                element_type="config",
                content=text,
                start_offset=0,
                end_offset=len(text),
            )]
        
        # Otherwise, fall back to line-based chunking
        return self._chunk_by_lines(file_path, text)
    
    def _chunk_by_lines(self, file_path: str, text: str) -> list[RawElement]:
        """
        Chunk by lines, trying to respect logical boundaries.
        """
        lines = text.split("\n")
        elements = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        line_num_start = 1
        
        for i, line in enumerate(lines, 1):
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > self.chunk_size and current_chunk:
                # Flush current chunk
                chunk_text = "\n".join(current_chunk)
                elements.append(RawElement(
                    source_path=file_path,
                    element_type="text",
                    content=chunk_text,
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(chunk_text),
                    line_start=line_num_start,
                    line_end=i - 1,
                ))
                chunk_start += len(chunk_text) + 1
                line_num_start = i
                current_chunk = []
                current_length = 0
            
            current_chunk.append(line)
            current_length += line_length
        
        # Flush remaining
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                elements.append(RawElement(
                    source_path=file_path,
                    element_type="text",
                    content=chunk_text,
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(chunk_text),
                    line_start=line_num_start,
                    line_end=len(lines),
                ))
        
        return elements
    
    def _split_long_text(self, text: str) -> list[str]:
        """
        Split long text into chunks, preferring sentence boundaries.
        """
        chunks = []
        
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if sentence_length > self.chunk_size:
                # Sentence too long - split at word boundaries
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Word-level splitting
                words = sentence.split()
                word_chunk = []
                word_length = 0
                
                for word in words:
                    if word_length + len(word) + 1 > self.chunk_size and word_chunk:
                        chunks.append(" ".join(word_chunk))
                        word_chunk = []
                        word_length = 0
                    word_chunk.append(word)
                    word_length += len(word) + 1
                
                if word_chunk:
                    chunks.append(" ".join(word_chunk))
                continue
            
            if current_length + sentence_length + 1 > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_length + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def to_chunks(self, elements: list[RawElement]) -> list[UniversalChunk]:
        """
        Convert raw elements to universal chunks.
        """
        chunks = []
        
        for element in elements:
            chunk_id = UniversalChunk.generate_id(
                element.source_path,
                element.start_offset,
                "generic",
            )
            
            # Determine chunk type
            chunk_type = self._map_element_type(element.element_type)
            
            # Detect language for syntax highlighting hint
            ext = Path(element.source_path).suffix.lower()
            language = self._detect_language(ext)
            
            chunks.append(UniversalChunk(
                chunk_id=chunk_id,
                domain=Domain.GENERIC,
                path=element.source_path,
                type=chunk_type,
                text=element.content,
                metadata=ChunkMetadata(
                    language=language,
                    line_start=element.line_start,
                    line_end=element.line_end,
                    depth=element.attributes.get("heading_level", 0),
                    extra={
                        k: v for k, v in element.attributes.items()
                        if k != "heading_level"
                    } if element.attributes else {},
                ),
                start_offset=element.start_offset,
                end_offset=element.end_offset,
            ))
        
        return chunks
    
    def _map_element_type(self, element_type: str) -> ChunkType:
        """Map element type to universal chunk type."""
        mapping = {
            "paragraph": ChunkType.PARAGRAPH,
            "section": ChunkType.SECTION,
            "heading": ChunkType.HEADING,
            "config": ChunkType.METADATA,
            "text": ChunkType.TEXT,
        }
        return mapping.get(element_type, ChunkType.TEXT)
    
    def _detect_language(self, ext: str) -> Optional[str]:
        """Detect language/format from file extension."""
        mapping = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".rst": "restructuredtext",
            ".txt": "text",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".toml": "toml",
            ".xml": "xml",
            ".html": "html",
            ".htm": "html",
            ".csv": "csv",
            ".log": "log",
        }
        return mapping.get(ext)
    
    def get_prompt_template(self) -> str:
        """Generic text prompt template."""
        return GENERIC_PROMPT_TEMPLATE
    
    def get_system_prompt(self) -> str:
        """Generic system prompt."""
        return GENERIC_SYSTEM_PROMPT


GENERIC_SYSTEM_PROMPT = """You are a documentation assistant. Your task is to generate 
clear, accurate, and helpful documentation for the given content. Adapt your style 
to the type of content:

- For technical documentation: Be precise and structured
- For prose: Summarize key points and themes
- For data/config: Explain the structure and purpose
- For logs/outputs: Identify patterns and key events

Always focus on helping the reader understand the content quickly."""

GENERIC_PROMPT_TEMPLATE = """Analyze and document the following content:

**File:** {path}
**Type:** {type}

```
{text}
```

{context}

Generate documentation in JSON format:
{{
    "summary": "A one-line description of what this content is about",
    "description": "A detailed explanation covering purpose and key points",
    "key_topics": ["Main topics or concepts covered"],
    "notes": ["Important observations or warnings"]
}}"""
