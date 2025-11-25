"""
Code Domain Ingestor - Handles source code repositories.

This ingestor detects code projects and uses the Rust AST parser
for extraction, then normalizes the output to UniversalChunks.

Supports: Rust, Python, JavaScript/TypeScript, Go, Java, C/C++, etc.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from .base import (
    DomainIngestor,
    Domain,
    ChunkType,
    ChunkMetadata,
    UniversalChunk,
    RawElement,
)

logger = logging.getLogger(__name__)

# File extensions that indicate code
CODE_EXTENSIONS = {
    # Rust
    ".rs", ".toml",
    # Python
    ".py", ".pyx", ".pyi",
    # JavaScript/TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Go
    ".go", ".mod",
    # Java/Kotlin
    ".java", ".kt", ".kts",
    # C/C++
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    # C#
    ".cs",
    # Ruby
    ".rb",
    # PHP
    ".php",
    # Swift
    ".swift",
    # Scala
    ".scala",
    # Shell
    ".sh", ".bash", ".zsh",
    # Config files that indicate code projects
    ".json", ".yaml", ".yml",
}

# Project marker files (strong indicators of code)
PROJECT_MARKERS = {
    # Rust
    "Cargo.toml", "Cargo.lock",
    # Python
    "setup.py", "pyproject.toml", "requirements.txt", "Pipfile",
    # Node.js
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    # Go
    "go.mod", "go.sum",
    # Java/Gradle/Maven
    "pom.xml", "build.gradle", "build.gradle.kts",
    # .NET
    "*.csproj", "*.sln",
    # Makefile-based
    "Makefile", "CMakeLists.txt",
    # Generic
    ".gitignore", ".editorconfig",
}

# Map file extensions to languages
EXTENSION_TO_LANGUAGE = {
    ".rs": "rust",
    ".py": "python",
    ".pyx": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}


class CodeIngestor(DomainIngestor):
    """
    Ingestor for source code repositories.
    
    Uses the Rust doctown binary for AST-based extraction,
    then normalizes output to UniversalChunks.
    """
    
    name = "code"
    domain = Domain.CODE
    priority = 100  # High priority - check code first
    
    def __init__(
        self,
        rust_binary_path: Optional[Path] = None,
        chunk_size: int = 240,
    ):
        self.rust_binary_path = rust_binary_path
        self.chunk_size = chunk_size
        self._rust_dir: Optional[Path] = None
    
    def _find_rust_dir(self) -> Path:
        """Find the Rust project directory."""
        if self._rust_dir is not None:
            return self._rust_dir
        
        # Try relative to this file
        candidates = [
            Path(__file__).parent.parent.parent / "rust",
            Path(__file__).parent.parent.parent.parent / "rust",
            Path("doctown/rust"),
        ]
        
        for candidate in candidates:
            if (candidate / "Cargo.toml").exists():
                self._rust_dir = candidate.resolve()
                return self._rust_dir
        
        raise FileNotFoundError(
            "Could not find Rust project directory. "
            "Expected to find doctown/rust/Cargo.toml"
        )
    
    def detect(self, files: dict[str, bytes]) -> bool:
        """
        Detect if this looks like a code repository.
        
        Checks for:
        1. Project marker files (Cargo.toml, package.json, etc.)
        2. High ratio of code file extensions
        """
        file_paths = set(files.keys())
        
        # Check for project markers
        for marker in PROJECT_MARKERS:
            if "*" in marker:
                # Glob pattern
                pattern = marker.replace("*", "")
                if any(p.endswith(pattern) for p in file_paths):
                    logger.debug(f"CodeIngestor: detected project marker pattern {marker}")
                    return True
            elif marker in file_paths or any(p.endswith(f"/{marker}") for p in file_paths):
                logger.debug(f"CodeIngestor: detected project marker {marker}")
                return True
        
        # Check extension ratio
        code_files = sum(
            1 for p in file_paths 
            if Path(p).suffix.lower() in CODE_EXTENSIONS
        )
        total_files = len(file_paths)
        
        if total_files > 0 and code_files / total_files > 0.3:
            logger.debug(f"CodeIngestor: {code_files}/{total_files} files are code")
            return True
        
        return False
    
    def extract_elements(self, files: dict[str, bytes]) -> list[RawElement]:
        """
        Extract code elements using the Rust AST parser.
        
        This writes files to a temp directory, creates a zip, and runs the Rust binary.
        """
        import zipfile
        
        # Write files to temp directory
        with tempfile.TemporaryDirectory(prefix="doctown_code_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a directory structure
            repo_dir = temp_path / "repo"
            repo_dir.mkdir()
            
            # Write all files
            for file_path, content in files.items():
                full_path = repo_dir / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_bytes(content)
            
            # Create a zip file (Rust tool expects a zip)
            zip_path = temp_path / "repo.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path, content in files.items():
                    # Add with "repo/" prefix to match GitHub zip structure
                    zipf.writestr(f"repo/{file_path}", content)
            
            # Run Rust ingest with the zip file
            output_dir = temp_path / "_output"
            output_dir.mkdir()
            
            rust_dir = self._find_rust_dir()
            cmd = [
                "cargo", "run", "--quiet", "--",
                str(zip_path),  # Pass zip file path, not directory
                "--chunks",
                "--chunk-size", str(self.chunk_size),
                "--output-dir", str(output_dir),
            ]
            
            logger.debug(f"Running Rust ingest: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=rust_dir,
                capture_output=True,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning(f"Rust ingest failed: {result.stderr}")
                # Fall back to simple text extraction
                return self._fallback_extract(files)
            
            # Read chunks from Rust output
            chunks_path = output_dir / "chunks.json"
            if not chunks_path.exists():
                logger.warning("Rust ingest did not produce chunks.json")
                return self._fallback_extract(files)
            
            with open(chunks_path) as f:
                chunks_data = json.load(f)
            
            # Convert to RawElements
            elements = []
            for chunk in chunks_data.get("chunks", []):
                # Include symbol_metadata if present
                attributes = {
                    "chunk_id": chunk.get("chunk_id"),
                }
                if "symbol_metadata" in chunk:
                    attributes["symbol_metadata"] = chunk["symbol_metadata"]
                
                elements.append(RawElement(
                    source_path=chunk["file_path"],
                    element_type="code_chunk",
                    content=chunk["text"],
                    start_offset=chunk.get("start", 0),
                    end_offset=chunk.get("end", 0),
                    attributes=attributes,
                ))
            
            return elements
    
    def _fallback_extract(self, files: dict[str, bytes]) -> list[RawElement]:
        """
        Fallback extraction when Rust parser isn't available.
        
        Does simple line-based chunking of code files.
        """
        elements = []
        
        for file_path, content in files.items():
            ext = Path(file_path).suffix.lower()
            if ext not in CODE_EXTENSIONS:
                continue
            
            try:
                text = content.decode("utf-8", errors="replace")
            except Exception:
                continue
            
            # Simple chunking by character count
            chunk_size = self.chunk_size
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i + chunk_size]
                elements.append(RawElement(
                    source_path=file_path,
                    element_type="code_chunk",
                    content=chunk_text,
                    start_offset=i,
                    end_offset=min(i + chunk_size, len(text)),
                    attributes={"language": EXTENSION_TO_LANGUAGE.get(ext)},
                ))
        
        return elements
    
    def to_chunks(self, elements: list[RawElement]) -> list[UniversalChunk]:
        """
        Convert raw code elements to universal chunks.
        """
        chunks = []
        
        for element in elements:
            ext = Path(element.source_path).suffix.lower()
            language = element.attributes.get("language") or EXTENSION_TO_LANGUAGE.get(ext)
            
            # Check if we have symbol metadata from Rust parser
            symbol_meta = element.attributes.get("symbol_metadata")
            
            # Determine chunk type
            if symbol_meta and "symbol_kind" in symbol_meta:
                chunk_type = self._symbol_kind_to_chunk_type(symbol_meta["symbol_kind"])
            else:
                chunk_type = self._infer_chunk_type(element.content, language)
            
            chunk_id = element.attributes.get("chunk_id") or UniversalChunk.generate_id(
                element.source_path,
                element.start_offset,
                "code",
            )
            
            # Build metadata from symbol info if available, otherwise infer
            if symbol_meta:
                metadata = self._build_metadata_from_symbol(symbol_meta, language)
            else:
                # Fallback: basic metadata
                line_start = element.content[:element.start_offset].count("\n") + 1 if element.start_offset > 0 else 1
                line_count = element.content.count("\n")
                metadata = ChunkMetadata(
                    language=language,
                    line_start=line_start,
                    line_end=line_start + line_count,
                    semantic_role=self._infer_semantic_role(element.content, language),
                )
            
            chunks.append(UniversalChunk(
                chunk_id=chunk_id,
                domain=Domain.CODE,
                path=element.source_path,
                type=chunk_type,
                text=element.content,
                metadata=metadata,
                start_offset=element.start_offset,
                end_offset=element.end_offset,
            ))
        
        return chunks
    
    def _symbol_kind_to_chunk_type(self, symbol_kind: str) -> ChunkType:
        """Map symbol kind to ChunkType."""
        mapping = {
            "function": ChunkType.FUNCTION,
            "method": ChunkType.FUNCTION,
            "class": ChunkType.CLASS,
            "struct": ChunkType.CLASS,
            "enum": ChunkType.CLASS,
            "trait": ChunkType.CLASS,
            "interface": ChunkType.CLASS,
            "module": ChunkType.MODULE,
            "const": ChunkType.VARIABLE,
            "variable": ChunkType.VARIABLE,
        }
        return mapping.get(symbol_kind.lower(), ChunkType.TEXT)
    
    def _build_metadata_from_symbol(self, symbol_meta: dict, language: str) -> ChunkMetadata:
        """Build ChunkMetadata from symbol metadata extracted by Rust parser."""
        # Extract signature info
        params = None
        if "signature" in symbol_meta and symbol_meta["signature"]:
            sig = symbol_meta["signature"]
            if "parameters" in sig:
                params = sig["parameters"]
        
        return ChunkMetadata(
            language=language,
            line_start=symbol_meta.get("line_start"),
            line_end=symbol_meta.get("line_end"),
            symbol_name=symbol_meta.get("symbol_name"),
            qualified_name=symbol_meta.get("qualified_name"),
            symbol_kind=symbol_meta.get("symbol_kind"),
            visibility=symbol_meta.get("visibility"),
            parent_name=symbol_meta.get("parent"),
            parameters=params,
            return_type=symbol_meta.get("signature", {}).get("return_type") if "signature" in symbol_meta else None,
            modifiers=symbol_meta.get("signature", {}).get("modifiers") if "signature" in symbol_meta else None,
            doc_comment=symbol_meta.get("doc_comment"),
            semantic_role=symbol_meta.get("symbol_kind"),  # Use symbol_kind as semantic_role
        )
    
    def _infer_chunk_type(self, content: str, language: Optional[str]) -> ChunkType:
        """
        Infer the chunk type from content.
        
        This is a heuristic - the Rust AST parser provides more accurate info.
        """
        content_lower = content.lower().strip()
        
        # Function patterns
        if any(pattern in content_lower for pattern in [
            "def ", "fn ", "func ", "function ", "public static",
            "private static", "async def", "async fn"
        ]):
            return ChunkType.FUNCTION
        
        # Class patterns
        if any(pattern in content_lower for pattern in [
            "class ", "struct ", "interface ", "enum ", "trait "
        ]):
            return ChunkType.CLASS
        
        # Import patterns
        if any(pattern in content_lower for pattern in [
            "import ", "from ", "use ", "require(", "#include", "using "
        ]):
            return ChunkType.IMPORT
        
        # Comment patterns
        if content_lower.startswith(("//", "#", "/*", "'''", '"""')):
            return ChunkType.COMMENT
        
        return ChunkType.TEXT
    
    def _infer_semantic_role(self, content: str, language: Optional[str]) -> Optional[str]:
        """
        Infer the semantic role of code.
        """
        content_lower = content.lower()
        
        # Constructor patterns
        if any(p in content_lower for p in ["__init__", "new(", "constructor", "::new"]):
            return "constructor"
        
        # Test patterns
        if any(p in content_lower for p in ["#[test]", "def test_", "@test", "it(", "describe("]):
            return "test"
        
        # Error handling
        if any(p in content_lower for p in ["try:", "catch", "except", "error", "panic!"]):
            return "error_handling"
        
        # Main/entry point
        if any(p in content_lower for p in ["fn main(", "def main(", "if __name__", "public static void main"]):
            return "entry_point"
        
        return None
    
    def get_prompt_template(self) -> str:
        """Code-specific prompt template."""
        return CODE_PROMPT_TEMPLATE
    
    def get_system_prompt(self) -> str:
        """Code-specific system prompt."""
        return CODE_SYSTEM_PROMPT


CODE_SYSTEM_PROMPT = """You are an expert code documentation assistant. Your task is to generate 
clear, accurate, and helpful documentation for source code. Focus on:

1. WHAT the code does (functionality)
2. WHY it exists (purpose/motivation)
3. HOW to use it (API, parameters, return values)
4. WHEN to use it (use cases, constraints)

Be precise about types, parameters, and return values. Reference related code when relevant.
Write documentation that would help a developer understand and use this code correctly."""

CODE_PROMPT_TEMPLATE = """Generate documentation for the following code:

**File:** {path}
**Language:** {language}
**Type:** {type}

```{language}
{text}
```

{context}

Generate comprehensive documentation in JSON format:
{{
    "summary": "A one-line description of what this code does",
    "description": "A detailed explanation (2-3 paragraphs) covering purpose, behavior, and implementation details",
    "parameters": [
        {{"name": "param_name", "type": "type", "description": "what it does"}}
    ],
    "returns": {{"type": "return_type", "description": "what is returned"}},
    "examples": ["// Example usage..."],
    "notes": ["Important considerations, edge cases, or warnings"],
    "see_also": ["Related functions, classes, or files"]
}}

Only include fields that are applicable. For example, don't include "parameters" for a class definition."""
