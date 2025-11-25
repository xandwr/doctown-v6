"""
Utility functions for the Doctown pipeline.
Provides helpers for batching, file I/O, JSON handling, and error management.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar, Union

# Type definitions
PathLike = Union[str, Path]
T = TypeVar("T")

logger = logging.getLogger(__name__)


# ============================================================================
# Batching Utilities
# ============================================================================

def batch_iter(
    items: list[T],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """
    Iterate over items in batches.
    
    Args:
        items: List of items to batch
        batch_size: Maximum items per batch
    
    Yields:
        Lists of items, each up to batch_size in length
    
    Example:
        for batch in batch_iter(texts, batch_size=32):
            embeddings = model.encode(batch)
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def batch_process(
    items: list[T],
    processor: Callable[[list[T]], list],
    batch_size: int,
    show_progress: bool = True,
) -> list:
    """
    Process items in batches with a given processor function.
    
    Args:
        items: List of items to process
        processor: Function that takes a batch and returns results
        batch_size: Maximum items per batch
        show_progress: Whether to log progress
    
    Returns:
        Flattened list of all results
    """
    results = []
    total = len(items)
    
    for i, batch in enumerate(batch_iter(items, batch_size)):
        if show_progress:
            processed = min((i + 1) * batch_size, total)
            logger.info(f"Processing batch {i+1}: {processed}/{total} items")
        
        batch_results = processor(batch)
        results.extend(batch_results)
    
    return results


# ============================================================================
# File I/O Utilities
# ============================================================================

def read_json(path: PathLike) -> Any:
    """
    Read and parse a JSON file.
    
    Args:
        path: Path to the JSON file
    
    Returns:
        Parsed JSON data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file isn't valid JSON
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any,
    path: PathLike,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> Path:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to serialize
        path: Output path
        indent: Indentation level (None for compact)
        ensure_ascii: Whether to escape non-ASCII characters
    
    Returns:
        Path to written file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
    
    return path


def safe_read_text(
    path: PathLike,
    default: str = "",
    encodings: tuple[str, ...] = ("utf-8", "latin-1", "cp1252"),
) -> str:
    """
    Read text file with fallback encodings.
    
    Args:
        path: Path to the text file
        default: Value to return if file can't be read
        encodings: Encodings to try in order
    
    Returns:
        File contents or default value
    """
    path = Path(path)
    
    if not path.exists():
        return default
    
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    logger.warning(f"Could not decode file: {path}")
    return default


# ============================================================================
# Validation Utilities
# ============================================================================

def validate_json_structure(
    data: dict,
    required_keys: list[str],
    name: str = "data",
) -> bool:
    """
    Validate that a dict has required keys.
    
    Args:
        data: Dictionary to validate
        required_keys: List of required key names
        name: Name for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If any required keys are missing
    """
    missing = [key for key in required_keys if key not in data]
    
    if missing:
        raise ValueError(
            f"{name} is missing required keys: {missing}"
        )
    
    return True


def is_text_file(path: PathLike) -> bool:
    """
    Check if a file is likely a text file (vs binary).
    
    Uses file extension heuristics.
    """
    text_extensions = {
        ".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml",
        ".py", ".rs", ".js", ".ts", ".jsx", ".tsx", ".vue", ".svelte",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
        ".java", ".kt", ".scala", ".clj",
        ".go", ".rb", ".php", ".pl", ".pm",
        ".sh", ".bash", ".zsh", ".fish",
        ".sql", ".graphql", ".proto",
        ".xml", ".svg", ".csv", ".tsv",
        ".gitignore", ".dockerignore", ".env",
        ".cfg", ".ini", ".conf", ".config",
        "Makefile", "Dockerfile", "Vagrantfile",
        ".lock", ".sum",
    }
    
    path = Path(path)
    suffix = path.suffix.lower()
    
    # Check extension
    if suffix in text_extensions:
        return True
    
    # Check filename (for files without extensions)
    if path.name in text_extensions:
        return True
    
    # Common config files
    if path.name.startswith(".") and path.suffix == "":
        return True  # Likely a dotfile
    
    return False


# ============================================================================
# Error Handling
# ============================================================================

class DocpackError(Exception):
    """Base exception for Docpack-related errors."""
    pass


class IngestError(DocpackError):
    """Error during repository ingestion."""
    pass


class EmbeddingError(DocpackError):
    """Error during embedding generation."""
    pass


class PackagingError(DocpackError):
    """Error during docpack packaging."""
    pass