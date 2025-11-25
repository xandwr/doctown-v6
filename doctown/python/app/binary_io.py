"""
Binary I/O utilities for docpack format.
Handles reading/writing embeddings in efficient binary format.
"""
from pathlib import Path
from typing import Union, Tuple
import struct
import numpy as np

# Type alias for path-like inputs
PathLike = Union[str, Path]


def write_embeddings_bin(
    embeddings: np.ndarray,
    output_path: PathLike,
) -> dict:
    """
    Write embeddings to binary file in row-major float32 format.
    
    Format:
        - Contiguous float32 values, row-major order
        - N rows (number of chunks), D columns (embedding dimensions)
        - Total size: N * D * 4 bytes
    
    Args:
        embeddings: NumPy array of shape (N, D) with float32 dtype
        output_path: Path to write the binary file
    
    Returns:
        dict with metadata about the written file
    
    Raises:
        ValueError: If embeddings array has invalid shape or dtype
    """
    output_path = Path(output_path)
    
    # Validate input
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D array, got {embeddings.ndim}D")
    
    n_chunks, n_dims = embeddings.shape
    
    # Ensure float32 dtype for consistent binary format
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    
    # Ensure contiguous memory layout (row-major/C-order)
    if not embeddings.flags['C_CONTIGUOUS']:
        embeddings = np.ascontiguousarray(embeddings)
    
    # Write raw bytes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings.tofile(output_path)
    
    file_size = output_path.stat().st_size
    expected_size = n_chunks * n_dims * 4  # 4 bytes per float32
    
    if file_size != expected_size:
        raise RuntimeError(
            f"File size mismatch: wrote {file_size} bytes, "
            f"expected {expected_size} bytes"
        )
    
    return {
        "path": str(output_path),
        "n_chunks": n_chunks,
        "n_dims": n_dims,
        "dtype": "float32",
        "size_bytes": file_size,
    }


def read_embeddings_bin(
    input_path: PathLike,
    n_dims: int,
    n_chunks: int = -1,
) -> np.ndarray:
    """
    Read embeddings from binary file.
    
    Args:
        input_path: Path to the binary file
        n_dims: Number of embedding dimensions (required to reshape)
        n_chunks: Number of chunks (-1 to infer from file size)
    
    Returns:
        NumPy array of shape (N, D) with float32 dtype
    
    Raises:
        ValueError: If file size doesn't match expected dimensions
    """
    input_path = Path(input_path)
    
    # Read raw bytes
    embeddings = np.fromfile(input_path, dtype=np.float32)
    
    total_floats = len(embeddings)
    
    if total_floats % n_dims != 0:
        raise ValueError(
            f"File contains {total_floats} floats, which is not "
            f"divisible by n_dims={n_dims}"
        )
    
    inferred_chunks = total_floats // n_dims
    
    if n_chunks > 0 and inferred_chunks != n_chunks:
        raise ValueError(
            f"Expected {n_chunks} chunks, but file contains {inferred_chunks}"
        )
    
    return embeddings.reshape(inferred_chunks, n_dims)


def tensor_to_numpy(tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to NumPy array, handling device transfer.
    
    Args:
        tensor: PyTorch tensor (can be on CPU or CUDA)
    
    Returns:
        NumPy array with float32 dtype
    """
    # Import torch only when needed (might not be installed)
    import torch
    
    if isinstance(tensor, np.ndarray):
        return tensor.astype(np.float32)
    
    if isinstance(tensor, torch.Tensor):
        # Move to CPU if on GPU, then convert
        return tensor.detach().cpu().numpy().astype(np.float32)
    
    # Try to convert anything else via numpy
    return np.array(tensor, dtype=np.float32)
