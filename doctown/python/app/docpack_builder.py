"""
Docpack Builder - Creates .docpack files from Git repositories.

A .docpack is a self-contained artifact containing:
- manifest.json: Metadata about the source and generation
- filestructure.json: Hierarchical file tree
- chunks.json: Extracted text chunks with stable IDs
- embeddings.bin: Binary embeddings (float32, row-major)
- graph.json: Semantic relationships between files
- documentation.json: Generated documentation (stubs)
- README.md: Human-readable summary
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .binary_io import tensor_to_numpy, write_embeddings_bin
from .embedder import EmbeddingModel

# Configure module logger
logger = logging.getLogger(__name__)

# Version of the docpack format
DOCPACK_VERSION = "6.0.0"


@dataclass
class DocpackConfig:
    """Configuration for docpack generation."""
    
    # Input
    input_source: str  # GitHub URL or local zip path
    branch: str = "main"
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    output_name: Optional[str] = None  # Auto-generated if None
    
    # Chunking
    chunk_size: int = 240
    
    # Embedding
    embedding_model: str = "fast"  # Preset name or model ID
    embedding_batch_size: int = 32
    
    # Paths (auto-resolved)
    rust_binary: Optional[Path] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.rust_binary:
            self.rust_binary = Path(self.rust_binary)


@dataclass
class DocpackResult:
    """Result of docpack generation."""
    
    success: bool
    docpack_path: Optional[Path] = None
    manifest: Optional[dict] = None
    error: Optional[str] = None
    stats: dict = field(default_factory=dict)


class DocpackBuilder:
    """
    Builds .docpack files from Git repositories.
    
    Pipeline:
        1. Run Rust ingest → filestructure.json + chunks.json
        2. Extract texts from chunks → embed via embedder
        3. Write embeddings.bin (contiguous float32)
        4. Build graph.json (semantic similarity edges)
        5. Generate documentation.json (stubs)
        6. Create manifest.json and README.md
        7. Package everything into .docpack (zip)
    
    Example:
        builder = DocpackBuilder(DocpackConfig(
            input_source="https://github.com/owner/repo",
            output_dir="./docpacks"
        ))
        result = builder.build()
        print(f"Created: {result.docpack_path}")
    """
    
    def __init__(self, config: DocpackConfig):
        self.config = config
        self._embedder: Optional[EmbeddingModel] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        
    def build(self) -> DocpackResult:
        """
        Execute the full docpack build pipeline.
        
        Returns:
            DocpackResult with success status and path to generated .docpack
        """
        logger.info(f"Starting docpack build for: {self.config.input_source}")
        
        try:
            # Create temp directory for intermediate files
            self._temp_dir = tempfile.TemporaryDirectory(prefix="docpack_")
            temp_path = Path(self._temp_dir.name)
            
            # Step 1: Run Rust ingest
            logger.info("Step 1/6: Running Rust ingest...")
            ingest_result = self._run_ingest(temp_path)
            
            # Step 2: Generate embeddings
            logger.info("Step 2/6: Generating embeddings...")
            embeddings_meta = self._generate_embeddings(
                ingest_result["chunks"],
                temp_path,
            )
            
            # Step 3: Build semantic graph
            logger.info("Step 3/6: Building semantic graph...")
            graph = self._build_graph(
                ingest_result["chunks"],
                embeddings_meta["embeddings_array"],
            )
            
            # Step 4: Generate documentation stubs
            logger.info("Step 4/6: Generating documentation...")
            docs = self._generate_documentation(
                ingest_result["chunks"],
                ingest_result["filestructure"],
            )
            
            # Step 5: Create manifest
            logger.info("Step 5/6: Creating manifest...")
            manifest = self._create_manifest(
                ingest_result,
                embeddings_meta,
            )
            
            # Step 6: Package into .docpack
            logger.info("Step 6/6: Packaging docpack...")
            docpack_path = self._package_docpack(
                temp_path,
                ingest_result,
                embeddings_meta,
                graph,
                docs,
                manifest,
            )
            
            logger.info(f"✅ Docpack created: {docpack_path}")
            
            return DocpackResult(
                success=True,
                docpack_path=docpack_path,
                manifest=manifest,
                stats={
                    "total_files": manifest["total_files"],
                    "total_chunks": manifest["total_chunks"],
                    "embedding_dims": manifest["embedding_dimensions"],
                },
            )
            
        except Exception as e:
            logger.exception(f"Docpack build failed: {e}")
            return DocpackResult(
                success=False,
                error=str(e),
            )
        finally:
            # Cleanup temp directory
            if self._temp_dir:
                self._temp_dir.cleanup()
                self._temp_dir = None
    
    def _find_rust_binary(self) -> Path:
        """Find the Rust doctown binary."""
        if self.config.rust_binary and self.config.rust_binary.exists():
            return self.config.rust_binary
        
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent.parent.parent / "target" / "debug" / "doctown",
            Path(__file__).parent.parent.parent.parent.parent / "target" / "release" / "doctown",
            Path("target/debug/doctown"),
            Path("target/release/doctown"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                logger.debug(f"Found Rust binary at: {candidate}")
                return candidate
        
        # Try cargo run as fallback
        raise FileNotFoundError(
            "Could not find doctown binary. Run 'cargo build' first, "
            "or specify rust_binary in config."
        )
    
    def _run_ingest(self, temp_path: Path) -> dict:
        """
        Run the Rust ingest tool to get filestructure and chunks.
        
        The Rust binary writes filestructure.json and chunks.json to the output directory.
        
        Returns:
            dict with 'filestructure' and 'chunks' data
        """
        rust_dir = Path(__file__).parent.parent.parent / "rust"
        
        # Create output directory for ingest results
        ingest_output_dir = temp_path / "ingest"
        ingest_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run cargo with chunks enabled, outputting to temp directory
        cmd = [
            "cargo", "run", "--quiet", "--",
            self.config.input_source,
            "--branch", self.config.branch,
            "--chunks",
            "--chunk-size", str(self.config.chunk_size),
            "--output-dir", str(ingest_output_dir),
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=rust_dir,
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Rust ingest failed: {result.stderr}")
        
        # Read the JSON files written by Rust
        filestructure_path = ingest_output_dir / "filestructure.json"
        chunks_path = ingest_output_dir / "chunks.json"
        
        if not filestructure_path.exists():
            raise RuntimeError(f"Rust ingest did not produce filestructure.json at {filestructure_path}")
        
        if not chunks_path.exists():
            raise RuntimeError(f"Rust ingest did not produce chunks.json at {chunks_path}")
        
        with open(filestructure_path, "r") as f:
            filestructure = json.load(f)
        
        with open(chunks_path, "r") as f:
            chunks_data = json.load(f)
        
        # Count files from structure
        total_files = self._count_files(filestructure.get("structure", {}))
        
        return {
            "filestructure": filestructure,
            "chunks": chunks_data.get("chunks", []),
            "total_files": total_files,
            "total_chunks": len(chunks_data.get("chunks", [])),
            "repo_url": filestructure.get("repo_url", self.config.input_source),
            "branch": filestructure.get("branch", self.config.branch),
        }
    
    def _count_files(self, node: dict) -> int:
        """Recursively count files in a filestructure node."""
        if node.get("type") == "file":
            return 1
        
        count = 0
        for child in node.get("children", []):
            count += self._count_files(child)
        return count
    
    def _get_embedder(self) -> EmbeddingModel:
        """Get or create the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(model_id=self.config.embedding_model)
            self._embedder.load_model()
        return self._embedder
    
    def _generate_embeddings(
        self,
        chunks: list[dict],
        temp_path: Path,
    ) -> dict:
        """
        Generate embeddings for all chunks.
        
        Returns:
            dict with embedding metadata and numpy array
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return {
                "n_chunks": 0,
                "n_dims": 0,
                "embeddings_array": np.array([], dtype=np.float32).reshape(0, 0),
            }
        
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embedder = self._get_embedder()
        embeddings_tensor = embedder.embed(
            texts,
            batch_size=self.config.embedding_batch_size,
        )
        
        # Convert to numpy
        embeddings_array = tensor_to_numpy(embeddings_tensor)
        
        n_chunks, n_dims = embeddings_array.shape
        
        logger.info(f"Generated {n_chunks} embeddings of dimension {n_dims}")
        
        return {
            "n_chunks": n_chunks,
            "n_dims": n_dims,
            "embeddings_array": embeddings_array,
            "model_id": embedder.model_id,
            "model_info": embedder.get_model_info(),
        }
    
    def _build_graph(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.7,
        max_edges_per_node: int = 5,
    ) -> dict:
        """
        Build a semantic graph from embeddings.
        
        Creates file-level nodes with edges based on chunk similarity.
        """
        if embeddings.size == 0:
            return {"nodes": [], "edges": []}
        
        # Group chunks by file
        file_chunks: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            file_path = chunk["file_path"]
            if file_path not in file_chunks:
                file_chunks[file_path] = []
            file_chunks[file_path].append(i)
        
        # Create file nodes
        nodes = []
        for file_path, chunk_indices in file_chunks.items():
            chunk_ids = [chunks[i]["chunk_id"] for i in chunk_indices]
            nodes.append({
                "id": f"file:{file_path}",
                "type": "file",
                "name": Path(file_path).name,
                "path": file_path,
                "chunks": chunk_ids,
                "pagerank": 0.0,  # Placeholder - computed later
            })
        
        # Compute file-level embeddings (mean of chunk embeddings)
        file_embeddings = {}
        for file_path, chunk_indices in file_chunks.items():
            file_emb = embeddings[chunk_indices].mean(axis=0)
            # Normalize
            norm = np.linalg.norm(file_emb)
            if norm > 0:
                file_emb = file_emb / norm
            file_embeddings[file_path] = file_emb
        
        # Compute pairwise similarity and create edges
        edges = []
        file_paths = list(file_embeddings.keys())
        
        for i, path_a in enumerate(file_paths):
            emb_a = file_embeddings[path_a]
            similarities = []
            
            for j, path_b in enumerate(file_paths):
                if i >= j:  # Skip self and duplicates
                    continue
                
                emb_b = file_embeddings[path_b]
                similarity = float(np.dot(emb_a, emb_b))
                
                if similarity >= similarity_threshold:
                    similarities.append((path_b, similarity))
            
            # Keep top N edges per file
            similarities.sort(key=lambda x: x[1], reverse=True)
            for path_b, score in similarities[:max_edges_per_node]:
                edges.append({
                    "from": f"file:{path_a}",
                    "to": f"file:{path_b}",
                    "type": "semantic_similarity",
                    "score": round(score, 4),
                })
        
        # Simple PageRank approximation
        edge_count = {}
        for edge in edges:
            edge_count[edge["from"]] = edge_count.get(edge["from"], 0) + 1
            edge_count[edge["to"]] = edge_count.get(edge["to"], 0) + 1
        
        total_edges = sum(edge_count.values()) or 1
        for node in nodes:
            node["pagerank"] = round(
                edge_count.get(node["id"], 0) / total_edges,
                4,
            )
        
        logger.info(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return {
            "nodes": nodes,
            "edges": edges,
        }
    
    def _generate_documentation(
        self,
        chunks: list[dict],
        filestructure: dict,
    ) -> dict:
        """
        Generate rule-based documentation (stubs for now).
        
        This is where LLM-based generation would go in the future.
        """
        # Group first chunks by file as summaries
        summaries = []
        seen_files = set()
        
        for chunk in chunks:
            file_path = chunk["file_path"]
            if file_path in seen_files:
                continue
            seen_files.add(file_path)
            
            # Use first chunk as a basic summary
            text_preview = chunk["text"][:200].replace('\n', ' ').strip()
            if len(chunk["text"]) > 200:
                text_preview += "..."
            
            summaries.append({
                "symbol_id": f"file:{file_path}",
                "summary": f"File: {file_path}",
                "details": text_preview,
                "related": [],
            })
        
        return {
            "summaries": summaries,
            "architecture_overview": "Architecture documentation not yet generated.",
            "highlights": [],
        }
    
    def _create_manifest(
        self,
        ingest_result: dict,
        embeddings_meta: dict,
    ) -> dict:
        """Create the manifest.json content."""
        return {
            "docpack_version": DOCPACK_VERSION,
            "source_repo": ingest_result["repo_url"],
            "branch": ingest_result["branch"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_files": ingest_result["total_files"],
            "total_chunks": ingest_result["total_chunks"],
            "embedding_dimensions": embeddings_meta["n_dims"],
            "generator": {
                "builder_version": "v6",
                "embedder": embeddings_meta.get("model_id", "unknown"),
                "gpu_used": embeddings_meta.get("model_info", {}).get("device", "").startswith("cuda"),
            },
        }
    
    def _generate_readme(self, manifest: dict) -> str:
        """Generate README.md content for the docpack."""
        return f"""# Project Documentation (Generated by Doctown v6)

- **Source:** {manifest['source_repo']}
- **Branch:** {manifest['branch']}
- **Generated:** {manifest['generated_at'][:10]}
- **Files:** {manifest['total_files']}
- **Chunks:** {manifest['total_chunks']}
- **Embedding dimensions:** {manifest['embedding_dimensions']}

## About This Docpack

This `.docpack` file contains a complete knowledge representation of the source repository,
including file structure, text chunks, semantic embeddings, and relationship graphs.

## Contents

| File | Description |
|------|-------------|
| `manifest.json` | Metadata about the source and generation |
| `filestructure.json` | Hierarchical file tree |
| `chunks.json` | Extracted text chunks with stable IDs |
| `embeddings.bin` | Binary embeddings (float32, row-major) |
| `graph.json` | Semantic relationships between files |
| `documentation.json` | Generated documentation |
| `README.md` | This file |

## Usage

Docpacks can be loaded by tools that understand the format. The embeddings can be used
for semantic search, the graph for understanding relationships, and the documentation
for quick project overview.

---
*Generated by [Doctown](https://github.com/xandwr/doctown-v6)*
"""
    
    def _package_docpack(
        self,
        temp_path: Path,
        ingest_result: dict,
        embeddings_meta: dict,
        graph: dict,
        docs: dict,
        manifest: dict,
    ) -> Path:
        """Package all components into a .docpack file."""
        # Determine output name
        if self.config.output_name:
            name = self.config.output_name
        else:
            # Extract repo name from URL
            repo_url = ingest_result["repo_url"]
            if "/" in repo_url:
                name = repo_url.rstrip("/").split("/")[-1]
                name = name.replace(".git", "")
            else:
                name = "project"
        
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the docpack file
        docpack_path = self.config.output_dir / f"{name}.docpack"
        
        # Write intermediate files to temp
        (temp_path / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        (temp_path / "filestructure.json").write_text(
            json.dumps(ingest_result["filestructure"], indent=2)
        )
        (temp_path / "chunks.json").write_text(
            json.dumps(ingest_result["chunks"], indent=2)
        )
        (temp_path / "graph.json").write_text(
            json.dumps(graph, indent=2)
        )
        (temp_path / "documentation.json").write_text(
            json.dumps(docs, indent=2)
        )
        (temp_path / "README.md").write_text(
            self._generate_readme(manifest)
        )
        
        # Write embeddings binary
        if embeddings_meta["n_chunks"] > 0:
            write_embeddings_bin(
                embeddings_meta["embeddings_array"],
                temp_path / "embeddings.bin",
            )
        else:
            # Write empty file
            (temp_path / "embeddings.bin").write_bytes(b"")
        
        # Create zip (docpack is just a renamed zip)
        with zipfile.ZipFile(docpack_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_name in [
                "manifest.json",
                "filestructure.json",
                "chunks.json",
                "embeddings.bin",
                "graph.json",
                "documentation.json",
                "README.md",
            ]:
                file_path = temp_path / file_name
                if file_path.exists():
                    zf.write(file_path, file_name)
        
        return docpack_path


def build_docpack(
    input_source: str,
    output_dir: str = "./output",
    branch: str = "main",
    **kwargs,
) -> DocpackResult:
    """
    Convenience function to build a docpack.
    
    Args:
        input_source: GitHub URL or local zip path
        output_dir: Directory to write the .docpack file
        branch: Branch to use (for GitHub URLs)
        **kwargs: Additional DocpackConfig options
    
    Returns:
        DocpackResult with success status and path
    
    Example:
        result = build_docpack("https://github.com/owner/repo")
        if result.success:
            print(f"Created: {result.docpack_path}")
    """
    config = DocpackConfig(
        input_source=input_source,
        output_dir=Path(output_dir),
        branch=branch,
        **kwargs,
    )
    builder = DocpackBuilder(config)
    return builder.build()
