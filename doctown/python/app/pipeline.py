"""
Unified Documentation Pipeline - The orchestrator for the document normalization engine.

This module ties together:
1. Domain detection & ingestion (pluggable)
2. Embedding generation (universal)
3. Graph building (universal)
4. Documentation generation (domain templates + LLM)
5. Docpack packaging (universal)

The key architectural principle:
    Raw Input → Domain Parser → Standardized Chunk Format → Embeddings → Graph → Docs → Docpack
                 (pluggable)      (universal IR)            (universal)  (universal)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Callable

import numpy as np

from .binary_io import tensor_to_numpy, write_embeddings_bin
from .embedder import EmbeddingModel
from .ingestors import (
    DomainIngestor,
    UniversalChunk,
    select_ingestor,
    get_registry,
)
from .ingestors.base import Domain
from .llm import OpenAIDocGenerator, create_doc_generator, BatchDocumentationResult

logger = logging.getLogger(__name__)

# Version of the docpack format
DOCPACK_VERSION = "6.1.0"  # Bumped for universal chunk schema


@dataclass
class PipelineConfig:
    """Configuration for the documentation pipeline."""
    
    # Input
    input_source: str  # GitHub URL, local path, or zip path
    branch: str = "main"
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    output_name: Optional[str] = None
    
    # Ingestion
    ingestor_name: Optional[str] = None  # Auto-detect if None
    chunk_size: int = 240
    
    # Embedding
    embedding_model: str = "fast"
    embedding_batch_size: int = 32
    
    # Documentation
    use_llm: bool = True  # Use OpenAI for docs if available
    llm_model: Optional[str] = None  # Uses OPENAI_MODEL env var if None
    llm_max_concurrent: int = 10
    llm_max_chunks: Optional[int] = None  # Limit chunks for LLM (cost control)
    
    # Graph
    similarity_threshold: float = 0.7
    max_edges_per_node: int = 5
    
    # Rust binary (for code ingestor)
    rust_binary: Optional[Path] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        if self.rust_binary:
            self.rust_binary = Path(self.rust_binary)


@dataclass
class PipelineResult:
    """Result of running the pipeline."""
    
    success: bool
    docpack_path: Optional[Path] = None
    manifest: Optional[dict] = None
    error: Optional[str] = None
    stats: dict = field(default_factory=dict)


class DocumentationPipeline:
    """
    Unified documentation pipeline with pluggable domain ingestors.
    
    This is the "compiler" that transforms any input into a standardized docpack.
    
    Pipeline stages:
        1. Load files from source (GitHub, local, zip)
        2. Detect domain and select ingestor (pluggable)
        3. Extract universal chunks (IR)
        4. Generate embeddings
        5. Build semantic graph
        6. Generate documentation (rule-based or LLM)
        7. Package into .docpack
    
    Example:
        pipeline = DocumentationPipeline(PipelineConfig(
            input_source="https://github.com/owner/repo",
            use_llm=True,
        ))
        result = await pipeline.run_async()
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._embedder: Optional[EmbeddingModel] = None
        self._doc_generator: Optional[OpenAIDocGenerator] = None
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
    
    def run(self) -> PipelineResult:
        """
        Run the pipeline synchronously.
        
        For better performance with LLM docs, use run_async() instead.
        """
        return asyncio.run(self.run_async())
    
    async def run_async(self) -> PipelineResult:
        """
        Run the pipeline asynchronously.
        
        This allows concurrent LLM API calls for faster documentation generation.
        """
        logger.info(f"Starting pipeline for: {self.config.input_source}")
        
        try:
            # Create temp directory
            self._temp_dir = tempfile.TemporaryDirectory(prefix="docpack_pipeline_")
            temp_path = Path(self._temp_dir.name)
            
            # Step 1: Load files
            logger.info("Step 1/7: Loading files...")
            files, source_info = await self._load_files(temp_path)
            
            # Step 2: Detect domain and select ingestor
            logger.info("Step 2/7: Detecting domain...")
            ingestor = self._select_ingestor(files)
            logger.info(f"  Using ingestor: {ingestor.name} (domain={ingestor.domain.value})")
            
            # Step 3: Extract universal chunks
            logger.info("Step 3/7: Extracting chunks...")
            chunks = ingestor.ingest(files)
            logger.info(f"  Extracted {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            logger.info("Step 4/7: Generating embeddings...")
            embeddings_meta = self._generate_embeddings(chunks)
            
            # Step 5: Build semantic graph
            logger.info("Step 5/7: Building semantic graph...")
            graph = self._build_graph(chunks, embeddings_meta["embeddings_array"])
            
            # Step 6: Generate documentation
            logger.info("Step 6/7: Generating documentation...")
            docs = await self._generate_documentation(chunks, ingestor)
            
            # Step 7: Package into docpack
            logger.info("Step 7/7: Packaging docpack...")
            manifest = self._create_manifest(source_info, chunks, embeddings_meta, ingestor)
            docpack_path = self._package_docpack(
                temp_path, source_info, chunks, embeddings_meta, graph, docs, manifest
            )
            
            logger.info(f"✅ Pipeline complete: {docpack_path}")
            
            return PipelineResult(
                success=True,
                docpack_path=docpack_path,
                manifest=manifest,
                stats={
                    "domain": ingestor.domain.value,
                    "total_files": len(files),
                    "total_chunks": len(chunks),
                    "embedding_dims": embeddings_meta["n_dims"],
                    "llm_docs_generated": docs.get("generation_stats", {}).get("successful", 0),
                },
            )
            
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return PipelineResult(success=False, error=str(e))
        finally:
            if self._temp_dir:
                self._temp_dir.cleanup()
                self._temp_dir = None
    
    async def _load_files(self, temp_path: Path) -> tuple[dict[str, bytes], dict]:
        """
        Load files from the input source.
        
        Supports:
        - GitHub URLs (clones repo)
        - Local directories (reads files)
        - Zip files (extracts)
        
        Returns:
            Tuple of (files dict, source info dict)
        """
        source = self.config.input_source
        
        if source.startswith(("http://", "https://")):
            # GitHub URL - use Rust to clone
            return await self._load_from_github(source, temp_path)
        elif source.endswith(".zip"):
            return self._load_from_zip(source)
        else:
            return self._load_from_local(source)
    
    async def _load_from_github(
        self, url: str, temp_path: Path
    ) -> tuple[dict[str, bytes], dict]:
        """Clone a GitHub repo and load its files."""
        # Use Rust binary to fetch repo (it handles the clone)
        # Path: app/pipeline.py -> app -> python -> doctown -> doctown/rust
        rust_dir = Path(__file__).parent.parent.parent / "rust"
        output_dir = temp_path / "repo"
        output_dir.mkdir()
        
        cmd = [
            "cargo", "run", "--quiet", "--",
            url,
            "--branch", self.config.branch,
            "--output-dir", str(output_dir),
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=rust_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to fetch repo: {result.stderr}")
        
        # Read the filestructure to get the actual repo path
        filestructure_path = output_dir / "filestructure.json"
        if filestructure_path.exists():
            with open(filestructure_path) as f:
                fs = json.load(f)
            repo_path = fs.get("repo_path", str(output_dir))
        else:
            repo_path = str(output_dir)
        
        files = self._read_directory(Path(repo_path))
        
        return files, {
            "type": "github",
            "url": url,
            "branch": self.config.branch,
            "total_files": len(files),
        }
    
    def _load_from_zip(self, zip_path: str) -> tuple[dict[str, bytes], dict]:
        """Extract and load files from a zip archive."""
        files = {}
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.endswith("/"):  # Skip directories
                    continue
                try:
                    files[name] = zf.read(name)
                except Exception as e:
                    logger.warning(f"Failed to read {name} from zip: {e}")
        
        return files, {
            "type": "zip",
            "path": zip_path,
            "total_files": len(files),
        }
    
    def _load_from_local(self, path: str) -> tuple[dict[str, bytes], dict]:
        """Load files from a local directory."""
        local_path = Path(path)
        if not local_path.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        if local_path.is_file():
            # Single file
            files = {local_path.name: local_path.read_bytes()}
        else:
            files = self._read_directory(local_path)
        
        return files, {
            "type": "local",
            "path": str(local_path.absolute()),
            "total_files": len(files),
        }
    
    def _read_directory(self, root: Path) -> dict[str, bytes]:
        """Recursively read all files in a directory."""
        files = {}
        
        # Patterns to skip
        skip_patterns = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            "target", "build", "dist", ".idea", ".vscode",
        }
        
        for path in root.rglob("*"):
            if path.is_dir():
                continue
            
            # Skip if in a skipped directory
            if any(part in skip_patterns for part in path.parts):
                continue
            
            # Skip large files (>1MB)
            try:
                if path.stat().st_size > 1_000_000:
                    continue
                
                rel_path = str(path.relative_to(root))
                files[rel_path] = path.read_bytes()
            except Exception as e:
                logger.debug(f"Skipping {path}: {e}")
        
        return files
    
    def _select_ingestor(self, files: dict[str, bytes]) -> DomainIngestor:
        """Select the appropriate ingestor for the files."""
        if self.config.ingestor_name:
            # Use explicitly specified ingestor
            registry = get_registry()
            ingestor = registry.get(self.config.ingestor_name)
            if ingestor:
                return ingestor
            logger.warning(f"Ingestor '{self.config.ingestor_name}' not found, auto-detecting")
        
        return select_ingestor(files)
    
    def _get_embedder(self) -> EmbeddingModel:
        """Get or create the embedding model."""
        if self._embedder is None:
            self._embedder = EmbeddingModel(model_id=self.config.embedding_model)
            self._embedder.load_model()
        return self._embedder
    
    def _generate_embeddings(self, chunks: list[UniversalChunk]) -> dict:
        """Generate embeddings for all chunks."""
        if not chunks:
            return {
                "n_chunks": 0,
                "n_dims": 0,
                "embeddings_array": np.array([], dtype=np.float32).reshape(0, 0),
            }
        
        texts = [chunk.text for chunk in chunks]
        embedder = self._get_embedder()
        
        embeddings_tensor = embedder.embed(texts, batch_size=self.config.embedding_batch_size)
        embeddings_array = tensor_to_numpy(embeddings_tensor)
        
        n_chunks, n_dims = embeddings_array.shape
        logger.info(f"  Generated {n_chunks} embeddings of dimension {n_dims}")
        
        return {
            "n_chunks": n_chunks,
            "n_dims": n_dims,
            "embeddings_array": embeddings_array,
            "model_id": embedder.model_id,
            "model_info": embedder.get_model_info(),
        }
    
    def _build_graph(
        self,
        chunks: list[UniversalChunk],
        embeddings: np.ndarray,
    ) -> dict:
        """Build a semantic graph from embeddings."""
        if embeddings.size == 0:
            return {"nodes": [], "edges": []}
        
        # Group chunks by file/path
        path_chunks: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            path = chunk.path
            if path not in path_chunks:
                path_chunks[path] = []
            path_chunks[path].append(i)
        
        # Create file nodes
        nodes = []
        for path, chunk_indices in path_chunks.items():
            chunk_ids = [chunks[i].chunk_id for i in chunk_indices]
            nodes.append({
                "id": f"file:{path}",
                "type": "file",
                "name": Path(path).name,
                "path": path,
                "chunks": chunk_ids,
                "domain": chunks[chunk_indices[0]].domain.value,
                "pagerank": 0.0,
            })
        
        # Compute path-level embeddings (mean of chunk embeddings)
        path_embeddings = {}
        for path, chunk_indices in path_chunks.items():
            path_emb = embeddings[chunk_indices].mean(axis=0)
            norm = np.linalg.norm(path_emb)
            if norm > 0:
                path_emb = path_emb / norm
            path_embeddings[path] = path_emb
        
        # Compute similarity edges
        edges = []
        paths = list(path_embeddings.keys())
        
        for i, path_a in enumerate(paths):
            emb_a = path_embeddings[path_a]
            similarities = []
            
            for j, path_b in enumerate(paths):
                if i >= j:
                    continue
                
                emb_b = path_embeddings[path_b]
                similarity = float(np.dot(emb_a, emb_b))
                
                if similarity >= self.config.similarity_threshold:
                    similarities.append((path_b, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            for path_b, score in similarities[:self.config.max_edges_per_node]:
                edges.append({
                    "from": f"file:{path_a}",
                    "to": f"file:{path_b}",
                    "type": "semantic_similarity",
                    "score": round(score, 4),
                })
        
        # Simple PageRank
        edge_count = {}
        for edge in edges:
            edge_count[edge["from"]] = edge_count.get(edge["from"], 0) + 1
            edge_count[edge["to"]] = edge_count.get(edge["to"], 0) + 1
        
        total_edges = sum(edge_count.values()) or 1
        for node in nodes:
            node["pagerank"] = round(edge_count.get(node["id"], 0) / total_edges, 4)
        
        logger.info(f"  Built graph with {len(nodes)} nodes and {len(edges)} edges")
        
        return {"nodes": nodes, "edges": edges}
    
    async def _generate_documentation(
        self,
        chunks: list[UniversalChunk],
        ingestor: DomainIngestor,
    ) -> dict:
        """
        Generate documentation for chunks.
        
        Uses LLM if available and enabled, otherwise falls back to rule-based.
        """
        if self.config.use_llm:
            doc_generator = self._get_doc_generator()
            if doc_generator:
                return await self._generate_llm_docs(chunks, doc_generator)
        
        # Fallback to rule-based
        return self._generate_rule_based_docs(chunks)
    
    def _get_doc_generator(self) -> Optional[OpenAIDocGenerator]:
        """Get or create the LLM doc generator."""
        if self._doc_generator is None:
            self._doc_generator = create_doc_generator(model=self.config.llm_model)
        return self._doc_generator
    
    async def _generate_llm_docs(
        self,
        chunks: list[UniversalChunk],
        generator: OpenAIDocGenerator,
    ) -> dict:
        """Generate documentation using LLM."""
        # Optionally limit chunks for cost control
        if self.config.llm_max_chunks and len(chunks) > self.config.llm_max_chunks:
            logger.info(f"  Limiting LLM docs to {self.config.llm_max_chunks} chunks (have {len(chunks)})")
            # Prioritize first chunk of each file
            seen_paths = set()
            priority_chunks = []
            other_chunks = []
            
            for chunk in chunks:
                if chunk.path not in seen_paths:
                    priority_chunks.append(chunk)
                    seen_paths.add(chunk.path)
                else:
                    other_chunks.append(chunk)
            
            chunks_to_document = priority_chunks[:self.config.llm_max_chunks]
            remaining = self.config.llm_max_chunks - len(chunks_to_document)
            if remaining > 0:
                chunks_to_document.extend(other_chunks[:remaining])
        else:
            chunks_to_document = chunks
        
        # Show cost estimate
        estimate = generator.estimate_cost(chunks_to_document)
        logger.info(f"  Estimated LLM cost: ${estimate['estimated_cost_usd']:.4f} for {len(chunks_to_document)} chunks")
        logger.info(f"    Model: {estimate['model']}, ~{estimate['estimated_total_tokens']:,} tokens")
        
        # Generate docs
        def progress(done: int, total: int):
            if done % 10 == 0 or done == total:
                logger.info(f"    Progress: {done}/{total} chunks")
        
        result = await generator.generate_batch_async(
            chunks_to_document,
            max_concurrent=self.config.llm_max_concurrent,
            progress_callback=progress,
        )
        
        # Show detailed cost breakdown
        logger.info(f"  LLM docs: {result.successful} successful, {result.failed} failed")
        logger.info(f"  Token usage: {result.input_tokens:,} input + {result.output_tokens:,} output = {result.total_tokens:,} total")
        if result.cached_tokens > 0:
            logger.info(f"  Cached tokens: {result.cached_tokens:,}")
        logger.info(f"  💰 Actual cost: ${result.total_cost:.4f}")
        
        return result.to_documentation_json()
    
    def _generate_rule_based_docs(self, chunks: list[UniversalChunk]) -> dict:
        """Generate simple rule-based documentation (fallback)."""
        summaries = []
        seen_paths = set()
        
        for chunk in chunks:
            if chunk.path in seen_paths:
                continue
            seen_paths.add(chunk.path)
            
            # Use first 200 chars as preview
            text_preview = chunk.text[:200].replace('\n', ' ').strip()
            if len(chunk.text) > 200:
                text_preview += "..."
            
            summaries.append({
                "symbol_id": f"file:{chunk.path}",
                "summary": f"File: {chunk.path}",
                "description": text_preview,
                "details": {
                    "domain": chunk.domain.value,
                    "type": chunk.type.value if hasattr(chunk.type, 'value') else str(chunk.type),
                },
                "related": [],
            })
        
        return {
            "summaries": summaries,
            "architecture_overview": "Documentation generated using rule-based extraction.",
            "highlights": [],
            "generation_stats": {
                "method": "rule_based",
                "total_chunks": len(chunks),
            },
        }
    
    def _create_manifest(
        self,
        source_info: dict,
        chunks: list[UniversalChunk],
        embeddings_meta: dict,
        ingestor: DomainIngestor,
    ) -> dict:
        """Create the manifest.json content."""
        return {
            "docpack_version": DOCPACK_VERSION,
            "source": source_info,
            "domain": ingestor.domain.value,
            "ingestor": ingestor.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_files": source_info.get("total_files", 0),
            "total_chunks": len(chunks),
            "embedding_dimensions": embeddings_meta["n_dims"],
            "chunk_schema_version": "1.0.0",  # Universal chunk schema version
            "generator": {
                "builder_version": "v6.1",
                "embedder": embeddings_meta.get("model_id", "unknown"),
                "gpu_used": embeddings_meta.get("model_info", {}).get("device", "").startswith("cuda"),
            },
        }
    
    def _generate_readme(self, manifest: dict) -> str:
        """Generate README.md content."""
        source = manifest.get("source", {})
        source_str = source.get("url", source.get("path", "unknown"))
        
        return f"""# Project Documentation (Generated by Doctown v6.1)

- **Source:** {source_str}
- **Domain:** {manifest.get('domain', 'unknown')}
- **Generated:** {manifest['generated_at'][:10]}
- **Files:** {manifest['total_files']}
- **Chunks:** {manifest['total_chunks']}
- **Embedding dimensions:** {manifest['embedding_dimensions']}

## About This Docpack

This `.docpack` file contains a complete knowledge representation including:
- Universal chunk format (domain-agnostic IR)
- Semantic embeddings
- Relationship graph
- Generated documentation

## Domain: {manifest.get('domain', 'unknown').title()}

This content was processed using the `{manifest.get('ingestor', 'unknown')}` ingestor,
which specializes in {manifest.get('domain', 'unknown')} domain content.

---
*Generated by [Doctown](https://github.com/xandwr/doctown-v6)*
"""
    
    def _package_docpack(
        self,
        temp_path: Path,
        source_info: dict,
        chunks: list[UniversalChunk],
        embeddings_meta: dict,
        graph: dict,
        docs: dict,
        manifest: dict,
    ) -> Path:
        """Package everything into a .docpack file."""
        # Determine output name
        if self.config.output_name:
            name = self.config.output_name
        else:
            source = self.config.input_source
            if "/" in source:
                name = source.rstrip("/").split("/")[-1]
                name = name.replace(".git", "").replace(".zip", "")
            else:
                name = Path(source).stem or "project"
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        docpack_path = self.config.output_dir / f"{name}.docpack"
        
        # Convert chunks to dict format
        chunks_data = [chunk.to_dict() for chunk in chunks]
        
        # Build filestructure from chunks
        filestructure = self._build_filestructure(chunks)
        
        # Write intermediate files
        (temp_path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (temp_path / "filestructure.json").write_text(json.dumps(filestructure, indent=2))
        (temp_path / "chunks.json").write_text(json.dumps(chunks_data, indent=2))
        (temp_path / "graph.json").write_text(json.dumps(graph, indent=2))
        (temp_path / "documentation.json").write_text(json.dumps(docs, indent=2))
        (temp_path / "README.md").write_text(self._generate_readme(manifest))
        
        # Write embeddings binary
        if embeddings_meta["n_chunks"] > 0:
            write_embeddings_bin(embeddings_meta["embeddings_array"], temp_path / "embeddings.bin")
        else:
            (temp_path / "embeddings.bin").write_bytes(b"")
        
        # Create zip
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
    
    def _build_filestructure(self, chunks: list[UniversalChunk]) -> dict:
        """Build a filestructure from chunks."""
        paths = sorted(set(chunk.path for chunk in chunks))
        
        root = {"name": "", "type": "directory", "children": []}
        
        for path in paths:
            parts = Path(path).parts
            current = root
            
            for i, part in enumerate(parts):
                is_file = i == len(parts) - 1
                
                # Find or create child
                existing = None
                for child in current.get("children", []):
                    if child["name"] == part:
                        existing = child
                        break
                
                if existing:
                    current = existing
                else:
                    new_node = {
                        "name": part,
                        "type": "file" if is_file else "directory",
                    }
                    if not is_file:
                        new_node["children"] = []
                    current.setdefault("children", []).append(new_node)
                    current = new_node
        
        return {
            "structure": root,
            "total_files": len(paths),
        }


def run_pipeline(
    input_source: str,
    output_dir: str = "./output",
    use_llm: bool = True,
    **kwargs,
) -> PipelineResult:
    """
    Convenience function to run the pipeline.
    
    Args:
        input_source: GitHub URL, local path, or zip path
        output_dir: Output directory for .docpack file
        use_llm: Whether to use LLM for documentation
        **kwargs: Additional PipelineConfig options
    
    Returns:
        PipelineResult with success status and path
    
    Example:
        result = run_pipeline("https://github.com/owner/repo")
        if result.success:
            print(f"Created: {result.docpack_path}")
    """
    config = PipelineConfig(
        input_source=input_source,
        output_dir=Path(output_dir),
        use_llm=use_llm,
        **kwargs,
    )
    pipeline = DocumentationPipeline(config)
    return pipeline.run()


async def run_pipeline_async(
    input_source: str,
    output_dir: str = "./output",
    use_llm: bool = True,
    **kwargs,
) -> PipelineResult:
    """
    Async convenience function to run the pipeline.
    """
    config = PipelineConfig(
        input_source=input_source,
        output_dir=Path(output_dir),
        use_llm=use_llm,
        **kwargs,
    )
    pipeline = DocumentationPipeline(config)
    return await pipeline.run_async()
