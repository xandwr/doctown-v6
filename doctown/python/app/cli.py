#!/usr/bin/env python3
"""
Doctown CLI - Command-line interface for building docpacks.

Usage:
    python -m app.cli build <input> [options]
    python -m app.cli build-llm <input> [options]  # Uses LLM for documentation
    python -m app.cli info <docpack>
    python -m app.cli ingestors  # List available ingestors
    
Examples:
    # Build from GitHub repo (rule-based docs)
    python -m app.cli build https://github.com/owner/repo
    
    # Build with LLM documentation (requires OPENAI_API_KEY)
    python -m app.cli build-llm https://github.com/owner/repo
    
    # Build with options
    python -m app.cli build https://github.com/owner/repo --output ./docpacks --branch develop
    
    # Inspect a docpack
    python -m app.cli info myproject.docpack
    
    # List available domain ingestors
    python -m app.cli ingestors
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def cmd_build(args: argparse.Namespace) -> int:
    """Build a docpack from a repository (rule-based docs)."""
    from .docpack_builder import DocpackConfig, DocpackBuilder
    
    config = DocpackConfig(
        input_source=args.input,
        branch=args.branch,
        output_dir=Path(args.output),
        output_name=args.name,
        chunk_size=args.chunk_size,
        embedding_model=args.model,
        embedding_batch_size=args.batch_size,
    )
    
    logger.info(f"Building docpack from: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    builder = DocpackBuilder(config)
    result = builder.build()
    
    if result.success:
        print(f"\n✅ Docpack created successfully!")
        print(f"   Path: {result.docpack_path}")
        print(f"   Files: {result.stats.get('total_files', 'N/A')}")
        print(f"   Chunks: {result.stats.get('total_chunks', 'N/A')}")
        print(f"   Embedding dims: {result.stats.get('embedding_dims', 'N/A')}")
        return 0
    else:
        print(f"\n❌ Docpack build failed!")
        print(f"   Error: {result.error}")
        return 1


def cmd_build_llm(args: argparse.Namespace) -> int:
    """Build a docpack using the new pipeline with LLM documentation."""
    import asyncio
    from .pipeline import PipelineConfig, DocumentationPipeline
    
    # Parse max VRAM setting
    max_memory = {0: args.local_max_vram, "cpu": "30GB"}
    
    config = PipelineConfig(
        input_source=args.input,
        branch=args.branch,
        output_dir=Path(args.output),
        output_name=args.name,
        chunk_size=args.chunk_size,
        embedding_model=args.model,
        embedding_batch_size=args.batch_size,
        use_llm=not args.no_llm,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        local_model_id=args.local_model if args.local_model else os.getenv("LOCAL_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
        local_quantization=args.local_quantization if args.local_quantization else os.getenv("LOCAL_QUANTIZATION", "4bit"),
        local_max_memory=max_memory,
        llm_batch_mode=args.llm_batch_mode,
        llm_batch_size=args.llm_batch_size,
        llm_batch_workers=args.llm_batch_workers,
        llm_max_batch_tokens=args.max_batch_tokens,
        include_semantic_context=args.include_semantic_context,
        llm_max_concurrent=args.llm_concurrent,
        llm_max_chunks=args.llm_max_chunks,
        llm_semantic_neighbors=args.llm_semantic_neighbors,
        ingestor_name=args.ingestor,
        use_doc_scorer=not args.no_doc_scorer,
        doc_score_threshold=args.doc_score_threshold,
        doc_score_structural_weight=args.doc_score_structural,
        doc_score_semantic_weight=args.doc_score_semantic,
        doc_score_complexity_weight=args.doc_score_complexity,
    )
    
    logger.info(f"Building docpack from: {args.input}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"LLM documentation: {'disabled' if args.no_llm else 'enabled'}")
    
    pipeline = DocumentationPipeline(config)
    result = asyncio.run(pipeline.run_async())
    
    if result.success:
        print(f"\n✅ Docpack created successfully!")
        print(f"   Path: {result.docpack_path}")
        print(f"   Domain: {result.stats.get('domain', 'N/A')}")
        print(f"   Files: {result.stats.get('total_files', 'N/A')}")
        print(f"   Chunks: {result.stats.get('total_chunks', 'N/A')}")
        print(f"   Embedding dims: {result.stats.get('embedding_dims', 'N/A')}")
        if result.stats.get('llm_docs_generated'):
            print(f"   LLM docs: {result.stats.get('llm_docs_generated', 0)} generated")
        if result.stats.get('doc_worthy_chunks'):
            print(f"   DocScore: {result.stats.get('doc_worthy_chunks', 0)} doc-worthy chunks "
                  f"({result.stats.get('doc_worthy_percent', 0):.1f}%)")
        return 0
    else:
        print(f"\n❌ Pipeline failed!")
        print(f"   Error: {result.error}")
        return 1


def cmd_ingestors(args: argparse.Namespace) -> int:
    """List available domain ingestors."""
    from .ingestors import get_registry
    
    registry = get_registry()
    ingestors = registry.list_ingestors()
    
    print("\n📋 Available Domain Ingestors:")
    print("-" * 50)
    
    for ing in sorted(ingestors, key=lambda x: -x['priority']):
        fallback = " (fallback)" if ing['is_fallback'] else ""
        print(f"  {ing['name']:<15} domain={ing['domain']:<10} priority={ing['priority']}{fallback}")
    
    print("\n💡 To use a specific ingestor:")
    print("   python -m app.cli build-llm <input> --ingestor <name>")
    
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Display information about a docpack."""
    docpack_path = Path(args.docpack)
    
    if not docpack_path.exists():
        print(f"❌ File not found: {docpack_path}")
        return 1
    
    if not docpack_path.suffix == ".docpack":
        print(f"⚠️  Warning: File does not have .docpack extension")
    
    try:
        with zipfile.ZipFile(docpack_path, "r") as zf:
            # List contents
            print(f"\n📦 Docpack: {docpack_path.name}")
            print(f"   Size: {docpack_path.stat().st_size / 1024:.1f} KB")
            print(f"\n📂 Contents:")
            
            for info in zf.infolist():
                size_kb = info.file_size / 1024
                print(f"   {info.filename:<25} {size_kb:>8.1f} KB")
            
            # Read manifest
            if "manifest.json" in zf.namelist():
                manifest = json.loads(zf.read("manifest.json"))
                print(f"\n📋 Manifest:")
                print(f"   Version: {manifest.get('docpack_version', 'N/A')}")
                print(f"   Source: {manifest.get('source_repo', 'N/A')}")
                print(f"   Branch: {manifest.get('branch', 'N/A')}")
                print(f"   Generated: {manifest.get('generated_at', 'N/A')[:10]}")
                print(f"   Files: {manifest.get('total_files', 'N/A')}")
                print(f"   Chunks: {manifest.get('total_chunks', 'N/A')}")
                print(f"   Embedding dims: {manifest.get('embedding_dimensions', 'N/A')}")
                
                gen = manifest.get("generator", {})
                print(f"\n🔧 Generator:")
                print(f"   Builder: {gen.get('builder_version', 'N/A')}")
                print(f"   Embedder: {gen.get('embedder', 'N/A')}")
                print(f"   GPU: {'Yes' if gen.get('gpu_used') else 'No'}")
        
        return 0
        
    except zipfile.BadZipFile:
        print(f"❌ Invalid docpack file (not a valid zip)")
        return 1
    except Exception as e:
        print(f"❌ Error reading docpack: {e}")
        return 1


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract a docpack to a directory."""
    docpack_path = Path(args.docpack)
    output_dir = Path(args.output)
    
    if not docpack_path.exists():
        print(f"❌ File not found: {docpack_path}")
        return 1
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(docpack_path, "r") as zf:
            zf.extractall(output_dir)
        
        print(f"✅ Extracted to: {output_dir}")
        return 0
        
    except Exception as e:
        print(f"❌ Error extracting: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="doctown",
        description="Build and inspect .docpack files from any content source",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Build command (legacy, rule-based)
    build_parser = subparsers.add_parser(
        "build",
        help="Build a docpack (rule-based docs, uses old pipeline)",
    )
    build_parser.add_argument(
        "input",
        help="GitHub URL or local zip path",
    )
    build_parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    build_parser.add_argument(
        "-n", "--name",
        default=None,
        help="Output file name (without extension)",
    )
    build_parser.add_argument(
        "-b", "--branch",
        default="main",
        help="Git branch (default: main)",
    )
    build_parser.add_argument(
        "--chunk-size",
        type=int,
        default=240,
        help="Chunk size in characters (default: 240)",
    )
    build_parser.add_argument(
        "-m", "--model",
        default="fast",
        help="Embedding model preset or HuggingFace ID (default: fast)",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    build_parser.set_defaults(func=cmd_build)
    
    # Build-LLM command (new pipeline with LLM)
    build_llm_parser = subparsers.add_parser(
        "build-llm",
        help="Build a docpack with LLM documentation (new pipeline)",
    )
    build_llm_parser.add_argument(
        "input",
        help="GitHub URL, local path, or zip path",
    )
    build_llm_parser.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    build_llm_parser.add_argument(
        "-n", "--name",
        default=None,
        help="Output file name (without extension)",
    )
    build_llm_parser.add_argument(
        "-b", "--branch",
        default="main",
        help="Git branch (default: main)",
    )
    build_llm_parser.add_argument(
        "--chunk-size",
        type=int,
        default=240,
        help="Chunk size in characters (default: 240)",
    )
    build_llm_parser.add_argument(
        "-m", "--model",
        default="fast",
        help="Embedding model preset or HuggingFace ID (default: fast)",
    )
    build_llm_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    build_llm_parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM documentation (use rule-based)",
    )
    build_llm_parser.add_argument(
        "--llm-backend",
        choices=["openai", "local"],
        default="local",
        help="LLM backend: 'openai' for API or 'local' for on-device (default: local)",
    )
    build_llm_parser.add_argument(
        "--llm-model",
        default=None,
        help="OpenAI model to use (only for --llm-backend=openai, default: gpt-4o-mini via env)",
    )
    build_llm_parser.add_argument(
        "--local-model",
        default=None,
        help="Local model ID (only for --llm-backend=local, default: from LOCAL_MODEL env or meta-llama/Llama-3.1-8B-Instruct)",
    )
    build_llm_parser.add_argument(
        "--local-quantization",
        choices=["4bit", "8bit", "gptq", "none"],
        default="4bit",
        help="Quantization for local model (default: 4bit)",
    )
    build_llm_parser.add_argument(
        "--local-max-vram",
        default="11GB",
        help="Max VRAM for local model (e.g., '11GB' for RTX 4070ti Super)",
    )
    build_llm_parser.add_argument(
        "--llm-batch-mode",
        action="store_true",
        help="Use batch mode (multiple symbols per request, better for latency)",
    )
    build_llm_parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=15,
        help="Symbols per batch request in batch mode (default: 15)",
    )
    build_llm_parser.add_argument(
        "--llm-batch-workers",
        type=int,
        default=3,
        help="Parallel workers for batch mode processing (default: 3)",
    )
    build_llm_parser.add_argument(
        "--llm-concurrent",
        type=int,
        default=10,
        help="Max concurrent LLM requests in per-symbol mode (default: 10)",
    )
    build_llm_parser.add_argument(
        "--llm-max-chunks",
        type=int,
        default=None,
        help="Limit chunks for LLM docs (cost control)",
    )
    build_llm_parser.add_argument(
        "--llm-semantic-neighbors",
        type=int,
        default=3,
        help="Number of semantic neighbors to include as context (default: 3)",
    )
    build_llm_parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=30000,
        help="Max input tokens per batch in batch mode (default: 30000)",
    )
    build_llm_parser.add_argument(
        "--include-semantic-context",
        action="store_true",
        help="Include semantic relationships from embeddings in LLM prompts",
    )
    build_llm_parser.add_argument(
        "--doc-score-threshold",
        type=float,
        default=0.65,
        help="Minimum DocScore for LLM documentation (0.0-1.0, default: 0.65)",
    )
    build_llm_parser.add_argument(
        "--no-doc-scorer",
        action="store_true",
        help="Disable DocScore filtering (document all chunks)",
    )
    build_llm_parser.add_argument(
        "--doc-score-structural",
        type=float,
        default=0.50,
        help="Structural signal weight in DocScore (default: 0.50)",
    )
    build_llm_parser.add_argument(
        "--doc-score-semantic",
        type=float,
        default=0.35,
        help="Semantic signal weight in DocScore (default: 0.35)",
    )
    build_llm_parser.add_argument(
        "--doc-score-complexity",
        type=float,
        default=0.15,
        help="Complexity signal weight in DocScore (default: 0.15)",
    )
    build_llm_parser.add_argument(
        "--ingestor",
        default=None,
        help="Force specific ingestor (auto-detect if not set)",
    )
    build_llm_parser.set_defaults(func=cmd_build_llm)
    
    # Ingestors command
    ingestors_parser = subparsers.add_parser(
        "ingestors",
        help="List available domain ingestors",
    )
    ingestors_parser.set_defaults(func=cmd_ingestors)
    
    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about a docpack",
    )
    info_parser.add_argument(
        "docpack",
        help="Path to the .docpack file",
    )
    info_parser.set_defaults(func=cmd_info)
    
    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract a docpack to a directory",
    )
    extract_parser.add_argument(
        "docpack",
        help="Path to the .docpack file",
    )
    extract_parser.add_argument(
        "-o", "--output",
        default="./extracted",
        help="Output directory (default: ./extracted)",
    )
    extract_parser.set_defaults(func=cmd_extract)
    
    # Parse args
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
