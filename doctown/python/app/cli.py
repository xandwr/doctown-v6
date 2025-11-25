#!/usr/bin/env python3
"""
Doctown CLI - Command-line interface for building docpacks.

Usage:
    python -m app.cli build <input> [options]
    python -m app.cli info <docpack>
    
Examples:
    # Build from GitHub repo
    python -m app.cli build https://github.com/owner/repo
    
    # Build with options
    python -m app.cli build https://github.com/owner/repo --output ./docpacks --branch develop
    
    # Inspect a docpack
    python -m app.cli info myproject.docpack
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


def cmd_build(args: argparse.Namespace) -> int:
    """Build a docpack from a repository."""
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
        description="Build and inspect .docpack files from Git repositories",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Build command
    build_parser = subparsers.add_parser(
        "build",
        help="Build a docpack from a repository",
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
