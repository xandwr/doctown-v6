#!/usr/bin/env python3
"""
CLI tool to pre-download embedding models.
Makes it easy to prepare models before running the pipeline.
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import EmbeddingModel, MODEL_PRESETS, resolve_model_id


def list_presets():
    """List all available model presets."""
    print("\nAvailable Model Presets:")
    print("=" * 70)
    
    # Load descriptions from config if available
    config_path = Path(__file__).parent.parent / "config" / "embedding_models.json"
    descriptions = {}
    
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            descriptions = {k: v.get("description", "") for k, v in config.get("presets", {}).items()}
    
    for preset_name, model_id in MODEL_PRESETS.items():
        desc = descriptions.get(preset_name, "")
        print(f"\n  {preset_name:12s} → {model_id}")
        if desc:
            print(f"                {desc}")
    
    print("\n" + "=" * 70)
    print("\nUsage:")
    print(f"  python {os.path.basename(__file__)} --preset fast")
    print(f"  python {os.path.basename(__file__)} --model your-model-id")
    print()


def download_model(model_id: str, cache_dir: str = "", use_auth_token: str = ""):
    """Download a specific model."""
    resolved_id = resolve_model_id(model_id)
    
    print(f"\nDownloading model: {resolved_id}")
    if resolved_id != model_id:
        print(f"   (resolved from preset: {model_id})")
    
    embedder = EmbeddingModel(
        model_id=resolved_id,
        cache_dir=cache_dir,
        use_auth_token=use_auth_token
    )
    
    try:
        model_path = embedder.download_model()
        print(f"\n✅ Model successfully downloaded!")
        print(f"   Location: {model_path}")
        print(f"   Cache dir: {embedder.cache_dir}")
        
        # Try to load and get info
        print(f"\n🔍 Verifying model...")
        embedder.load_model()
        info = embedder.get_model_info()
        
        print(f"\n📊 Model Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download and manage embedding models for doctown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all presets
  python download_model.py --list
  
  # Download using preset
  python download_model.py --preset fast
  python download_model.py --preset quality
  
  # Download specific model
  python download_model.py --model sentence-transformers/all-MiniLM-L6-v2
  python download_model.py --model BAAI/bge-large-en-v1.5
  
  # Download to custom cache
  python download_model.py --preset fast --cache ./my-models
  
  # Download private model
  python download_model.py --model private/model --token your-hf-token

Environment Variables:
  EMBEDDING_MODEL        - Model ID or preset name
  EMBEDDING_CACHE_DIR    - Cache directory
  HF_TOKEN              - HuggingFace token
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available model presets"
    )
    
    parser.add_argument(
        "--preset", "-p",
        type=str,
        help="Model preset name (fast, balanced, quality, multilingual, code)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="HuggingFace model ID"
    )
    
    parser.add_argument(
        "--cache", "-c",
        type=str,
        help="Cache directory path (default: ./models/embeddings)"
    )
    
    parser.add_argument(
        "--token", "-t",
        type=str,
        help="HuggingFace token for private models"
    )
    
    args = parser.parse_args()
    
    # Show header
    print("\n" + "=" * 70)
    print("Doctown Embedding Model Downloader")
    print("=" * 70)
    
    # List presets
    if args.list:
        list_presets()
        return
    
    # Determine model to download
    model_id = None
    if args.preset:
        model_id = args.preset
    elif args.model:
        model_id = args.model
    else:
        # No model specified, show help
        print("\n❌ No model specified!")
        print("\nUse --preset, --model, or --list to get started.")
        parser.print_help()
        return
    
    # Download the model
    success = download_model(
        model_id=model_id,
        cache_dir=args.cache,
        use_auth_token=args.token
    )
    
    if success:
        print(f"\nAll done! Model {model_id} is ready to use.\n")
    else:
        print("\n💥 Download failed. Check the error above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
