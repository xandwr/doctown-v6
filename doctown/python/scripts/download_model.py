#!/usr/bin/env python3
"""
CLI tool to pre-download embedding and LLM models.
Makes it easy to prepare models before running the pipeline.

Supports:
- Embedding models (sentence-transformers)
- LLM models (Qwen2.5-Coder, etc.) for local inference
"""
import sys
import os
import argparse
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.embedder import EmbeddingModel, MODEL_PRESETS, resolve_model_id

# Popular LLM models for code documentation
LLM_PRESETS = {
    "qwen-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "qwen-32b-gptq": "Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4",  # Pre-quantized
    "qwen-14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "codellama-34b": "codellama/CodeLlama-34b-Instruct-hf",
    "deepseek-33b": "deepseek-ai/deepseek-coder-33b-instruct",
}


def list_presets():
    """List all available model presets."""
    print("\n📦 Available Embedding Model Presets:")
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
    
    print("\n")
    print("🤖 Available LLM Model Presets:")
    print("=" * 70)
    
    llm_descriptions = {
        "qwen-32b": "32B params, best quality (requires 12GB+ VRAM with 4-bit)",
        "qwen-32b-gptq": "32B params, pre-quantized GPTQ (fastest loading)",
        "qwen-14b": "14B params, good balance (8-10GB VRAM with 4-bit)",
        "qwen-7b": "7B params, fastest inference (6GB VRAM with 4-bit)",
        "codellama-34b": "Meta's CodeLlama 34B (requires 13GB+ VRAM)",
        "deepseek-33b": "DeepSeek Coder 33B (excellent code quality)",
    }
    
    for preset_name, model_id in LLM_PRESETS.items():
        desc = llm_descriptions.get(preset_name, "")
        print(f"\n  {preset_name:15s} → {model_id}")
        if desc:
            print(f"                   {desc}")
    
    print("\n" + "=" * 70)
    print("\n💡 Usage:")
    print(f"  # Download embedding model")
    print(f"  python {os.path.basename(__file__)} --type embedding --preset fast")
    print(f"  # Download LLM model")
    print(f"  python {os.path.basename(__file__)} --type llm --preset qwen-32b")
    print(f"  # Download custom model")
    print(f"  python {os.path.basename(__file__)} --type llm --model Qwen/Qwen2.5-Coder-32B-Instruct")
    print()


def download_embedding_model(model_id: str, cache_dir: str = "", use_auth_token: str = ""):
    """Download a specific embedding model."""
    resolved_id = resolve_model_id(model_id)
    
    print(f"\n📥 Downloading embedding model: {resolved_id}")
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


def download_llm_model(model_id: str, cache_dir: str = "", use_auth_token: str = ""):
    """Download a specific LLM model using HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    
    # Resolve preset if applicable
    resolved_id = LLM_PRESETS.get(model_id, model_id)
    
    print(f"\n📥 Downloading LLM model: {resolved_id}")
    if resolved_id != model_id:
        print(f"   (resolved from preset: {model_id})")
    
    print(f"   ⚠️  This may take a while for large models (32B = ~20GB download)")
    
    # Use default cache if not specified
    if not cache_dir:
        cache_dir = str(Path.home() / ".cache" / "huggingface" / "hub")
    
    try:
        # Download model files
        token = use_auth_token if use_auth_token else None
        
        print(f"\n⏳ Downloading to: {cache_dir}")
        model_path = snapshot_download(
            repo_id=resolved_id,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,  # Resume if interrupted
        )
        
        print(f"\n✅ LLM model successfully downloaded!")
        print(f"   Model ID: {resolved_id}")
        print(f"   Location: {model_path}")
        print(f"   Cache dir: {cache_dir}")
        
        # Estimate size
        model_size = sum(f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file())
        print(f"   Size: {model_size / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading LLM model: {e}")
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
        "--type",
        choices=["embedding", "llm"],
        default="embedding",
        help="Type of model to download (default: embedding)"
    )
    
    parser.add_argument(
        "--preset", "-p",
        type=str,
        help="Model preset name (see --list for options)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        help="HuggingFace model ID"
    )
    
    parser.add_argument(
        "--cache", "-c",
        type=str,
        help="Cache directory path (default: auto-detected)"
    )
    
    parser.add_argument(
        "--token", "-t",
        type=str,
        help="HuggingFace token for private models"
    )
    
    args = parser.parse_args()
    
    # Show header
    print("\n" + "=" * 70)
    print("Doctown Model Downloader")
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
    
    # Download the appropriate model type
    if args.type == "llm":
        success = download_llm_model(
            model_id=model_id,
            cache_dir=args.cache,
            use_auth_token=args.token
        )
    else:
        success = download_embedding_model(
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
