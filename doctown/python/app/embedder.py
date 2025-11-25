"""
Hackable embedding model loader with runtime model selection.
Uses HuggingFace snapshot_download for flexible model fetching.
"""
import os
import json
from pathlib import Path
from typing import Optional, List
import torch
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer


# Model presets for quick selection
MODEL_PRESETS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "balanced": "sentence-transformers/all-mpnet-base-v2",
    "quality": "BAAI/bge-large-en-v1.5",
    "multilingual": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "code": "microsoft/codebert-base",
}


def resolve_model_id(model_id: Optional[str] = None) -> str:
    """
    Resolve model ID from various sources with fallback chain.
    Priority: direct parameter > preset env var > model env var > default
    """
    # Check for preset first
    preset = os.getenv("EMBEDDING_MODEL_PRESET")
    if preset and preset in MODEL_PRESETS:
        return MODEL_PRESETS[preset]
    
    # Check direct model ID
    if model_id:
        # Check if it's a preset name
        if model_id in MODEL_PRESETS:
            return MODEL_PRESETS[model_id]
        return model_id
    
    # Check environment variable
    env_model = os.getenv("EMBEDDING_MODEL")
    if env_model:
        # Check if env var is a preset name
        if env_model in MODEL_PRESETS:
            return MODEL_PRESETS[env_model]
        return env_model
    
    # Default fallback
    return MODEL_PRESETS["fast"]


class EmbeddingModel:
    """
    Flexible embedding model manager that downloads and caches any HuggingFace model.
    
    Environment Variables:
    - EMBEDDING_MODEL: HuggingFace model ID (default: "sentence-transformers/all-MiniLM-L6-v2")
    - EMBEDDING_MODEL_PRESET: Preset name (fast, balanced, quality, multilingual, code)
    - EMBEDDING_CACHE_DIR: Local cache directory (default: "./models/embeddings")
    - EMBEDDING_DEVICE: Device to use - "cuda", "cpu", or "auto" (default: "auto")
    - HF_TOKEN: HuggingFace token for private models (optional)
    
    Examples:
        # Use default model
        embedder = EmbeddingModel()
        
        # Use preset
        embedder = EmbeddingModel(model_id="fast")
        # or: export EMBEDDING_MODEL_PRESET=fast
        
        # Use custom model
        embedder = EmbeddingModel(model_id="BAAI/bge-large-en-v1.5")
        
        # Use model from environment
        export EMBEDDING_MODEL="intfloat/e5-large-v2"
        embedder = EmbeddingModel()
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize the embedding model with runtime configuration.
        
        Args:
            model_id: HuggingFace model identifier or preset name (fast, balanced, quality, etc.)
            cache_dir: Directory to cache downloaded models
            device: Device to load model on ("cuda", "cpu", "auto")
            use_auth_token: HuggingFace token for private models
        """
        self.model_id = resolve_model_id(model_id)
        self.cache_dir = Path(cache_dir or os.getenv("EMBEDDING_CACHE_DIR", "./models/embeddings"))
        self.device = device or os.getenv("EMBEDDING_DEVICE", "auto")
        self.use_auth_token = use_auth_token or os.getenv("HF_TOKEN")
        
        self.model = None
        self._is_loaded = False
        
        print(f"🔧 EmbeddingModel configured:")
        print(f"   Model: {self.model_id}")
        print(f"   Cache: {self.cache_dir}")
        print(f"   Device: {self.device}")
    
    def download_model(self) -> Path:
        """
        Download model from HuggingFace using snapshot_download.
        This gives us the full model directory for maximum flexibility.
        
        Returns:
            Path to the downloaded model directory
        """
        print(f"📥 Downloading model: {self.model_id}")
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the entire model snapshot - hackable AF
        model_path = snapshot_download(
            repo_id=self.model_id,
            cache_dir=str(self.cache_dir),
            token=self.use_auth_token,
            # Allow custom file patterns if needed
            allow_patterns=None,  # Download everything by default
            ignore_patterns=None,
        )
        
        print(f"✅ Model downloaded to: {model_path}")
        return Path(model_path)
    
    def load_model(self, force_reload: bool = False):
        """
        Load the embedding model. Downloads if not cached.
        
        Args:
            force_reload: Force re-download even if cached
        """
        if self._is_loaded and not force_reload:
            print("♻️  Model already loaded")
            return
        
        print(f"🚀 Loading embedding model...")
        
        # Download model (uses cache if available)
        model_path = self.download_model()
        
        # Determine device
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        print(f"🎯 Loading model on device: {device}")
        
        # Load with sentence-transformers for convenience
        # But you could swap this out for raw transformers, ONNX, etc.
        self.model = SentenceTransformer(str(model_path), device=device)
        self._is_loaded = True
        
        print(f"✨ Model loaded successfully!")
        print(f"   Dimensions: {self.model.get_sentence_embedding_dimension()}")
    
    def embed(self, texts: List[str], batch_size: int = 32, **kwargs) -> torch.Tensor:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            **kwargs: Additional arguments passed to model.encode()
        
        Returns:
            Tensor of embeddings with shape (len(texts), embedding_dim)
        """
        if not self._is_loaded:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Embedding model failed to load. 'self.model' is None.")
        
        print(f"🔮 Generating embeddings for {len(texts)} texts...")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
            **kwargs
        )
        
        return embeddings
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"loaded": False, "model_id": self.model_id}
        
        return {
            "loaded": True,
            "model_id": self.model_id,
            "device": str(self.model.device) if self.model is not None and hasattr(self.model, "device") else str(self.device),
            "embedding_dim": self.model.get_sentence_embedding_dimension() if self.model is not None else None,
            "max_seq_length": self.model.max_seq_length if self.model is not None else None,
            "cache_dir": str(self.cache_dir),
        }


# Global instance - loaded once, reused across requests
_embedder_instance: Optional[EmbeddingModel] = None


def get_embedder() -> EmbeddingModel:
    """
    Get or create the global embedder instance.
    This ensures we only load the model once.
    """
    global _embedder_instance
    
    if _embedder_instance is None:
        _embedder_instance = EmbeddingModel()
        _embedder_instance.load_model()
    
    return _embedder_instance


# Convenience function for quick embedding
def embed_texts(texts: List[str], **kwargs) -> torch.Tensor:
    """
    Quick embedding function that uses the global embedder instance.
    
    Args:
        texts: List of texts to embed
        **kwargs: Additional arguments passed to embedder.embed()
    
    Returns:
        Tensor of embeddings
    """
    embedder = get_embedder()
    return embedder.embed(texts, **kwargs)