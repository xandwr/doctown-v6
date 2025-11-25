"""
Local LLM Client - Documentation generation using on-device models.

Supports Qwen2.5-Coder-32B with 4-bit GPTQ or FP8 quantization for efficient
inference on consumer GPUs (12GB+ VRAM). Uses transformers library with
BitsAndBytes quantization or AutoGPTQ for maximum compatibility.

This client mirrors the OpenAI client interface for drop-in compatibility
with the existing pipeline architecture.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Callable, TYPE_CHECKING

from pydantic import BaseModel, Field

from ..ingestors.base import Domain, UniversalChunk
from .prompts import get_prompt_for_domain, PromptTemplate

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ~= 4 chars for English)."""
    return len(text) // 4


def truncate_chunk_text(text: str, max_tokens: int = 800) -> tuple[str, bool]:
    """
    Intelligently truncate chunk text to fit within token budget.
    
    Reuses the same strategy as OpenAI client for consistency.
    
    Args:
        text: Full source code text
        max_tokens: Maximum tokens to allow (default: 800 = ~3200 chars)
        
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text, False
    
    # Calculate character budget (4 chars per token)
    max_chars = max_tokens * 4
    
    # Reserve space for truncation marker
    marker = "\n... [truncated] ...\n"
    available_chars = max_chars - len(marker)
    
    # Split 70% beginning, 30% ending to preserve signature and conclusion
    head_chars = int(available_chars * 0.70)
    tail_chars = int(available_chars * 0.30)
    
    head = text[:head_chars].rstrip()
    tail = text[-tail_chars:].lstrip()
    
    truncated = f"{head}{marker}{tail}"
    return truncated, True


class LocalLLMModel:
    """
    Wrapper for local LLM inference using transformers.
    
    Supports:
    - 4-bit quantization via BitsAndBytes (automatic, no pre-quantized model needed)
    - AutoGPTQ for pre-quantized GPTQ models
    - FP16/BF16 for high-end GPUs with sufficient VRAM
    
    The model is loaded lazily on first use to avoid startup overhead.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization: Literal["4bit", "8bit", "gptq", "none"] = "4bit",
        device_map: str = "auto",
        max_memory: Optional[dict[int, str]] = None,
        use_flash_attention: bool = True,
    ):
        """
        Initialize local LLM model.
        
        Args:
            model_id: HuggingFace model ID (e.g., "Qwen/Qwen2.5-Coder-32B-Instruct")
            quantization: Quantization method ("4bit", "8bit", "gptq", or "none")
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
            max_memory: Maximum memory per device (e.g., {0: "11GB", "cpu": "30GB"})
            use_flash_attention: Use Flash Attention 2 if available (faster, lower VRAM)
        """
        self.model_id = model_id
        self.quantization = quantization
        self.device_map = device_map
        self.max_memory = max_memory or {0: "11GB", "cpu": "30GB"}
        self.use_flash_attention = use_flash_attention
        
        self.model: Optional[Any] = None  # PreTrainedModel
        self.tokenizer: Optional[Any] = None  # PreTrainedTokenizer
        self._loaded = False
        
        logger.info(f"Configured LocalLLM: {model_id} with {quantization} quantization")
    
    def load_model(self):
        """Load the model and tokenizer (lazy loading)."""
        if self._loaded:
            return
        
        logger.info(f"Loading local LLM: {self.model_id}")
        start_time = time.time()
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            
            # Configure quantization
            model_kwargs = {
                "device_map": self.device_map,
                "trust_remote_code": True,
                "dtype": torch.bfloat16,  # Use BF16 for modern GPUs
            }
            
            if self.quantization == "4bit":
                # 4-bit quantization with BitsAndBytes
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,  # Double quantization for better compression
                    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["max_memory"] = self.max_memory
                
            elif self.quantization == "8bit":
                # 8-bit quantization with BitsAndBytes
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["max_memory"] = self.max_memory
                
            elif self.quantization == "gptq":
                # Pre-quantized GPTQ model
                try:
                    from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-not-found]
                    logger.info("Using AutoGPTQ for pre-quantized model")
                    self.model = AutoGPTQForCausalLM.from_quantized(
                        self.model_id,
                        device_map=self.device_map,
                        trust_remote_code=True,
                        use_safetensors=True,
                    )
                    self._loaded = True
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Loaded GPTQ model in {elapsed:.1f}s")
                    return
                except ImportError:
                    logger.warning("auto-gptq not installed, falling back to 4-bit")
                    self.quantization = "4bit"
                    return self.load_model()
            
            # Flash Attention 2 (much faster, lower VRAM)
            if self.use_flash_attention:
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    logger.info("Using Flash Attention 2")
                except Exception:
                    logger.info("Flash Attention 2 not available, using default")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self._loaded = True
            elapsed = time.time() - start_time
            
            # Log memory usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"✅ Model loaded in {elapsed:.1f}s (VRAM: {vram_used:.2f}GB)")
            else:
                logger.info(f"✅ Model loaded in {elapsed:.1f}s")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling threshold
            do_sample: Use sampling (vs greedy decoding)
            
        Returns:
            Generated text (decoded)
        """
        if not self._loaded:
            self.load_model()
        
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        import torch
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=30000)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode output (remove input prompt)
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        return response.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the tokenizer."""
        if not self._loaded:
            self.load_model()
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return len(self.tokenizer.encode(text, add_special_tokens=True))


# Reuse DocumentationResult and BatchDocumentationResult from openai_client
@dataclass
class DocumentationResult:
    """Result of documenting a single chunk."""
    success: bool
    chunk_id: str
    documentation: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency_ms: int = 0


@dataclass
class BatchDocumentationResult:
    """Result of batch documentation generation."""
    results: list[DocumentationResult]
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency_ms: int = 0
    successful: int = 0
    failed: int = 0


class LocalDocGenerator:
    """
    Local LLM-based documentation generator.
    
    Compatible with OpenAIDocGenerator interface for drop-in replacement.
    Uses on-device models instead of API calls, so no cost but requires GPU.
    """
    
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        quantization: Literal["4bit", "8bit", "gptq", "none"] = "4bit",
        max_memory: Optional[dict[int, str]] = None,
        use_flash_attention: bool = True,
    ):
        """
        Initialize local documentation generator.
        
        Args:
            model_id: HuggingFace model ID
            quantization: Quantization method for memory efficiency
            max_memory: Maximum memory allocation per device
            use_flash_attention: Use Flash Attention 2 if available
        """
        self.model = LocalLLMModel(
            model_id=model_id,
            quantization=quantization,
            max_memory=max_memory,
            use_flash_attention=use_flash_attention,
        )
        self.total_tokens = 0
        self.total_requests = 0
    
    def _format_prompt_for_chat(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format prompts into chat template for Qwen2.5.
        
        Qwen2.5 uses the ChatML format.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Use tokenizer's chat template if available
        if self.model.tokenizer is not None and hasattr(self.model.tokenizer, "apply_chat_template"):
            return self.model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        # Fallback to manual ChatML formatting
        chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        chat_prompt += f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        chat_prompt += "<|im_start|>assistant\n"
        
        return chat_prompt
    
    def _parse_json_response(self, response: str) -> Optional[dict]:
        """
        Extract JSON from model response.
        
        The model might wrap JSON in markdown code blocks or add explanatory text.
        We need to extract just the JSON portion.
        """
        # Try direct JSON parse first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        import re
        
        # Look for ```json ... ``` blocks
        json_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        matches = re.findall(json_block_pattern, response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Try to find any { ... } block
        brace_pattern = r"\{[\s\S]*\}"
        matches = re.findall(brace_pattern, response)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        logger.warning("Could not extract JSON from response")
        return None
    
    async def generate_documentation(
        self,
        chunk: UniversalChunk,
        semantic_context: str = "",
        retry_with_truncation: bool = True,
    ) -> DocumentationResult:
        """
        Generate documentation for a single chunk.
        
        Args:
            chunk: The code chunk to document
            semantic_context: Optional semantic neighbor context
            retry_with_truncation: Retry with truncated text if too long
            
        Returns:
            DocumentationResult with success status and generated docs
        """
        start_time = time.time()
        
        try:
            # Get prompt template for domain
            template = get_prompt_for_domain(chunk.domain)
            
            # Format prompts
            # Convert metadata to dict if it's a ChunkMetadata object
            from dataclasses import asdict, is_dataclass
            metadata_dict = chunk.metadata if isinstance(chunk.metadata, dict) else (asdict(chunk.metadata) if is_dataclass(chunk.metadata) else {})
            
            user_prompt = template.format_user_prompt(
                text=chunk.text,
                path=chunk.path,
                chunk_type=chunk.type,
                metadata=metadata_dict,
                context=semantic_context,
            )
            
            # Add JSON format instruction
            user_prompt += "\n\nRespond with ONLY a valid JSON object following this schema:\n"
            user_prompt += json.dumps({
                "summary": "Brief one-sentence summary",
                "description": "Detailed explanation of functionality",
                "parameters": [{"name": "param", "type": "Type", "description": "What it does"}],
                "returns": {"type": "ReturnType", "description": "What is returned"},
                "examples": ["Example usage or test case"],
                "notes": ["Important implementation details or warnings"],
            }, indent=2)
            
            # Format for chat
            full_prompt = self._format_prompt_for_chat(template.system_prompt, user_prompt)
            
            # Estimate tokens
            input_tokens = self.model.count_tokens(full_prompt)
            
            # Generate response (run in thread pool to avoid blocking event loop)
            response = await asyncio.to_thread(
                self.model.generate,
                full_prompt,
                max_new_tokens=2048,
                temperature=0.1,  # Low temperature for structured output
                top_p=0.9,
                do_sample=True,
            )
            
            # Parse JSON response
            doc_json = self._parse_json_response(response)
            
            if doc_json is None:
                return DocumentationResult(
                    success=False,
                    chunk_id=chunk.chunk_id,
                    error="Failed to parse JSON from model response",
                    tokens_used=input_tokens,
                    latency_ms=int((time.time() - start_time) * 1000),
                )
            
            # Calculate tokens
            output_tokens = self.model.count_tokens(response)
            total_tokens = input_tokens + output_tokens
            self.total_tokens += total_tokens
            self.total_requests += 1
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return DocumentationResult(
                success=True,
                chunk_id=chunk.chunk_id,
                documentation=doc_json,
                tokens_used=total_tokens,
                latency_ms=latency_ms,
            )
            
        except Exception as e:
            logger.error(f"Error generating docs for {chunk.chunk_id}: {e}")
            return DocumentationResult(
                success=False,
                chunk_id=chunk.chunk_id,
                error=str(e),
                latency_ms=int((time.time() - start_time) * 1000),
            )
    
    async def generate_batch_async(
        self,
        chunks: list[UniversalChunk],
        max_concurrent: int = 1,  # Sequential by default (local model can't parallelize easily)
        semantic_neighbors: Optional[dict[str, list[tuple[str, float]]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchDocumentationResult:
        """
        Generate documentation for multiple chunks.
        
        Note: For local models, we process sequentially since the model occupies
        the GPU. Set max_concurrent=1 unless you have multiple GPUs.
        
        Args:
            chunks: List of chunks to document
            max_concurrent: Number of concurrent generations (usually 1 for local)
            semantic_neighbors: Precomputed semantic neighbors for context
            progress_callback: Optional callback for progress updates
            
        Returns:
            BatchDocumentationResult with all results
        """
        results = []
        total_tokens = 0
        total_latency = 0
        successful = 0
        failed = 0
        
        # Build semantic context map
        semantic_contexts = {}
        if semantic_neighbors:
            chunk_map = {c.chunk_id: c for c in chunks}
            for chunk in chunks:
                if chunk.chunk_id in semantic_neighbors:
                    context_lines = ["**Semantically Related Code:**"]
                    for neighbor_id, similarity in semantic_neighbors[chunk.chunk_id]:
                        if neighbor_id in chunk_map:
                            neighbor = chunk_map[neighbor_id]
                            context_lines.append(
                                f"- {neighbor.path} (similarity: {similarity:.2f})"
                            )
                            # Include snippet
                            snippet = neighbor.text[:200].replace("\n", " ")
                            context_lines.append(f"  {snippet}...")
                    semantic_contexts[chunk.chunk_id] = "\n".join(context_lines)
        
        # Process chunks sequentially or with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_chunk(idx: int, chunk: UniversalChunk) -> DocumentationResult:
            async with semaphore:
                context = semantic_contexts.get(chunk.chunk_id, "")
                result = await self.generate_documentation(chunk, context)
                
                if progress_callback:
                    progress_callback(idx + 1, len(chunks))
                
                return result
        
        # Run all tasks
        tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        
        # Aggregate results
        for result in results:
            total_tokens += result.tokens_used
            total_latency += result.latency_ms
            if result.success:
                successful += 1
            else:
                failed += 1
        
        return BatchDocumentationResult(
            results=results,
            total_tokens=total_tokens,
            total_cost=0.0,  # No cost for local inference
            total_latency_ms=total_latency,
            successful=successful,
            failed=failed,
        )
    
    def estimate_cost_batch_mode(
        self,
        chunks: list[UniversalChunk],
        batch_size: int = 10,
    ) -> dict[str, Any]:
        """
        Estimate the cost for batch mode documentation generation.
        
        For local models, there is no monetary cost, but we still provide
        token usage estimates for transparency.
        
        Args:
            chunks: Chunks to estimate for
            batch_size: Number of symbols per batch (not used for local sequential processing)
        
        Returns:
            Dict with token estimates and zero cost
        """
        # Estimate tokens (similar to OpenAI estimator)
        tokens_per_symbol = 400  # Truncated content
        prompt_overhead = 400  # System prompt
        per_symbol_overhead = 150  # Formatting + context
        avg_output_per_symbol = 400  # Output tokens
        
        total_input = len(chunks) * (tokens_per_symbol + per_symbol_overhead) + prompt_overhead
        total_output = len(chunks) * avg_output_per_symbol
        
        return {
            "model": self.model.model_id,
            "num_chunks": len(chunks),
            "num_batches": len(chunks),  # Sequential processing for local
            "batch_size": 1,  # Process one at a time
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_total_tokens": total_input + total_output,
            "estimated_cost_usd": 0.0,  # No cost for local inference
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "batch_discount": "N/A (local inference)",
        }
    
    async def generate_multi_batch_async(
        self,
        chunks: list[UniversalChunk],
        batch_size: int = 15,
        context_map: Optional[dict[str, str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        max_workers: int = 1,  # Local models process sequentially
    ) -> BatchDocumentationResult:
        """
        Generate documentation in batch mode for local LLM.
        
        For local models, we process sequentially since the GPU can only handle
        one inference at a time. This method provides API compatibility with
        OpenAI's batch processing interface.
        
        Args:
            chunks: List of chunks to document
            batch_size: Ignored for local processing (process one at a time)
            context_map: Optional dict mapping chunk_id to context string
            progress_callback: Called with (completed, total)
            max_workers: Ignored for local processing (always 1)
        
        Returns:
            BatchDocumentationResult with all results
        """
        logger.info(f"Processing {len(chunks)} chunks sequentially (local LLM)")
        
        # Convert context_map to semantic_neighbors format if provided
        semantic_neighbors = None
        if context_map:
            # Context map is already built, we'll pass it directly in the loop
            pass
        
        results = []
        total_tokens = 0
        successful = 0
        failed = 0
        
        for idx, chunk in enumerate(chunks):
            # Get context for this chunk
            context = context_map.get(chunk.chunk_id, "") if context_map else ""
            
            # Generate documentation
            result = await self.generate_documentation(chunk, context)
            results.append(result)
            
            total_tokens += result.tokens_used
            if result.success:
                successful += 1
            else:
                failed += 1
            
            # Update progress
            if progress_callback:
                progress_callback(idx + 1, len(chunks))
        
        return BatchDocumentationResult(
            results=results,
            total_tokens=total_tokens,
            total_cost=0.0,
            total_latency_ms=sum(r.latency_ms for r in results),
            successful=successful,
            failed=failed,
        )
    
    async def generate_batch_documentation(
        self,
        chunks: list[UniversalChunk],
        batch_size: int = 10,
        max_input_tokens: int = 30000,
        semantic_neighbors: Optional[dict[str, list[tuple[str, float]]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchDocumentationResult:
        """
        Generate documentation for multiple chunks in batches.
        
        This method batches multiple symbols into a single prompt for efficiency.
        Useful for processing many small functions/classes at once.
        
        Args:
            chunks: List of chunks to document
            batch_size: Number of symbols per batch request
            max_input_tokens: Maximum input tokens per batch
            semantic_neighbors: Precomputed semantic neighbors
            progress_callback: Progress callback function
            
        Returns:
            BatchDocumentationResult with all documentation
        """
        # For now, delegate to generate_batch_async with sequential processing
        # In the future, we could implement true batch prompting here
        logger.info(f"Processing {len(chunks)} chunks in batch mode (local LLM)")
        
        return await self.generate_batch_async(
            chunks=chunks,
            max_concurrent=1,  # Sequential for local model
            semantic_neighbors=semantic_neighbors,
            progress_callback=progress_callback,
        )


def create_local_generator(
    model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
    quantization: Literal["4bit", "8bit", "gptq", "none"] = "4bit",
    max_memory: Optional[dict] = None,
) -> LocalDocGenerator:
    """
    Factory function to create a local documentation generator.
    
    Args:
        model_id: HuggingFace model ID
        quantization: Quantization method ("4bit", "8bit", "gptq", "none")
        max_memory: Maximum memory per device
        
    Returns:
        Configured LocalDocGenerator instance
    """
    if max_memory is None:
        max_memory = {0: "11GB", "cpu": "30GB"}
    
    return LocalDocGenerator(
        model_id=model_id,
        quantization=quantization,
        max_memory=max_memory,
        use_flash_attention=False,
    )
