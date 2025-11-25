"""
OpenAI Client - Structured documentation generation via OpenAI API.

Uses gpt-5-nano (or configurable model) for fast, cheap documentation generation
with structured JSON output via response_format.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Callable
from concurrent.futures import ThreadPoolExecutor

from ..ingestors.base import Domain, UniversalChunk
from .prompts import get_prompt_for_domain, PromptTemplate

logger = logging.getLogger(__name__)

# Try to import openai
try:
    import openai
    from openai import OpenAI, AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    AsyncOpenAI = None


@dataclass
class DocumentationResult:
    """Result of documenting a single chunk."""
    
    chunk_id: str
    success: bool
    documentation: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "success": self.success,
            "documentation": self.documentation,
            "error": self.error,
            "tokens_used": self.tokens_used,
        }


@dataclass
class BatchDocumentationResult:
    """Result of documenting multiple chunks."""
    
    results: list[DocumentationResult] = field(default_factory=list)
    total_tokens: int = 0
    successful: int = 0
    failed: int = 0
    
    def add(self, result: DocumentationResult) -> None:
        """Add a result to the batch."""
        self.results.append(result)
        self.total_tokens += result.tokens_used
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
    
    def to_documentation_json(self) -> dict[str, Any]:
        """Convert to documentation.json format."""
        summaries = []
        
        for result in self.results:
            if not result.success:
                continue
            
            doc = result.documentation
            summaries.append({
                "symbol_id": result.chunk_id,
                "summary": doc.get("summary", ""),
                "description": doc.get("description", ""),
                "details": doc,  # Include full structured output
                "related": doc.get("see_also", doc.get("related", doc.get("cross_references", []))),
            })
        
        return {
            "summaries": summaries,
            "architecture_overview": "Generated via LLM analysis.",
            "highlights": [],
            "generation_stats": {
                "total_chunks": len(self.results),
                "successful": self.successful,
                "failed": self.failed,
                "total_tokens": self.total_tokens,
            },
        }


class OpenAIDocGenerator:
    """
    Documentation generator using OpenAI API.
    
    Features:
    - Structured JSON output via response_format
    - Domain-specific prompt templates
    - Batch processing with concurrency control
    - Token tracking for cost estimation
    - Graceful fallback when API unavailable
    
    Usage:
        generator = OpenAIDocGenerator()
        
        # Single chunk
        result = generator.generate_for_chunk(chunk)
        
        # Batch with concurrency
        results = await generator.generate_batch_async(chunks, max_concurrent=10)
        
        # Or sync batch
        results = generator.generate_batch(chunks)
    """
    
    # Default model - gpt-5-nano is the smallest/fastest modern model
    DEFAULT_MODEL = "gpt-5-nano"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        """
        Initialize the OpenAI documentation generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to gpt-5-nano)
            base_url: Custom API base URL (for Azure, proxies, etc.)
            max_tokens: Maximum tokens for each response
            temperature: Sampling temperature (lower = more deterministic)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.model = model or os.getenv("OPENAI_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize clients
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)
        
        logger.info(f"Initialized OpenAI doc generator with model: {self.model}")
    
    def generate_for_chunk(
        self,
        chunk: UniversalChunk,
        context: str = "",
        prompt_override: Optional[PromptTemplate] = None,
    ) -> DocumentationResult:
        """
        Generate documentation for a single chunk.
        
        Args:
            chunk: The chunk to document
            context: Additional context (e.g., related chunks)
            prompt_override: Custom prompt template (uses domain default if None)
        
        Returns:
            DocumentationResult with the generated documentation
        """
        try:
            # Get prompt template
            prompt = prompt_override or get_prompt_for_domain(chunk.domain)
            
            # Format the user prompt
            user_content = prompt.format_user_prompt(
                text=chunk.text,
                path=chunk.path,
                chunk_type=chunk.type.value if hasattr(chunk.type, 'value') else str(chunk.type),
                metadata=chunk.metadata.to_dict(),
                context=f"**Context:**\n{context}" if context else "",
            )
            
            # Call OpenAI API
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            # Parse response
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            try:
                documentation = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response for {chunk.chunk_id}: {e}")
                documentation = {"raw_response": content}
            
            return DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=True,
                documentation=documentation,
                tokens_used=tokens_used,
            )
            
        except Exception as e:
            logger.error(f"Error generating docs for {chunk.chunk_id}: {e}")
            return DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=False,
                error=str(e),
            )
    
    async def generate_for_chunk_async(
        self,
        chunk: UniversalChunk,
        context: str = "",
        prompt_override: Optional[PromptTemplate] = None,
    ) -> DocumentationResult:
        """
        Async version of generate_for_chunk.
        """
        try:
            prompt = prompt_override or get_prompt_for_domain(chunk.domain)
            
            user_content = prompt.format_user_prompt(
                text=chunk.text,
                path=chunk.path,
                chunk_type=chunk.type.value if hasattr(chunk.type, 'value') else str(chunk.type),
                metadata=chunk.metadata.to_dict(),
                context=f"**Context:**\n{context}" if context else "",
            )
            
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            try:
                documentation = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response for {chunk.chunk_id}: {e}")
                documentation = {"raw_response": content}
            
            return DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=True,
                documentation=documentation,
                tokens_used=tokens_used,
            )
            
        except Exception as e:
            logger.error(f"Error generating docs for {chunk.chunk_id}: {e}")
            return DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=False,
                error=str(e),
            )
    
    def generate_batch(
        self,
        chunks: list[UniversalChunk],
        context_map: Optional[dict[str, str]] = None,
        max_workers: int = 5,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchDocumentationResult:
        """
        Generate documentation for multiple chunks (sync, uses thread pool).
        
        Args:
            chunks: List of chunks to document
            context_map: Optional dict mapping chunk_id to context string
            max_workers: Maximum concurrent requests
            progress_callback: Called with (completed, total) after each chunk
        
        Returns:
            BatchDocumentationResult with all results
        """
        batch_result = BatchDocumentationResult()
        context_map = context_map or {}
        
        def process_chunk(chunk: UniversalChunk) -> DocumentationResult:
            context = context_map.get(chunk.chunk_id, "")
            return self.generate_for_chunk(chunk, context)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            
            for i, future in enumerate(futures):
                result = future.result()
                batch_result.add(result)
                
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
        
        return batch_result
    
    async def generate_batch_async(
        self,
        chunks: list[UniversalChunk],
        context_map: Optional[dict[str, str]] = None,
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchDocumentationResult:
        """
        Generate documentation for multiple chunks (async with concurrency limit).
        
        Args:
            chunks: List of chunks to document
            context_map: Optional dict mapping chunk_id to context string
            max_concurrent: Maximum concurrent API requests
            progress_callback: Called with (completed, total) after each chunk
        
        Returns:
            BatchDocumentationResult with all results
        """
        batch_result = BatchDocumentationResult()
        context_map = context_map or {}
        semaphore = asyncio.Semaphore(max_concurrent)
        completed = 0
        
        async def process_with_semaphore(chunk: UniversalChunk) -> DocumentationResult:
            nonlocal completed
            async with semaphore:
                context = context_map.get(chunk.chunk_id, "")
                result = await self.generate_for_chunk_async(chunk, context)
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(chunks))
                return result
        
        tasks = [process_with_semaphore(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        
        for result in results:
            batch_result.add(result)
        
        return batch_result
    
    def estimate_cost(
        self,
        chunks: list[UniversalChunk],
        avg_input_tokens: int = 500,
        avg_output_tokens: int = 300,
    ) -> dict[str, Any]:
        """
        Estimate the cost of documenting chunks.
        
        Note: Prices are approximate and may vary. Check OpenAI pricing for accuracy.
        
        Args:
            chunks: Chunks to estimate for
            avg_input_tokens: Average input tokens per chunk
            avg_output_tokens: Average output tokens per chunk
        
        Returns:
            Dict with cost estimates
        """
        # Approximate prices per 1M tokens (as of late 2025, adjust as needed)
        # gpt-5-nano is assumed to be their cheapest modern model
        prices = {
            "gpt-5-nano": {"input": 0.10, "output": 0.30},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        }
        
        model_prices = prices.get(self.model, prices["gpt-5-nano"])
        
        total_input = len(chunks) * avg_input_tokens
        total_output = len(chunks) * avg_output_tokens
        
        input_cost = (total_input / 1_000_000) * model_prices["input"]
        output_cost = (total_output / 1_000_000) * model_prices["output"]
        
        return {
            "model": self.model,
            "num_chunks": len(chunks),
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_cost_usd": round(input_cost + output_cost, 4),
            "input_cost_usd": round(input_cost, 4),
            "output_cost_usd": round(output_cost, 4),
            "note": "Estimates only. Actual costs may vary.",
        }


def create_doc_generator(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[OpenAIDocGenerator]:
    """
    Create an OpenAI doc generator if the API key is available.
    
    Returns None if:
    - openai package is not installed
    - No API key is available
    
    This allows graceful degradation to rule-based docs.
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not installed. LLM docs will be disabled.")
        return None
    
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("No OpenAI API key. LLM docs will be disabled.")
        return None
    
    try:
        return OpenAIDocGenerator(api_key=api_key, model=model)
    except Exception as e:
        logger.warning(f"Failed to create OpenAI doc generator: {e}")
        return None
