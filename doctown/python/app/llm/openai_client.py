"""
OpenAI Client - Structured documentation generation via OpenAI API.

Uses gpt-5-nano (or configurable model) for fast, cheap documentation generation
with structured JSON output via Pydantic models and structured outputs API.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, Literal, Callable
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

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


# OpenAI Pricing Table (as of November 2025)
# Prices are in USD per 1M tokens
OPENAI_PRICING = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-5.1-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-chat-latest": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-codex": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-pro": {"input": 15.00, "output": 120.00},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-realtime": {"input": 4.00, "cached_input": 0.40, "output": 16.00},
    "gpt-realtime-mini": {"input": 0.60, "cached_input": 0.06, "output": 2.40},
    "gpt-4o-realtime-preview": {"input": 5.00, "cached_input": 2.50, "output": 20.00},
    "gpt-4o-mini-realtime-preview": {"input": 0.60, "cached_input": 0.30, "output": 2.40},
    "gpt-audio": {"input": 2.50, "output": 10.00},
    "gpt-audio-mini": {"input": 0.60, "output": 2.40},
    "gpt-4o-audio-preview": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini-audio-preview": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "cached_input": 7.50, "output": 60.00},
    "o1-pro": {"input": 150.00, "output": 600.00},
    "o3-pro": {"input": 20.00, "output": 80.00},
    "o3": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-deep-research": {"input": 10.00, "cached_input": 2.50, "output": 40.00},
    "o4-mini": {"input": 1.10, "cached_input": 0.275, "output": 4.40},
    "o4-mini-deep-research": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "o3-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "o1-mini": {"input": 1.10, "cached_input": 0.55, "output": 4.40},
    "gpt-5.1-codex-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "codex-mini-latest": {"input": 1.50, "cached_input": 0.375, "output": 6.00},
    "gpt-5-search-api": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-4o-mini-search-preview": {"input": 0.15, "output": 0.60},
    "gpt-4o-search-preview": {"input": 2.50, "output": 10.00},
    "computer-use-preview": {"input": 3.00, "output": 12.00},
    "gpt-image-1": {"input": 5.00, "cached_input": 1.25},
    "gpt-image-1-mini": {"input": 2.00, "cached_input": 0.20},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> dict[str, float]:
    """
    Calculate the cost of an OpenAI API call based on actual token usage.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens (excluding cached)
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens (if applicable)
    
    Returns:
        Dictionary with cost breakdown
    """
    pricing = OPENAI_PRICING.get(model)
    if not pricing:
        # Use gpt-5-nano as fallback for unknown models
        logger.warning(f"Unknown model '{model}', using gpt-5-nano pricing as fallback")
        pricing = OPENAI_PRICING["gpt-5-nano"]
    
    # Calculate costs per million tokens
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    cached_cost = 0.0
    
    if cached_tokens > 0 and "cached_input" in pricing:
        cached_cost = (cached_tokens / 1_000_000) * pricing["cached_input"]
    
    total_cost = input_cost + output_cost + cached_cost
    
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "cached_cost": cached_cost,
        "total_cost": total_cost,
    }


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================

class Parameter(BaseModel):
    """Model for function/method parameters."""
    name: str
    type: str
    description: str

class ReturnValue(BaseModel):
    """Model for return value documentation."""
    type: str
    description: str

class CodeDocumentation(BaseModel):
    """Structured output model for code documentation."""
    summary: str = Field(description="One-line description of what this code does")
    description: str = Field(description="Detailed explanation (2-3 paragraphs) of purpose, behavior, implementation")
    parameters: Optional[list[Parameter]] = Field(default=None, description="Function/method parameters")
    returns: Optional[ReturnValue] = Field(default=None, description="Return value description")
    examples: Optional[list[str]] = Field(default=None, description="Example usage code")
    notes: Optional[list[str]] = Field(default=None, description="Edge cases, warnings, important considerations")
    see_also: Optional[list[str]] = Field(default=None, description="Related functions/classes/files")

class GenericDocumentation(BaseModel):
    """Structured output model for generic content documentation."""
    summary: str = Field(description="One-line description")
    description: str = Field(description="Detailed explanation of content and purpose")
    key_topics: Optional[list[str]] = Field(default=None, description="Main topics or concepts")
    notes: Optional[list[str]] = Field(default=None, description="Important observations")

class FinanceDocumentation(BaseModel):
    """Structured output model for finance documentation."""
    summary: str = Field(description="One-line description")
    description: str = Field(description="Detailed explanation")
    financial_concepts: Optional[list[str]] = Field(default=None, description="Key financial concepts")
    metrics: Optional[list[str]] = Field(default=None, description="Financial metrics mentioned")
    notes: Optional[list[str]] = Field(default=None, description="Important observations")

class LegalDocumentation(BaseModel):
    """Structured output model for legal documentation."""
    summary: str = Field(description="One-line description")
    description: str = Field(description="Detailed explanation")
    legal_concepts: Optional[list[str]] = Field(default=None, description="Key legal concepts")
    obligations: Optional[list[str]] = Field(default=None, description="Obligations or requirements")
    notes: Optional[list[str]] = Field(default=None, description="Important observations")

class ResearchDocumentation(BaseModel):
    """Structured output model for research documentation."""
    summary: str = Field(description="One-line description")
    description: str = Field(description="Detailed explanation")
    key_findings: Optional[list[str]] = Field(default=None, description="Main findings or results")
    methodology: Optional[str] = Field(default=None, description="Research methodology")
    notes: Optional[list[str]] = Field(default=None, description="Important observations")


def get_model_for_domain(domain: Domain) -> type[BaseModel]:
    """Get the appropriate Pydantic model for a domain."""
    domain_models = {
        Domain.CODE: CodeDocumentation,
        Domain.FINANCE: FinanceDocumentation,
        Domain.LEGAL: LegalDocumentation,
        Domain.RESEARCH: ResearchDocumentation,
        Domain.GENERIC: GenericDocumentation,
    }
    return domain_models.get(domain, GenericDocumentation)


@dataclass
class DocumentationResult:
    """Result of documenting a single chunk."""
    
    chunk_id: str
    success: bool
    documentation: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_used: int = 0
    _input_tokens: int = 0
    _output_tokens: int = 0
    _cached_tokens: int = 0
    
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
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    successful: int = 0
    failed: int = 0
    model: str = ""
    total_cost: float = 0.0
    
    def add(self, result: DocumentationResult, input_tokens: int = 0, output_tokens: int = 0, cached_tokens: int = 0) -> None:
        """Add a result to the batch."""
        self.results.append(result)
        self.total_tokens += result.tokens_used
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cached_tokens += cached_tokens
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
    
    def calculate_cost(self) -> None:
        """Calculate the total cost based on actual token usage."""
        if self.model:
            cost_breakdown = calculate_cost(
                self.model,
                self.input_tokens,
                self.output_tokens,
                self.cached_tokens
            )
            self.total_cost = cost_breakdown["total_cost"]
    
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
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cached_tokens": self.cached_tokens,
                "total_cost_usd": round(self.total_cost, 4),
                "model": self.model,
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
        max_completion_tokens: int = 4096,
        temperature: float = 1.0,
    ):
        """
        Initialize the OpenAI documentation generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (defaults to gpt-5-nano)
            base_url: Custom API base URL (for Azure, proxies, etc.)
            max_completion_tokens: Maximum tokens for each response
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
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        
        # Initialize clients
        if OpenAI is None or AsyncOpenAI is None:
            raise ImportError(
                "openai package not installed or failed to import OpenAI client classes."
            )

        # Only pass supported arguments to OpenAI/AsyncOpenAI constructors
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url if self.base_url else None)
        self._async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url if self.base_url else None)

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
            
            # Get the appropriate Pydantic model for this domain
            response_model = get_model_for_domain(chunk.domain)
            
            # Call OpenAI API with structured outputs
            response = self._client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                response_format=response_model,
            )
            
            # Get the parsed object
            parsed_obj = response.choices[0].message.parsed
            
            # Extract token usage details
            if response.usage:
                tokens_used = response.usage.total_tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(response.usage, 'prompt_tokens_details') else 0
            else:
                tokens_used = 0
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
            
            # Check if parsing succeeded
            if parsed_obj is None:
                logger.warning(f"Failed to parse structured output for {chunk.chunk_id}")
                logger.debug(f"Full response: {response}")
                return DocumentationResult(
                    chunk_id=chunk.chunk_id,
                    success=False,
                    error="Failed to parse structured output",
                    tokens_used=tokens_used,
                )
            
            # Convert Pydantic model to dict
            documentation = parsed_obj.model_dump(exclude_none=True)
            
            result = DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=True,
                documentation=documentation,
                tokens_used=tokens_used,
            )
            
            # Store token breakdown in metadata for batch cost calculation
            result._input_tokens = input_tokens
            result._output_tokens = output_tokens
            result._cached_tokens = cached_tokens
            
            return result
            
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
            
            # Get the appropriate Pydantic model for this domain
            response_model = get_model_for_domain(chunk.domain)
            
            response = await self._async_client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                response_format=response_model,
            )
            
            # Get the parsed object
            parsed_obj = response.choices[0].message.parsed
            
            # Extract token usage details
            if response.usage:
                tokens_used = response.usage.total_tokens
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens
                cached_tokens = getattr(response.usage.prompt_tokens_details, 'cached_tokens', 0) if hasattr(response.usage, 'prompt_tokens_details') else 0
            else:
                tokens_used = 0
                input_tokens = 0
                output_tokens = 0
                cached_tokens = 0
            
            # Check if parsing succeeded
            if parsed_obj is None:
                logger.warning(f"Failed to parse structured output for {chunk.chunk_id}")
                logger.debug(f"Full response: {response}")
                return DocumentationResult(
                    chunk_id=chunk.chunk_id,
                    success=False,
                    error="Failed to parse structured output",
                    tokens_used=tokens_used,
                )
            
            # Convert Pydantic model to dict
            documentation = parsed_obj.model_dump(exclude_none=True)
            
            result = DocumentationResult(
                chunk_id=chunk.chunk_id,
                success=True,
                documentation=documentation,
                tokens_used=tokens_used,
            )
            
            # Store token breakdown in metadata for batch cost calculation
            result._input_tokens = input_tokens
            result._output_tokens = output_tokens
            result._cached_tokens = cached_tokens
            
            return result
            
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
        
        batch_result.model = self.model
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
            
            for i, future in enumerate(futures):
                result = future.result()
                input_tokens = getattr(result, '_input_tokens', 0)
                output_tokens = getattr(result, '_output_tokens', 0)
                cached_tokens = getattr(result, '_cached_tokens', 0)
                batch_result.add(result, input_tokens, output_tokens, cached_tokens)
                
                if progress_callback:
                    progress_callback(i + 1, len(chunks))
        
        # Calculate total cost
        batch_result.calculate_cost()
        
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
        
        batch_result.model = self.model
        for result in results:
            input_tokens = getattr(result, '_input_tokens', 0)
            output_tokens = getattr(result, '_output_tokens', 0)
            cached_tokens = getattr(result, '_cached_tokens', 0)
            batch_result.add(result, input_tokens, output_tokens, cached_tokens)
        
        # Calculate total cost
        batch_result.calculate_cost()
        
        return batch_result
    
    def estimate_cost(
        self,
        chunks: list[UniversalChunk],
        avg_input_tokens: int = 650,
        avg_output_tokens: int = 350,
    ) -> dict[str, Any]:
        """
        Estimate the cost of documenting chunks before generation.
        
        Uses official OpenAI pricing table (November 2025).
        
        Args:
            chunks: Chunks to estimate for
            avg_input_tokens: Average input tokens per chunk (default: 650)
            avg_output_tokens: Average output tokens per chunk (default: 350)
        
        Returns:
            Dict with cost estimates
        """
        total_input = len(chunks) * avg_input_tokens
        total_output = len(chunks) * avg_output_tokens
        
        cost_breakdown = calculate_cost(self.model, total_input, total_output)
        
        return {
            "model": self.model,
            "num_chunks": len(chunks),
            "estimated_input_tokens": total_input,
            "estimated_output_tokens": total_output,
            "estimated_total_tokens": total_input + total_output,
            "estimated_cost_usd": round(cost_breakdown["total_cost"], 4),
            "input_cost_usd": round(cost_breakdown["input_cost"], 4),
            "output_cost_usd": round(cost_breakdown["output_cost"], 4),
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
