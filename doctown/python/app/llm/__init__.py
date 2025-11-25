"""
LLM Integration - Documentation generation via OpenAI API or local models.

This module provides structured output generation using either:
- OpenAI API (cloud-based, requires API key)
- Local LLM (on-device, requires GPU)

Both implementations share the same interface for drop-in compatibility.
"""
from .openai_client import (
    OpenAIDocGenerator,
    DocumentationResult,
    BatchDocumentationResult,
    create_doc_generator,
)
from .local_client import (
    LocalDocGenerator,
    create_local_generator,
)
from .prompts import PromptTemplateRegistry, get_prompt_for_domain

__all__ = [
    "OpenAIDocGenerator",
    "LocalDocGenerator",
    "DocumentationResult",
    "BatchDocumentationResult",
    "create_doc_generator",
    "create_local_generator",
    "PromptTemplateRegistry",
    "get_prompt_for_domain",
]
