"""
LLM Integration - Documentation generation via OpenAI API.

This module provides structured output generation using OpenAI's API,
with domain-specific prompt templates for customization.
"""
from .openai_client import (
    OpenAIDocGenerator,
    DocumentationResult,
    BatchDocumentationResult,
    create_doc_generator,
)
from .prompts import PromptTemplateRegistry, get_prompt_for_domain

__all__ = [
    "OpenAIDocGenerator",
    "DocumentationResult",
    "BatchDocumentationResult",
    "create_doc_generator",
    "PromptTemplateRegistry",
    "get_prompt_for_domain",
]
