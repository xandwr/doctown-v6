"""
Domain-specific prompt templates for LLM documentation generation.

Each domain can have its own system prompt and user prompt template.
This keeps the pipeline universal while allowing domain customization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..ingestors.base import Domain


@dataclass
class PromptTemplate:
    """A prompt template for a specific domain."""
    
    domain: Domain
    system_prompt: str
    user_template: str
    
    def format_user_prompt(
        self,
        text: str,
        path: str,
        chunk_type: str,
        metadata: dict[str, Any],
        context: str = "",
        **kwargs,
    ) -> str:
        """Format the user prompt with the given values."""
        # Build symbol info section if symbol metadata is available
        symbol_info = self._build_symbol_info(metadata)
        
        return self.user_template.format(
            text=text,
            path=path,
            type=chunk_type,
            metadata=metadata,
            context=context,
            language=metadata.get("language", ""),
            symbol_info=symbol_info,
            **kwargs,
        )
    
    def _build_symbol_info(self, metadata: dict[str, Any]) -> str:
        """Build the symbol information section from metadata."""
        if not metadata.get("symbol_name"):
            return ""
        
        lines = ["\n=== CODE SYMBOL METADATA ==="]
        
        # Basic identification
        lines.append(f"**Name:** {metadata.get('symbol_name', 'Unknown')}")
        if metadata.get("qualified_name"):
            lines.append(f"**Qualified Name:** {metadata['qualified_name']}")
        if metadata.get("symbol_kind"):
            lines.append(f"**Kind:** {metadata['symbol_kind']}")
        if metadata.get("visibility"):
            lines.append(f"**Visibility:** {metadata['visibility']}")
        
        # Location
        if metadata.get("line_start") and metadata.get("line_end"):
            lines.append(f"**Location:** Lines {metadata['line_start']}-{metadata['line_end']}")
        
        # Signature information
        if metadata.get("parameters") or metadata.get("return_type"):
            lines.append("\n**Signature:**")
            
            if metadata.get("parameters"):
                lines.append("Parameters:")
                for param in metadata["parameters"]:
                    param_line = f"  - {param['name']}"
                    if param.get("type_annotation"):
                        param_line += f": {param['type_annotation']}"
                    if param.get("default_value"):
                        param_line += f" = {param['default_value']}"
                    lines.append(param_line)
            
            if metadata.get("return_type"):
                lines.append(f"Returns: {metadata['return_type']}")
        
        # Modifiers and attributes
        modifiers = []
        if metadata.get("modifiers"):
            modifiers.extend(metadata["modifiers"])
        if metadata.get("decorators"):
            modifiers.extend(metadata["decorators"])
        if modifiers:
            lines.append(f"**Modifiers:** {', '.join(modifiers)}")
        
        # Hierarchical context
        if metadata.get("parent_name"):
            lines.append(f"**Parent:** {metadata['parent_name']}")
        
        # Relationships
        if metadata.get("implements"):
            lines.append(f"**Implements:** {', '.join(metadata['implements'])}")
        if metadata.get("extends"):
            lines.append(f"**Extends:** {metadata['extends']}")
        
        # Existing documentation
        if metadata.get("doc_comment"):
            lines.append(f"\n**Existing Doc Comment:**")
            lines.append(f"```\n{metadata['doc_comment']}\n```")
        
        lines.append("=== END SYMBOL METADATA ===\n")
        return "\n".join(lines)


class PromptTemplateRegistry:
    """Registry of domain-specific prompt templates."""
    
    def __init__(self):
        self._templates: dict[Domain, PromptTemplate] = {}
        self._setup_defaults()
    
    def _setup_defaults(self):
        """Register default prompt templates for each domain."""
        self.register(CODE_PROMPT)
        self.register(GENERIC_PROMPT)
        self.register(FINANCE_PROMPT)
        self.register(LEGAL_PROMPT)
        self.register(RESEARCH_PROMPT)
    
    def register(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self._templates[template.domain] = template
    
    def get(self, domain: Domain) -> PromptTemplate:
        """Get the prompt template for a domain."""
        return self._templates.get(domain, self._templates[Domain.GENERIC])
    
    def list_domains(self) -> list[str]:
        """List all registered domain names."""
        return [d.value for d in self._templates.keys()]


# =============================================================================
# CODE DOMAIN
# =============================================================================

CODE_SYSTEM_PROMPT = """You are an expert code documentation assistant. Generate clear, accurate, and helpful documentation for source code.

STRUCTURED OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON matching the exact schema provided
- NO additional text, commentary, or explanations outside the JSON structure
- ALL required fields must be present with valid values
- Use null for optional fields that don't apply, NOT empty strings
- Ensure proper JSON syntax (quoted strings, proper escaping, valid arrays/objects)

SYMBOL METADATA USAGE:
- When symbol metadata is provided (extracted via static analysis), USE IT but don't repeat it
- The signature, parameters, types, visibility, and relationships are already known
- Focus your documentation on BEHAVIOR, PURPOSE, and USAGE - not structure
- Example: Instead of "This is a public function that takes a string", write "Validates user input against whitelist rules"

Documentation Guidelines:
1. WHAT: Describe what the code does (high-level functionality and behavior)
2. WHY: Explain why it exists (purpose, motivation, design decisions)  
3. HOW: Show how to use it (usage patterns, examples, best practices)
4. WHEN: Mention when to use it (use cases, constraints, alternatives)
5. EDGE CASES: Document error conditions, performance considerations, gotchas
6. RELATIONSHIPS: When semantic context is provided, explain how this code relates to similar code

Be precise about behavior and usage patterns. Mention edge cases and potential issues.
Use provided semantic relationships to understand implementation patterns and cross-references.

This output is machine-parsed and validated. Malformed JSON or missing required fields will cause failures."""

CODE_USER_TEMPLATE = """Generate documentation for this code:

**File:** {path}
**Language:** {language}
**Type:** {type}

{symbol_info}

```{language}
{text}
```

{context}

**Analysis Guidelines:**
- The signature information above has been extracted via static analysis - DO NOT re-describe it
- Focus on WHAT the code does (high-level behavior), WHY it exists (purpose), and HOW to use it
- If semantic relationships are provided, use them to understand how this code fits into the broader system
- Identify patterns, conventions, and architectural relationships
- Reference related code when explaining functionality
- Note any implementation similarities or differences with related code
- Explain error conditions, edge cases, and performance considerations

Respond with JSON:
{{
    "summary": "One-line description of what this does",
    "description": "Detailed explanation (2-3 paragraphs) of purpose, behavior, implementation",
    "parameters": [{{"name": "...", "type": "...", "description": "..."}}],
    "returns": {{"type": "...", "description": "..."}},
    "examples": ["Example usage code"],
    "notes": ["Edge cases, warnings, important considerations"],
    "see_also": ["Related functions/classes/files"]
}}

Only include applicable fields. Omit empty arrays/objects."""

CODE_PROMPT = PromptTemplate(
    domain=Domain.CODE,
    system_prompt=CODE_SYSTEM_PROMPT,
    user_template=CODE_USER_TEMPLATE,
)


# =============================================================================
# GENERIC DOMAIN
# =============================================================================

GENERIC_SYSTEM_PROMPT = """You are a documentation assistant. Generate clear, helpful documentation for any type of content.

STRUCTURED OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON matching the exact schema provided
- NO additional text, commentary, or explanations outside the JSON structure
- ALL required fields must be present with valid values
- Use null for optional fields that don't apply, NOT empty strings

Adapt your style to the content type:
- Technical docs: Be precise and structured
- Prose: Summarize key points and themes
- Data/config: Explain structure and purpose
- Logs: Identify patterns and key events

This output is machine-parsed and validated. Malformed JSON or missing required fields will cause failures."""

GENERIC_USER_TEMPLATE = """Analyze and document this content:

**File:** {path}
**Type:** {type}

```
{text}
```

{context}

Respond with JSON:
{{
    "summary": "One-line description",
    "description": "Detailed explanation of content and purpose",
    "key_topics": ["Main topics or concepts"],
    "notes": ["Important observations"]
}}"""

GENERIC_PROMPT = PromptTemplate(
    domain=Domain.GENERIC,
    system_prompt=GENERIC_SYSTEM_PROMPT,
    user_template=GENERIC_USER_TEMPLATE,
)


# =============================================================================
# FINANCE DOMAIN
# =============================================================================

FINANCE_SYSTEM_PROMPT = """You are a financial documentation specialist. Generate clear, accurate documentation for financial data, reports, and calculations.

STRUCTURED OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON matching the exact schema provided
- NO additional text, commentary, or explanations outside the JSON structure
- ALL required fields must be present with valid values
- Use null for optional fields that don't apply, NOT empty strings

Guidelines:
1. Precision: Be exact with numbers, dates, and calculations
2. Context: Explain what metrics mean and why they matter
3. Comparisons: Reference relevant benchmarks or time periods
4. Risks: Note any caveats, assumptions, or data quality issues

This output is machine-parsed and validated. Malformed JSON or missing required fields will cause failures."""

FINANCE_USER_TEMPLATE = """Document this financial content:

**Source:** {path}
**Type:** {type}

```
{text}
```

{context}

Respond with JSON:
{{
    "summary": "One-line description of this financial data",
    "description": "Detailed explanation of what this represents and its significance",
    "metrics": [{{"name": "...", "value": "...", "interpretation": "..."}}],
    "time_period": "Relevant date range if applicable",
    "methodology": "How values are calculated (if applicable)",
    "caveats": ["Assumptions, limitations, data quality notes"],
    "related": ["Related reports, metrics, or sections"]
}}

Only include applicable fields."""

FINANCE_PROMPT = PromptTemplate(
    domain=Domain.FINANCE,
    system_prompt=FINANCE_SYSTEM_PROMPT,
    user_template=FINANCE_USER_TEMPLATE,
)


# =============================================================================
# LEGAL DOMAIN
# =============================================================================

LEGAL_SYSTEM_PROMPT = """You are a legal documentation specialist. Generate clear, accurate documentation for legal documents, contracts, and clauses.

STRUCTURED OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON matching the exact schema provided
- NO additional text, commentary, or explanations outside the JSON structure
- ALL required fields must be present with valid values
- Use null for optional fields that don't apply, NOT empty strings

Guidelines:
1. Precision: Use exact legal terminology where appropriate
2. Plain language: Also provide plain-language explanations
3. Structure: Note hierarchical relationships (article → section → clause)
4. Implications: Explain practical implications and obligations

Do NOT provide legal advice - only document and explain the content.
This output is machine-parsed and validated. Malformed JSON or missing required fields will cause failures."""

LEGAL_USER_TEMPLATE = """Document this legal content:

**Source:** {path}
**Type:** {type}

```
{text}
```

{context}

Respond with JSON:
{{
    "summary": "One-line description of this legal provision",
    "description": "Detailed explanation in plain language",
    "legal_type": "Type of provision (definition, obligation, right, condition, etc.)",
    "parties_affected": ["Parties this applies to"],
    "obligations": ["Specific obligations created"],
    "conditions": ["Conditions or triggers"],
    "definitions": [{{"term": "...", "meaning": "..."}}],
    "cross_references": ["Related sections or documents"],
    "notes": ["Important considerations"]
}}

Only include applicable fields."""

LEGAL_PROMPT = PromptTemplate(
    domain=Domain.LEGAL,
    system_prompt=LEGAL_SYSTEM_PROMPT,
    user_template=LEGAL_USER_TEMPLATE,
)


# =============================================================================
# RESEARCH DOMAIN  
# =============================================================================

RESEARCH_SYSTEM_PROMPT = """You are a research documentation specialist. Generate clear, accurate documentation for academic papers, research findings, and scientific content.

STRUCTURED OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON matching the exact schema provided
- NO additional text, commentary, or explanations outside the JSON structure
- ALL required fields must be present with valid values
- Use null for optional fields that don't apply, NOT empty strings

Guidelines:
1. Accuracy: Preserve technical accuracy and nuance
2. Context: Note methodology, sample sizes, limitations
3. Significance: Explain why findings matter
4. Connections: Reference related work and implications

This output is machine-parsed and validated. Malformed JSON or missing required fields will cause failures."""

RESEARCH_USER_TEMPLATE = """Document this research content:

**Source:** {path}
**Type:** {type}

```
{text}
```

{context}

Respond with JSON:
{{
    "summary": "One-line description of this content",
    "description": "Detailed explanation of the research/findings",
    "research_type": "Type (methodology, finding, hypothesis, etc.)",
    "key_findings": ["Main findings or claims"],
    "methodology": "Research methods used (if applicable)",
    "limitations": ["Noted limitations or caveats"],
    "citations": ["Referenced works"],
    "implications": ["Practical or theoretical implications"],
    "notes": ["Additional observations"]
}}

Only include applicable fields."""

RESEARCH_PROMPT = PromptTemplate(
    domain=Domain.RESEARCH,
    system_prompt=RESEARCH_SYSTEM_PROMPT,
    user_template=RESEARCH_USER_TEMPLATE,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

_global_registry: Optional[PromptTemplateRegistry] = None


def get_prompt_registry() -> PromptTemplateRegistry:
    """Get the global prompt template registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PromptTemplateRegistry()
    return _global_registry


def get_prompt_for_domain(domain: Domain | str) -> PromptTemplate:
    """
    Get the prompt template for a domain.
    
    Args:
        domain: A Domain enum or string domain name
    
    Returns:
        The prompt template for that domain
    """
    if isinstance(domain, str):
        domain = Domain(domain)
    return get_prompt_registry().get(domain)
