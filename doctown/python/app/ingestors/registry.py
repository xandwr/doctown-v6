"""
Ingestor Registry - Plugin system for domain ingestors.

This module provides automatic domain detection and ingestor selection.
Add new ingestors by registering them with the global registry.
"""
from __future__ import annotations

import logging
from typing import Optional, Type

from .base import DomainIngestor, Domain

logger = logging.getLogger(__name__)


class IngestorRegistry:
    """
    Registry for domain ingestors.
    
    Handles:
    - Registration of ingestor classes
    - Auto-detection of appropriate ingestor for given files
    - Priority-based selection when multiple ingestors match
    
    Usage:
        # Register a custom ingestor
        registry = IngestorRegistry()
        registry.register(MyCustomIngestor)
        
        # Auto-select ingestor for files
        ingestor = registry.select(files)
        chunks = ingestor.ingest(files)
    """
    
    def __init__(self):
        self._ingestors: list[DomainIngestor] = []
        self._fallback: Optional[DomainIngestor] = None
    
    def register(
        self,
        ingestor: DomainIngestor | Type[DomainIngestor],
        as_fallback: bool = False,
    ) -> None:
        """
        Register an ingestor.
        
        Args:
            ingestor: An ingestor instance or class
            as_fallback: If True, use this as the fallback when nothing matches
        """
        # Instantiate if given a class
        if isinstance(ingestor, type):
            ingestor = ingestor()
        
        if as_fallback:
            self._fallback = ingestor
            logger.debug(f"Registered fallback ingestor: {ingestor.name}")
        else:
            self._ingestors.append(ingestor)
            # Keep sorted by priority (highest first)
            self._ingestors.sort(key=lambda x: x.priority, reverse=True)
            logger.debug(f"Registered ingestor: {ingestor.name} (priority={ingestor.priority})")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister an ingestor by name.
        
        Returns:
            True if an ingestor was removed, False otherwise
        """
        original_count = len(self._ingestors)
        self._ingestors = [i for i in self._ingestors if i.name != name]
        
        if self._fallback and self._fallback.name == name:
            self._fallback = None
            return True
        
        return len(self._ingestors) < original_count
    
    def get(self, name: str) -> Optional[DomainIngestor]:
        """Get an ingestor by name."""
        for ingestor in self._ingestors:
            if ingestor.name == name:
                return ingestor
        if self._fallback and self._fallback.name == name:
            return self._fallback
        return None
    
    def list_ingestors(self) -> list[dict]:
        """List all registered ingestors with their metadata."""
        result = []
        for ingestor in self._ingestors:
            result.append({
                "name": ingestor.name,
                "domain": ingestor.domain.value,
                "priority": ingestor.priority,
                "is_fallback": False,
            })
        if self._fallback:
            result.append({
                "name": self._fallback.name,
                "domain": self._fallback.domain.value,
                "priority": self._fallback.priority,
                "is_fallback": True,
            })
        return result
    
    def select(self, files: dict[str, bytes]) -> DomainIngestor:
        """
        Select the appropriate ingestor for the given files.
        
        Checks ingestors in priority order, returns the first match.
        Falls back to the fallback ingestor if nothing matches.
        
        Args:
            files: Dict mapping file paths to file contents (as bytes)
        
        Returns:
            The selected ingestor
        
        Raises:
            ValueError: If no ingestor matches and no fallback is registered
        """
        logger.debug(f"Selecting ingestor for {len(files)} files")
        
        for ingestor in self._ingestors:
            try:
                if ingestor.detect(files):
                    logger.info(f"Selected ingestor: {ingestor.name}")
                    return ingestor
            except Exception as e:
                logger.warning(f"Ingestor {ingestor.name} detection failed: {e}")
                continue
        
        if self._fallback:
            logger.info(f"Using fallback ingestor: {self._fallback.name}")
            return self._fallback
        
        raise ValueError(
            "No ingestor could handle the given files. "
            "Register a fallback ingestor to handle unknown content."
        )
    
    def detect_domain(self, files: dict[str, bytes]) -> Domain:
        """
        Detect the domain of the given files without instantiating an ingestor.
        
        Returns:
            The detected Domain
        """
        ingestor = self.select(files)
        return ingestor.domain


# Global registry instance
_global_registry: Optional[IngestorRegistry] = None


def get_registry() -> IngestorRegistry:
    """Get the global ingestor registry, creating it if needed."""
    global _global_registry
    
    if _global_registry is None:
        _global_registry = IngestorRegistry()
        _setup_default_ingestors(_global_registry)
    
    return _global_registry


def _setup_default_ingestors(registry: IngestorRegistry) -> None:
    """Register the default ingestors."""
    # Import here to avoid circular imports
    from .code import CodeIngestor
    from .generic import GenericTextIngestor
    
    # Register built-in ingestors
    registry.register(CodeIngestor())
    
    # Generic is the fallback
    registry.register(GenericTextIngestor(), as_fallback=True)
    
    logger.info(f"Registered {len(registry._ingestors)} ingestors + fallback")


def select_ingestor(files: dict[str, bytes]) -> DomainIngestor:
    """
    Convenience function to select an ingestor using the global registry.
    
    Args:
        files: Dict mapping file paths to file contents (as bytes)
    
    Returns:
        The selected ingestor
    """
    return get_registry().select(files)


def register_ingestor(
    ingestor: DomainIngestor | Type[DomainIngestor],
    as_fallback: bool = False,
) -> None:
    """
    Convenience function to register an ingestor in the global registry.
    
    Args:
        ingestor: An ingestor instance or class
        as_fallback: If True, use this as the fallback when nothing matches
    """
    get_registry().register(ingestor, as_fallback=as_fallback)
