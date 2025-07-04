"""Provider registry map (ShnifterBB version)."""

from typing import Any, Dict, List, Optional

from .registry import Registry

class RegistryMap:
    """Class to store information about providers in the registry (ShnifterBB version)."""

    def __init__(self, registry: Optional[Registry] = None) -> None:
        """Initialize Registry Map."""
        self._registry = registry or Registry()
        self._available_providers = list(self._registry.providers.keys())

    @property
    def registry(self) -> Registry:
        """Get the registry."""
        return self._registry

    @property
    def available_providers(self) -> List[str]:
        """Get list of available providers."""
        return self._available_providers

    # Add more methods as needed for your ShnifterBB provider logic
