"""Provider Registry Module (ShnifterBB version)."""

from typing import Dict

class Registry:
    """Maintain registry of providers (ShnifterBB version)."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._providers: Dict[str, object] = {}

    @property
    def providers(self):
        """Return a dictionary of providers."""
        return self._providers

    def include_provider(self, provider: object) -> None:
        """Include a provider in the registry."""
        self._providers[getattr(provider, 'name', '').lower()] = provider

class LoadingError(Exception):
    """Error loading provider (ShnifterBB version)."""
    pass

class RegistryLoader:
    """Stub for loading providers (ShnifterBB version)."""
    @staticmethod
    def from_extensions() -> Registry:
        # In ShnifterBB, you can manually register providers as needed
        return Registry()
