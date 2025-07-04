from core.events import EventLog, EventBus

class ProviderRegistry:
    """Registry for Shnifter providers and strategies."""
    def __init__(self):
        self._registry = {}
        EventLog.emit("INFO", "ProviderRegistry initialized.")

    def register(self, name, provider):
        self._registry[name] = provider
        EventLog.emit("INFO", f"Registered provider: {name}")

    def get(self, name):
        provider = self._registry.get(name)
        if provider is None:
            EventLog.emit("ERROR", f"Provider not found: {name}")
        return provider

    def list_providers(self):
        return list(self._registry.keys())

    def unregister(self, name):
        if name in self._registry:
            del self._registry[name]
            EventLog.emit("INFO", f"Unregistered provider: {name}")
