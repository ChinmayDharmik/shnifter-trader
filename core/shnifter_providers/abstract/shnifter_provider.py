# This is a template for shnifterizing provider base classes.
from core.events import EventLog, EventBus

class ShnifterProvider:
    """Base class for all Shnifter data providers."""
    def __init__(self, name):
        self.name = name
        EventLog.emit("INFO", f"Initialized provider: {self.name}")

    def fetch(self, *args, **kwargs):
        EventLog.emit("DEBUG", f"Fetching data from {self.name} with args={args}, kwargs={kwargs}")
        raise NotImplementedError("fetch() must be implemented by subclasses.")

    def log_event(self, message, level="INFO"):
        EventLog.emit(level, f"[{self.name}] {message}")
        EventBus.publish(level, {"provider": self.name, "message": message})
