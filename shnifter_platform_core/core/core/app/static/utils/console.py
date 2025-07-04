# Generated: 2025-07-04T09:50:40.471426
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Console module."""

from shnifter_core.env import Env


class Console:
    """Console to be used by builder and linters."""

    def __init__(self, verbose: bool):
        """Initialize the console."""
        self.verbose = verbose

    def log(self, message: str, **kwargs):
        """Console log method."""
        if self.verbose or Env().DEBUG_MODE:
            print(message, **kwargs)  # noqa: T201
