# Generated: 2025-07-04T09:50:40.475993
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Error."""

from typing import Optional, Union


class ShnifterError(Exception):
    """Shnifter Error."""

    def __init__(self, original: Optional[Union[str, Exception]] = None):
        """Initialize the ShnifterError."""
        self.original = original
        super().__init__(str(original))
