# Generated: 2025-07-04T09:50:40.479475
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Module for warnings."""

from warnings import WarningMessage

from pydantic import BaseModel


class Warning_(BaseModel):
    """Model for Warning."""

    category: str
    message: str


def cast_warning(w: WarningMessage) -> Warning_:
    """Cast a warning to a pydantic model."""
    return Warning_(
        category=w.category.__name__,
        message=str(w.message),
    )


class ShnifterWarning(Warning):
    """Base class for Shnifter warnings."""
