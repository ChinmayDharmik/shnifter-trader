# Generated: 2025-07-04T09:50:40.478661
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Core App Abstract Model Tagged."""

from pydantic import BaseModel, Field
from uuid_extensions import uuid7str  # type: ignore


class Tagged(BaseModel):
    """Model for Tagged."""

    id: str = Field(default_factory=uuid7str, alias="_id")
