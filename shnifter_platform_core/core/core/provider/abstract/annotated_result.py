# Generated: 2025-07-04T09:50:40.242305
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Annotated result."""

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class AnnotatedResult(BaseModel, Generic[T]):
    """Annotated result allows fetchers to return metadata along with the data."""

    result: Optional[T] = Field(
        default=None,
        description="Serializable results.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Metadata.",
    )
