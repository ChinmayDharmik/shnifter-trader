# Generated: 2025-07-04T09:50:40.412301
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Commitment of Traders Reports Search Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field


class SymbolMapQueryParams(QueryParams):
    """Commitment of Traders Reports Search Query."""

    query: str = Field(description="Search query.")
    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether or not to use cache. If True, cache will store for seven days.",
    )
