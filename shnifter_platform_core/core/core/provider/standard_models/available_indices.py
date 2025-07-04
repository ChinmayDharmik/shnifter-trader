# Generated: 2025-07-04T09:50:40.252614
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Available Indices Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field


class AvailableIndicesQueryParams(QueryParams):
    """Available Indices Query."""


class AvailableIndicesData(Data):
    """Available Indices Data.

    Returns the list of available indices from a provider.
    """

    name: Optional[str] = Field(default=None, description="Name of the index.")
    currency: Optional[str] = Field(
        default=None, description="Currency the index is traded in."
    )
