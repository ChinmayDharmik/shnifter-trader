# Generated: 2025-07-04T09:50:40.288615
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Currency Available Pairs Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class CurrencyPairsQueryParams(QueryParams):
    """Currency Available Pairs Query."""

    query: Optional[str] = Field(
        default=None, description="Query to search for currency pairs."
    )


class CurrencyPairsData(Data):
    """Currency Available Pairs Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    name: Optional[str] = Field(default=None, description="Name of the currency pair.")
