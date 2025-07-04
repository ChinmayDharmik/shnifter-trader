# Generated: 2025-07-04T09:50:40.416455
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Top Retail Standard Model."""

from datetime import date as DateType

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class TopRetailQueryParams(QueryParams):
    """Top Retail Search Query."""

    limit: int = Field(description=QUERY_DESCRIPTIONS.get("limit", ""), default=5)


class TopRetailData(Data):
    """Top Retail Search Data."""

    date: DateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    activity: float = Field(description="Activity of the symbol.")
    sentiment: float = Field(
        description="Sentiment of the symbol. 1 is bullish, -1 is bearish."
    )
