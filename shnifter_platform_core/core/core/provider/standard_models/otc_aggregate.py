# Generated: 2025-07-04T09:50:40.383543
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""OTC Aggregate Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field


class OTCAggregateQueryParams(QueryParams):
    """OTC Aggregate Query."""

    symbol: Optional[str] = Field(
        description=QUERY_DESCRIPTIONS.get("symbol", ""),
        default=None,
    )


class OTCAggregateData(Data):
    """OTC Aggregate Data."""

    update_date: dateType = Field(
        description="Most recent date on which total trades is updated based on data received from each ATS/OTC."
    )
    share_quantity: float = Field(
        description="Aggregate weekly total number of shares reported by each ATS for the Symbol."
    )
    trade_quantity: float = Field(
        description="Aggregate weekly total number of trades reported by each ATS for the Symbol"
    )
