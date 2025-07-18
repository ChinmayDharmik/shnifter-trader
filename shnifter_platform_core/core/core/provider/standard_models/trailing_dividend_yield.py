# Generated: 2025-07-04T09:50:40.417669
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Trailing Dividend Yield Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class TrailingDivYieldQueryParams(QueryParams):
    """Trailing Dividend Yield Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))
    limit: Optional[int] = Field(
        default=252,
        description=f"{QUERY_DESCRIPTIONS.get('limit', '')}"
        " Default is 252, the number of trading days in a year.",
    )


class TrailingDivYieldData(Data):
    """Trailing Dividend Yield Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    trailing_dividend_yield: float = Field(description="Trailing dividend yield.")
