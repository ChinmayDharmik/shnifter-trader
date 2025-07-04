# Generated: 2025-07-04T09:50:40.294336
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Earnings Call Transcript Standard Model."""

from datetime import datetime
from typing import List, Set, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class EarningsCallTranscriptQueryParams(QueryParams):
    """Earnings Call Transcript rating Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))
    year: Union[int, str] = Field(description="Year of the earnings call transcript.")

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EarningsCallTranscriptData(Data):
    """Earnings Call Transcript Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    quarter: int = Field(description="Quarter of the earnings call transcript.")
    year: int = Field(description="Year of the earnings call transcript.")
    date: datetime = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    content: str = Field(description="Content of the earnings call transcript.")

    @field_validator("symbol", mode="before")
    @classmethod
    def to_upper(cls, v: Union[str, List[str], Set[str]]):
        """Convert field to uppercase."""
        if isinstance(v, str):
            return v.upper()
        return ",".join([symbol.upper() for symbol in list(v)])
