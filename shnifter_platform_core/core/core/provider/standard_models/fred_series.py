# Generated: 2025-07-04T09:50:40.336705
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FRED Series Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class SeriesQueryParams(QueryParams):
    """FRED Series Query."""

    symbol: str = Field(
        description=QUERY_DESCRIPTIONS.get("symbol", ""),
    )
    start_date: Optional[dateType] = Field(
        description=QUERY_DESCRIPTIONS.get("start_date", ""), default=None
    )
    end_date: Optional[dateType] = Field(
        description=QUERY_DESCRIPTIONS.get("end_date", ""), default=None
    )
    limit: Optional[int] = Field(
        description=QUERY_DESCRIPTIONS.get("limit", ""), default=100000
    )

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class SeriesData(Data):
    """FRED Series Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
