# Generated: 2025-07-04T09:50:40.402322
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Share Price Index Standard Model."""

from datetime import date as dateType
from typing import Literal, Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class SharePriceIndexQueryParams(QueryParams):
    """Share Price Index Query."""

    country: str = Field(
        description=QUERY_DESCRIPTIONS.get("country", ""),
        default="united_states",
    )
    frequency: Literal["monthly", "quarter", "annual"] = Field(
        description=QUERY_DESCRIPTIONS.get("frequency", ""),
        default="monthly",
        json_schema_extra={"choices": ["monthly", "quarter", "annual"]},
    )
    start_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("start_date")
    )
    end_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("end_date")
    )


class SharePriceIndexData(Data):
    """Share Price Index Data."""

    date: Optional[dateType] = Field(
        default=None, description=DATA_DESCRIPTIONS.get("date")
    )
    country: Optional[str] = Field(
        default=None,
        description=DATA_DESCRIPTIONS.get("country", ""),
    )
    value: Optional[float] = Field(
        default=None,
        description="Share price index value.",
    )
