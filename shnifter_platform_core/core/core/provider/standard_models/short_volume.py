# Generated: 2025-07-04T09:50:40.405974
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Short Volume Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class ShortVolumeQueryParams(QueryParams):
    """Short Volume Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol"))


class ShortVolumeData(Data):
    """Short Volume Data."""

    date: Optional[dateType] = Field(
        default=None, description=DATA_DESCRIPTIONS.get("date")
    )

    market: Optional[str] = Field(
        default=None,
        description="Reporting Facility ID. N=NYSE TRF, Q=NASDAQ TRF Carteret, B=NASDAQ TRY Chicago, D=FINRA ADF",
    )

    short_volume: Optional[int] = Field(
        default=None,
        description=(
            "Aggregate reported share volume of executed short sale "
            "and short sale exempt trades during regular trading hours"
        ),
    )

    short_exempt_volume: Optional[int] = Field(
        default=None,
        description="Aggregate reported share volume of executed short sale exempt trades during regular trading hours",
    )

    total_volume: Optional[int] = Field(
        default=None,
        description="Aggregate reported share volume of executed trades during regular trading hours",
    )
