# Generated: 2025-07-04T09:50:40.264641
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Earnings Calendar Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class CalendarEarningsQueryParams(QueryParams):
    """Earnings Calendar Query."""

    start_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("start_date", "")
    )
    end_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("end_date", "")
    )


class CalendarEarningsData(Data):
    """Earnings Calendar Data."""

    report_date: dateType = Field(description="The date of the earnings report.")
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    name: Optional[str] = Field(description="Name of the entity.", default=None)
    eps_previous: Optional[float] = Field(
        default=None,
        description="The earnings-per-share from the same previously reported period.",
    )
    eps_consensus: Optional[float] = Field(
        default=None,
        description="The analyst conesus earnings-per-share estimate.",
    )
