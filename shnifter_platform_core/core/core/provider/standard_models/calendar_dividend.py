# Generated: 2025-07-04T09:50:40.263791
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Dividend Calendar Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class CalendarDividendQueryParams(QueryParams):
    """Dividend Calendar Query."""

    start_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("start_date", "")
    )
    end_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("end_date", "")
    )


class CalendarDividendData(Data):
    """Dividend Calendar Data."""

    ex_dividend_date: dateType = Field(
        description="The ex-dividend date - the date on which the stock begins trading without rights to the dividend."
    )
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    amount: Optional[float] = Field(
        default=None, description="The dividend amount per share."
    )
    name: Optional[str] = Field(default=None, description="Name of the entity.")
    record_date: Optional[dateType] = Field(
        default=None,
        description="The record date of ownership for eligibility.",
    )
    payment_date: Optional[dateType] = Field(
        default=None,
        description="The payment date of the dividend.",
    )
    declaration_date: Optional[dateType] = Field(
        default=None,
        description="Declaration date of the dividend.",
    )
