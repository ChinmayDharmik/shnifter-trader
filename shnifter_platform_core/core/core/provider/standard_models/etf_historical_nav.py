# Generated: 2025-07-04T09:50:40.315635
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Historical NAV model."""

from datetime import date as dateType

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class EtfHistoricalNavQueryParams(QueryParams):
    """ETF Historical NAV Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol")
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EtfHistoricalNavData(Data):
    """ETF Historical NAV Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    nav: float = Field(description="The net asset value on the date.")
