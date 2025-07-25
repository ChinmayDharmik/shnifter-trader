# Generated: 2025-07-04T09:50:40.287752
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Currency Historical Price Standard Model."""

from datetime import (
    date as dateType,
    datetime,
)
from typing import List, Optional, Set, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, PositiveFloat, field_validator


class CurrencyHistoricalQueryParams(QueryParams):
    """Currency Historical Price Query."""

    symbol: str = Field(
        description=QUERY_DESCRIPTIONS.get("symbol", "")
        + " Can use CURR1-CURR2 or CURR1CURR2 format."
    )
    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )

    @field_validator("symbol", mode="before", check_fields=False)
    def validate_symbol(
        cls, v: Union[str, List[str], Set[str]]
    ):  # pylint: disable=E0213
        """Convert field to uppercase and remove '-'."""
        if isinstance(v, str):
            return v.upper().replace("-", "")
        return ",".join([symbol.upper().replace("-", "") for symbol in list(v)])


class CurrencyHistoricalData(Data):
    """Currency Historical Price Data."""

    date: Union[dateType, datetime] = Field(
        description=DATA_DESCRIPTIONS.get("date", "")
    )
    open: PositiveFloat = Field(description=DATA_DESCRIPTIONS.get("open", ""))
    high: PositiveFloat = Field(description=DATA_DESCRIPTIONS.get("high", ""))
    low: PositiveFloat = Field(description=DATA_DESCRIPTIONS.get("low", ""))
    close: PositiveFloat = Field(description=DATA_DESCRIPTIONS.get("close", ""))
    volume: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("volume", ""), default=None
    )
    vwap: Optional[PositiveFloat] = Field(
        description=DATA_DESCRIPTIONS.get("vwap", ""), default=None
    )
