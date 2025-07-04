# Generated: 2025-07-04T09:50:40.296948
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Economic Indicators Standard Model."""

from datetime import date as dateType
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class EconomicIndicatorsQueryParams(QueryParams):
    """Economic Indicators Query."""

    country: Optional[str] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("country", "")
        + " The country represented by the indicator, if available.",
    )
    start_date: Optional[dateType] = Field(
        description=QUERY_DESCRIPTIONS.get("start_date", ""), default=None
    )
    end_date: Optional[dateType] = Field(
        description=QUERY_DESCRIPTIONS.get("end_date", ""), default=None
    )


class EconomicIndicatorsData(Data):
    """Economic Indicators Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    symbol_root: Optional[str] = Field(
        default=None, description="The root symbol for the indicator (e.g. GDP)."
    )
    symbol: Optional[str] = Field(
        default=None, description=DATA_DESCRIPTIONS.get("symbol", "")
    )
    country: Optional[str] = Field(
        default=None, description="The country represented by the data."
    )
    value: Optional[Union[int, float]] = Field(
        default=None, description=DATA_DESCRIPTIONS.get("value", "")
    )
