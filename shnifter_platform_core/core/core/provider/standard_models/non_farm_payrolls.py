# Generated: 2025-07-04T09:50:40.379001
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""NonFarm Payrolls Standard Model."""

from datetime import date as dateType
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class NonFarmPayrollsQueryParams(QueryParams):
    """NonFarm Payrolls Query."""

    date: Optional[Union[dateType, str]] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("date", "")
        + " Default is the latest report.",
    )


class NonFarmPayrollsData(Data):
    """NonFarm Payrolls Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    value: float = Field(description=DATA_DESCRIPTIONS.get("value", ""))
