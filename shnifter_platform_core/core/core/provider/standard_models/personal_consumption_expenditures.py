# Generated: 2025-07-04T09:50:40.385353
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Personal Consumption Expenditures Standard Model."""

from datetime import date as dateType
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class PersonalConsumptionExpendituresQueryParams(QueryParams):
    """Personal Consumption Expenditures Query."""

    date: Optional[Union[dateType, str]] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("date", "")
        + " Default is the latest report.",
    )


class PersonalConsumptionExpendituresData(Data):
    """Personal Consumption Expenditures Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    value: float = Field(description=DATA_DESCRIPTIONS.get("value", ""))
