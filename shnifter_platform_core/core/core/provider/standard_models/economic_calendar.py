# Generated: 2025-07-04T09:50:40.296075
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Economic Calendar Standard Model."""

from datetime import (
    date as dateType,
    datetime,
)
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class EconomicCalendarQueryParams(QueryParams):
    """Economic Calendar Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )


class EconomicCalendarData(Data):
    """Economic Calendar Data."""

    date: Optional[datetime] = Field(
        default=None, description=DATA_DESCRIPTIONS.get("date", "")
    )
    country: Optional[str] = Field(default=None, description="Country of event.")
    category: Optional[str] = Field(default=None, description="Category of event.")
    event: Optional[str] = Field(default=None, description="Event name.")
    importance: Optional[str] = Field(
        default=None, description="The importance level for the event."
    )
    source: Optional[str] = Field(default=None, description="Source of the data.")
    currency: Optional[str] = Field(default=None, description="Currency of the data.")
    unit: Optional[str] = Field(default=None, description="Unit of the data.")
    consensus: Optional[Union[str, float]] = Field(
        default=None,
        description="Average forecast among a representative group of economists.",
    )
    previous: Optional[Union[str, float]] = Field(
        default=None,
        description="Value for the previous period after the revision (if revision is applicable).",
    )
    revised: Optional[Union[str, float]] = Field(
        default=None,
        description="Revised previous value, if applicable.",
    )
    actual: Optional[Union[str, float]] = Field(
        default=None, description="Latest released value."
    )
