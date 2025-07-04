# Generated: 2025-07-04T09:50:40.423075
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""University Of Michigan Survey Standard Model."""

from datetime import (
    date as dateType,
)
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class UofMichiganQueryParams(QueryParams):
    """University Of Michigan Survey Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )


class UofMichiganData(Data):
    """University Of Michigan Survey Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    consumer_sentiment: Optional[float] = Field(
        default=None,
        description="Index of the results of the University of Michigan's monthly Survey of Consumers,"
        + " which is used to estimate future spending and saving.  (1966:Q1=100).",
    )
    inflation_expectation: Optional[float] = Field(
        default=None,
        description="Median expected price change next 12 months, Surveys of Consumers.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
