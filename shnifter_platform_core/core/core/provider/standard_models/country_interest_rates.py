# Generated: 2025-07-04T09:50:40.283725
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Country Interest Rates Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class CountryInterestRatesQueryParams(QueryParams):
    """Country Interest Rates Query."""

    country: str = Field(
        default="united_states",
        description=QUERY_DESCRIPTIONS.get("country"),
    )
    start_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("start_date")
    )
    end_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("end_date")
    )


class CountryInterestRatesData(Data):
    """Country Interest Rates Data."""

    date: dateType = Field(default=None, description=DATA_DESCRIPTIONS.get("date"))
    value: float = Field(
        default=None,
        description="The interest rate value.",
        json_schema_extra={"x-unit_measurment": "percent", "x-frontend_multiply": 100},
    )
    country: Optional[str] = Field(
        default=None,
        description="Country for which the interest rate is given.",
    )
