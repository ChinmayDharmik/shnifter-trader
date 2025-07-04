# Generated: 2025-07-04T09:50:40.331630
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Forward EPS Estimates Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class ForwardEpsEstimatesQueryParams(QueryParams):
    """Forward EPS Estimates Query Parameters."""

    symbol: Optional[str] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS["symbol"],
    )

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v):
        """Convert field to uppercase."""
        return v.upper() if v else None


class ForwardEpsEstimatesData(Data):
    """Forward EPS Estimates Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    name: Optional[str] = Field(default=None, description="Name of the entity.")
    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    fiscal_year: Optional[int] = Field(
        default=None, description="Fiscal year for the estimate."
    )
    fiscal_period: Optional[str] = Field(
        default=None, description="Fiscal quarter for the estimate."
    )
    calendar_year: Optional[int] = Field(
        default=None, description="Calendar year for the estimate."
    )
    calendar_period: Optional[str] = Field(
        default=None, description="Calendar quarter for the estimate."
    )
    low_estimate: Optional[float] = Field(
        default=None, description="Estimated EPS low for the period."
    )
    high_estimate: Optional[float] = Field(
        default=None, description="Estimated EPS high for the period."
    )
    mean: Optional[float] = Field(
        default=None, description="Estimated EPS mean for the period."
    )
    median: Optional[float] = Field(
        default=None, description="Estimated EPS median for the period."
    )
    standard_deviation: Optional[float] = Field(
        default=None, description="Estimated EPS standard deviation for the period."
    )
    number_of_analysts: Optional[int] = Field(
        default=None,
        description="Number of analysts providing estimates for the period.",
    )
