# Generated: 2025-07-04T09:50:40.333574
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Forward Sales Estimates Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data, ForceInt
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class ForwardSalesEstimatesQueryParams(QueryParams):
    """Forward Sales Estimates Query Parameters."""

    symbol: Optional[str] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS["symbol"],
    )

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v):
        """Convert field to uppercase."""
        return v.upper() if v else None


class ForwardSalesEstimatesData(Data):
    """Forward Sales Estimates Data."""

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
    low_estimate: Optional[ForceInt] = Field(
        default=None, description="The sales estimate low for the period."
    )
    high_estimate: Optional[ForceInt] = Field(
        default=None, description="The sales estimate high for the period."
    )
    mean: Optional[ForceInt] = Field(
        default=None, description="The sales estimate mean for the period."
    )
    median: Optional[ForceInt] = Field(
        default=None, description="The sales estimate median for the period."
    )
    standard_deviation: Optional[ForceInt] = Field(
        default=None,
        description="The sales estimate standard deviation for the period.",
    )
    number_of_analysts: Optional[int] = Field(
        default=None,
        description="Number of analysts providing estimates for the period.",
    )
