# Generated: 2025-07-04T09:50:40.396377
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Revenue by Geographic Segments Standard Model."""

from datetime import date as dateType
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class RevenueGeographicQueryParams(QueryParams):
    """Revenue by Geographic Segments Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str):
        """Convert field to uppercase."""
        return v.upper()


class RevenueGeographicData(Data):
    """Revenue by Geographic Segments Data."""

    period_ending: dateType = Field(description="The end date of the reporting period.")
    fiscal_period: Optional[str] = Field(
        default=None, description="The fiscal period of the reporting period."
    )
    fiscal_year: Optional[int] = Field(
        default=None, description="The fiscal year of the reporting period."
    )
    filing_date: Optional[dateType] = Field(
        default=None, description="The filing date of the report."
    )
    region: Optional[str] = Field(
        default=None,
        description="The region represented by the revenue data.",
    )
    revenue: Union[int, float] = Field(
        description="The total revenue attributed to the region.",
        json_schema_extra={"x-unit_measurement": "currency"},
    )
