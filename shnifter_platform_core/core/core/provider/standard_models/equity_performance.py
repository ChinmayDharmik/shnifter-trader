# Generated: 2025-07-04T09:50:40.303164
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Equity Performance Standard Model."""

from typing import Literal, Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field, field_validator


class EquityPerformanceQueryParams(QueryParams):
    """Equity Performance Query."""

    sort: Literal["asc", "desc"] = Field(
        default="desc",
        description="Sort order. Possible values: 'asc', 'desc'. Default: 'desc'.",
    )

    @field_validator("sort", mode="before", check_fields=False)
    @classmethod
    def to_lower(cls, v: Optional[str]) -> Optional[str]:
        """Convert field to lowercase."""
        return v.lower() if v else v


class EquityPerformanceData(Data):
    """Equity Performance Data."""

    symbol: str = Field(
        description=DATA_DESCRIPTIONS.get("symbol", ""),
    )
    name: Optional[str] = Field(
        default=None,
        description="Name of the entity.",
    )
    price: float = Field(
        description="Last price.",
        json_schema_extra={"x-unit_measurement": "currency"},
    )
    change: float = Field(
        description="Change in price.",
        json_schema_extra={"x-unit_measurement": "currency"},
    )
    percent_change: float = Field(
        description="Percent change.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
    volume: Union[int, float] = Field(
        description=DATA_DESCRIPTIONS.get("volume", ""),
    )
