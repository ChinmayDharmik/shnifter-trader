# Generated: 2025-07-04T09:50:40.410273
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Spot Rate Standard Model."""

from datetime import (
    date as dateType,
)
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class SpotRateQueryParams(QueryParams):
    """Spot Rate Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )
    maturity: Union[float, str] = Field(
        default=10.0, description="Maturities in years."
    )
    category: str = Field(
        default="spot_rate",
        description="Rate category. Options: spot_rate, par_yield.",
    )

    @field_validator("category", mode="before", check_fields=False)
    @classmethod
    def to_lower(cls, v: Optional[str]) -> Optional[str]:
        """Convert field to lowercase."""
        return v.lower() if v else v


class SpotRateData(Data):
    """Spot Rate Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    rate: Optional[float] = Field(description="Spot Rate.")
