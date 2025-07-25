# Generated: 2025-07-04T09:50:40.415270
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Treasury Constant Maturity Model."""

from datetime import (
    date as dateType,
)
from typing import Literal, Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class TreasuryConstantMaturityQueryParams(QueryParams):
    """Treasury Constant Maturity Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )
    maturity: Optional[Literal["3m", "2y"]] = Field(
        default="3m",
        description="The maturity",
    )

    @field_validator("maturity", mode="before", check_fields=False)
    @classmethod
    def to_lower(cls, v: Optional[str]) -> Optional[str]:
        """Convert field to lowercase."""
        return v.lower() if v else v


class TreasuryConstantMaturityData(Data):
    """Treasury Constant Maturity Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    rate: Optional[float] = Field(description="TreasuryConstantMaturity Rate.")
