# Generated: 2025-07-04T09:50:40.382684
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Unusual Options Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class OptionsUnusualQueryParams(QueryParams):
    """Unusual Options Query."""

    symbol: Optional[str] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("symbol", "") + " (the underlying symbol)",
    )

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str):
        """Convert field to uppercase."""
        return v.upper() if v else None


class OptionsUnusualData(Data):
    """Unusual Options Data."""

    underlying_symbol: Optional[str] = Field(
        description=DATA_DESCRIPTIONS.get("symbol", "") + " (the underlying symbol)",
        default=None,
    )
    contract_symbol: str = Field(description="Contract symbol for the option.")
