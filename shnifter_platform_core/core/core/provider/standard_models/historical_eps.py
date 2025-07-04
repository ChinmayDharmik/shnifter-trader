# Generated: 2025-07-04T09:50:40.348531
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Historical EPS Standard Model."""

from datetime import date as dateType
from typing import Optional

from dateutil import parser
from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field, field_validator


class HistoricalEpsQueryParams(QueryParams):
    """Historical EPS Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class HistoricalEpsData(Data):
    """Historical EPS Data."""

    date: dateType = Field(default=None, description=DATA_DESCRIPTIONS.get("date", ""))
    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    announce_time: Optional[str] = Field(
        default=None, description="Timing of the earnings announcement."
    )
    eps_actual: Optional[float] = Field(
        default=None, description="Actual EPS from the earnings date."
    )
    eps_estimated: Optional[float] = Field(
        default=None, description="Estimated EPS for the earnings date."
    )

    @field_validator("date", mode="before", check_fields=False)
    def date_validate(cls, v):  # pylint: disable=E0213
        """Return formatted datetime."""
        return parser.isoparse(str(v))
