# Generated: 2025-07-04T09:50:40.366475
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Key Executives Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data, ForceInt
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class KeyExecutivesQueryParams(QueryParams):
    """Key Executives Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class KeyExecutivesData(Data):
    """Key Executives Data."""

    title: str = Field(description="Designation of the key executive.")
    name: str = Field(description="Name of the key executive.")
    pay: Optional[ForceInt] = Field(
        default=None, description="Pay of the key executive."
    )
    currency_pay: Optional[str] = Field(
        default=None, description="Currency of the pay."
    )
    gender: Optional[str] = Field(
        default=None, description="Gender of the key executive."
    )
    year_born: Optional[ForceInt] = Field(
        default=None, description="Birth year of the key executive."
    )
    title_since: Optional[ForceInt] = Field(
        default=None, description="Date the tile was held since."
    )
