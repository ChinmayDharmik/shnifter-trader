# Generated: 2025-07-04T09:50:40.313042
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Countries Standard Model."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class EtfCountriesQueryParams(QueryParams):
    """ETF Countries Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", "") + " (ETF)")

    @field_validator("symbol")
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EtfCountriesData(Data):
    """ETF Countries Data."""

    country: str = Field(
        description="The country of the exposure.  Corresponding values are normalized percentage points."
    )
