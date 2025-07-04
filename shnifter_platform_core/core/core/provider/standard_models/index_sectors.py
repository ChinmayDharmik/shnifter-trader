# Generated: 2025-07-04T09:50:40.360645
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Index Sectors Standard Model."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class IndexSectorsQueryParams(QueryParams):
    """Index Sectors Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol")
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class IndexSectorsData(Data):
    """Index Sectors Data."""

    sector: str = Field(description="The sector name.")
    weight: float = Field(description="The weight of the sector in the index.")
