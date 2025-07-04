# Generated: 2025-07-04T09:50:40.321186
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Sectors Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class EtfSectorsQueryParams(QueryParams):
    """ETF Sectors Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", "") + " (ETF)")

    @field_validator("symbol")
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EtfSectorsData(Data):
    """ETF Sectors Data."""

    sector: str = Field(description="Sector of exposure.")
    weight: Optional[float] = Field(
        description="Exposure of the ETF to the sector in normalized percentage points."
    )
