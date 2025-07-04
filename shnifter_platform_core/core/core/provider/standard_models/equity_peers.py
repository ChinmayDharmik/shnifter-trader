# Generated: 2025-07-04T09:50:40.302211
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Equity Peers Standard Model."""

from typing import List

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class EquityPeersQueryParams(QueryParams):
    """Equity Peers Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", ""))

    @field_validator("symbol", mode="before", check_fields=False)
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EquityPeersData(Data):
    """Equity Peers Data."""

    peers_list: List[str] = Field(
        default_factory=list,
        description="A list of equity peers based on sector, exchange and market cap.",
    )
