# Generated: 2025-07-04T09:50:40.320324
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Search Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class EtfSearchQueryParams(QueryParams):
    """ETF Search Query."""

    query: Optional[str] = Field(description="Search query.", default="")


class EtfSearchData(Data):
    """ETF Search Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", "") + "(ETF)")
    name: Optional[str] = Field(description="Name of the ETF.", default=None)
