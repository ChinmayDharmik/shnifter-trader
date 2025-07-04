# Generated: 2025-07-04T09:50:40.339274
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Futures Info Standard Model."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class FuturesInfoQueryParams(QueryParams):
    """Futures Info Query."""

    # leaving this empty to let the provider create custom symbol docstrings.


class FuturesInfoData(Data):
    """Futures Instruments Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
