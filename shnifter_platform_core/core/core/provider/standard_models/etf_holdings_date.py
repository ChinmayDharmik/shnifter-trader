# Generated: 2025-07-04T09:50:40.317325
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Holdings Date Standard Model."""

from datetime import date as dateType

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class EtfHoldingsDateQueryParams(QueryParams):
    """ETF Holdings Date Query."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", "") + " (ETF)")


class EtfHoldingsDateData(Data):
    """ETF Holdings Date Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date"))
