# Generated: 2025-07-04T09:50:40.374404
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Market Snapshots Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data, ForceInt
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class MarketSnapshotsQueryParams(QueryParams):
    """Market Snapshots Query."""


class MarketSnapshotsData(Data):
    """Market Snapshots Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    open: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("open", ""),
        default=None,
    )
    high: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("high", ""),
        default=None,
    )
    low: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("low", ""),
        default=None,
    )
    close: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("close", ""),
        default=None,
    )
    volume: Optional[ForceInt] = Field(
        description=DATA_DESCRIPTIONS.get("volume", ""), default=None
    )
    prev_close: Optional[float] = Field(
        description=DATA_DESCRIPTIONS.get("prev_close", ""),
        default=None,
    )
    change: Optional[float] = Field(
        description="The change in price from the previous close.",
        default=None,
    )
    change_percent: Optional[float] = Field(
        description="The change in price from the previous close, as a normalized percent.",
        default=None,
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
