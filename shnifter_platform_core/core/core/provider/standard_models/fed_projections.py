# Generated: 2025-07-04T09:50:40.325887
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""PROJECTION Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class PROJECTIONQueryParams(QueryParams):
    """PROJECTION Query."""


class PROJECTIONData(Data):
    """PROJECTION Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    range_high: Optional[float] = Field(description="High projection of rates.")
    central_tendency_high: Optional[float] = Field(
        description="Central tendency of high projection of rates."
    )
    median: Optional[float] = Field(description="Median projection of rates.")
    range_midpoint: Optional[float] = Field(description="Midpoint projection of rates.")
    central_tendency_midpoint: Optional[float] = Field(
        description="Central tendency of midpoint projection of rates."
    )
    range_low: Optional[float] = Field(description="Low projection of rates.")
    central_tendency_low: Optional[float] = Field(
        description="Central tendency of low projection of rates."
    )
