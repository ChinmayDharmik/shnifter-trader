# Generated: 2025-07-04T09:50:40.400602
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Sector Performance Standard Model."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field


class SectorPerformanceQueryParams(QueryParams):
    """Sector Performance Query."""


class SectorPerformanceData(Data):
    """Sector Performance Data."""

    sector: str = Field(description="The name of the sector.")
    change_percent: float = Field(description="The change in percent from open.")
