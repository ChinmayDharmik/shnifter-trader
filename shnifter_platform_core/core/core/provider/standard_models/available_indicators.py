# Generated: 2025-07-04T09:50:40.251710
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Available Indicators Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class AvailableIndicesQueryParams(QueryParams):
    """Available Indicators Query."""


class AvailableIndicatorsData(Data):
    """Available Indicators Data.

    Returns the list of available economic indicators from a provider.
    """

    symbol_root: Optional[str] = Field(
        default=None, description="The root symbol representing the indicator."
    )
    symbol: Optional[str] = Field(
        default=None,
        description=DATA_DESCRIPTIONS.get("symbol", "")
        + " The root symbol with additional codes.",
    )
    country: Optional[str] = Field(
        default=None,
        description="The name of the country, region, or entity represented by the symbol.",
    )
    iso: Optional[str] = Field(
        default=None,
        description="The ISO code of the country, region, or entity represented by the symbol.",
    )
    description: Optional[str] = Field(
        default=None, description="The description of the indicator."
    )
    frequency: Optional[str] = Field(
        default=None, description="The frequency of the indicator data."
    )
