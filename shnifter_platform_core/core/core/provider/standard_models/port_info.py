# Generated: 2025-07-04T09:50:40.387025
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Port information and metadata."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field


class PortInfoQueryParams(QueryParams):
    """Port Information Query."""


class PortInfoData(Data):
    """Port Information Data."""

    port_code: str = Field(description="Unique ID assigned to the port by the source.")
