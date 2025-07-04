# Generated: 2025-07-04T09:50:40.371633
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Maritime chokepoint information and metadata."""

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field


class MaritimeChokePointInfoQueryParams(QueryParams):
    """MaritimeChokepointInfo Query."""


class MaritimeChokePointInfoData(Data):
    """MaritimeChokepointInfo Data."""

    chokepoint_code: str = Field(
        description="Unique ID assigned to the chokepoint by the source."
    )
