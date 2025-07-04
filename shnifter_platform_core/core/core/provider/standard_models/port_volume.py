# Generated: 2025-07-04T09:50:40.387875
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Port Volume Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class PortVolumeQueryParams(QueryParams):
    """Port Volume Query."""

    start_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("start_date", "")
    )
    end_date: Optional[dateType] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("end_date", "")
    )


class PortVolumeData(Data):
    """Port Volume Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    port_code: Optional[str] = Field(default=None, description="Port code.")
    port_name: Optional[str] = Field(default=None, description="Port name.")
    country: Optional[str] = Field(
        default=None, description="Country where the port is located."
    )
