# Generated: 2025-07-04T09:50:40.362753
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Industry P/E Ratio Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class IndustryPEQueryParams(QueryParams):
    """Industry P/E Ratio Query."""


class IndustryPEData(Data):
    """Industry P/E Ratio Data."""

    date: Optional[dateType] = Field(
        description=DATA_DESCRIPTIONS.get("date", ""), default=None
    )
    exchange: Optional[str] = Field(
        default=None, description="The exchange where the data is from."
    )
    industry: str = Field(description="The name of the industry.")
    pe: float = Field(description="The P/E ratio of the industry.")
