# Generated: 2025-07-04T09:50:40.344993
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""High Quality Market Corporate Bond Standard Model."""

from datetime import (
    date as dateType,
)
from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class HighQualityMarketCorporateBondQueryParams(QueryParams):
    """High Quality Market Corporate Bond Query."""

    date: Optional[Union[dateType, str]] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("date", ""),
    )


class HighQualityMarketCorporateBondData(Data):
    """High Quality Market Corporate Bond Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    rate: float = Field(
        description="Interest rate.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
    maturity: str = Field(description="Maturity.")
