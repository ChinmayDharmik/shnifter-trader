# Generated: 2025-07-04T09:50:40.414351
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""TIPS (Treasury Inflation-Protected Securities) Yields Standard Model."""

from datetime import date as dateType
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class TipsYieldsQueryParams(QueryParams):
    """TIPS Yields Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )


class TipsYieldsData(Data):
    """TIPS Yields Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    symbol: Optional[str] = Field(
        default=None,
        description=DATA_DESCRIPTIONS.get("symbol", ""),
    )
    due: Optional[dateType] = Field(
        default=None,
        description="The due date (maturation date) of the security.",
    )
    name: Optional[str] = Field(
        default=None,
        description="The name of the security.",
    )
    value: float = Field(
        default=None,
        description="The yield value.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
