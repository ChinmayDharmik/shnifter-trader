# Generated: 2025-07-04T09:50:40.269831
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Central Bank Holdings Standard Model."""

from datetime import (
    date as dateType,
)
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class CentralBankHoldingsQueryParams(QueryParams):
    """Central Bank Holdings Query."""

    date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("date", ""),
    )


class CentralBankHoldingsData(Data):
    """Central Bank Holdings Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
