# Generated: 2025-07-04T09:50:40.370780
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Manufacturing Outlook - Texas - Standard Model."""

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


class ManufacturingOutlookTexasQueryParams(QueryParams):
    """Manufacturing Outlook - Texas - Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )


class ManufacturingOutlookTexasData(Data):
    """Manufacturing Outlook - Texas - Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    topic: Optional[str] = Field(
        default=None, description="Topic of the survey response."
    )
    diffusion_index: Optional[float] = Field(
        default=None, description="Diffusion Index."
    )
    percent_reporting_increase: Optional[float] = Field(
        default=None,
        description="Percent of respondents reporting an increase over the last month.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
    percent_reporting_decrease: Optional[float] = Field(
        default=None,
        description="Percent of respondents reporting a decrease over the last month.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
    percent_reporting_no_change: Optional[float] = Field(
        default=None,
        description="Percent of respondents reporting no change over the last month.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
