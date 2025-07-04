# Generated: 2025-07-04T09:50:40.323954
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Export Destinations Standard Model."""

from typing import Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field


class ExportDestinationsQueryParams(QueryParams):
    """Export Destinations Query."""

    country: str = Field(description=QUERY_DESCRIPTIONS.get("country", ""))


class ExportDestinationsData(Data):
    """Export Destinations Data."""

    origin_country: str = Field(
        description="The country of origin.",
    )
    destination_country: str = Field(
        description="The destination country.",
    )
    value: Union[float, int] = Field(
        description="The value of the export.",
        json_schema_extra={"x-unit_measurement": "currency"},
    )
