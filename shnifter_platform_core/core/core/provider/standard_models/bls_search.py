# Generated: 2025-07-04T09:50:40.256908
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""BLS Search Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class SearchQueryParams(QueryParams):
    """BLS Search Query Params."""

    query: str = Field(
        default="",
        description="The search word(s). Use semi-colon to separate multiple queries as an & operator.",
    )


class SearchData(Data):
    """BLS Search Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))
    title: Optional[str] = Field(default=None, description="The title of the series.")
    survey_name: Optional[str] = Field(
        default=None, description="The name of the survey."
    )
