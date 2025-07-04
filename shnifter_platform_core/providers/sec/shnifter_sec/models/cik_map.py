# Generated: 2025-07-04T09:50:39.542446
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""SEC CIK Mapping Model."""

# pylint: disable=unused-argument

from typing import Any, Dict, Optional

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.cik_map import CikMapData, CikMapQueryParams
from pydantic import Field


class SecCikMapQueryParams(CikMapQueryParams):
    """SEC CIK Mapping Query.

    Source: https://sec.gov/
    """

    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether or not to use cache for the request, default is True.",
    )


class SecCikMapData(CikMapData):
    """SEC CIK Mapping Data."""


class SecCikMapFetcher(
    Fetcher[
        SecCikMapQueryParams,
        SecCikMapData,
    ]
):
    """SEC CIK Map Fetcher."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> SecCikMapQueryParams:
        """Transform the query."""
        return SecCikMapQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: SecCikMapQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict:
        """Return the raw data from the SEC endpoint."""
        # pylint: disable=import-outside-toplevel
        from shnifter_sec.utils.helpers import symbol_map

        results = {"cik": await symbol_map(query.symbol, query.use_cache)}
        if not results:
            return {"Error": "Symbol not found."}
        return results

    @staticmethod
    def transform_data(
        query: SecCikMapQueryParams, data: Dict, **kwargs: Any
    ) -> SecCikMapData:
        """Transform the data to the standard format."""
        return SecCikMapData.model_validate(data)
