# Generated: 2025-07-04T09:50:39.746720
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FMP Analyst Estimates Model."""

import asyncio
from typing import Any, Dict, List, Literal, Optional
from warnings import warn

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.analyst_estimates import (
    AnalystEstimatesData,
    AnalystEstimatesQueryParams,
)
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from shnifter_core.provider.utils.errors import EmptyDataError
from shnifter_core.provider.utils.helpers import amake_request
from shnifter_fmp.utils.helpers import create_url, response_callback
from pydantic import Field


class FMPAnalystEstimatesQueryParams(AnalystEstimatesQueryParams):
    """FMP Analyst Estimates Query.

    Source: https://site.financialmodelingprep.com/developer/docs/analyst-estimates-api/
    """

    __json_schema_extra__ = {"symbol": {"multiple_items_allowed": True}}

    period: Literal["quarter", "annual"] = Field(
        default="annual", description=QUERY_DESCRIPTIONS.get("period", "")
    )
    limit: Optional[int] = Field(
        default=None, description=QUERY_DESCRIPTIONS.get("limit", "")
    )


class FMPAnalystEstimatesData(AnalystEstimatesData):
    """FMP Analyst Estimates Data."""


class FMPAnalystEstimatesFetcher(
    Fetcher[
        FMPAnalystEstimatesQueryParams,
        List[FMPAnalystEstimatesData],
    ]
):
    """Transform the query, extract and transform the data from the FMP endpoints."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> FMPAnalystEstimatesQueryParams:
        """Transform the query params."""
        return FMPAnalystEstimatesQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: FMPAnalystEstimatesQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> List[Dict]:
        """Return the raw data from the FMP endpoint."""
        api_key = credentials.get("fmp_api_key") if credentials else ""

        symbols = query.symbol.split(",")  # type: ignore

        results: List[dict] = []

        async def get_one(symbol):
            """Get data for one symbol."""
            url = create_url(
                3, f"analyst-estimates/{symbol}", api_key, query, ["symbol"]
            )
            result = await amake_request(
                url, response_callback=response_callback, **kwargs
            )
            if not result or len(result) == 0:
                warn(f"Symbol Error: No data found for {symbol}")
            if result:
                results.extend(result)

        await asyncio.gather(*[get_one(symbol) for symbol in symbols])

        if not results:
            raise EmptyDataError("No data returned for the given symbols.")

        return sorted(results, key=lambda x: (x["date"], x["symbol"]), reverse=False)

    @staticmethod
    def transform_data(
        query: FMPAnalystEstimatesQueryParams, data: List[Dict], **kwargs: Any
    ) -> List[FMPAnalystEstimatesData]:
        """Return the transformed data."""
        return [FMPAnalystEstimatesData.model_validate(d) for d in data]
