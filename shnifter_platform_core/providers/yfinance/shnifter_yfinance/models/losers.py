# Generated: 2025-07-04T09:50:39.454042
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Yahoo Finance Top Losers Model."""

# pylint: disable=unused-argument

from typing import Any, Optional

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.equity_performance import (
    EquityPerformanceQueryParams,
)
from shnifter_yfinance.utils.references import YFPredefinedScreenerData
from pydantic import Field


class YFLosersQueryParams(EquityPerformanceQueryParams):
    """Yahoo Finance Losers Query.

    Source: https://finance.yahoo.com/screener/predefined/day_losers
    """

    limit: Optional[int] = Field(
        default=200,
        description="Limit the number of results.",
    )


class YFLosersData(YFPredefinedScreenerData):
    """Yahoo Finance Losers Data."""


class YFLosersFetcher(Fetcher[YFLosersQueryParams, list[YFLosersData]]):
    """Yahoo Finance Losers Fetcher."""

    @staticmethod
    def transform_query(params: dict[str, Any]) -> YFLosersQueryParams:
        """Transform query params."""
        return YFLosersQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: YFLosersQueryParams,
        credentials: Optional[dict[str, str]],
        **kwargs: Any,
    ) -> list[dict]:
        """Get data from YF."""
        # pylint: disable=import-outside-toplevel
        from shnifter_yfinance.utils.helpers import get_custom_screener

        body = {
            "offset": 0,
            "size": 250,
            "sortField": "percentchange",
            "sortType": "asc",
            "quoteType": "equity",
            "query": {
                "operator": "and",
                "operands": [
                    {"operator": "gt", "operands": ["intradaymarketcap", 500000000]},
                    {
                        "operator": "or",
                        "operands": [
                            {"operator": "eq", "operands": ["exchange", "NMS"]},
                            {"operator": "eq", "operands": ["exchange", "NYQ"]},
                        ],
                    },
                    {"operator": "gt", "operands": ["percentchange", -3]},
                    {"operator": "gt", "operands": ["intradayprice", 5]},
                ],
            },
            "userId": "",
            "userIdType": "guid",
        }

        return await get_custom_screener(body=body, limit=query.limit)

    @staticmethod
    def transform_data(
        query: EquityPerformanceQueryParams,
        data: list[dict],
        **kwargs: Any,
    ) -> list[YFLosersData]:
        """Transform data."""
        return [
            YFLosersData.model_validate(d)
            for d in sorted(
                data,
                key=lambda x: x["regularMarketChangePercent"],
                reverse=query.sort == "desc",
            )
        ]
