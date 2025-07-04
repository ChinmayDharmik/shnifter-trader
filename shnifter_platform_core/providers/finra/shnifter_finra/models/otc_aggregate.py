# Generated: 2025-07-04T09:50:39.829567
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FINRA OTC Aggregate Model."""

from typing import Any, Dict, List, Literal, Optional

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.otc_aggregate import (
    OTCAggregateData,
    OTCAggregateQueryParams,
)
from pydantic import Field


class FinraOTCAggregateQueryParams(OTCAggregateQueryParams):
    """FINRA OTC Aggregate Query."""

    tier: Literal["T1", "T2", "OTCE"] = Field(
        default="T1",
        description=""""T1 - Securities included in the S&P 500, Russell 1000 and selected exchange-traded products;
        T2 - All other NMS stocks; OTC - Over-the-Counter equity securities""",
    )
    is_ats: bool = Field(
        default=True, description="ATS data if true, NON-ATS otherwise"
    )


class FinraOTCAggregateData(OTCAggregateData):
    """FINRA OTC Aggregate Data."""

    __alias_dict__ = {
        "share_quantity": "totalWeeklyShareQuantity",
        "trade_quantity": "totalWeeklyTradeCount",
        "update_date": "lastUpdateDate",
    }


class FinraOTCAggregateFetcher(
    Fetcher[FinraOTCAggregateQueryParams, List[FinraOTCAggregateData]]
):
    """Transform the query, extract and transform the data from the FINRA endpoints."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> FinraOTCAggregateQueryParams:
        """Transform query params."""
        return FinraOTCAggregateQueryParams(**params)

    # pylint: disable=unused-argument
    @staticmethod
    def extract_data(
        query: FinraOTCAggregateQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> List[Dict]:
        """Extract the data from the FINRA endpoint."""
        # pylint: disable=import-outside-toplevel
        from shnifter_finra.utils.helpers import get_full_data

        return get_full_data(query.symbol, query.tier, query.is_ats)

    @staticmethod
    def transform_data(
        query: FinraOTCAggregateQueryParams, data: List[Dict], **kwargs: Any
    ) -> List[FinraOTCAggregateData]:
        """Transform the data."""
        return [FinraOTCAggregateData.model_validate(d) for d in data if d]
