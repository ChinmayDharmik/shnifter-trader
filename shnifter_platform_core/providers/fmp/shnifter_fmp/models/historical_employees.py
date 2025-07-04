# Generated: 2025-07-04T09:50:39.785542
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FMP Historical Employees Model."""

from typing import Any, Dict, List, Optional

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.historical_employees import (
    HistoricalEmployeesData,
    HistoricalEmployeesQueryParams,
)
from shnifter_fmp.utils.helpers import create_url, get_data_many


class FMPHistoricalEmployeesQueryParams(HistoricalEmployeesQueryParams):
    """FMP Historical Employees Query.

    Source: https://site.financialmodelingprep.com/developer/docs/historical-numer-of-employees-api/
    """


class FMPHistoricalEmployeesData(HistoricalEmployeesData):
    """FMP Historical Employees Data."""


class FMPHistoricalEmployeesFetcher(
    Fetcher[
        FMPHistoricalEmployeesQueryParams,
        List[FMPHistoricalEmployeesData],
    ]
):
    """Transform the query, extract and transform the data from the FMP endpoints."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> FMPHistoricalEmployeesQueryParams:
        """Transform the query params."""
        return FMPHistoricalEmployeesQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: FMPHistoricalEmployeesQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> List[Dict]:
        """Return the raw data from the FMP endpoint."""
        api_key = credentials.get("fmp_api_key") if credentials else ""

        url = create_url(4, "historical/employee_count", api_key, query)

        return await get_data_many(url, **kwargs)

    @staticmethod
    def transform_data(
        query: FMPHistoricalEmployeesQueryParams, data: List[Dict], **kwargs: Any
    ) -> List[FMPHistoricalEmployeesData]:
        """Return the transformed data."""
        return [FMPHistoricalEmployeesData.model_validate(d) for d in data]
