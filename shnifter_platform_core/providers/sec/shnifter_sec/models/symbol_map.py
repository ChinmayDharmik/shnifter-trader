# Generated: 2025-07-04T09:50:39.559815
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""SEC Symbol Mapping Model."""

# pylint: disable=unused-argument

from typing import Any, Dict, Optional

from shnifter_core.app.model.abstract.error import ShnifterError
from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.symbol_map import SymbolMapQueryParams
from shnifter_core.provider.utils.descriptions import DATA_DESCRIPTIONS
from pydantic import Field


class SecSymbolMapQueryParams(SymbolMapQueryParams):
    """SEC Symbol Mapping Query.

    Source: https://sec.gov/
    """


class SecSymbolMapData(Data):
    """SEC symbol map Data."""

    symbol: str = Field(description=DATA_DESCRIPTIONS.get("symbol", ""))


class SecSymbolMapFetcher(
    Fetcher[
        SecSymbolMapQueryParams,
        SecSymbolMapData,
    ]
):
    """Transform the query, extract and transform the data from the SEC endpoints."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> SecSymbolMapQueryParams:
        """Transform the query."""
        return SecSymbolMapQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: SecSymbolMapQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict:
        """Return the raw data from the SEC endpoint."""
        # pylint: disable=import-outside-toplevel
        from shnifter_sec.utils.helpers import cik_map

        if not query.query.isdigit():
            raise ShnifterError("Query is required and must be a valid CIK.")
        symbol = await cik_map(int(query.query), query.use_cache)
        response = {"symbol": symbol}
        return response

    @staticmethod
    def transform_data(
        query: SecSymbolMapQueryParams, data: Dict, **kwargs: Any
    ) -> SecSymbolMapData:
        """Transform the data to the standard format."""
        return SecSymbolMapData.model_validate(data)
