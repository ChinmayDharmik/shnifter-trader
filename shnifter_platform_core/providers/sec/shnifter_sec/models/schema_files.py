# Generated: 2025-07-04T09:50:39.556398
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""SEC Schema Files List Model."""

# pylint: disable=unused-argument

from typing import Any, Dict, List, Optional

from shnifter_core.app.model.abstract.error import ShnifterError
from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.cot_search import CotSearchQueryParams
from pydantic import Field


class SecSchemaFilesQueryParams(CotSearchQueryParams):
    """SEC Schema Files List Query.

    Source: https://sec.gov/
    """

    url: Optional[str] = Field(
        description="Enter an optional URL path to fetch the next level.", default=None
    )
    use_cache: Optional[bool] = Field(
        default=True,
        description="Whether or not to use cache.",
    )


class SecSchemaFilesData(Data):
    """SEC Schema Files List Data."""

    files: List[str] = Field(description="Dictionary of URLs to SEC Schema Files")


class SecSchemaFilesFetcher(Fetcher[SecSchemaFilesQueryParams, SecSchemaFilesData]):
    """SEC Schema Files Fetcher."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> SecSchemaFilesQueryParams:
        """Transform the query."""
        return SecSchemaFilesQueryParams(**params)

    @staticmethod
    def extract_data(
        query: SecSchemaFilesQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> Dict:
        """Return the raw data from the SEC endpoint."""
        # pylint: disable=import-outside-toplevel
        from shnifter_sec.utils.helpers import get_schema_filelist

        if query.url and ".xsd" in query.url or query.url and ".xml" in query.url:
            raise ShnifterError("Invalid URL. This endpoint does not parse the files.")
        results = get_schema_filelist(query.query, query.url)

        return {"files": results}

    @staticmethod
    def transform_data(
        query: SecSchemaFilesQueryParams, data: Dict, **kwargs: Any
    ) -> SecSchemaFilesData:
        """Transform the data to the standard format."""
        return SecSchemaFilesData.model_validate(data)
