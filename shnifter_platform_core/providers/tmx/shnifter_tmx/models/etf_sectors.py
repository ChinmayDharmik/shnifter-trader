# Generated: 2025-07-04T09:50:39.501310
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""TMX ETF Sectors fetcher."""

# pylint: disable=unused-argument

from typing import Any, Dict, List, Optional

from shnifter_core.provider.abstract.fetcher import Fetcher
from shnifter_core.provider.standard_models.etf_sectors import (
    EtfSectorsData,
    EtfSectorsQueryParams,
)
from pydantic import Field


class TmxEtfSectorsQueryParams(EtfSectorsQueryParams):
    """TMX ETF Sectors Query Params"""

    use_cache: bool = Field(
        default=True,
        description="Whether to use a cached request. All ETF data comes from a single JSON file that is updated daily."
        + " To bypass, set to False. If True, the data will be cached for 4 hours.",
    )


class TmxEtfSectorsData(EtfSectorsData):
    """TMX ETF Sectors Data."""


class TmxEtfSectorsFetcher(
    Fetcher[
        TmxEtfSectorsQueryParams,
        List[TmxEtfSectorsData],
    ]
):
    """TMX ETF Sectors Fetcher."""

    @staticmethod
    def transform_query(params: Dict[str, Any]) -> TmxEtfSectorsQueryParams:
        """Transform the query."""
        return TmxEtfSectorsQueryParams(**params)

    @staticmethod
    async def aextract_data(
        query: TmxEtfSectorsQueryParams,
        credentials: Optional[Dict[str, str]],
        **kwargs: Any,
    ) -> List[Dict]:
        """Return the raw data from the TMX endpoint."""
        # pylint: disable=import-outside-toplevel
        from shnifter_core.provider.utils.errors import EmptyDataError  # noqa
        from shnifter_tmx.utils.helpers import get_all_etfs
        from pandas import DataFrame

        target = DataFrame()
        _data = DataFrame(await get_all_etfs(use_cache=query.use_cache))
        symbol = (
            query.symbol.upper()
            .replace("-", ".")
            .replace(".TO", "")
            .replace(".TSX", "")
        )
        _target = _data[_data["symbol"] == symbol]["sectors"]

        if len(_target) > 0:
            target = DataFrame.from_records(_target.iloc[0]).rename(
                columns={"name": "sector", "percent": "weight"}
            )
            return target.to_dict(orient="records")
        raise EmptyDataError(f"No sectors info found for ETF symbol: {query.symbol}.")

    @staticmethod
    def transform_data(
        query: TmxEtfSectorsQueryParams,
        data: List[Dict],
        **kwargs: Any,
    ) -> List[TmxEtfSectorsData]:
        """Return the transformed data."""
        # pylint: disable=import-outside-toplevel
        from shnifter_core.provider.utils.errors import EmptyDataError  # noqa
        from numpy import nan
        from pandas import DataFrame

        target = DataFrame(data)
        try:
            target["weight"] = target["weight"] / 100
        except KeyError:
            raise EmptyDataError(  # pylint: disable=raise-missing-from
                f"No sectors info found for ETF symbol: {query.symbol}."
            )

        target = target.replace({nan: None})
        return [
            TmxEtfSectorsData.model_validate(d)
            for d in target.to_dict(orient="records")
        ]
