# Generated: 2025-07-04T09:50:40.313873
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Equity Exposure Standard Model."""

from typing import Optional, Union

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import QUERY_DESCRIPTIONS
from pydantic import Field, field_validator


class EtfEquityExposureQueryParams(QueryParams):
    """ETF Equity Exposure Query Params."""

    symbol: str = Field(description=QUERY_DESCRIPTIONS.get("symbol", "") + " (Stock)")

    @field_validator("symbol")
    @classmethod
    def to_upper(cls, v: str) -> str:
        """Convert field to uppercase."""
        return v.upper()


class EtfEquityExposureData(Data):
    """ETF Equity Exposure Data."""

    equity_symbol: str = Field(description="The symbol of the equity requested.")
    etf_symbol: str = Field(
        description="The symbol of the ETF with exposure to the requested equity."
    )
    shares: Optional[float] = Field(
        default=None,
        description="The number of shares held in the ETF.",
    )
    weight: Optional[float] = Field(
        default=None,
        description="The weight of the equity in the ETF, as a normalized percent.",
        json_schema_extra={"x-unit_measurement": "percent", "x-frontend_multiply": 100},
    )
    market_value: Optional[Union[int, float]] = Field(
        default=None,
        description="The market value of the equity position in the ETF.",
    )
