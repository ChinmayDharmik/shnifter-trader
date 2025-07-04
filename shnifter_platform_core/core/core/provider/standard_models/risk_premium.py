# Generated: 2025-07-04T09:50:40.397207
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Risk Premium Standard Model."""

from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from pydantic import Field, NonNegativeFloat, PositiveFloat


class RiskPremiumQueryParams(QueryParams):
    """Risk Premium Query."""


class RiskPremiumData(Data):
    """Risk Premium Data."""

    country: str = Field(description="Market country.")
    continent: Optional[str] = Field(
        default=None, description="Continent of the country."
    )
    total_equity_risk_premium: Optional[PositiveFloat] = Field(
        default=None, description="Total equity risk premium for the country."
    )
    country_risk_premium: Optional[NonNegativeFloat] = Field(
        default=None, description="Country-specific risk premium."
    )
