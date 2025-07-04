# Generated: 2025-07-04T09:50:40.411296
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Survey Of Economic Conditions - Chicago - Standard Model."""

from datetime import (
    date as dateType,
)
from typing import Optional

from shnifter_core.provider.abstract.data import Data
from shnifter_core.provider.abstract.query_params import QueryParams
from shnifter_core.provider.utils.descriptions import (
    DATA_DESCRIPTIONS,
    QUERY_DESCRIPTIONS,
)
from pydantic import Field


class SurveyOfEconomicConditionsChicagoQueryParams(QueryParams):
    """Survey Of Economic Conditions - Chicago - Query."""

    start_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("start_date", ""),
    )
    end_date: Optional[dateType] = Field(
        default=None,
        description=QUERY_DESCRIPTIONS.get("end_date", ""),
    )


class SurveyOfEconomicConditionsChicagoData(Data):
    """Survey Of Economic Conditions - Chicago - Data."""

    date: dateType = Field(description=DATA_DESCRIPTIONS.get("date", ""))
    activity_index: Optional[float] = Field(default=None, description="Activity Index.")
    one_year_outlook: Optional[float] = Field(
        default=None, description="One Year Outlook Index."
    )
    manufacturing_activity: Optional[float] = Field(
        default=None, description="Manufacturing Activity Index."
    )
    non_manufacturing_activity: Optional[float] = Field(
        default=None, description="Non-Manufacturing Activity Index."
    )
    capital_expenditures_expectations: Optional[float] = Field(
        default=None, description="Capital Expenditures Expectations Index."
    )
    hiring_expectations: Optional[float] = Field(
        default=None, description="Hiring Expectations Index."
    )
    current_hiring: Optional[float] = Field(
        default=None, description="Current Hiring Index."
    )
    labor_costs: Optional[float] = Field(default=None, description="Labor Costs Index.")
    non_labor_costs: Optional[float] = Field(
        default=None, description="Non-Labor Costs Index."
    )
