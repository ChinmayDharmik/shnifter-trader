# Generated: 2025-07-04T09:50:39.532178
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Seeking Alpha Provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_seeking_alpha.models.calendar_earnings import SACalendarEarningsFetcher
from shnifter_seeking_alpha.models.forward_eps_estimates import (
    SAForwardEpsEstimatesFetcher,
)
from shnifter_seeking_alpha.models.forward_sales_estimates import (
    SAForwardSalesEstimatesFetcher,
)

seeking_alpha_provider = Provider(
    name="seeking_alpha",
    website="https://seekingalpha.com",
    description="""Seeking Alpha is a data provider with access to news, analysis, and
real-time alerts on stocks.""",
    fetcher_dict={
        "CalendarEarnings": SACalendarEarningsFetcher,
        "ForwardEpsEstimates": SAForwardEpsEstimatesFetcher,
        "ForwardSalesEstimates": SAForwardSalesEstimatesFetcher,
    },
    repr_name="Seeking Alpha",
)
