# Generated: 2025-07-04T09:50:39.620646
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Multpl Provider Module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_multpl.models.sp500_multiples import MultplSP500MultiplesFetcher

multpl_provider = Provider(
    name="multpl",
    website="https://www.multpl.com/",
    description="""Public broad-market data published to https://multpl.com.""",
    fetcher_dict={
        "SP500Multiples": MultplSP500MultiplesFetcher,
    },
)
