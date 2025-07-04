# Generated: 2025-07-04T09:50:39.826906
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FINRA provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_finra.models.equity_short_interest import FinraShortInterestFetcher
from shnifter_finra.models.otc_aggregate import FinraOTCAggregateFetcher

finra_provider = Provider(
    name="finra",
    website="https://www.finra.org/finra-data",
    description="""FINRA Data provides centralized access to the abundance of data FINRA
makes available to the public, media, researchers and member firms.""",
    credentials=None,
    fetcher_dict={
        "OTCAggregate": FinraOTCAggregateFetcher,
        "EquityShortInterest": FinraShortInterestFetcher,
    },
    repr_name="Financial Industry Regulatory Authority (FINRA)",
)
