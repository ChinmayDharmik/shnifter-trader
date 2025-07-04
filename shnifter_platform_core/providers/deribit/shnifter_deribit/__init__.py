# Generated: 2025-07-04T09:50:39.886730
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Deribit Provider Module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_deribit.models.futures_curve import DeribitFuturesCurveFetcher
from shnifter_deribit.models.futures_historical import DeribitFuturesHistoricalFetcher
from shnifter_deribit.models.futures_info import DeribitFuturesInfoFetcher
from shnifter_deribit.models.futures_instruments import DeribitFuturesInstrumentsFetcher
from shnifter_deribit.models.options_chains import DeribitOptionsChainsFetcher

deribit_provider = Provider(
    name="deribit",
    website="https://deribit.com/",
    description="""Unofficial Python interfaceent for public data published by Deribit.""",
    credentials=None,
    fetcher_dict={
        "FuturesCurve": DeribitFuturesCurveFetcher,
        "FuturesHistorical": DeribitFuturesHistoricalFetcher,
        "FuturesInfo": DeribitFuturesInfoFetcher,
        "FuturesInstruments": DeribitFuturesInstrumentsFetcher,
        "OptionsChains": DeribitOptionsChainsFetcher,
    },
    repr_name="Deribit Public Data",
    instructions="This provider does not require any credentials and is not meant for trading.",
)
