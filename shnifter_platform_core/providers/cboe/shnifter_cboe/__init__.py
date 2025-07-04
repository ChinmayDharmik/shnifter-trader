# Generated: 2025-07-04T09:50:39.904263
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Cboe provider module."""

from shnifter_cboe.models.available_indices import CboeAvailableIndicesFetcher
from shnifter_cboe.models.equity_historical import CboeEquityHistoricalFetcher
from shnifter_cboe.models.equity_quote import CboeEquityQuoteFetcher
from shnifter_cboe.models.equity_search import CboeEquitySearchFetcher
from shnifter_cboe.models.futures_curve import CboeFuturesCurveFetcher
from shnifter_cboe.models.index_constituents import (
    CboeIndexConstituentsFetcher,
)
from shnifter_cboe.models.index_historical import (
    CboeIndexHistoricalFetcher,
)
from shnifter_cboe.models.index_search import CboeIndexSearchFetcher
from shnifter_cboe.models.index_snapshots import CboeIndexSnapshotsFetcher
from shnifter_cboe.models.options_chains import CboeOptionsChainsFetcher
from shnifter_core.provider.abstract.provider import Provider

cboe_provider = Provider(
    name="cboe",
    website="https://www.cboe.com",
    description="""Cboe is the world's go-to derivatives and exchange network,
delivering cutting-edge trading, clearing and investment solutions to people
around the world.""",
    credentials=None,
    fetcher_dict={
        "AvailableIndices": CboeAvailableIndicesFetcher,
        "EquityHistorical": CboeEquityHistoricalFetcher,
        "EquityQuote": CboeEquityQuoteFetcher,
        "EquitySearch": CboeEquitySearchFetcher,
        "EtfHistorical": CboeEquityHistoricalFetcher,
        "IndexConstituents": CboeIndexConstituentsFetcher,
        "FuturesCurve": CboeFuturesCurveFetcher,
        "IndexHistorical": CboeIndexHistoricalFetcher,
        "IndexSearch": CboeIndexSearchFetcher,
        "IndexSnapshots": CboeIndexSnapshotsFetcher,
        "OptionsChains": CboeOptionsChainsFetcher,
    },
    repr_name="Chicago Board Options Exchange (CBOE)",
)
