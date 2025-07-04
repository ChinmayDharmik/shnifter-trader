# Generated: 2025-07-04T09:50:39.485369
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""TMX Provider Module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_tmx.models.available_indices import TmxAvailableIndicesFetcher
from shnifter_tmx.models.bond_prices import TmxBondPricesFetcher
from shnifter_tmx.models.calendar_earnings import TmxCalendarEarningsFetcher
from shnifter_tmx.models.company_filings import TmxCompanyFilingsFetcher
from shnifter_tmx.models.company_news import TmxCompanyNewsFetcher
from shnifter_tmx.models.equity_historical import TmxEquityHistoricalFetcher
from shnifter_tmx.models.equity_profile import TmxEquityProfileFetcher
from shnifter_tmx.models.equity_quote import TmxEquityQuoteFetcher
from shnifter_tmx.models.equity_search import TmxEquitySearchFetcher
from shnifter_tmx.models.etf_countries import TmxEtfCountriesFetcher
from shnifter_tmx.models.etf_holdings import TmxEtfHoldingsFetcher
from shnifter_tmx.models.etf_info import TmxEtfInfoFetcher
from shnifter_tmx.models.etf_search import TmxEtfSearchFetcher
from shnifter_tmx.models.etf_sectors import TmxEtfSectorsFetcher
from shnifter_tmx.models.gainers import TmxGainersFetcher
from shnifter_tmx.models.historical_dividends import TmxHistoricalDividendsFetcher
from shnifter_tmx.models.index_constituents import TmxIndexConstituentsFetcher
from shnifter_tmx.models.index_sectors import TmxIndexSectorsFetcher
from shnifter_tmx.models.index_snapshots import TmxIndexSnapshotsFetcher
from shnifter_tmx.models.insider_trading import TmxInsiderTradingFetcher
from shnifter_tmx.models.options_chains import TmxOptionsChainsFetcher
from shnifter_tmx.models.price_target_consensus import TmxPriceTargetConsensusFetcher
from shnifter_tmx.models.treasury_prices import TmxTreasuryPricesFetcher

tmx_provider = Provider(
    name="tmx",
    website="https://www.tmx.com",
    description="""Unofficial TMX Data Provider Extension
    TMX Group Companies
        - Toronto Stock Exchange
        - TSX Venture Exchange
        - TSX Trust
        - Montréal Exchange
        - TSX Alpha Exchange
        - Shorcan
        - CDCC
        - CDS
        - TMX Datalinx
        - Trayport
    """,
    fetcher_dict={
        "AvailableIndices": TmxAvailableIndicesFetcher,
        "BondPrices": TmxBondPricesFetcher,
        "CalendarEarnings": TmxCalendarEarningsFetcher,
        "CompanyFilings": TmxCompanyFilingsFetcher,
        "CompanyNews": TmxCompanyNewsFetcher,
        "EquityHistorical": TmxEquityHistoricalFetcher,
        "EquityInfo": TmxEquityProfileFetcher,
        "EquityQuote": TmxEquityQuoteFetcher,
        "EquitySearch": TmxEquitySearchFetcher,
        "EtfSearch": TmxEtfSearchFetcher,
        "EtfHoldings": TmxEtfHoldingsFetcher,
        "EtfSectors": TmxEtfSectorsFetcher,
        "EtfCountries": TmxEtfCountriesFetcher,
        "EtfHistorical": TmxEquityHistoricalFetcher,
        "EtfInfo": TmxEtfInfoFetcher,
        "EquityGainers": TmxGainersFetcher,
        "HistoricalDividends": TmxHistoricalDividendsFetcher,
        "IndexConstituents": TmxIndexConstituentsFetcher,
        "IndexSectors": TmxIndexSectorsFetcher,
        "IndexSnapshots": TmxIndexSnapshotsFetcher,
        "InsiderTrading": TmxInsiderTradingFetcher,
        "OptionsChains": TmxOptionsChainsFetcher,
        "PriceTargetConsensus": TmxPriceTargetConsensusFetcher,
        "TreasuryPrices": TmxTreasuryPricesFetcher,
    },
    repr_name="TMX",
)
