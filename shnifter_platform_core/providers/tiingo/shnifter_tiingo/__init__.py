# Generated: 2025-07-04T09:50:39.517687
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Tiingo provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_tiingo.models.company_news import TiingoCompanyNewsFetcher
from shnifter_tiingo.models.crypto_historical import TiingoCryptoHistoricalFetcher
from shnifter_tiingo.models.currency_historical import TiingoCurrencyHistoricalFetcher
from shnifter_tiingo.models.equity_historical import TiingoEquityHistoricalFetcher
from shnifter_tiingo.models.trailing_dividend_yield import TiingoTrailingDivYieldFetcher
from shnifter_tiingo.models.world_news import TiingoWorldNewsFetcher

tiingo_provider = Provider(
    name="tiingo",
    website="https://tiingo.com",
    description="""A Reliable, Enterprise-Grade Financial Markets API. Tiingo's APIs
power hedge funds, tech companies, and individuals.""",
    credentials=["token"],
    fetcher_dict={
        "EquityHistorical": TiingoEquityHistoricalFetcher,
        "EtfHistorical": TiingoEquityHistoricalFetcher,
        "CompanyNews": TiingoCompanyNewsFetcher,
        "WorldNews": TiingoWorldNewsFetcher,
        "CryptoHistorical": TiingoCryptoHistoricalFetcher,
        "CurrencyHistorical": TiingoCurrencyHistoricalFetcher,
        "TrailingDividendYield": TiingoTrailingDivYieldFetcher,
    },
    repr_name="Tiingo",
)
