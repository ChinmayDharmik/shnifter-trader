# Generated: 2025-07-04T09:50:39.952068
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Alpha Vantage Provider module."""

from shnifter_alpha_vantage.models.equity_historical import AVEquityHistoricalFetcher
from shnifter_alpha_vantage.models.historical_eps import AVHistoricalEpsFetcher
from shnifter_core.provider.abstract.provider import Provider

alpha_vantage_provider = Provider(
    name="alpha_vantage",
    website="https://www.alphavantage.co",
    description="""Alpha Vantage provides realtime and historical
financial market data through a set of powerful and developer-friendly data APIs
and spreadsheets. From traditional asset classes (e.g., stocks, ETFs, mutual funds)
to economic indicators, from foreign exchange rates to commodities,
from fundamental data to technical indicators, Alpha Vantage
is your one-stop-shop for enterprise-grade global market data delivered through
cloud-based APIs, Excel, and Google Sheets. """,
    credentials=["api_key"],
    fetcher_dict={
        "EquityHistorical": AVEquityHistoricalFetcher,
        "HistoricalEps": AVHistoricalEpsFetcher,
        "EtfHistorical": AVEquityHistoricalFetcher,
    },
    repr_name="Alpha Vantage",
    deprecated_credentials={"API_KEY_ALPHAVANTAGE": "alpha_vantage_api_key"},
    instructions='Go to: https://www.alphavantage.co/support/#api-key\n\n![AlphaVantage](https://user-images.gitcoreusercontent.com/46355364/207820936-46c2ba00-81ff-4cd3-98a4-4fa44412996f.png)\n\nFill out the form, pass Captcha, and interfaceck on, "GET FREE API KEY".',  # noqa: E501  pylint: disable=line-too-long
)
