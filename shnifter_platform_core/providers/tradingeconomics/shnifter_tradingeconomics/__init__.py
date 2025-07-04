# Generated: 2025-07-04T09:50:39.469996
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Trading Economics provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_tradingeconomics.models.economic_calendar import TEEconomicCalendarFetcher

tradingeconomics_provider = Provider(
    name="tradingeconomics",
    website="https://tradingeconomics.com",
    description="""Trading Economics provides its users with accurate information for
196 countries including historical data and forecasts for more than 20 million economic
indicators, exchange rates, stock market indexes, government bond yields and commodity
prices. Our data for economic indicators is based on official sources, not third party
data providers, and our facts are regularly checked for inconsistencies.
Trading Economics has received nearly 2 billion page views from all around the
world.""",
    credentials=["api_key"],
    fetcher_dict={"EconomicCalendar": TEEconomicCalendarFetcher},
    repr_name="Trading Economics",
)
