from providers.yfinance_provider import YFinanceProvider
# from providers.benzinga_provider import BenzingaProvider  # For future expansion
from core.data_models import ShnifterData, ShnifterNewsData
from datetime import datetime
import pandas as pd

# Import the missing pieces from the original shnifter_bb.py
class TechnicalAnalysisToolkit:
    def calculate_sma(self, data_obj, length):
        return data_obj

class PriceRouter:
    def __init__(self, providers):
        self._providers = providers
    
    def historical(self, ticker, start_date, end_date):
        provider = self._providers.get("yfinance")
        if provider:
            df = provider.get_historical_price(ticker, start_date, end_date)
            return ShnifterData(results=df, provider="yfinance")
        else:
            raise ValueError("No provider available for historical data")

class EquityRouter:
    def __init__(self, providers):
        self.price = PriceRouter(providers)

class NewsRouter:
    """Routes news-related queries to the appropriate provider."""
    def __init__(self, providers):
        self._providers = providers

    def get(self, ticker: str, provider: str, limit: int = 20) -> list:
        provider_name = provider.lower()
        if provider_name not in self._providers:
            raise ValueError(f"Provider '{provider_name}' is not supported for news.")
        news_list = self._providers[provider_name].get_news(ticker, limit=limit)
        return news_list

class ShnifterBB:
    """
    The main entry point for our custom financial platform.
    """
    def __init__(self):
        self._providers = {
            "yfinance": YFinanceProvider(),
            # "benzinga": BenzingaProvider(),  # Add more as implemented
        }
        self.technicals = TechnicalAnalysisToolkit()
        self.equity = EquityRouter(self._providers)
        self.news = NewsRouter(self._providers)
        print("ShnifterBB Platform Initialized.")