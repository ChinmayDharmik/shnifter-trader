from providers.yfinance_provider import YFinanceProvider
from core.data_models import ShnifterData
from toolkits.technicals_toolkit import TechnicalAnalysisToolkit
from datetime import datetime
import pandas as pd

class PriceRouter:
    def __init__(self, parent_equity):
        self._parent = parent_equity

    def historical(self, ticker: str, start_date: str, end_date: str) -> ShnifterData:
        provider_name = 'yfinance'
        print(f"Fetching historical data for {ticker} via {provider_name}...")
        df = self._parent._providers[provider_name].get_historical_price(ticker, start_date, end_date)
        return ShnifterData(results=df, provider=provider_name)

class EquityRouter:
    def __init__(self, providers):
        self._providers = providers
        self.price = PriceRouter(self)

class ShnifterBB:
    """
    The main entry point for our custom financial platform.
    """
    def __init__(self):
        self._providers = {
            "yfinance": YFinanceProvider()
        }
        self.technicals = TechnicalAnalysisToolkit()
        self.equity = EquityRouter(self._providers)
        print("ShnifterBB Platform Initialized.")

if __name__ == '__main__':
    shnifter = ShnifterBB()
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
    price_data_obj = shnifter.equity.price.historical(ticker="AAPL", start_date=start, end_date=end)
    price_with_sma = shnifter.technicals.calculate_sma(price_data_obj, length=50)
    print("\n--- Price data with 50-day SMA ---")
    print(price_with_sma.to_df().tail())
