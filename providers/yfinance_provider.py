import yfinance as yf
from datetime import datetime
from typing import List
import pandas as pd
from core.data_models import ShnifterNewsData

class YFinanceProvider:
    """
    A provider to fetch data specifically from the yfinance library.
    This class handles the 'Extract' and initial 'Transform' steps.
    """
    
    @staticmethod
    def get_historical_price(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches historical OHLCV data."""
        try:
            data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
            if data.empty:
                raise ValueError(f"No data found for ticker {ticker} on yfinance.")
            # --- Data Standardization ---
            data.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume"
            }, inplace=True)
            # Patch: Ensure all expected columns exist
            for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                if col not in data.columns:
                    data[col] = float('nan')
            return data
        except Exception as e:
            print(f"yfinance provider error: {e}")
            return pd.DataFrame() # Return empty DataFrame on error

    @staticmethod
    def get_news(ticker: str, limit: int = 20) -> List[ShnifterNewsData]:
        """Fetches news and returns a list of standardized ShnifterNewsData objects."""
        try:
            news_obj = yf.Ticker(ticker)
            news_list = news_obj.news
            if not news_list:
                return []
            standardized_news = []
            for item in news_list[:limit]:
                published_date = datetime.fromtimestamp(item.get('providerPublishTime', 0))
                standardized_news.append(
                    ShnifterNewsData(
                        date=published_date,
                        title=item.get('title', ''),
                        url=item.get('link', ''),
                        provider='yfinance',
                        symbols=item.get('relatedTickers', [])
                    )
                )
            return standardized_news
        except Exception as e:
            print(f"yfinance news provider error: {e}")
            return []
