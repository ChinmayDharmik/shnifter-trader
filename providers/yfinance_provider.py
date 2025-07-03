import yfinance as yf
import pandas as pd

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
    def get_news(ticker: str, limit: int = 20) -> pd.DataFrame:
        """Fetches news headlines."""
        try:
            news_obj = yf.Ticker(ticker)
            news_data = news_obj.news
            if not news_data:
                return pd.DataFrame(columns=['title', 'provider', 'url'])
            df = pd.DataFrame(news_data)
            # Ensure required columns exist
            for col in ['title', 'publisher', 'link']:
                if col not in df.columns:
                    df[col] = ""
            df.rename(columns={"title": "title", "publisher": "provider", "link": "url"}, inplace=True)
            return df[['title', 'provider', 'url']].head(limit)
        except Exception as e:
            print(f"yfinance news provider error: {e}")
            return pd.DataFrame(columns=['title', 'provider', 'url'])
