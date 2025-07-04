from core.events import EventLog
import requests
import os

class ShnifterBenzingaCompanyNews:
    """Fetches company news from Benzinga."""
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key = os.getenv('BENZINGA_API_KEY')
        EventLog.emit("INFO", f"Initialized ShnifterBenzingaCompanyNews for {ticker}")

    def fetch(self, limit=10):
        EventLog.emit("DEBUG", f"Fetching company news for {self.ticker}, limit={limit}")
        url = f"https://api.benzinga.com/api/v2/news"
        params = {
            'tickers': self.ticker,
            'limit': limit,
            'token': self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            EventLog.emit("SUCCESS", f"Fetched company news for {self.ticker}")
            return data.get('news', data)
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to fetch company news for {self.ticker}: {e}")
            return []
