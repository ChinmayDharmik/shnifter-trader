from core.events import EventLog
import requests
import os

class ShnifterFmpEquityQuote:
    """Fetches real-time equity quote data from FMP."""
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key = os.getenv('FMP_API_KEY')
        EventLog.emit("INFO", f"Initialized ShnifterFmpEquityQuote for {ticker}")

    def fetch(self):
        EventLog.emit("DEBUG", f"Fetching real-time quote for {self.ticker}")
        url = f"https://financialmodelingprep.com/api/v3/quote/{self.ticker}"
        params = {'apikey': self.api_key}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            EventLog.emit("SUCCESS", f"Fetched real-time quote for {self.ticker}")
            return data[0] if data else {}
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to fetch real-time quote for {self.ticker}: {e}")
            return {}
