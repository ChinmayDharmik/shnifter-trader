from core.events import EventLog
import requests
import os

class ShnifterIntrinioBalanceSheet:
    """Fetches balance sheet data from Intrinio."""
    def __init__(self, ticker):
        self.ticker = ticker
        self.api_key = os.getenv('INTRINIO_API_KEY')
        EventLog.emit("INFO", f"Initialized ShnifterIntrinioBalanceSheet for {ticker}")

    def fetch(self, period='annual'):
        EventLog.emit("DEBUG", f"Fetching balance sheet for {self.ticker}, period={period}")
        url = f"https://api-v2.intrinio.com/financials/standardized"
        params = {
            'identifier': self.ticker,
            'statement': 'balance_sheet',
            'fiscal_period_type': period,
            'api_key': self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            EventLog.emit("SUCCESS", f"Fetched balance sheet for {self.ticker}")
            return data
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to fetch balance sheet for {self.ticker}: {e}")
            return None
