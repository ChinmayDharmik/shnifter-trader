from core.events import EventLog
import requests
import os

class ShnifterTiingoCryptoHistorical:
    """Fetches historical crypto data from Tiingo."""
    def __init__(self, symbol):
        self.symbol = symbol
        self.api_key = os.getenv('TIINGO_API_KEY')
        EventLog.emit("INFO", f"Initialized ShnifterTiingoCryptoHistorical for {symbol}")

    def fetch(self, start_date, end_date):
        EventLog.emit("DEBUG", f"Fetching crypto historical data for {self.symbol} from {start_date} to {end_date}")
        url = f"https://api.tiingo.com/tiingo/crypto/prices"
        params = {
            'tickers': self.symbol,
            'startDate': start_date,
            'endDate': end_date,
            'token': self.api_key
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            EventLog.emit("SUCCESS", f"Fetched crypto historical data for {self.symbol}")
            return data
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to fetch crypto historical data for {self.symbol}: {e}")
            return []
