from core.events import EventLog
import yfinance as yf

class ShnifterYFinanceEquityHistorical:
    """Fetches historical equity data from yfinance."""
    def __init__(self, ticker):
        self.ticker = ticker
        EventLog.emit("INFO", f"Initialized ShnifterYFinanceEquityHistorical for {ticker}")

    def fetch(self, start_date, end_date):
        EventLog.emit("DEBUG", f"Fetching historical data for {self.ticker} from {start_date} to {end_date}")
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date)
            EventLog.emit("SUCCESS", f"Fetched historical data for {self.ticker}")
            return data.to_dict() if hasattr(data, 'to_dict') else data
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to fetch historical data for {self.ticker}: {e}")
            return []
