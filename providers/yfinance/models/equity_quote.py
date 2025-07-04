"""
Refactored YFinance Equity Quote Model for Shnifter Trader.
Wires fetch events to EventBus.
"""
from typing import Any, Dict, List, Optional
from warnings import warn
from pydantic import BaseModel, Field
import yfinance as yf
from core.events import EventBus, EventLog

class YFinanceEquityQuoteQueryParams(BaseModel):
    symbol: str
    # Add more fields as needed for your use case

class YFinanceEquityQuoteData(BaseModel):
    symbol: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    last_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    ma_50d: Optional[float] = None
    ma_200d: Optional[float] = None
    volume_average: Optional[float] = None
    volume_average_10d: Optional[float] = None
    currency: Optional[str] = None

class YFinanceEquityQuoteFetcher:
    @staticmethod
    def fetch(params: YFinanceEquityQuoteQueryParams) -> List[YFinanceEquityQuoteData]:
        symbols = params.symbol.split(",")
        results = []
        fields = {
            "symbol": "symbol",
            "longName": "name",
            "quoteType": "asset_type",
            "currentPrice": "last_price",
            "dayHigh": "high",
            "dayLow": "low",
            "previousClose": "prev_close",
            "fiftyTwoWeekHigh": "year_high",
            "fiftyTwoWeekLow": "year_low",
            "fiftyDayAverage": "ma_50d",
            "twoHundredDayAverage": "ma_200d",
            "averageVolume": "volume_average",
            "averageDailyVolume10Day": "volume_average_10d",
            "currency": "currency",
        }
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                data = {fields[k]: info.get(k) for k in fields if k in info}
                data["symbol"] = symbol
                results.append(YFinanceEquityQuoteData(**data))
                EventLog.emit("INFO", f"Fetched equity quote for {symbol}")
            except Exception as e:
                warn(f"Error getting data for {symbol}: {e}")
                EventLog.emit("ERROR", f"Error fetching equity quote for {symbol}: {e}")
        EventBus.publish("INFO", {"event": "yfinance_equity_quote_fetch", "symbols": symbols, "count": len(results)})
        return results
