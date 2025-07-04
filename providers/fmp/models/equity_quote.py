"""
Refactored FMP Equity Quote Model for Shnifter Trader.
Wires fetch events to EventBus.
"""
from typing import List, Optional
from warnings import warn
from pydantic import BaseModel, Field
import requests
from core.events import EventBus, EventLog

class FMPEquityQuoteQueryParams(BaseModel):
    symbol: str
    api_key: Optional[str] = None

class FMPEquityQuoteData(BaseModel):
    symbol: str
    last_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    price_avg50: Optional[float] = None
    price_avg200: Optional[float] = None
    avg_volume: Optional[int] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[int] = None
    eps: Optional[float] = None
    pe: Optional[float] = None
    earnings_announcement: Optional[str] = None
    change_percent: Optional[float] = None

class FMPEquityQuoteFetcher:
    @staticmethod
    def fetch(params: FMPEquityQuoteQueryParams) -> List[FMPEquityQuoteData]:
        symbols = params.symbol.split(",")
        api_key = params.api_key or "demo"
        base_url = "https://financialmodelingprep.com/api/v3"
        results = []
        for symbol in symbols:
            url = f"{base_url}/quote/{symbol}?apikey={api_key}"
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    warn(f"No data found for {symbol}")
                    EventLog.emit("WARNING", f"No FMP data for {symbol}")
                    continue
                for item in data:
                    mapped = {
                        "symbol": item.get("symbol"),
                        "last_price": item.get("price"),
                        "high": item.get("dayHigh"),
                        "low": item.get("dayLow"),
                        "prev_close": item.get("previousClose"),
                        "price_avg50": item.get("priceAvg50"),
                        "price_avg200": item.get("priceAvg200"),
                        "avg_volume": item.get("avgVolume"),
                        "market_cap": item.get("marketCap"),
                        "shares_outstanding": item.get("sharesOutstanding"),
                        "eps": item.get("eps"),
                        "pe": item.get("pe"),
                        "earnings_announcement": item.get("earningsAnnouncement"),
                        "change_percent": item.get("changesPercentage"),
                    }
                    results.append(FMPEquityQuoteData(**mapped))
                    EventLog.emit("INFO", f"Fetched FMP equity quote for {symbol}")
            except Exception as e:
                warn(f"Error fetching FMP data for {symbol}: {e}")
                EventLog.emit("ERROR", f"Error fetching FMP equity quote for {symbol}: {e}")
        EventBus.publish("INFO", {"event": "fmp_equity_quote_fetch", "symbols": symbols, "count": len(results)})
        return results
