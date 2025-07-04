"""
Intrinio Equity Quote Model for Shnifter Trader.
wires fetch events to EventBus.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import requests
from core.events import EventBus, EventLog

class IntrinioEquityQuoteQueryParams(BaseModel):
    symbol: str
    api_key: Optional[str] = None
    source: str = "iex"

class IntrinioEquityQuoteData(BaseModel):
    symbol: str
    last_price: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    open: Optional[float] = None
    volume: Optional[int] = None
    updated_on: Optional[str] = None
    source: Optional[str] = None

class IntrinioEquityQuoteFetcher:
    @staticmethod
    def fetch(params: IntrinioEquityQuoteQueryParams) -> List[IntrinioEquityQuoteData]:
        base_url = "https://api-v2.intrinio.com"
        api_key = params.api_key or "demo"
        results = []
        for symbol in params.symbol.split(","):
            url = f"{base_url}/securities/{symbol.strip()}/prices/realtime?source={params.source}&api_key={api_key}"
            try:
                resp = requests.get(url)
                resp.raise_for_status()
                data = resp.json()
                if not data or "last_price" not in data:
                    EventLog.emit("WARNING", f"No Intrinio data for {symbol}")
                    continue
                mapped = {
                    "symbol": symbol,
                    "last_price": data.get("last_price"),
                    "high": data.get("high_price"),
                    "low": data.get("low_price"),
                    "prev_close": data.get("close_price"),
                    "open": data.get("open_price"),
                    "volume": data.get("market_volume"),
                    "updated_on": data.get("updated_on"),
                    "source": data.get("source"),
                }
                results.append(IntrinioEquityQuoteData(**mapped))
                EventLog.emit("INFO", f"Fetched Intrinio equity quote for {symbol}")
            except Exception as e:
                EventLog.emit("ERROR", f"Error fetching Intrinio equity quote for {symbol}: {e}")
        EventBus.publish("INFO", {"event": "intrinio_equity_quote_fetch", "symbols": params.symbol, "count": len(results)})
        return results

# This script is now adapted for direct use with Shnifter. All OpenBB dependencies removed.
