"""
Benzinga Company News Model for Shnifter Trader.
wires fetch events to EventBus.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import requests
from core.events import EventBus, EventLog

class BenzingaCompanyNewsQueryParams(BaseModel):
    symbol: str
    api_key: Optional[str] = None
    limit: int = 10
    sort: str = "created"
    order: str = "desc"

class BenzingaCompanyNewsData(BaseModel):
    id: str
    author: Optional[str] = None
    teaser: Optional[str] = None
    images: Optional[list] = None
    channels: Optional[str] = None
    stocks: Optional[str] = None
    tags: Optional[str] = None
    updated: Optional[str] = None
    date: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None

class BenzingaCompanyNewsFetcher:
    @staticmethod
    def fetch(params: BenzingaCompanyNewsQueryParams) -> List[BenzingaCompanyNewsData]:
        base_url = "https://api.benzinga.com/api/v2/news"
        api_key = params.api_key or "demo"
        results = []
        url = f"{base_url}?tickers={params.symbol}&pageSize={params.limit}&sort={params.sort}:{params.order}&token={api_key}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            if not data or not data.get("news"):
                EventLog.emit("WARNING", f"No Benzinga news for {params.symbol}")
                return results
            for item in data["news"]:
                mapped = {
                    "id": str(item.get("id")),
                    "author": item.get("author"),
                    "teaser": item.get("teaser"),
                    "images": item.get("images"),
                    "channels": item.get("channels"),
                    "stocks": item.get("stocks"),
                    "tags": item.get("tags"),
                    "updated": item.get("updated"),
                    "date": item.get("created"),
                    "text": item.get("body"),
                    "title": item.get("title"),
                }
                results.append(BenzingaCompanyNewsData(**mapped))
            EventLog.emit("INFO", f"Fetched Benzinga news for {params.symbol}")
        except Exception as e:
            EventLog.emit("ERROR", f"Error fetching Benzinga news for {params.symbol}: {e}")
        EventBus.publish("INFO", {"event": "benzinga_company_news_fetch", "symbol": params.symbol, "count": len(results)})
        return results

