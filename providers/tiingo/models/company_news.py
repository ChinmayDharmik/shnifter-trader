"""
Tiingo Company News Model for Shnifter Trader.
wires fetch events to EventBus.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
import requests
from core.events import EventBus, EventLog

class TiingoCompanyNewsQueryParams(BaseModel):
    symbol: str
    api_key: Optional[str] = None
    limit: int = 10
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class TiingoCompanyNewsData(BaseModel):
    article_id: str
    title: Optional[str] = None
    text: Optional[str] = None
    date: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[str] = None
    crawl_date: Optional[str] = None

class TiingoCompanyNewsFetcher:
    @staticmethod
    def fetch(params: TiingoCompanyNewsQueryParams) -> List[TiingoCompanyNewsData]:
        base_url = "https://api.tiingo.com/tiingo/news"
        api_key = params.api_key or "demo"
        results = []
        url = f"{base_url}?tickers={params.symbol}&limit={params.limit}&token={api_key}"
        if params.start_date:
            url += f"&startDate={params.start_date}"
        if params.end_date:
            url += f"&endDate={params.end_date}"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                EventLog.emit("WARNING", f"No Tiingo news for {params.symbol}")
                return results
            for item in data:
                mapped = {
                    "article_id": str(item.get("id")),
                    "title": item.get("title"),
                    "text": item.get("description"),
                    "date": item.get("publishedDate"),
                    "source": item.get("source"),
                    "tags": ",".join(item.get("tags", [])) if item.get("tags") else None,
                    "crawl_date": item.get("crawlDate"),
                }
                results.append(TiingoCompanyNewsData(**mapped))
            EventLog.emit("INFO", f"Fetched Tiingo news for {params.symbol}")
        except Exception as e:
            EventLog.emit("ERROR", f"Error fetching Tiingo news for {params.symbol}: {e}")
        EventBus.publish("INFO", {"event": "tiingo_company_news_fetch", "symbol": params.symbol, "count": len(results)})
        return results
