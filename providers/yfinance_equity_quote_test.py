"""
Test for refactored YFinanceEquityQuoteFetcher (Shnifter version).
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from yfinance_equity_quote import YFinanceEquityQuoteFetcher, YFinanceEquityQuoteQueryParams

def test_yfinance_equity_quote_fetcher():
    params = YFinanceEquityQuoteQueryParams(symbol="AAPL")
    results = YFinanceEquityQuoteFetcher.fetch(params)
    assert results, "No results returned."
    assert results[0].symbol == "AAPL"
    print(results[0])

if __name__ == "__main__":
    test_yfinance_equity_quote_fetcher()
    print("Test passed.")
