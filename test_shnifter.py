"""
Integration and unit tests for The Shnifter Trader core module.
Covers signal logic, trend/ML/sentiment models, and edge cases.
Compatible with pytest, unittest, and direct execution.
"""

# --- Core Signal Logic Tests ---
def test_signal_logic():
    """Test overall signal logic with simulated price data."""
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from Multi_Model_Trading_Bot import AnalysisWorker

    # Simulate fake price data for testing
    dates = pd.date_range(end=datetime.now(), periods=100)
    prices = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
    df = pd.DataFrame({'close': prices}, index=dates)

    worker = AnalysisWorker("TEST")
    trend_signal, _ = worker.get_trend_signal(df.copy())
    ml_signal, _ = worker.get_ml_signal(df.copy())

    assert trend_signal in ["BUY", "SELL"], f"Unexpected trend signal: {trend_signal}"
    assert ml_signal in ["BUY", "SELL", "HOLD"], f"Unexpected ML signal: {ml_signal}"

# --- Trend Model Tests ---
def test_trend_signal_buy():
    """Test trend model returns BUY for upward trend."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np
    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(100, 120, 60)})
    signal, log = worker.get_trend_signal(df)
    assert signal == "BUY", f"Expected BUY, got {signal}"

def test_trend_signal_sell():
    """Test trend model returns SELL for downward trend."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np
    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(120, 100, 60)})
    signal, log = worker.get_trend_signal(df)
    assert signal == "SELL", f"Expected SELL, got {signal}"

# --- ML Model Tests ---
def test_ml_signal_buy_sell():
    """Test ML model returns a valid signal."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np
    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)})
    signal, log = worker.get_ml_signal(df)
    assert signal in ["BUY", "SELL", "HOLD"], f"Unexpected ML signal: {signal}"

# --- Sentiment Model Tests ---
def test_sentiment_signal_buy():
    """Test sentiment model returns BUY for positive news."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch
    worker = AnalysisWorker("TEST")
    with patch("Multi_Model_Trading_Bot.analyzer") as mock_analyzer:
        # Mock the news provider to avoid external calls
        with patch.object(worker.shnifter.news, 'get') as mock_news:
            # Create mock news articles with title attribute
            class MockArticle:
                def __init__(self, title):
                    self.title = title
            mock_news.return_value = [MockArticle("Great earnings!"), MockArticle("Stock surges")]
            mock_analyzer.polarity_scores.side_effect = lambda h: {'compound': 0.2}
            signal, log = worker.get_sentiment_signal("TEST")
            assert signal == "BUY", f"Expected BUY, got {signal}"

def test_sentiment_signal_sell():
    """Test sentiment model returns SELL for negative news."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch
    worker = AnalysisWorker("TEST")
    with patch("Multi_Model_Trading_Bot.analyzer") as mock_analyzer:
        # Mock the news provider to avoid external calls
        with patch.object(worker.shnifter.news, 'get') as mock_news:
            # Create mock news articles with title attribute
            class MockArticle:
                def __init__(self, title):
                    self.title = title
            mock_news.return_value = [MockArticle("Bad quarter"), MockArticle("Stock plunges")]
            mock_analyzer.polarity_scores.side_effect = lambda h: {'compound': -0.2}
            signal, log = worker.get_sentiment_signal("TEST")
            assert signal == "SELL", f"Expected SELL, got {signal}"

def test_sentiment_signal_hold_on_no_news():
    """Test sentiment model returns HOLD when no news is available."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch
    worker = AnalysisWorker("TEST")
    with patch.object(worker.shnifter.news, 'get') as mock_news:
        mock_news.return_value = []  # No news articles
        signal, log = worker.get_sentiment_signal("TEST")
        assert signal == "HOLD", f"Expected HOLD, got {signal}"

# --- Edge Case Tests ---
def test_empty_dataframe():
    """Test all models handle empty DataFrame gracefully."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': []})
    trend_signal, _ = worker.get_trend_signal(df)
    ml_signal, _ = worker.get_ml_signal(df)
    assert trend_signal in ["BUY", "SELL", "HOLD", None]
    assert ml_signal in ["BUY", "SELL", "HOLD", None]

# --- Main block for direct execution ---
if __name__ == "__main__":
    print("[INFO] Running all Shnifter Trader tests...")
    import sys
    import inspect
    # Run all test functions in this file
    test_functions = [obj for name, obj in globals().items() if name.startswith("test_") and inspect.isfunction(obj)]
    failed = 0
    for test in test_functions:
        try:
            test()
            print(f"✅ {test.__name__} passed.")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) failed.")
