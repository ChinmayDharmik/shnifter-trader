# test_shnifter.py - simple integration and unit tests for The Shnifter Trader core module
# Each test prints status and asserts expected model behavior.

def test_signal_logic():
    """Test overall signal logic with simulated price data."""
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Simulate fake price data for testing
    dates = pd.date_range(end=datetime.now(), periods=100)
    prices = np.linspace(100, 110, 100) + np.random.normal(0, 1, 100)
    df = pd.DataFrame({'close': prices}, index=dates)

    # Import the signal functions from core file
    from Multi_Model_Trading_Bot import AnalysisWorker
    worker = AnalysisWorker("TEST")

    trend_signal, _ = worker.get_trend_signal(df.copy())
    ml_signal, _ = worker.get_ml_signal(df.copy())

    assert trend_signal in ["BUY", "SELL"], f"Unexpected trend signal: {trend_signal}"
    assert ml_signal in ["BUY", "SELL", "HOLD"], f"Unexpected ML signal: {ml_signal}"
    print("✅ test_signal_logic passed.")

def test_trend_signal_buy():
    """Test trend model returns BUY for upward trend."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np

    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(100, 120, 60)})
    signal, log = worker.get_trend_signal(df)
    assert signal == "BUY", f"Expected BUY, got {signal}"
    print("✅ test_trend_signal_buy passed.")

def test_trend_signal_sell():
    """Test trend model returns SELL for downward trend."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np

    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(120, 100, 60)})
    signal, log = worker.get_trend_signal(df)
    assert signal == "SELL", f"Expected SELL, got {signal}"
    print("✅ test_trend_signal_sell passed.")

def test_ml_signal_buy_sell():
    """Test ML model returns a valid signal."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    import numpy as np

    worker = AnalysisWorker("TEST")
    df = pd.DataFrame({'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)})
    signal, log = worker.get_ml_signal(df)
    assert signal in ["BUY", "SELL", "HOLD"], f"Unexpected ML signal: {signal}"
    print("✅ test_ml_signal_buy_sell passed.")

def test_sentiment_signal_buy():
    """Test sentiment model returns BUY for positive news."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch

    worker = AnalysisWorker("TEST")
    # Patch YFinanceProvider.get_news directly
    with patch("providers.yfinance_provider.YFinanceProvider.get_news") as mock_news, \
         patch("Multi_Model_Trading_Bot.analyzer") as mock_analyzer:
        mock_df = pd.DataFrame({'title': ["Great earnings!", "Stock surges"]})
        mock_news.return_value = mock_df
        mock_analyzer.polarity_scores.side_effect = lambda h: {'compound': 0.2}
        signal, log = worker.get_sentiment_signal("TEST")
        assert signal == "BUY", f"Expected BUY, got {signal}"
    print("✅ test_sentiment_signal_buy passed.")

def test_sentiment_signal_sell():
    """Test sentiment model returns SELL for negative news."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch

    worker = AnalysisWorker("TEST")
    # Patch YFinanceProvider.get_news directly
    with patch("providers.yfinance_provider.YFinanceProvider.get_news") as mock_news, \
         patch("Multi_Model_Trading_Bot.analyzer") as mock_analyzer:
        mock_df = pd.DataFrame({'title': ["Bad quarter", "Stock plunges"]})
        mock_news.return_value = mock_df
        mock_analyzer.polarity_scores.side_effect = lambda h: {'compound': -0.2}
        signal, log = worker.get_sentiment_signal("TEST")
        assert signal == "SELL", f"Expected SELL, got {signal}"
    print("✅ test_sentiment_signal_sell passed.")

def test_sentiment_signal_hold_on_no_news():
    """Test sentiment model returns HOLD when no news is available."""
    from Multi_Model_Trading_Bot import AnalysisWorker
    import pandas as pd
    from unittest.mock import patch

    worker = AnalysisWorker("TEST")
    # Patch YFinanceProvider.get_news directly
    with patch("providers.yfinance_provider.YFinanceProvider.get_news") as mock_news:
        mock_news.return_value = pd.DataFrame({'title': []})
        signal, log = worker.get_sentiment_signal("TEST")
        assert signal == "HOLD", f"Expected HOLD, got {signal}"
    print("✅ test_sentiment_signal_hold_on_no_news passed.")

if __name__ == "__main__":
    print("[INFO] Running all Shnifter Trader tests...")
    test_signal_logic()
    test_trend_signal_buy()
    test_trend_signal_sell()
    test_ml_signal_buy_sell()
    test_sentiment_signal_buy()
    test_sentiment_signal_sell()
    test_sentiment_signal_hold_on_no_news()
    print("All tests passed.")
