import unittest
import pandas as pd
import numpy as np
from core.data_models import ShnifterData
from providers.yfinance_provider import YFinanceProvider
from toolkits.technicals_toolkit import TechnicalAnalysisToolkit
from shnifter_bb import ShnifterBB
from Multi_Model_Trading_Bot import AnalysisWorker

class TestShnifterData(unittest.TestCase):
    def test_to_df_and_to_dict(self):
        df = pd.DataFrame({'close': [1, 2, 3]})
        data = ShnifterData(results=df, provider='test')
        self.assertTrue(isinstance(data.to_df(), pd.DataFrame))
        self.assertEqual(data.to_dict(), [{'close': 1}, {'close': 2}, {'close': 3}])

    def test_warning_on_missing_column(self):
        df = pd.DataFrame({'open': [1, 2, 3]})
        data = ShnifterData(results=df)
        toolkit = TechnicalAnalysisToolkit()
        result = toolkit.calculate_sma(data, 5)
        self.assertIn("SMA calculation failed: 'close' column not found.", result.warnings)

class TestYFinanceProvider(unittest.TestCase):
    def test_get_historical_price_structure(self):
        # Use a known ticker and short range for speed
        df = YFinanceProvider.get_historical_price('AAPL', '2024-01-01', '2024-01-10')
        # Only require the essential columns, but all will be present (may be NaN)
        for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
            self.assertIn(col, df.columns)

    def test_get_news_structure(self):
        df = YFinanceProvider.get_news('AAPL', limit=2)
        self.assertTrue(set(['title', 'provider', 'url']).issubset(df.columns))

class TestTechnicalAnalysisToolkit(unittest.TestCase):
    def test_calculate_sma(self):
        df = pd.DataFrame({'close': np.arange(10)})
        data = ShnifterData(results=df)
        toolkit = TechnicalAnalysisToolkit()
        result = toolkit.calculate_sma(data, 3)
        self.assertIn('SMA_3', result.results.columns)

class TestShnifterBB(unittest.TestCase):
    def test_integration_price_and_sma(self):
        shnifter = ShnifterBB()
        end = pd.Timestamp.now().strftime('%Y-%m-%d')
        start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
        price_data_obj = shnifter.equity.price.historical('AAPL', start, end)
        price_with_sma = shnifter.technicals.calculate_sma(price_data_obj, 5)
        self.assertIn('SMA_5', price_with_sma.results.columns)

class TestAnalysisWorker(unittest.TestCase):
    def test_trend_signal(self):
        df = pd.DataFrame({'close': np.linspace(100, 120, 60)})
        worker = AnalysisWorker('TEST')
        signal, _ = worker.get_trend_signal(df)
        self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])

    def test_ml_signal(self):
        df = pd.DataFrame({'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)})
        worker = AnalysisWorker('TEST')
        signal, _ = worker.get_ml_signal(df)
        self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])

if __name__ == '__main__':
    unittest.main()
