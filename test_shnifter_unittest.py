"""
Unittest-based tests for Shnifter core modules and providers.
Covers data models, provider structure, technical t        event_log.emit("INFO", "test_run_start", {"test": "TestShnifterBB.test_integration_price_and_sma"})
        try:
            shnifter = ShnifterBB()
            end = pd.Timestamp.now().strftime('%Y-%m-%d')
            start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            price_data_obj = shnifter.equity.price.historical('AAPL', start, end)
            price_with_sma = shnifter.technicals.calculate_sma(price_data_obj, 5)
            self.assertIn('SMA_5', price_with_sma.results.columns)
            event_log.emit("INFO", "test_run_success", {"test": "TestShnifterBB.test_integration_price_and_sma"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestShnifterBB.test_integration_price_and_sma", "error": str(e)})egration, and analysis worker logic.
"""
import unittest
import pandas as pd
import numpy as np
import logging
from core.events import EventLog  # Import only EventLog, Event does not exist
from core.data_models import ShnifterData
from providers.yfinance_provider import YFinanceProvider
from toolkits.technicals_toolkit import TechnicalAnalysisToolkit
from shnifter_bb import ShnifterBB
from Multi_Model_Trading_Bot import AnalysisWorker

# Initialize the central event log for capturing test results
event_log = EventLog()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestShnifterData(unittest.TestCase):
    """Tests for the ShnifterData data model."""

    def test_to_df_and_to_dict(self):
        """
        Test ShnifterData conversion to DataFrame and dictionary.
        Ensures that the core data structure can be reliably transformed into common formats.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestShnifterData.test_to_df_and_to_dict"})
        try:
            df = pd.DataFrame({'close': [1, 2, 3]})
            data = ShnifterData(results=df, provider='test')
            self.assertTrue(isinstance(data.to_df(), pd.DataFrame))
            self.assertEqual(data.to_dict(), [{'close': 1}, {'close': 2}, {'close': 3}])
            event_log.emit("INFO", "test_run_success", {"test": "TestShnifterData.test_to_df_and_to_dict"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestShnifterData.test_to_df_and_to_dict", "error": str(e)})
            raise

    def test_warning_on_missing_column(self):
        """
        Test for a warning when a required column is missing for a technical analysis calculation.
        This verifies that the data model correctly handles incomplete data and provides feedback.
        """
        event_log.emit("INFO", "TestShnifterData.test_warning_on_missing_column started")
        try:
            df = pd.DataFrame({'open': [1, 2, 3]})
            data = ShnifterData(results=df)
            toolkit = TechnicalAnalysisToolkit()
            result = toolkit.calculate_sma(data, 5)
            self.assertIn("SMA calculation failed: 'close' column not found.", result.warnings)
            event_log.emit("INFO", "test_run_success", {"test": "TestShnifterData.test_warning_on_missing_column"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestShnifterData.test_warning_on_missing_column", "error": str(e)})
            raise

class TestYFinanceProvider(unittest.TestCase):
    """Tests for the YFinanceProvider to ensure it fetches data correctly."""

    def test_get_historical_price_structure(self):
        """
        Test that the yfinance provider returns a DataFrame with the expected columns.
        This ensures the data source maintains a consistent and expected structure.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestYFinanceProvider.test_get_historical_price_structure"})
        try:
            df = YFinanceProvider.get_historical_price('AAPL', '2024-01-01', '2024-01-10')
            for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume']:
                self.assertIn(col, df.columns)
            event_log.emit("INFO", "test_run_success", {"test": "TestYFinanceProvider.test_get_historical_price_structure"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestYFinanceProvider.test_get_historical_price_structure", "error": str(e)})
            raise

    def test_get_news_structure(self):
        """
        Test that the yfinance provider returns news as a list of ShnifterNewsData objects.
        Verifies that the news fetching mechanism is working and data is correctly parsed.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestYFinanceProvider.test_get_news_structure"})
        try:
            news = YFinanceProvider.get_news('AAPL', limit=2)
            self.assertTrue(isinstance(news, list))
            if news:
                self.assertTrue(hasattr(news[0], 'title'))
                self.assertTrue(hasattr(news[0], 'provider'))
                self.assertTrue(hasattr(news[0], 'url'))
            event_log.emit("INFO", "test_run_success", {"test": "TestYFinanceProvider.test_get_news_structure"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestYFinanceProvider.test_get_news_structure", "error": str(e)})
            raise

class TestTechnicalAnalysisToolkit(unittest.TestCase):
    """Tests for the TechnicalAnalysisToolkit."""

    def test_calculate_sma(self):
        """
        Test the Simple Moving Average (SMA) calculation.
        Ensures that the technical indicator is calculated correctly and added to the DataFrame.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestTechnicalAnalysisToolkit.test_calculate_sma"})
        try:
            df = pd.DataFrame({'close': np.arange(10)})
            data = ShnifterData(results=df)
            toolkit = TechnicalAnalysisToolkit()
            result = toolkit.calculate_sma(data, 3)
            self.assertIn('SMA_3', result.results.columns)
            event_log.emit("INFO", "test_run_success", {"test": "TestTechnicalAnalysisToolkit.test_calculate_sma"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestTechnicalAnalysisToolkit.test_calculate_sma", "error": str(e)})
            raise

class TestShnifterBB(unittest.TestCase):
    """High-level integration tests for the ShnifterBB facade."""

    def test_integration_price_and_sma(self):
        """
        Integration test: fetch historical price data and then calculate the SMA.
        This tests the end-to-end workflow of data retrieval and analysis.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestShnifterBB.test_integration_price_and_sma"})
        try:
            shnifter = ShnifterBB()
            end = pd.Timestamp.now().strftime('%Y-%m-%d')
            start = (pd.Timestamp.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')
            price_data_obj = shnifter.equity.price.historical('AAPL', start, end)
            price_with_sma = shnifter.technicals.calculate_sma(price_data_obj, 5)
            self.assertIn('SMA_5', price_with_sma.results.columns)
            event_log.emit("INFO", "test_run_success", {"test": "TestShnifterBB.test_integration_price_and_sma"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestShnifterBB.test_integration_price_and_sma", "error": str(e)})
            raise

class TestAnalysisWorker(unittest.TestCase):
    """Tests for the AnalysisWorker, which handles trading signal generation."""

    def test_trend_signal(self):
        """
        Test the trend-based signal generation.
        Ensures the worker produces a valid signal ('BUY', 'SELL', 'HOLD').
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestAnalysisWorker.test_trend_signal"})
        try:
            df = pd.DataFrame({'close': np.linspace(100, 120, 60)})
            worker = AnalysisWorker('TEST')
            signal, _ = worker.get_trend_signal(df)
            self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])
            event_log.emit("INFO", "test_run_success", {"test": "TestAnalysisWorker.test_trend_signal"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestAnalysisWorker.test_trend_signal", "error": str(e)})
            raise

    def test_ml_signal(self):
        """
        Test the machine learning-based signal generation.
        Ensures the ML model interface produces a valid signal.
        """
        event_log.emit("INFO", "test_run_start", {"test": "TestAnalysisWorker.test_ml_signal"})
        try:
            df = pd.DataFrame({'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)})
            worker = AnalysisWorker('TEST')
            signal, _ = worker.get_ml_signal(df)
            self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])
            event_log.emit("INFO", "test_run_success", {"test": "TestAnalysisWorker.test_ml_signal"})
        except Exception as e:
            event_log.emit("ERROR", "test_run_failure", {"test": "TestAnalysisWorker.test_ml_signal", "error": str(e)})
            raise

if __name__ == '__main__':
    # Main execution block to run the test suite
    event_log.emit("INFO", "Shnifter Unittest Suite started")
    logging.info("[INFO] Running Shnifter unittest suite...")
    
    # Create a TestSuite
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    
    # Add tests to the suite
    suite.addTest(loader.loadTestsFromTestCase(TestShnifterData))
    suite.addTest(loader.loadTestsFromTestCase(TestYFinanceProvider))
    suite.addTest(loader.loadTestsFromTestCase(TestTechnicalAnalysisToolkit))
    suite.addTest(loader.loadTestsFromTestCase(TestShnifterBB))
    suite.addTest(loader.loadTestsFromTestCase(TestAnalysisWorker))
    
    # Run the suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Log the overall results
    if result.wasSuccessful():
        event_log.emit("INFO", "Shnifter Unittest Suite passed")
        logging.info("[SUCCESS] All tests passed.")
    else:
        event_log.emit("ERROR", f"Shnifter Unittest Suite failed. Errors: {len(result.errors)}, Failures: {len(result.failures)}")
        logging.error("[FAILURE] Some tests failed.")
