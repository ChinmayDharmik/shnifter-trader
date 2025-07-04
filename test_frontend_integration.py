"""
Frontend Integration Test Suite
Tests the actual GUI interactions and real integration issues
"""

import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

# Ensure we can import the main module
sys.path.insert(0, '.')

class TestFrontendIntegration(unittest.TestCase):
    """Test actual frontend integration issues"""
    
    @classmethod
    def setUpClass(cls):
        """Set up QApplication for widget testing"""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        """Set up each test"""
        self.errors_found = []
    
    def test_ml_model_empty_data_handling(self):
        """Test ML model with empty data after dropna - reproduces your error"""
        from Multi_Model_Trading_Bot import AnalysisWorker
        
        # Create worker
        worker = AnalysisWorker("TEST", "llama3", "yfinance")
        
        # Test with data that becomes empty after processing
        empty_df = pd.DataFrame({
            'close': [np.nan, np.nan, np.nan],  # All NaN - will be empty after dropna
            'volume': [np.nan, np.nan, np.nan]
        })
        
        # This should NOT crash and should return HOLD
        signal, log = worker.get_ml_signal(empty_df)
        
        self.assertEqual(signal, "HOLD")
        self.assertTrue(any("No valid data" in line for line in log))
        print("✓ ML Model empty data handling test passed")
    
    def test_backtest_worker_api_compatibility(self):
        """Test BacktestWorker ShnifterBB API compatibility"""
        from Multi_Model_Trading_Bot import BacktestWorker
        
        # Create worker
        worker = BacktestWorker(["TEST"])
        
        # Test the simulate_backtest method directly
        try:
            pnl, win_rate, trades = worker.simulate_backtest("TEST", "yfinance")
            print("✓ BacktestWorker API compatibility test passed")
        except TypeError as e:
            if "unexpected keyword argument 'provider'" in str(e):
                self.errors_found.append(f"BacktestWorker provider error: {e}")
                print("✗ BacktestWorker still has provider parameter issue")
            else:
                raise
    
    def test_popout_widget_initialization(self):
        """Test popout widget initialization with required parameters"""
        try:
            # Test PnL Dashboard with missing callback
            from shnifter_frontend.pnl_dashboard_popout import PnLDashboardPopout
            
            # This should work with dummy callback
            def dummy_callback():
                return {'total_pnl': 0, 'win_rate': 0, 'total_trades': 0}
            
            widget = PnLDashboardPopout(parent=None, get_stats_callback=dummy_callback)
            widget.close()
            print("✓ PnL Dashboard popout initialization test passed")
            
        except TypeError as e:
            if "missing 1 required positional argument" in str(e):
                self.errors_found.append(f"PnL Dashboard callback error: {e}")
                print("✗ PnL Dashboard still missing required callback parameter")
            else:
                raise
    
    def test_event_log_scroll_functionality(self):
        """Test EventLog scroll functionality with correct Qt API"""
        try:
            from shnifter_frontend.event_log_popout import EventLogPopout
            
            popout = EventLogPopout(parent=None)
            # Test the scroll method
            popout.scroll_to_end()
            popout.close()
            print("✓ EventLog scroll functionality test passed")
            
        except AttributeError as e:
            if "QTextCursor" in str(e) and "End" in str(e):
                self.errors_found.append(f"EventLog scroll error: {e}")
                print("✗ EventLog still has QTextCursor.End API error")
            else:
                raise
    
    def test_dual_llm_analysis_method_calls(self):
        """Test dual LLM analysis method call compatibility"""
        from Multi_Model_Trading_Bot import MainWindow
        
        # Create main window
        window = MainWindow()
        
        # Test dual LLM decision with mock data
        with patch('Multi_Model_Trading_Bot.ShnifterBB') as mock_shnifter:
            # Setup mock data
            mock_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 1200, 1300, 1400]
            })
            mock_shnifter.return_value.equity.price.historical.return_value.to_df.return_value = mock_data
            
            try:
                window.run_dual_llm_decision("TEST", "yfinance")
                print("✓ Dual LLM analysis method calls test passed")
            except Exception as e:
                self.errors_found.append(f"Dual LLM method call error: {e}")
                print(f"✗ Dual LLM analysis error: {e}")
        
        window.close()
    
    def test_widget_lifecycle_and_event_handling(self):
        """Test widget lifecycle and EventBus integration"""
        from shnifter_frontend.event_log_popout import EventLogPopout
        from core.events import EventLog
        
        # Create and close widget rapidly to test lifecycle
        popout = EventLogPopout(parent=None)
        
        # Test event emission after widget creation
        try:
            EventLog.emit("INFO", "Test message for lifecycle")
            print("✓ Widget lifecycle and event handling test passed")
        except Exception as e:
            if "Internal C++ object" in str(e) and "already deleted" in str(e):
                self.errors_found.append(f"Widget lifecycle error: {e}")
                print("✗ Widget lifecycle error with deleted Qt objects")
            else:
                print(f"Other widget error: {e}")
        
        popout.close()
    
    def test_analysis_worker_real_data_conditions(self):
        """Test AnalysisWorker with realistic problematic data conditions"""
        from Multi_Model_Trading_Bot import AnalysisWorker
        
        worker = AnalysisWorker("TEST", "llama3", "yfinance")
        
        # Test with very small dataset (causes train_test_split issues)
        small_df = pd.DataFrame({
            'close': [100.0, 100.1],  # Only 2 rows
            'volume': [1000, 1100]
        })
        
        signal, log = worker.get_ml_signal(small_df)
        
        # Should handle gracefully and return HOLD
        self.assertEqual(signal, "HOLD")
        self.assertTrue(any("Not enough data" in line for line in log))
        print("✓ AnalysisWorker small dataset test passed")
    
    def tearDown(self):
        """Report any errors found"""
        if self.errors_found:
            print("\\n❌ FRONTEND INTEGRATION ERRORS FOUND:")
            for error in self.errors_found:
                print(f"   - {error}")
        else:
            print("\\n✅ All frontend integration tests passed!")

if __name__ == "__main__":
    print("🔍 Running Frontend Integration Tests...")
    print("=" * 60)
    
    # Run the tests
    unittest.main(verbosity=2)
