"""
Enhanced Integration and Unit Tests for Shnifter Trader
Complete test coverage with proper event system integration
Compatible with pytest, unittest, and direct execution
"""

import unittest
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Core imports
from core.events import EventLog, EventBus
from core.data_models import ShnifterData, ShnifterNewsData
from core.config import shnifter_config

# Test initialization
event_log = EventLog()

class TestEventSystemIntegration(unittest.TestCase):
    """Test proper event system integration"""
    
    def setUp(self):
        """Set up test environment with event system"""
        self.event_log = EventLog()
        self.test_events = []
        
        # Subscribe to test events
        EventBus.subscribe("test_event", self._capture_test_event)
        EventBus.subscribe("test_start", self._capture_test_event)
        EventBus.subscribe("test_complete", self._capture_test_event)
        
    def _capture_test_event(self, event_data):
        """Capture events for testing"""
        self.test_events.append(event_data)
        
    def test_event_log_emit(self):
        """Test EventLog emit functionality"""
        self.event_log.emit("INFO", "test_message", {"test_data": "test_value"})
        
        # Verify event was logged
        logs = self.event_log.get_recent_logs()
        self.assertTrue(any(log.get('message') == "test_message" for log in logs))
        
    def test_event_bus_publish_subscribe(self):
        """Test EventBus publish/subscribe mechanism"""
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
        
        EventBus.publish("test_event", test_data)
        
        # Verify event was received
        self.assertEqual(len(self.test_events), 1)
        self.assertEqual(self.test_events[0], test_data)
        
    def tearDown(self):
        """Clean up test environment"""
        EventBus.unsubscribe("test_event", self._capture_test_event)
        EventBus.unsubscribe("test_start", self._capture_test_event)
        EventBus.unsubscribe("test_complete", self._capture_test_event)

class TestShnifterData(unittest.TestCase):
    """Enhanced tests for the ShnifterData data model"""

    def setUp(self):
        """Set up test data and event logging"""
        self.event_log = EventLog()
        self.test_df = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'timestamp': pd.date_range(start='2025-01-01', periods=5, freq='D')
        })

    def test_to_df_and_to_dict(self):
        """Test ShnifterData conversion to DataFrame and dictionary"""
        self.event_log.emit("INFO", "test_start", {"test": "TestShnifterData.test_to_df_and_to_dict"})
        
        try:
            data = ShnifterData(results=self.test_df, provider='test')
            
            # Test DataFrame conversion
            df_result = data.to_df()
            self.assertTrue(isinstance(df_result, pd.DataFrame))
            self.assertEqual(len(df_result), 5)
            self.assertIn('close', df_result.columns)
            
            # Test dictionary conversion
            dict_result = data.to_dict()
            self.assertTrue(isinstance(dict_result, list))
            self.assertEqual(len(dict_result), 5)
            self.assertEqual(dict_result[0]['close'], 100)
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "TestShnifterData.test_to_df_and_to_dict",
                "status": "success"
            })
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "TestShnifterData.test_to_df_and_to_dict",
                "error": str(e)
            })
            raise

    def test_data_validation(self):
        """Test ShnifterData validation and error handling"""
        self.event_log.emit("INFO", "test_start", {"test": "TestShnifterData.test_data_validation"})
        
        try:
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            data = ShnifterData(results=empty_df, provider='test')
            self.assertTrue(data.to_df().empty)
            
            # Test with None data
            with self.assertRaises(Exception):
                ShnifterData(results=None, provider='test')
                
            self.event_log.emit("INFO", "test_complete", {
                "test": "TestShnifterData.test_data_validation",
                "status": "success"
            })
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "TestShnifterData.test_data_validation",
                "error": str(e)
            })
            raise

class TestShnifterAnalysisModules(unittest.TestCase):
    """Test the converted analysis modules"""
    
    def setUp(self):
        """Set up analysis module testing"""
        self.event_log = EventLog()
        self.test_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='H')
        })
        
    def test_portfolio_optimization_module(self):
        """Test portfolio optimization analysis module"""
        self.event_log.emit("INFO", "test_start", {"test": "portfolio_optimization_module"})
        
        try:
            # Import the module
            from shnifter_analysis_modules.portfolioOptimizationUsingModernPortfolioTheory_module import (
                ShnifterPortfoliooptimizationusingmodernportfoliotheory
            )
            
            # Create analyzer instance
            analyzer = ShnifterPortfoliooptimizationusingmodernportfoliotheory()
            
            # Run analysis
            results = analyzer.run_analysis(self.test_data)
            
            # Verify results structure
            self.assertIn('analysis_type', results)
            self.assertIn('results', results)
            self.assertIn('timestamp', results)
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "portfolio_optimization_module",
                "status": "success",
                "results_keys": list(results.keys())
            })
            
        except ImportError as e:
            self.event_log.emit("WARNING", "test_skip", {
                "test": "portfolio_optimization_module",
                "reason": f"Module not available: {e}"
            })
            self.skipTest(f"Module not available: {e}")
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "portfolio_optimization_module",
                "error": str(e)
            })
            raise

    def test_backtesting_momentum_module(self):
        """Test backtesting momentum analysis module"""
        self.event_log.emit("INFO", "test_start", {"test": "backtesting_momentum_module"})
        
        try:
            from shnifter_analysis_modules.BacktestingMomentumTrading_module import (
                ShnifterBacktestingmomentumtrading
            )
            
            analyzer = ShnifterBacktestingmomentumtrading()
            results = analyzer.run_analysis(self.test_data)
            
            self.assertIn('analysis_type', results)
            self.assertEqual(results['analysis_type'], 'Backtestingmomentumtrading')
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "backtesting_momentum_module",
                "status": "success"
            })
            
        except ImportError as e:
            self.event_log.emit("WARNING", "test_skip", {
                "test": "backtesting_momentum_module", 
                "reason": f"Module not available: {e}"
            })
            self.skipTest(f"Module not available: {e}")
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "backtesting_momentum_module",
                "error": str(e)
            })
            raise

class TestShnifterWidgets(unittest.TestCase):
    """Test the converted PySide6 widgets"""
    
    def setUp(self):
        """Set up widget testing"""
        self.event_log = EventLog()
        # Mock Qt Application for testing
        self.app = None
        try:
            from PySide6.QtWidgets import QApplication
            if not QApplication.instance():
                self.app = QApplication([])
        except ImportError:
            self.skipTest("PySide6 not available")
            
    def test_chart_widget_creation(self):
        """Test chart widget creation and basic functionality"""
        self.event_log.emit("INFO", "test_start", {"test": "chart_widget_creation"})
        
        try:
            from shnifter_frontend.shnifter_plotly_widgets.chart_widget import ShnifterChartWidget
            
            # Create widget
            widget = ShnifterChartWidget()
            
            # Test basic properties
            self.assertIsNotNone(widget.widget_id)
            self.assertEqual(widget.component_type, "plotly")
            self.assertTrue(hasattr(widget, 'data_cache'))
            
            # Test data update
            test_data = {"test": "data", "timestamp": datetime.now().isoformat()}
            widget.update_data(test_data)
            self.assertEqual(widget.data_cache['test'], "data")
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "chart_widget_creation",
                "status": "success",
                "widget_id": widget.widget_id
            })
            
        except ImportError as e:
            self.event_log.emit("WARNING", "test_skip", {
                "test": "chart_widget_creation",
                "reason": f"Widget not available: {e}"
            })
            self.skipTest(f"Widget not available: {e}")
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "chart_widget_creation",
                "error": str(e)
            })
            raise
            
    def test_table_widget_functionality(self):
        """Test table widget functionality"""
        self.event_log.emit("INFO", "test_start", {"test": "table_widget_functionality"})
        
        try:
            from shnifter_frontend.shnifter_table_widget import ShnifterTableWidget
            
            widget = ShnifterTableWidget()
            
            # Test data loading
            test_df = pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'Price': [150.0, 300.0, 2500.0],
                'Change': [1.5, -2.0, 10.0]
            })
            
            widget.load_data(test_df)
            
            # Verify data was loaded
            self.assertEqual(len(widget.table_data), 3)
            self.assertIn('Symbol', widget.table_data.columns)
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "table_widget_functionality",
                "status": "success",
                "data_rows": len(widget.table_data)
            })
            
        except ImportError as e:
            self.event_log.emit("WARNING", "test_skip", {
                "test": "table_widget_functionality",
                "reason": f"Widget not available: {e}"
            })
            self.skipTest(f"Widget not available: {e}")
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "table_widget_functionality",
                "error": str(e)
            })
            raise
            
    def tearDown(self):
        """Clean up widget testing"""
        if self.app:
            self.app.quit()

class TestShnifterIntegration(unittest.TestCase):
    """Integration tests for the complete Shnifter system"""
    
    def setUp(self):
        """Set up integration testing"""
        self.event_log = EventLog()
        
    def test_popout_registry_integration(self):
        """Test popout registry integration"""
        self.event_log.emit("INFO", "test_start", {"test": "popout_registry_integration"})
        
        try:
            from shnifter_frontend.shnifter_popout_registry import shnifter_popout_registry
            
            # Test registry exists and has widgets
            self.assertIsNotNone(shnifter_popout_registry)
            self.assertTrue(hasattr(shnifter_popout_registry, 'registered_widgets'))
            
            # Test registry methods
            widgets = shnifter_popout_registry.get_registered_widgets()
            self.assertTrue(isinstance(widgets, dict))
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "popout_registry_integration",
                "status": "success",
                "registered_widgets_count": len(widgets)
            })
            
        except ImportError as e:
            self.event_log.emit("WARNING", "test_skip", {
                "test": "popout_registry_integration",
                "reason": f"Registry not available: {e}"
            })
            self.skipTest(f"Registry not available: {e}")
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "popout_registry_integration",
                "error": str(e)
            })
            raise
            
    def test_event_system_end_to_end(self):
        """Test complete event system integration"""
        self.event_log.emit("INFO", "test_start", {"test": "event_system_end_to_end"})
        
        try:
            # Set up event capturing
            captured_events = []
            
            def capture_event(event_data):
                captured_events.append(event_data)
                
            EventBus.subscribe("integration_test", capture_event)
            
            # Publish test event
            test_event_data = {
                "test_type": "integration",
                "timestamp": datetime.now().isoformat(),
                "data": {"key": "value"}
            }
            
            EventBus.publish("integration_test", test_event_data)
            
            # Verify event was captured
            self.assertEqual(len(captured_events), 1)
            self.assertEqual(captured_events[0], test_event_data)
            
            # Clean up
            EventBus.unsubscribe("integration_test", capture_event)
            
            self.event_log.emit("INFO", "test_complete", {
                "test": "event_system_end_to_end",
                "status": "success",
                "events_captured": len(captured_events)
            })
            
        except Exception as e:
            self.event_log.emit("ERROR", "test_error", {
                "test": "event_system_end_to_end",
                "error": str(e)
            })
            raise

# Test suite runner
def run_enhanced_tests():
    """Run all enhanced tests with detailed event logging"""
    print("🧪 Running Enhanced Shnifter Test Suite...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestEventSystemIntegration,
        TestShnifterData,
        TestShnifterAnalysisModules,
        TestShnifterWidgets,
        TestShnifterIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY:")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            
    if result.errors:
        print("\n🚨 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_enhanced_tests()
    sys.exit(0 if success else 1)
