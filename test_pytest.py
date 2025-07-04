"""
Enhanced Pytest runner for Shnifter Trader
Integrates with event system and provides comprehensive test coverage
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Import core modules for testing
from core.events import EventLog, EventBus
from core.data_models import ShnifterData

# Initialize event logging for tests
event_log = EventLog()

class TestShnifterPytest:
    """Enhanced pytest test class with event system integration"""
    
    def setup_method(self):
        """Set up each test method with event logging"""
        self.event_log = EventLog()
        self.test_events = []
        
        # Subscribe to test events
        EventBus.subscribe("pytest_test", self._capture_event)
        
        self.event_log.emit("INFO", "test_setup", {
            "test_framework": "pytest",
            "timestamp": datetime.now().isoformat()
        })
        
    def _capture_event(self, event_data):
        """Capture events for testing"""
        self.test_events.append(event_data)
        
    def test_event_system_integration(self):
        """Test event system integration in pytest"""
        # Publish test event
        test_data = {"test": "pytest_integration", "value": 42}
        EventBus.publish("pytest_test", test_data)
        
        # Verify event was captured
        assert len(self.test_events) == 1
        assert self.test_events[0] == test_data
        
        self.event_log.emit("INFO", "test_complete", {
            "test": "event_system_integration",
            "events_captured": len(self.test_events)
        })
        
    def test_shnifter_data_creation(self):
        """Test ShnifterData creation and manipulation"""
        import pandas as pd
        
        # Create test data
        df = pd.DataFrame({
            'price': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        # Create ShnifterData object
        data = ShnifterData(results=df, provider='pytest')
        
        # Test basic properties
        assert data.provider == 'pytest'
        assert len(data.to_df()) == 3
        assert 'price' in data.to_df().columns
        
        self.event_log.emit("INFO", "test_complete", {
            "test": "shnifter_data_creation",
            "data_rows": len(data.to_df())
        })
        
    def test_widget_imports(self):
        """Test that widgets can be imported without errors"""
        widget_imports = []
        
        try:
            from shnifter_frontend.shnifter_plotly_widgets.chart_widget import ShnifterChartWidget
            widget_imports.append("ShnifterChartWidget")
        except ImportError as e:
            self.event_log.emit("WARNING", "import_skip", {
                "widget": "ShnifterChartWidget",
                "reason": str(e)
            })
            
        try:
            from shnifter_frontend.shnifter_table_widget import ShnifterTableWidget
            widget_imports.append("ShnifterTableWidget")
        except ImportError as e:
            self.event_log.emit("WARNING", "import_skip", {
                "widget": "ShnifterTableWidget", 
                "reason": str(e)
            })
            
        self.event_log.emit("INFO", "test_complete", {
            "test": "widget_imports",
            "widgets_imported": widget_imports
        })
        
        # At least one widget should be importable
        assert len(widget_imports) > 0, "No widgets could be imported"
        
    def teardown_method(self):
        """Clean up after each test method"""
        EventBus.unsubscribe("pytest_test", self._capture_event)

def run_pytest_html():
    """Run all tests and output HTML report with enhanced configuration."""
    print("[INFO] Running enhanced pytest suite with HTML report...")
    
    pytest_args = [
        "-v",                                    # Verbose output
        "--html=shnifter_test_report.html",     # HTML report
        "--self-contained-html",                 # Self-contained report
        "--tb=short",                           # Short traceback format
        "--strict-markers",                     # Strict marker handling
        "--maxfail=5",                          # Stop after 5 failures
        "--durations=10",                       # Show 10 slowest tests
    ]
    
    # Add this file and discover others
    pytest_args.extend([
        ".",                                    # Discover tests in current directory
        __file__                               # Include this file
    ])
    
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    print("🎯 PYTEST ENHANCED SUMMARY:")
    print("=" * 60)
    print(f"Report generated: shnifter_test_report.html")
    print(f"Exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_pytest_html()
    sys.exit(exit_code)



