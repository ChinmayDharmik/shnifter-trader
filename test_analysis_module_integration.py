"""
Test Analysis Module Integration with Event System
Tests the integration between analysis modules and the event wiring system
"""

import unittest
import sys
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.events import EventBus, EventLog
from shnifter_event_wiring import ShnifterEventWiring

class TestAnalysisModuleIntegration(unittest.TestCase):
    """Test integration between analysis modules and event system"""
    
    def setUp(self):
        """Set up test environment"""
        # Clear any existing subscribers
        EventBus.clear_subscribers()
        EventLog.clear_logs()
        
        # Initialize event wiring
        self.event_wiring = ShnifterEventWiring()
        self.test_events = []
        
        # Subscribe to test events
        EventBus.subscribe("analysis_module_test", self._capture_test_event)
        
    def _capture_test_event(self, event_data):
        """Capture events for testing"""
        self.test_events.append(event_data)
        
    def test_event_wiring_initialization(self):
        """Test that event wiring initializes correctly"""
        self.event_wiring.wire_all_components()
        
        # Check that components were wired
        self.assertGreater(len(self.event_wiring.wired_components), 50)
        
        # Check that specific analysis module events are wired
        analysis_events = [comp for comp in self.event_wiring.wired_components 
                          if comp.startswith('analysis_modules.')]
        self.assertGreater(len(analysis_events), 30)  # Should have many analysis module events
        
    def test_platform_standardization_events(self):
        """Test platform standardization module events"""
        self.event_wiring.wire_all_components()
        
        # Test platform standardization request
        test_data = {"symbols": ["AAPL", "MSFT"], "timeframe": "1year"}
        EventBus.publish("platform_standardization_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'platform_standardization_request' 
                           for log in logs))
        
    def test_portfolio_optimization_events(self):
        """Test portfolio optimization module events"""
        self.event_wiring.wire_all_components()
        
        # Test portfolio optimization request
        test_data = {"risk_tolerance": "moderate", "capital": 10000}
        EventBus.publish("portfolio_optimization_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'portfolio_optimization_request' 
                           for log in logs))
        
    def test_autogen_agents_events(self):
        """Test AutoGen agents module events"""
        self.event_wiring.wire_all_components()
        
        # Test AutoGen agents request
        test_data = {"trading_strategy": "momentum", "symbols": ["TSLA"]}
        EventBus.publish("autogen_agents_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'autogen_agents_request' 
                           for log in logs))
        
    def test_backtesting_momentum_events(self):
        """Test backtesting momentum module events"""
        self.event_wiring.wire_all_components()
        
        # Test backtesting momentum request
        test_data = {"timeframe": "1year", "symbols": ["AAPL"], "strategy": "SMA_crossover"}
        EventBus.publish("backtesting_momentum_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'backtesting_momentum_request' 
                           for log in logs))
        
    def test_financial_statements_events(self):
        """Test financial statements module events"""
        self.event_wiring.wire_all_components()
        
        # Test financial statements request
        test_data = {"symbol": "AAPL", "period": "quarterly", "years": 3}
        EventBus.publish("financial_statements_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'financial_statements_request' 
                           for log in logs))
        
    def test_ethereum_analysis_events(self):
        """Test Ethereum analysis module events"""
        self.event_wiring.wire_all_components()
        
        # Test Ethereum analysis request
        test_data = {"timeframe": "1month", "indicators": ["RSI", "MACD"]}
        EventBus.publish("ethereum_analysis_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'ethereum_analysis_request' 
                           for log in logs))
        
    def test_event_routing(self):
        """Test event routing between analysis modules"""
        self.event_wiring.wire_all_components()
        
        # Test that analysis_complete events route to other events
        test_data = {"analysis_type": "momentum", "signal": "BUY", "confidence": 0.8}
        EventBus.publish("analysis_complete", test_data)
        
        # Check that the event was logged and routed
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'analysis_complete' 
                           for log in logs))
        
    def test_analysis_module_workflow(self):
        """Test complete analysis module workflow"""
        self.event_wiring.wire_all_components()
        
        # Simulate complete analysis workflow
        workflow_events = [
            ("analysis_start", {"module": "portfolio_optimization", "timestamp": datetime.now().isoformat()}),
            ("portfolio_optimization_request", {"symbols": ["AAPL", "GOOGL"], "capital": 50000}),
            ("portfolio_optimization_complete", {"allocation": {"AAPL": 0.6, "GOOGL": 0.4}, "expected_return": 0.12}),
            ("analysis_complete", {"module": "portfolio_optimization", "success": True})
        ]
        
        for event_type, event_data in workflow_events:
            EventBus.publish(event_type, event_data)
        
        # Check that all events were processed
        logs = EventLog.get_recent_logs()
        for event_type, _ in workflow_events:
            self.assertTrue(any(log.get('message') == event_type for log in logs),
                           f"Event {event_type} was not logged")
        
    def test_error_handling(self):
        """Test error handling in analysis modules"""
        self.event_wiring.wire_all_components()
        
        # Test analysis error event
        error_data = {
            "module": "risk_return_analysis", 
            "error": "Insufficient data", 
            "timestamp": datetime.now().isoformat()
        }
        EventBus.publish("analysis_error", error_data)
        
        # Check that error was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'analysis_error' and log.get('level') == 'ERROR' 
                           for log in logs))
        
    def test_n8n_workflow_integration(self):
        """Test N8N workflow adapter events"""
        self.event_wiring.wire_all_components()
        
        # Test N8N workflow request
        test_data = {
            "workflow_id": "trading_automation_v1",
            "trigger": "market_open",
            "parameters": {"symbols": ["SPY", "QQQ"]}
        }
        EventBus.publish("n8n_workflow_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'n8n_workflow_request' 
                           for log in logs))
        
    def test_chroma_knowledge_base_events(self):
        """Test Chroma knowledge base events"""
        self.event_wiring.wire_all_components()
        
        # Test Chroma knowledge request
        test_data = {
            "query": "latest market trends for tech stocks",
            "collection": "financial_news",
            "limit": 10
        }
        EventBus.publish("chroma_knowledge_request", test_data)
        
        # Check that event was logged
        logs = EventLog.get_recent_logs()
        self.assertTrue(any(log.get('message') == 'chroma_knowledge_request' 
                           for log in logs))
        
    def tearDown(self):
        """Clean up test environment"""
        EventBus.unsubscribe("analysis_module_test", self._capture_test_event)
        EventBus.clear_subscribers()

class TestAnalysisModuleExecution(unittest.TestCase):
    """Test actual execution of analysis modules"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_data = {
            "AAPL": [100, 101, 102, 98, 99, 103, 105],
            "GOOGL": [2500, 2520, 2490, 2510, 2530, 2540, 2560]
        }
        
    def test_portfolio_optimization_module(self):
        """Test portfolio optimization module execution"""
        try:
            from shnifter_analysis_modules.portfolioOptimizationUsingModernPortfolioTheory_module import create_portfolioOptimizationUsingModernPortfolioTheory_analyzer
            
            analyzer = create_portfolioOptimizationUsingModernPortfolioTheory_analyzer()
            result = analyzer.run_analysis(self.test_data)
            
            self.assertIn("analysis_type", result)
            self.assertIn("results", result)
            self.assertIn("timestamp", result)
            
        except ImportError:
            self.skipTest("Portfolio optimization module not available")
            
    def test_backtesting_momentum_module(self):
        """Test backtesting momentum module execution"""
        try:
            from shnifter_analysis_modules.BacktestingMomentumTrading_module import create_BacktestingMomentumTrading_analyzer
            
            analyzer = create_BacktestingMomentumTrading_analyzer()
            result = analyzer.run_analysis(self.test_data)
            
            self.assertIn("analysis_type", result)
            self.assertIn("results", result)
            self.assertIn("timestamp", result)
            
        except ImportError:
            self.skipTest("Backtesting momentum module not available")
            
    def test_risk_return_analysis_module(self):
        """Test risk return analysis module execution"""
        try:
            from shnifter_analysis_modules.riskReturnAnalysis_module import create_riskReturnAnalysis_analyzer
            
            analyzer = create_riskReturnAnalysis_analyzer()
            result = analyzer.run_analysis(self.test_data)
            
            self.assertIn("analysis_type", result)
            self.assertIn("results", result)
            self.assertIn("timestamp", result)
            
        except ImportError:
            self.skipTest("Risk return analysis module not available")

if __name__ == '__main__':
    unittest.main()
