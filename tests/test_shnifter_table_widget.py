"""
Shnifter Table Widget Tests
Testing the LLM-integrated table functionality
"""
import unittest
import pandas as pd
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shnifter_frontend.shnifter_table_widget import ShnifterTableWidget, LLMTableAnalyzer
from core.events import EventLog

class TestShnifterTableWidget(unittest.TestCase):
    """Test cases for ShnifterTableWidget"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'ticker': ['AAPL', 'GOOGL', 'MSFT'],
            'price': [150.0, 2500.0, 300.0],
            'volume': [1000000, 500000, 750000],
            'change': [2.5, -1.2, 0.8]
        })
        
    def test_data_loading(self):
        """Test loading different data formats"""
        widget = ShnifterTableWidget()
        
        # Test DataFrame loading
        widget.load_data(self.sample_data)
        self.assertFalse(widget.table_data.empty)
        self.assertEqual(len(widget.table_data), 3)
        
        # Test dict loading
        dict_data = self.sample_data.to_dict('records')
        widget.load_data(dict_data)
        self.assertEqual(len(widget.table_data), 3)
        
    def test_filter_functionality(self):
        """Test table filtering"""
        widget = ShnifterTableWidget()
        widget.load_data(self.sample_data)
        
        # Test filtering
        widget._apply_filter("AAPL")
        # Check that filter was applied (would need UI testing for full verification)
        
    @patch('shnifter_frontend.shnifter_table_widget.OllamaProvider')
    def test_llm_analysis(self, mock_provider):
        """Test LLM analysis functionality"""
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.generate_response.return_value = "Test analysis result"
        mock_provider.return_value = mock_llm
        
        analyzer = LLMTableAnalyzer(self.sample_data, "test_analysis", mock_llm)
        
        # Test prompt building
        prompt = analyzer._build_analysis_prompt()
        self.assertIn("financial data table", prompt.lower())
        self.assertIn("AAPL", prompt)
        
    def test_insights_extraction(self):
        """Test insight extraction from analysis"""
        analyzer = LLMTableAnalyzer(self.sample_data)
        
        test_analysis = """
        The top performing stock is AAPL with strong gains.
        Risk alert: GOOGL showing volatility concerns.
        I recommend buying AAPL for portfolio growth.
        """
        
        insights = analyzer._extract_insights(test_analysis)
        
        self.assertTrue(len(insights['top_performers']) > 0)
        self.assertTrue(len(insights['risk_alerts']) > 0)
        self.assertTrue(len(insights['recommendations']) > 0)
        
if __name__ == '__main__':
    unittest.main()
