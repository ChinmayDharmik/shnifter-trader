import unittest
from shnifter_frontend.pnl_dashboard_popout import PnLDashboardPopout
from PySide6.QtWidgets import QApplication
import sys

def dummy_stats():
    return {
        'unrealized': 100,
        'realized': 50,
        'wins': 2,
        'losses': 1,
        'drawdown': 0.05,
        'balance': 10100
    }

class TestPnLDashboardPopout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.popout = PnLDashboardPopout(get_stats_callback=dummy_stats)

    def test_update_stats(self):
        self.popout.update_stats()
        # Check that the label text contains expected values
        text = self.popout.stats_label.text()
        self.assertIn('Unrealized', text)
        self.assertIn('Realized', text)
        self.assertIn('Wins', text)
        self.assertIn('Losses', text)
        self.assertIn('Drawdown', text)
        self.assertIn('Balance', text)

if __name__ == '__main__':
    unittest.main()
