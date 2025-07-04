import unittest
from core import shnifter_risk, shnifter_paper
from core.shnifter_trade import Portfolio, Trade

class TestShnifterRisk(unittest.TestCase):
    def test_calculate_position_size(self):
        size = shnifter_risk.calculate_position_size(confidence=0.8, balance=10000)
        self.assertTrue(size >= 0)
        size_zero = shnifter_risk.calculate_position_size(confidence=0, balance=10000)
        self.assertEqual(size_zero, 0)

    def test_get_portfolio_heat(self):
        portfolio = Portfolio(10000)
        trade = Trade('AAPL', entry=100, size=10, stop=90, take_profit=120)
        portfolio.add_trade(trade)
        price_lookup = {'AAPL': 110}
        heat = shnifter_risk.get_portfolio_heat(portfolio, price_lookup)
        self.assertTrue(heat > 0)

    def test_check_drawdown(self):
        portfolio = Portfolio(10000)
        trade = Trade('AAPL', entry=100, size=10, stop=90, take_profit=120)
        portfolio.add_trade(trade)
        # Simulate a loss to trigger drawdown
        portfolio.close_trade(trade, exit_price=80)
        result = shnifter_risk.check_drawdown(portfolio, max_drawdown=0.1)
        self.assertIsInstance(result, bool)

class TestPaperTradingEngine(unittest.TestCase):
    def test_place_and_close_trade(self):
        engine = shnifter_paper.PaperTradingEngine(10000)
        trade = engine.place_trade('AAPL', entry=100, size=10, stop=90, take_profit=120, direction='long', confidence=0.8)
        self.assertEqual(len(engine.portfolio.trades), 1)
        engine.close_trade(trade, exit_price=110)
        self.assertEqual(len(engine.portfolio.trades), 0)

    def test_update_trades_and_stats(self):
        engine = shnifter_paper.PaperTradingEngine(10000)
        trade = engine.place_trade('AAPL', entry=100, size=10, stop=90, take_profit=120, direction='long', confidence=0.8)
        price_lookup = {'AAPL': 110}
        engine.update_trades(price_lookup)
        stats = engine.get_stats(price_lookup)
        self.assertIn('unrealized', stats)
        self.assertIn('realized', stats)
        self.assertIn('wins', stats)
        self.assertIn('losses', stats)
        self.assertIn('drawdown', stats)
        self.assertIn('balance', stats)

if __name__ == '__main__':
    unittest.main()
