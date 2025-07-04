"""
Unit tests for Shnifter core modules: trade, risk, paper trading, config, credentials, and data models.
Covers logic not tested in other files.
"""
import unittest
import tempfile
import os
import time
from core.shnifter_trade import Trade, Portfolio
from core.shnifter_risk import calculate_position_size, get_portfolio_heat, check_drawdown
from core.shnifter_paper import PaperTradingEngine
from core.data_models import ShnifterNewsData
from core.credentials import CredentialsLoader
from core.config import ShnifterConfig

class TestTradeAndPortfolio(unittest.TestCase):
    def test_trade_pnl_and_exit(self):
        trade = Trade('AAPL', 100, 10, 90, 120)
        trade.direction = 'long'
        pnl = trade.update_pnl(110)
        self.assertEqual(pnl, 100)
        self.assertFalse(trade.check_exit(105))
        self.assertTrue(trade.check_exit(90))
        self.assertTrue(trade.check_exit(120))

    def test_portfolio_add_close_win_loss(self):
        p = Portfolio(1000)
        t1 = Trade('AAPL', 100, 1, 90, 120)
        t2 = Trade('MSFT', 200, 1, 180, 220)
        p.add_trade(t1)
        p.add_trade(t2)
        self.assertEqual(len(p.get_open_trades()), 2)
        p.close_trade(t1, 110)
        p.close_trade(t2, 190)
        wins, losses = p.get_win_loss()
        self.assertEqual(wins, 1)
        self.assertEqual(losses, 1)
        self.assertEqual(len(p.get_open_trades()), 0)
        self.assertEqual(len(p.closed_trades), 2)

class TestRiskManagement(unittest.TestCase):
    def test_calculate_position_size(self):
        size = calculate_position_size(0.7, 10000)
        self.assertTrue(size > 0)
        size2 = calculate_position_size(None, 10000)
        self.assertTrue(size2 > 0)

    def test_portfolio_heat_and_drawdown(self):
        p = Portfolio(10000)
        t = Trade('AAPL', 100, 10, 90, 120)
        p.add_trade(t)
        price_lookup = {'AAPL': 105}
        heat = get_portfolio_heat(p, price_lookup)
        self.assertTrue(heat > 0)
        p.balance = 7000
        p.update_drawdown()
        self.assertTrue(check_drawdown(p, max_drawdown=0.2))

class TestPaperTradingEngine(unittest.TestCase):
    def test_place_and_close_trade(self):
        engine = PaperTradingEngine(10000)
        trade = engine.place_trade('AAPL', 100, 1, 90, 120)
        self.assertEqual(len(engine.portfolio.trades), 1)
        price_lookup = {'AAPL': 120}
        engine.update_trades(price_lookup)
        self.assertEqual(len(engine.portfolio.trades), 0)
        self.assertEqual(len(engine.portfolio.closed_trades), 1)
        stats = engine.get_stats(price_lookup)
        self.assertIn('unrealized', stats)
        self.assertIn('realized', stats)

class TestShnifterNewsData(unittest.TestCase):
    def test_news_model(self):
        from datetime import datetime
        news = ShnifterNewsData(
            date=datetime.now(),
            title="Test News",
            text="Body",
            url="http://example.com",
            provider="test",
            symbols=["AAPL"]
        )
        self.assertEqual(news.title, "Test News")
        self.assertIn("AAPL", news.symbols)

class TestCredentialsLoader(unittest.TestCase):
    def test_save_and_load_credentials(self):
        creds = {"api_key": "123", "secret": "abc"}
        # Patch USER_SETTINGS_PATH to a temp file
        from core import constants
        orig_path = constants.USER_SETTINGS_PATH
        with tempfile.TemporaryDirectory() as tmpdir:
            constants.USER_SETTINGS_PATH = os.path.join(tmpdir, "user_settings.json")
            CredentialsLoader.save_credentials(creds)
            loaded = CredentialsLoader.load_credentials()
            self.assertEqual(loaded, creds)
        constants.USER_SETTINGS_PATH = orig_path

class TestShnifterConfig(unittest.TestCase):
    def test_config_get_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            config = ShnifterConfig(config_path)
            config.set("trading.default_ticker", "AAPL")
            self.assertEqual(config.get("trading.default_ticker"), "AAPL")
            config.set("ui.theme", "light")
            self.assertEqual(config.get("ui.theme"), "light")
            config.save_config()
            config2 = ShnifterConfig(config_path)
            self.assertEqual(config2.get("ui.theme"), "light")

if __name__ == "__main__":
    unittest.main(verbosity=2)
