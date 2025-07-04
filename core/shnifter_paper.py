from core.shnifter_trade import Trade, Portfolio
import time

class PaperTradingEngine:
    """
    Simulates order fills, manages trade log, and updates P&L for paper trading mode.
    """
    def __init__(self, starting_balance=100000):
        self.portfolio = Portfolio(starting_balance)
        self.paper_mode = True
        self.trade_log = []  # List of dicts for GUI display

    def place_trade(self, ticker, entry, size, stop, take_profit, direction='long', confidence=None):
        trade = Trade(ticker, entry, size, stop, take_profit)
        trade.direction = direction
        trade.open_time = time.time()
        trade.confidence = confidence
        self.portfolio.add_trade(trade)
        self.trade_log.append({
            'ticker': ticker,
            'entry': entry,
            'size': size,
            'stop': stop,
            'take_profit': take_profit,
            'direction': direction,
            'status': 'open',
            'open_time': trade.open_time,
            'confidence': confidence
        })
        return trade

    def update_trades(self, price_lookup):
        # price_lookup: dict of {ticker: current_price}
        for trade in self.portfolio.get_open_trades():
            current_price = price_lookup.get(trade.ticker, trade.entry)
            trade.update_pnl(current_price)
            if trade.check_exit(current_price):
                self.close_trade(trade, current_price)

    def close_trade(self, trade, exit_price):
        trade.close_time = time.time()
        self.portfolio.close_trade(trade, exit_price, close_time=trade.close_time)
        for log in self.trade_log:
            if log['ticker'] == trade.ticker and log['status'] == 'open':
                log['status'] = 'closed'
                log['exit'] = exit_price
                log['close_time'] = trade.close_time
                log['pnl'] = trade.pnl

    def get_stats(self, price_lookup):
        unrealized = self.portfolio.get_total_unrealized_pnl(price_lookup)
        realized = self.portfolio.get_total_realized_pnl()
        wins, losses = self.portfolio.get_win_loss()
        drawdown = self.portfolio.max_drawdown
        balance = self.portfolio.balance
        return {
            'unrealized': unrealized,
            'realized': realized,
            'wins': wins,
            'losses': losses,
            'drawdown': drawdown,
            'balance': balance
        }
