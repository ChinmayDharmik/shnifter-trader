class Trade:
    """
    Represents a single trade with all relevant risk and P&L fields.
    """
    def __init__(self, ticker, entry, size, stop, take_profit, status='open'):
        self.ticker = ticker
        self.entry = entry
        self.size = size
        self.stop = stop
        self.take_profit = take_profit
        self.status = status  # 'open' or 'closed'
        self.exit = None
        self.pnl = 0
        self.direction = 'long'  # or 'short', can be extended
        self.open_time = None
        self.close_time = None
        self.confidence = None  # LLM/ML confidence score

    def update_pnl(self, current_price):
        if self.direction == 'long':
            self.pnl = (current_price - self.entry) * self.size
        else:
            self.pnl = (self.entry - current_price) * self.size
        return self.pnl

    def check_exit(self, current_price):
        if self.direction == 'long':
            if current_price <= self.stop or current_price >= self.take_profit:
                return True
        else:
            if current_price >= self.stop or current_price <= self.take_profit:
                return True
        return False

class Portfolio:
    """
    Tracks all trades, account balance, drawdown, and risk metrics.
    """
    def __init__(self, balance):
        self.balance = balance
        self.trades = []  # List of Trade objects
        self.max_drawdown = 0
        self.peak_balance = balance
        self.closed_trades = []

    def add_trade(self, trade):
        self.trades.append(trade)

    def close_trade(self, trade, exit_price, close_time=None):
        trade.exit = exit_price
        trade.close_time = close_time
        trade.status = 'closed'
        trade.update_pnl(exit_price)
        self.balance += trade.pnl
        self.closed_trades.append(trade)
        self.trades.remove(trade)
        self.update_drawdown()

    def update_drawdown(self):
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
        self.max_drawdown = max(self.max_drawdown, drawdown)

    def get_open_trades(self):
        return [t for t in self.trades if t.status == 'open']

    def get_total_unrealized_pnl(self, price_lookup):
        return sum(t.update_pnl(price_lookup.get(t.ticker, t.entry)) for t in self.get_open_trades())

    def get_total_realized_pnl(self):
        return sum(t.pnl for t in self.closed_trades)

    def get_win_loss(self):
        wins = sum(1 for t in self.closed_trades if t.pnl > 0)
        losses = sum(1 for t in self.closed_trades if t.pnl <= 0)
        return wins, losses
