from core.shnifter_trade import Trade, Portfolio

# Example: Kelly, fixed fractional, or LLM confidence-based sizing

def calculate_position_size(confidence, balance, risk_per_trade=0.01, kelly_fraction=1.0):
    """
    Returns position size based on confidence (0-1), account balance, and risk per trade.
    If confidence is None, use fixed fractional sizing.
    """
    if confidence is not None:
        # Kelly formula: f* = (bp - q)/b, simplified for binary outcome
        # Here, confidence = probability of win, b = reward/risk (assume 1)
        kelly = (confidence - (1 - confidence)) * kelly_fraction
        kelly = max(0, min(kelly, 1))  # Clamp between 0 and 1
        size = balance * risk_per_trade * kelly
    else:
        size = balance * risk_per_trade
    return max(size, 0)

# Example: Portfolio heat and drawdown checks

def get_portfolio_heat(portfolio, price_lookup):
    """
    Returns total risk exposure as a fraction of balance.
    """
    total_risk = 0
    for trade in portfolio.get_open_trades():
        # Risk per trade = abs(entry - stop) * size
        risk = abs(trade.entry - trade.stop) * trade.size
        total_risk += risk
    return total_risk / portfolio.balance if portfolio.balance > 0 else 0

def check_drawdown(portfolio, max_drawdown=0.2):
    """
    Returns True if drawdown exceeds max_drawdown (e.g., 20%).
    """
    return portfolio.max_drawdown >= max_drawdown
