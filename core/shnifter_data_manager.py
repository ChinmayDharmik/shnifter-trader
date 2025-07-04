"""
Shnifter Data Manager - Centralized Backend for Real Trading Data

This module provides a unified data management system that connects
backend trading logic to frontend components. It manages:
- Real-time trading statistics and P&L
- Live market data and model performance
- Trade history and portfolio management
- Event streaming and notifications
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

from core.shnifter_trade import Trade, Portfolio
from core.events import EventLog
from shnifterBB.shnifter_bb import ShnifterBB


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    last_prediction: str
    confidence: float
    total_predictions: int
    correct_predictions: int
    last_updated: datetime


@dataclass
class TradingStats:
    """Current trading statistics"""
    total_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    best_trade: float
    worst_trade: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    active_positions: int
    last_updated: datetime


@dataclass
class MarketData:
    """Live market data"""
    ticker: str
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    bid: float
    ask: float
    high_52w: float
    low_52w: float
    market_cap: float
    pe_ratio: float
    last_updated: datetime


class ShnifterDataManager:
    """
    Centralized data manager that provides real-time data to frontend components.
    Manages trading stats, model performance, market data, and notifications.
    """
    
    def __init__(self, db_path: str = "data/shnifter_trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Data stores
        self.portfolio = Portfolio(balance=100000.0)  # Start with $100k
        self.trading_stats = TradingStats(
            total_pnl=0.0, unrealized_pnl=0.0, realized_pnl=0.0,
            win_rate=0.0, total_trades=0, winning_trades=0, losing_trades=0,
            best_trade=0.0, worst_trade=0.0, avg_win=0.0, avg_loss=0.0,
            max_drawdown=0.0, current_drawdown=0.0, sharpe_ratio=0.0,
            profit_factor=0.0, active_positions=0, last_updated=datetime.now()
        )
        
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.trade_history: List[Trade] = []
        
        # Callbacks for real-time updates
        self.stats_callbacks: List[Callable] = []
        self.model_callbacks: List[Callable] = []
        self.trade_callbacks: List[Callable] = []
        
        # ShnifterBB for market data
        self.shnifter = ShnifterBB()
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        self._load_data()
        
        EventLog.emit("INFO", "ShnifterDataManager initialized")

    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    size REAL NOT NULL,
                    direction TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    status TEXT NOT NULL,
                    pnl REAL DEFAULT 0,
                    confidence REAL,
                    open_time TIMESTAMP,
                    close_time TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    accuracy REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    last_prediction TEXT,
                    confidence REAL,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_pnl REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    best_trade REAL NOT NULL,
                    worst_trade REAL NOT NULL,
                    avg_win REAL NOT NULL,
                    avg_loss REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    current_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    profit_factor REAL NOT NULL,
                    active_positions INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_data(self):
        """Load existing data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Load recent trading stats
                stats_df = pd.read_sql_query(
                    "SELECT * FROM trading_stats ORDER BY timestamp DESC LIMIT 1",
                    conn
                )
                if not stats_df.empty:
                    row = stats_df.iloc[0]
                    self.trading_stats = TradingStats(
                        total_pnl=row['total_pnl'],
                        unrealized_pnl=row['unrealized_pnl'],
                        realized_pnl=row['realized_pnl'],
                        win_rate=row['win_rate'],
                        total_trades=row['total_trades'],
                        winning_trades=row['winning_trades'],
                        losing_trades=row['losing_trades'],
                        best_trade=row['best_trade'],
                        worst_trade=row['worst_trade'],
                        avg_win=row['avg_win'],
                        avg_loss=row['avg_loss'],
                        max_drawdown=row['max_drawdown'],
                        current_drawdown=row['current_drawdown'],
                        sharpe_ratio=row['sharpe_ratio'],
                        profit_factor=row['profit_factor'],
                        active_positions=row['active_positions'],
                        last_updated=datetime.now()
                    )
                
                # Load model performance
                model_df = pd.read_sql_query("SELECT * FROM model_performance", conn)
                for _, row in model_df.iterrows():
                    self.model_performance[row['model_name']] = ModelPerformance(
                        model_name=row['model_name'],
                        accuracy=row['accuracy'],
                        precision=row['precision_score'],
                        recall=row['recall_score'],
                        f1_score=row['f1_score'],
                        last_prediction=row['last_prediction'],
                        confidence=row['confidence'],
                        total_predictions=row['total_predictions'],
                        correct_predictions=row['correct_predictions'],
                        last_updated=datetime.now()
                    )
                
                # Load recent trades
                trades_df = pd.read_sql_query(
                    "SELECT * FROM trades ORDER BY created_at DESC LIMIT 100",
                    conn
                )
                for _, row in trades_df.iterrows():
                    trade = Trade(
                        ticker=row['ticker'],
                        entry=row['entry_price'],
                        size=row['size'],
                        stop=row['stop_loss'],
                        take_profit=row['take_profit'],
                        status=row['status']
                    )
                    trade.exit = row['exit_price']
                    trade.pnl = row['pnl']
                    trade.direction = row['direction']
                    trade.confidence = row['confidence']
                    trade.open_time = row['open_time']
                    trade.close_time = row['close_time']
                    
                    if trade.status == 'open':
                        self.portfolio.add_trade(trade)
                    else:
                        self.trade_history.append(trade)
        
        except Exception as e:
            EventLog.emit("WARNING", f"Could not load existing data: {e}")

    def register_stats_callback(self, callback: Callable):
        """Register callback for trading stats updates"""
        with self._lock:
            self.stats_callbacks.append(callback)

    def register_model_callback(self, callback: Callable):
        """Register callback for model performance updates"""
        with self._lock:
            self.model_callbacks.append(callback)

    def register_trade_callback(self, callback: Callable):
        """Register callback for trade updates"""
        with self._lock:
            self.trade_callbacks.append(callback)

    def get_trading_stats(self) -> Dict[str, Any]:
        """Get current trading statistics"""
        with self._lock:
            return asdict(self.trading_stats)

    def get_model_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get model performance metrics"""
        with self._lock:
            if model_name:
                return asdict(self.model_performance.get(model_name, {}))
            return {name: asdict(perf) for name, perf in self.model_performance.items()}

    def get_market_data(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get current market data"""
        with self._lock:
            if ticker:
                return asdict(self.market_data.get(ticker, {}))
            return {ticker: asdict(data) for ticker, data in self.market_data.items()}

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        with self._lock:
            recent_trades = (self.trade_history + self.portfolio.closed_trades)[-limit:]
            return [self._trade_to_dict(trade) for trade in recent_trades]

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions"""
        with self._lock:
            return [self._trade_to_dict(trade) for trade in self.portfolio.get_open_trades()]

    def add_trade(self, ticker: str, entry_price: float, size: float, 
                  direction: str = 'long', stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None, confidence: Optional[float] = None):
        """Add a new trade"""
        with self._lock:
            trade = Trade(
                ticker=ticker,
                entry=entry_price,
                size=size,
                stop=stop_loss,
                take_profit=take_profit
            )
            trade.direction = direction
            trade.confidence = confidence
            trade.open_time = datetime.now()
            
            self.portfolio.add_trade(trade)
            self._save_trade(trade)
            self._update_stats()
            self._notify_trade_callbacks()
            
            EventLog.emit("INFO", f"Added {direction} trade: {ticker} @ {entry_price}")

    def close_trade(self, ticker: str, exit_price: float):
        """Close an open trade"""
        with self._lock:
            open_trades = [t for t in self.portfolio.get_open_trades() if t.ticker == ticker]
            if not open_trades:
                EventLog.emit("WARNING", f"No open trade found for {ticker}")
                return
            
            trade = open_trades[0]  # Close first matching trade
            self.portfolio.close_trade(trade, exit_price, datetime.now())
            self.trade_history.append(trade)
            self._update_trade_in_db(trade)
            self._update_stats()
            self._notify_trade_callbacks()
            
            EventLog.emit("INFO", f"Closed trade: {ticker} @ {exit_price}, P&L: {trade.pnl:.2f}")

    def update_model_performance(self, model_name: str, accuracy: float, 
                               precision: float, recall: float, f1_score: float,
                               last_prediction: str = "", confidence: float = 0.0):
        """Update model performance metrics"""
        with self._lock:
            if model_name in self.model_performance:
                perf = self.model_performance[model_name]
                perf.accuracy = accuracy
                perf.precision = precision
                perf.recall = recall
                perf.f1_score = f1_score
                perf.last_prediction = last_prediction
                perf.confidence = confidence
                perf.total_predictions += 1
                perf.last_updated = datetime.now()
            else:
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    last_prediction=last_prediction,
                    confidence=confidence,
                    total_predictions=1,
                    correct_predictions=0,
                    last_updated=datetime.now()
                )
            
            self._save_model_performance(model_name)
            self._notify_model_callbacks()

    def update_market_data(self, ticker: str):
        """Update market data for a ticker"""
        try:
            # Get current price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            
            data_obj = self.shnifter.equity.price.historical(ticker, start_date=start_date, end_date=end_date)
            df = data_obj.to_df()
            
            if not df.empty:
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                self.market_data[ticker] = MarketData(
                    ticker=ticker,
                    current_price=latest['close'],
                    price_change=latest['close'] - prev['close'],
                    price_change_pct=((latest['close'] - prev['close']) / prev['close']) * 100,
                    volume=latest.get('volume', 0),
                    bid=latest['close'] * 0.999,  # Approximate bid/ask
                    ask=latest['close'] * 1.001,
                    high_52w=df['high'].max(),
                    low_52w=df['low'].min(),
                    market_cap=0.0,  # Would need additional API call
                    pe_ratio=0.0,    # Would need additional API call
                    last_updated=datetime.now()
                )
        
        except Exception as e:
            EventLog.emit("WARNING", f"Could not update market data for {ticker}: {e}")

    def _update_stats(self):
        """Recalculate trading statistics"""
        with self._lock:
            # Calculate unrealized P&L
            current_prices = {}
            for trade in self.portfolio.get_open_trades():
                if trade.ticker not in current_prices:
                    try:
                        market_data = self.market_data.get(trade.ticker)
                        if market_data:
                            current_prices[trade.ticker] = market_data.current_price
                        else:
                            current_prices[trade.ticker] = trade.entry  # Use entry as fallback
                    except:
                        current_prices[trade.ticker] = trade.entry
            
            unrealized_pnl = self.portfolio.get_total_unrealized_pnl(current_prices)
            realized_pnl = self.portfolio.get_total_realized_pnl()
            
            wins, losses = self.portfolio.get_win_loss()
            total_trades = wins + losses
            
            # Calculate additional metrics
            winning_trades = [t for t in self.portfolio.closed_trades if t.pnl > 0]
            losing_trades = [t for t in self.portfolio.closed_trades if t.pnl <= 0]
            
            best_trade = max([t.pnl for t in self.portfolio.closed_trades], default=0.0)
            worst_trade = min([t.pnl for t in self.portfolio.closed_trades], default=0.0)
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0.0
            
            # Update stats
            self.trading_stats = TradingStats(
                total_pnl=realized_pnl + unrealized_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                win_rate=(wins / total_trades * 100) if total_trades > 0 else 0.0,
                total_trades=total_trades,
                winning_trades=wins,
                losing_trades=losses,
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_win=avg_win,
                avg_loss=avg_loss,
                max_drawdown=self.portfolio.max_drawdown * 100,
                current_drawdown=(self.portfolio.peak_balance - self.portfolio.balance) / self.portfolio.peak_balance * 100 if self.portfolio.peak_balance > 0 else 0.0,
                sharpe_ratio=self._calculate_sharpe_ratio(),
                profit_factor=abs(avg_win / avg_loss) if avg_loss != 0 else 0.0,
                active_positions=len(self.portfolio.get_open_trades()),
                last_updated=datetime.now()
            )
            
            self._save_trading_stats()
            self._notify_stats_callbacks()

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade returns"""
        if not self.portfolio.closed_trades:
            return 0.0
        
        returns = [t.pnl / abs(t.entry * t.size) for t in self.portfolio.closed_trades if t.entry * t.size != 0]
        if not returns:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        return (avg_return / std_return) if std_return != 0 else 0.0

    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """Convert Trade object to dictionary"""
        return {
            'ticker': trade.ticker,
            'entry_price': trade.entry,
            'exit_price': trade.exit,
            'size': trade.size,
            'direction': trade.direction,
            'status': trade.status,
            'pnl': trade.pnl,
            'confidence': trade.confidence,
            'open_time': trade.open_time.isoformat() if trade.open_time else None,
            'close_time': trade.close_time.isoformat() if trade.close_time else None
        }

    def _save_trade(self, trade: Trade):
        """Save trade to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trades (ticker, entry_price, size, direction, 
                                      stop_loss, take_profit, status, confidence, open_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.ticker, trade.entry, trade.size, trade.direction,
                    trade.stop, trade.take_profit, trade.status, trade.confidence,
                    trade.open_time
                ))
        except Exception as e:
            EventLog.emit("ERROR", f"Could not save trade: {e}")

    def _update_trade_in_db(self, trade: Trade):
        """Update trade in database when closed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE trades 
                    SET exit_price = ?, status = ?, pnl = ?, close_time = ?
                    WHERE ticker = ? AND status = 'open'
                """, (trade.exit, trade.status, trade.pnl, trade.close_time, trade.ticker))
        except Exception as e:
            EventLog.emit("ERROR", f"Could not update trade: {e}")

    def _save_model_performance(self, model_name: str):
        """Save model performance to database"""
        try:
            perf = self.model_performance[model_name]
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_performance 
                    (model_name, accuracy, precision_score, recall_score, f1_score,
                     last_prediction, confidence, total_predictions, correct_predictions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    perf.model_name, perf.accuracy, perf.precision, perf.recall,
                    perf.f1_score, perf.last_prediction, perf.confidence,
                    perf.total_predictions, perf.correct_predictions
                ))
        except Exception as e:
            EventLog.emit("ERROR", f"Could not save model performance: {e}")

    def _save_trading_stats(self):
        """Save trading stats to database"""
        try:
            stats = self.trading_stats
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO trading_stats 
                    (total_pnl, unrealized_pnl, realized_pnl, win_rate, total_trades,
                     winning_trades, losing_trades, best_trade, worst_trade, avg_win,
                     avg_loss, max_drawdown, current_drawdown, sharpe_ratio, 
                     profit_factor, active_positions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats.total_pnl, stats.unrealized_pnl, stats.realized_pnl,
                    stats.win_rate, stats.total_trades, stats.winning_trades,
                    stats.losing_trades, stats.best_trade, stats.worst_trade,
                    stats.avg_win, stats.avg_loss, stats.max_drawdown,
                    stats.current_drawdown, stats.sharpe_ratio, stats.profit_factor,
                    stats.active_positions
                ))
        except Exception as e:
            EventLog.emit("ERROR", f"Could not save trading stats: {e}")

    def _notify_stats_callbacks(self):
        """Notify all registered stats callbacks"""
        for callback in self.stats_callbacks:
            try:
                callback(self.get_trading_stats())
            except Exception as e:
                EventLog.emit("WARNING", f"Stats callback error: {e}")

    def _notify_model_callbacks(self):
        """Notify all registered model callbacks"""
        for callback in self.model_callbacks:
            try:
                callback(self.get_model_performance())
            except Exception as e:
                EventLog.emit("WARNING", f"Model callback error: {e}")

    def _notify_trade_callbacks(self):
        """Notify all registered trade callbacks"""
        for callback in self.trade_callbacks:
            try:
                callback(self.get_trade_history(20))
            except Exception as e:
                EventLog.emit("WARNING", f"Trade callback error: {e}")

    def simulate_demo_trades(self):
        """Generate some demo trades for testing"""
        demo_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
        for i, ticker in enumerate(demo_tickers):
            # Add some closed trades
            entry_price = 100 + i * 20
            exit_price = entry_price * (1 + np.random.uniform(-0.1, 0.15))
            
            trade = Trade(
                ticker=ticker,
                entry=entry_price,
                size=10,
                stop=entry_price * 0.95,
                take_profit=entry_price * 1.1
            )
            trade.exit = exit_price
            trade.pnl = (exit_price - entry_price) * trade.size
            trade.status = 'closed'
            trade.open_time = datetime.now() - timedelta(hours=i*2)
            trade.close_time = datetime.now() - timedelta(hours=i*2-1)
            trade.confidence = np.random.uniform(0.6, 0.9)
            
            self.portfolio.closed_trades.append(trade)
            self.trade_history.append(trade)
        
        # Add some open trades
        for ticker in ['SPY', 'QQQ']:
            self.add_trade(ticker, 400 + np.random.uniform(-10, 10), 5, 'long', 
                          confidence=np.random.uniform(0.7, 0.9))
        
        # Update model performance
        models = ['RandomForest', 'XGBoost', 'LSTM', 'Transformer']
        for model in models:
            self.update_model_performance(
                model, 
                np.random.uniform(0.6, 0.85),
                np.random.uniform(0.5, 0.8),
                np.random.uniform(0.5, 0.8),
                np.random.uniform(0.5, 0.8),
                np.random.choice(['BUY', 'SELL', 'HOLD']),
                np.random.uniform(0.6, 0.9)
            )
        
        EventLog.emit("INFO", "Demo trades and data generated")


# Global instance
data_manager = ShnifterDataManager()
