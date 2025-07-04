from PySide6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QHBoxLayout, QPushButton,
                             QGroupBox, QGridLayout, QProgressBar, QTabWidget, QWidget,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from shnifter_frontend.shnifter_plotly_widget import ShnifterPlotlyWidget
import pandas as pd
from datetime import datetime, timedelta
import random
from core.shnifter_data_manager import data_manager
from core.events import EventLog

class PnLDashboardPopout(QDialog):
    """
    Enhanced PnL Dashboard with real-time trading statistics,
    performance charts, and detailed trade history.
    """
    def __init__(self, get_stats_callback=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("💰 PnL Dashboard - Shnifter Trader")
        self.setMinimumSize(800, 600)
        self.setWindowModality(Qt.NonModal)
        self.get_stats_callback = get_stats_callback or (lambda: data_manager.get_trading_stats())
        
        # Initialize data storage
        self.trade_history = []
        self.pnl_history = []
        self.performance_metrics = {}
        
        # Register for real-time updates
        data_manager.register_stats_callback(self.on_stats_update)
        data_manager.register_trade_callback(self.on_trade_update)
        
        self.setup_ui()
        self.setup_timers()
        self.update_all_data()
        
        EventLog.emit("INFO", "PnL Dashboard connected to real data manager")

    def setup_ui(self):
        """Setup the enhanced UI with tabs and real-time charts"""
        main_layout = QVBoxLayout()
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # === OVERVIEW TAB ===
        overview_tab = QWidget()
        overview_layout = QVBoxLayout()
        
        # Real-time stats cards
        stats_group = QGroupBox("📊 Live Portfolio Statistics")
        stats_layout = QGridLayout()
        
        # Create stat cards
        self.balance_card = self.create_stat_card("💵 Balance", "$0.00", "#4CAF50")
        self.unrealized_card = self.create_stat_card("📈 Unrealized P&L", "$0.00", "#2196F3")
        self.realized_card = self.create_stat_card("💰 Realized P&L", "$0.00", "#FF9800")
        self.total_card = self.create_stat_card("🎯 Total P&L", "$0.00", "#9C27B0")
        
        stats_layout.addWidget(self.balance_card, 0, 0)
        stats_layout.addWidget(self.unrealized_card, 0, 1)
        stats_layout.addWidget(self.realized_card, 1, 0)
        stats_layout.addWidget(self.total_card, 1, 1)
        
        stats_group.setLayout(stats_layout)
        overview_layout.addWidget(stats_group)
        
        # Performance metrics
        performance_group = QGroupBox("🎪 Performance Metrics")
        perf_layout = QGridLayout()
        
        # Win/Loss metrics
        self.win_rate_label = QLabel("Win Rate:")
        self.win_rate_value = QLabel("0%")
        self.win_rate_bar = QProgressBar()
        self.win_rate_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")
        
        self.profit_factor_label = QLabel("Profit Factor:")
        self.profit_factor_value = QLabel("0.00")
        
        self.sharpe_ratio_label = QLabel("Sharpe Ratio:")
        self.sharpe_ratio_value = QLabel("0.00")
        
        self.max_drawdown_label = QLabel("Max Drawdown:")
        self.max_drawdown_value = QLabel("0%")
        self.drawdown_bar = QProgressBar()
        self.drawdown_bar.setStyleSheet("QProgressBar::chunk { background-color: #F44336; }")
        
        perf_layout.addWidget(self.win_rate_label, 0, 0)
        perf_layout.addWidget(self.win_rate_value, 0, 1)
        perf_layout.addWidget(self.win_rate_bar, 0, 2)
        perf_layout.addWidget(self.profit_factor_label, 1, 0)
        perf_layout.addWidget(self.profit_factor_value, 1, 1, 1, 2)
        perf_layout.addWidget(self.sharpe_ratio_label, 2, 0)
        perf_layout.addWidget(self.sharpe_ratio_value, 2, 1, 1, 2)
        perf_layout.addWidget(self.max_drawdown_label, 3, 0)
        perf_layout.addWidget(self.max_drawdown_value, 3, 1)
        perf_layout.addWidget(self.drawdown_bar, 3, 2)
        
        performance_group.setLayout(perf_layout)
        overview_layout.addWidget(performance_group)
        
        overview_tab.setLayout(overview_layout)
        self.tab_widget.addTab(overview_tab, "📊 Overview")
        
        # === CHARTS TAB ===
        charts_tab = QWidget()
        charts_layout = QVBoxLayout()
        
        # PnL Chart
        self.pnl_chart = ShnifterPlotlyWidget()
        charts_layout.addWidget(self.pnl_chart)
        
        charts_tab.setLayout(charts_layout)
        self.tab_widget.addTab(charts_tab, "📈 Charts")
        
        # === TRADES TAB ===
        trades_tab = QWidget()
        trades_layout = QVBoxLayout()
        
        # Trade history table
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(8)
        self.trades_table.setHorizontalHeaderLabels([
            "Time", "Symbol", "Action", "Quantity", "Price", "P&L", "Status", "Strategy"
        ])
        
        # Auto-resize columns
        header = self.trades_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        
        trades_layout.addWidget(self.trades_table)
        
        trades_tab.setLayout(trades_layout)
        self.tab_widget.addTab(trades_tab, "📋 Trades")
        
        # === RISK TAB ===
        risk_tab = QWidget()
        risk_layout = QVBoxLayout()
        
        # Risk metrics
        risk_group = QGroupBox("⚠️ Risk Management")
        risk_grid = QGridLayout()
        
        # Position sizing
        self.position_size_label = QLabel("Position Size:")
        self.position_size_value = QLabel("0 shares")
        
        # Risk per trade
        self.risk_per_trade_label = QLabel("Risk per Trade:")
        self.risk_per_trade_value = QLabel("0%")
        
        # Portfolio exposure
        self.exposure_label = QLabel("Portfolio Exposure:")
        self.exposure_value = QLabel("0%")
        self.exposure_bar = QProgressBar()
        
        # VaR (Value at Risk)
        self.var_label = QLabel("VaR (95%):")
        self.var_value = QLabel("$0.00")
        
        risk_grid.addWidget(self.position_size_label, 0, 0)
        risk_grid.addWidget(self.position_size_value, 0, 1)
        risk_grid.addWidget(self.risk_per_trade_label, 1, 0)
        risk_grid.addWidget(self.risk_per_trade_value, 1, 1)
        risk_grid.addWidget(self.exposure_label, 2, 0)
        risk_grid.addWidget(self.exposure_value, 2, 1)
        risk_grid.addWidget(self.exposure_bar, 2, 2)
        risk_grid.addWidget(self.var_label, 3, 0)
        risk_grid.addWidget(self.var_value, 3, 1, 1, 2)
        
        risk_group.setLayout(risk_grid)
        risk_layout.addWidget(risk_group)
        
        risk_tab.setLayout(risk_layout)
        self.tab_widget.addTab(risk_tab, "⚠️ Risk")
        
        main_layout.addWidget(self.tab_widget)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("📊 Export Report")
        self.export_btn.clicked.connect(self.export_performance_report)
        
        self.reset_btn = QPushButton("🔄 Reset Stats")
        self.reset_btn.clicked.connect(self.reset_statistics)
        
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def create_stat_card(self, title, value, color):
        """Create a styled statistics card"""
        card = QGroupBox()
        card.setStyleSheet(f"""
            QGroupBox {{
                border: 2px solid {color};
                border-radius: 8px;
                font-weight: bold;
                padding: 10px;
                margin: 5px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(10)
        title_label.setFont(title_font)
        
        value_label = QLabel(value)
        value_label.setAlignment(Qt.AlignCenter)
        value_font = QFont()
        value_font.setPointSize(16)
        value_font.setBold(True)
        value_label.setFont(value_font)
        value_label.setStyleSheet(f"color: {color};")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        card.setLayout(layout)
        
        # Store reference to value label for updates
        card.value_label = value_label
        
        return card

    def setup_timers(self):
        """Setup timers for real-time updates"""
        # Main stats timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)  # Update every second
        
        # Chart update timer
        self.chart_timer = QTimer(self)
        self.chart_timer.timeout.connect(self.update_charts)
        self.chart_timer.start(5000)  # Update every 5 seconds
        
        # Trade history timer
        self.trades_timer = QTimer(self)
        self.trades_timer.timeout.connect(self.update_trade_history)
        self.trades_timer.start(10000)  # Update every 10 seconds

    def update_all_data(self):
        """Update all dashboard data"""
        self.update_stats()
        self.update_charts()
        self.update_trade_history()

    def update_stats(self):
        """Update real-time statistics using real data"""
        try:
            # Get real stats from data manager
            stats = data_manager.get_trading_stats()
            
            # Update stat cards
            self.balance_card.value_label.setText(f"${stats.get('total_pnl', 0):,.2f}")
            self.unrealized_card.value_label.setText(f"${stats.get('unrealized_pnl', 0):,.2f}")
            self.realized_card.value_label.setText(f"${stats.get('realized_pnl', 0):,.2f}")
            
            total_pnl = stats.get('total_pnl', 0)
            self.total_card.value_label.setText(f"${total_pnl:,.2f}")
            
            # Update performance metrics
            win_rate = stats.get('win_rate', 0)
            self.win_rate_value.setText(f"{win_rate:.1f}%")
            self.win_rate_bar.setValue(int(win_rate))
            
            self.profit_factor_value.setText(f"{stats.get('profit_factor', 0):.2f}")
            self.sharpe_ratio_value.setText(f"{stats.get('sharpe_ratio', 0):.2f}")
            
            max_dd = stats.get('max_drawdown', 0)
            self.max_drawdown_value.setText(f"{max_dd:.1f}%")
            self.drawdown_bar.setValue(int(max_dd))
            
            # Update risk metrics
            active_positions = stats.get('active_positions', 0)
            self.position_size_value.setText(f"{active_positions} positions")
            
            risk_per_trade = 2.0  # Default 2% risk per trade
            self.risk_per_trade_value.setText(f"{risk_per_trade:.1f}%")
            
            exposure = min(active_positions * 20, 100)  # Cap at 100%
            self.exposure_value.setText(f"{exposure:.1f}%")
            self.exposure_bar.setValue(int(exposure))
            
        except Exception as e:
            EventLog.emit("WARNING", f"PnL stats update error: {e}")
            # Fall back to mock data if real data fails
            self.update_stats_with_mock_data()
        
        self.var_value.setText(f"${stats.get('var_95', 0):,.2f}")

    def update_charts(self):
        """Update the PnL charts with improved error handling"""
        try:
            # Generate mock PnL history if no real data
            if len(self.pnl_history) < 100:
                self.generate_mock_pnl_history()
            
            # Create PnL chart
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Portfolio P&L Over Time', 'Daily Returns'),
                row_heights=[0.7, 0.3]
            )
            
            # Extract P&L values and dates
            dates = [datetime.now() - timedelta(days=i) for i in range(len(self.pnl_history))]
            dates.reverse()
            
            # Ensure pnl_history contains numeric values
            pnl_values = [float(val) if val is not None else 0 for val in self.pnl_history]
            
            # P&L line chart
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=pnl_values,
                    mode='lines+markers',
                    name='P&L',
                    line=dict(color='#4CAF50', width=2),
                    marker=dict(size=4)
                ),
                row=1, col=1
            )
            
            # Daily returns - handle empty or insufficient data
            if len(pnl_values) > 1:
                returns = [0] + [pnl_values[i] - pnl_values[i-1] 
                                for i in range(1, len(pnl_values))]
            else:
                returns = [0]
            
            colors = ['green' if r >= 0 else 'red' for r in returns]
            
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=returns,
                    name='Daily Return',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Portfolio Performance Dashboard",
                xaxis_title="Date",
                yaxis_title="P&L ($)",
                xaxis2_title="Date",
                yaxis2_title="Daily Return ($)",
                showlegend=False,
                height=400
            )
            
            self.pnl_chart.update_figure(fig)
            
        except Exception as e:
            EventLog.emit("ERROR", f"Chart update failed: {e}")
            # Create a simple error chart
            try:
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Chart Error: {str(e)[:100]}...",
                    showarrow=False,
                    x=0.5, y=0.5,
                    xref="paper", yref="paper"
                )
                self.pnl_chart.update_figure(error_fig)
            except:
                pass  # If even error display fails, just continue

    def update_trade_history(self):
        """Update trade history table with real data"""
        try:
            # Get real trade history from data manager
            trades = data_manager.get_trade_history(50)  # Get last 50 trades
            
            self.trades_table.setRowCount(len(trades))
            
            for row, trade in enumerate(trades):
                # Format timestamp
                open_time = trade.get('open_time', '')
                if open_time:
                    try:
                        dt = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = "N/A"
                else:
                    time_str = "N/A"
                
                # Determine action
                direction = trade.get('direction', 'long')
                action = "BUY" if direction == 'long' else "SELL"
                
                # Status with emoji
                status = trade.get('status', 'unknown')
                if status == 'open':
                    status_display = "🟢 Open"
                elif trade.get('pnl', 0) > 0:
                    status_display = "✅ Profit"
                else:
                    status_display = "❌ Loss"
                
                # Fill table
                self.trades_table.setItem(row, 0, QTableWidgetItem(time_str))
                self.trades_table.setItem(row, 1, QTableWidgetItem(trade.get('ticker', 'N/A')))
                self.trades_table.setItem(row, 2, QTableWidgetItem(action))
                self.trades_table.setItem(row, 3, QTableWidgetItem(f"{trade.get('size', 0):.0f}"))
                self.trades_table.setItem(row, 4, QTableWidgetItem(f"${trade.get('entry_price', 0):.2f}"))
                
                # Color-code P&L
                pnl_item = QTableWidgetItem(f"${trade.get('pnl', 0):.2f}")
                if trade.get('pnl', 0) >= 0:
                    pnl_item.setForeground(QColor('green'))
                else:
                    pnl_item.setForeground(QColor('red'))
                self.trades_table.setItem(row, 5, pnl_item)
                
                self.trades_table.setItem(row, 6, QTableWidgetItem(status_display))
                self.trades_table.setItem(row, 7, QTableWidgetItem("AI Model"))
        
        except Exception as e:
            EventLog.emit("WARNING", f"Trade history update error: {e}")
            # Fall back to mock data if real data fails
            self.update_trade_history_with_mock_data()

    def generate_mock_stats(self):
        """Generate mock statistics for demo purposes"""
        return {
            'balance': 50000 + random.uniform(-5000, 5000),
            'unrealized': random.uniform(-2000, 3000),
            'realized': random.uniform(-1000, 2000),
            'win_rate': random.uniform(0.45, 0.75),
            'profit_factor': random.uniform(0.8, 2.5),
            'sharpe_ratio': random.uniform(-0.5, 2.0),
            'max_drawdown': random.uniform(0.05, 0.25),
            'position_size': random.randint(10, 500),
            'risk_per_trade': random.uniform(1, 5),
            'exposure': random.uniform(0.3, 0.8),
            'var_95': random.uniform(500, 3000)
        }

    def generate_mock_pnl_history(self):
        """Generate mock P&L history"""
        base_value = 50000
        for i in range(100):
            if i == 0:
                self.pnl_history.append(base_value)
            else:
                change = random.uniform(-500, 700)  # Slight upward bias
                new_value = self.pnl_history[-1] + change
                self.pnl_history.append(max(new_value, 10000))  # Don't go below 10k

    def add_mock_trade(self):
        """Add a mock trade to history"""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'AMZN']
        actions = ['BUY', 'SELL']
        strategies = ['Dual-LLM', 'ML Signal', 'Trend Following', 'Mean Reversion']
        
        trade = {
            'time': datetime.now().strftime('%H:%M:%S'),
            'symbol': random.choice(symbols),
            'action': random.choice(actions),
            'quantity': random.randint(10, 200),
            'price': random.uniform(50, 300),
            'pnl': random.uniform(-200, 400),
            'status': random.choice(['Completed', 'Pending', 'Cancelled']),
            'strategy': random.choice(strategies)
        }
        
        self.trade_history.append(trade)

    def export_performance_report(self):
        """Export performance report to CSV"""
        import csv
        from datetime import datetime
        
        filename = f"shnifter_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow(['Shnifter Trader Performance Report'])
                writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                writer.writerow([])
                
                # Write trade history
                writer.writerow(['Trade History'])
                writer.writerow(['Time', 'Symbol', 'Action', 'Quantity', 'Price', 'P&L', 'Status', 'Strategy'])
                
                for trade in self.trade_history:
                    writer.writerow([
                        trade['time'], trade['symbol'], trade['action'],
                        trade['quantity'], trade['price'], trade['pnl'],
                        trade['status'], trade['strategy']
                    ])
            
            from core.events import EventLog
            EventLog.emit("INFO", f"Performance report exported to {filename}")
            
        except Exception as e:
            from core.events import EventLog
            EventLog.emit("ERROR", f"Failed to export report: {e}")

    def reset_statistics(self):
        """Reset all statistics and trade history"""
        self.trade_history.clear()
        self.pnl_history.clear()
        self.performance_metrics.clear()
        
        # Clear table
        self.trades_table.setRowCount(0)
        
        from core.events import EventLog
        EventLog.emit("INFO", "PnL statistics reset")

    def on_stats_update(self, stats_data):
        """Callback for real-time stats updates from data manager"""
        try:
            # Update stat cards with real data
            self.balance_card.value_label.setText(f"${stats_data.get('total_pnl', 0):,.2f}")
            self.unrealized_card.value_label.setText(f"${stats_data.get('unrealized_pnl', 0):,.2f}")
            self.realized_card.value_label.setText(f"${stats_data.get('realized_pnl', 0):,.2f}")
            
            total_pnl = stats_data.get('total_pnl', 0)
            self.total_card.value_label.setText(f"${total_pnl:,.2f}")
            
            # Update performance metrics
            win_rate = stats_data.get('win_rate', 0)
            self.win_rate_value.setText(f"{win_rate:.1f}%")
            self.win_rate_bar.setValue(int(win_rate))
            
            self.profit_factor_value.setText(f"{stats_data.get('profit_factor', 0):.2f}")
            self.sharpe_ratio_value.setText(f"{stats_data.get('sharpe_ratio', 0):.2f}")
            
            max_dd = stats_data.get('max_drawdown', 0)
            self.max_drawdown_value.setText(f"{max_dd:.1f}%")
            self.drawdown_bar.setValue(int(max_dd))
            
            # Update risk metrics
            active_positions = stats_data.get('active_positions', 0)
            self.position_size_value.setText(f"{active_positions} positions")
            
            # Calculate risk per trade as percentage of total portfolio
            risk_per_trade = 2.0  # Default 2% risk per trade
            self.risk_per_trade_value.setText(f"{risk_per_trade:.1f}%")
            
            # Calculate exposure based on active positions
            exposure = min(active_positions * 20, 100)  # Cap at 100%
            self.exposure_value.setText(f"{exposure:.1f}%")
            self.exposure_bar.setValue(int(exposure))
            
            # Update PnL history for charting - store just the numeric value
            self.pnl_history.append(total_pnl)
            
            # Keep only last 100 points
            if len(self.pnl_history) > 100:
                self.pnl_history = self.pnl_history[-100:]
                
        except Exception as e:
            EventLog.emit("WARNING", f"PnL Dashboard stats update error: {e}")

    def on_trade_update(self, trade_data):
        """Callback for real-time trade updates from data manager"""
        try:
            self.trade_history = trade_data
            # Refresh trade table
            self.update_trade_history()
        except Exception as e:
            EventLog.emit("WARNING", f"PnL Dashboard trade update error: {e}")

    def get_real_trading_stats(self):
        """Get real trading statistics from data manager"""
        try:
            return data_manager.get_trading_stats()
        except Exception as e:
            EventLog.emit("WARNING", f"Could not get real trading stats: {e}")
            return self.generate_mock_stats()
