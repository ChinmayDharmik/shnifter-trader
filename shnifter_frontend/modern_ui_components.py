"""
Advanced UI Components adapted from Streamlit/Dash for PySide6
Modern, responsive trading dashboard components with real-time updates.
"""

import sys
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

class ModernTradingCard(QFrame):
    """
    Modern card-style widget for displaying trading metrics.
    Inspired by Streamlit's metric cards.
    """
    
    def __init__(self, title: str, value: str = "", delta: str = "", 
                 delta_color: str = "green", parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.delta = delta
        self.delta_color = delta_color
        
        self.setup_ui()
        self.apply_modern_styling()
    
    def setup_ui(self):
        """Set up the card UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)
        
        # Title label
        self.title_label = QLabel(self.title)
        self.title_label.setObjectName("cardTitle")
        layout.addWidget(self.title_label)
        
        # Value label
        self.value_label = QLabel(self.value)
        self.value_label.setObjectName("cardValue")
        layout.addWidget(self.value_label)
        
        # Delta container
        delta_layout = QHBoxLayout()
        delta_layout.setContentsMargins(0, 0, 0, 0)
        
        self.delta_label = QLabel(self.delta)
        self.delta_label.setObjectName("cardDelta")
        delta_layout.addWidget(self.delta_label)
        
        delta_layout.addStretch()
        layout.addLayout(delta_layout)
    
    def apply_modern_styling(self):
        """Apply modern card styling."""
        self.setStyleSheet("""
            ModernTradingCard {
                background-color: #ffffff;
                border: 1px solid #e1e5e9;
                border-radius: 8px;
                margin: 4px;
            }
            ModernTradingCard:hover {
                border-color: #0066cc;
                box-shadow: 0 2px 8px rgba(0, 102, 204, 0.1);
            }
            QLabel#cardTitle {
                color: #666666;
                font-size: 12px;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QLabel#cardValue {
                color: #1a1a1a;
                font-size: 24px;
                font-weight: 700;
                margin: 2px 0;
            }
            QLabel#cardDelta {
                font-size: 11px;
                font-weight: 600;
                padding: 2px 6px;
                border-radius: 4px;
                background-color: #f0f0f0;
            }
        """)
        
        # Set delta color
        if self.delta_color == "green":
            self.delta_label.setStyleSheet("QLabel { color: #28a745; background-color: #d4edda; }")
        elif self.delta_color == "red":
            self.delta_label.setStyleSheet("QLabel { color: #dc3545; background-color: #f8d7da; }")
        else:
            self.delta_label.setStyleSheet("QLabel { color: #6c757d; background-color: #f8f9fa; }")
    
    def update_value(self, value: str, delta: str = "", delta_color: str = "green"):
        """Update the card values."""
        self.value = value
        self.delta = delta
        self.delta_color = delta_color
        
        self.value_label.setText(value)
        self.delta_label.setText(delta)
        self.apply_modern_styling()


class InteractiveChart(QWidget):
    """
    Interactive chart widget using Plotly.
    Supports multiple chart types and real-time updates.
    """
    
    def __init__(self, chart_type: str = "candlestick", parent=None):
        super().__init__(parent)
        self.chart_type = chart_type
        self.data = []
        self.layout_config = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the chart UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d", "1w"])
        self.timeframe_combo.setCurrentText("1h")
        controls_layout.addWidget(QLabel("Timeframe:"))
        controls_layout.addWidget(self.timeframe_combo)
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems(["candlestick", "line", "area", "ohlc"])
        self.chart_type_combo.setCurrentText(self.chart_type)
        controls_layout.addWidget(QLabel("Type:"))
        controls_layout.addWidget(self.chart_type_combo)
        
        controls_layout.addStretch()
        
        # Indicators toggle
        self.indicators_group = QGroupBox("Indicators")
        indicators_layout = QHBoxLayout()
        
        self.ma_checkbox = QCheckBox("Moving Averages")
        self.rsi_checkbox = QCheckBox("RSI")
        self.macd_checkbox = QCheckBox("MACD")
        self.volume_checkbox = QCheckBox("Volume")
        
        indicators_layout.addWidget(self.ma_checkbox)
        indicators_layout.addWidget(self.rsi_checkbox)
        indicators_layout.addWidget(self.macd_checkbox)
        indicators_layout.addWidget(self.volume_checkbox)
        self.indicators_group.setLayout(indicators_layout)
        
        controls_layout.addWidget(self.indicators_group)
        
        layout.addLayout(controls_layout)
        
        # Chart placeholder (would integrate with Plotly in real implementation)
        self.chart_placeholder = QLabel("Interactive Chart\n(Plotly integration)")
        self.chart_placeholder.setMinimumHeight(400)
        self.chart_placeholder.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                border: 2px dashed #dee2e6;
                border-radius: 8px;
                color: #6c757d;
                font-size: 16px;
                text-align: center;
            }
        """)
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.chart_placeholder)
        
        # Connect signals
        self.timeframe_combo.currentTextChanged.connect(self.update_chart)
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        self.ma_checkbox.toggled.connect(self.update_indicators)
        self.rsi_checkbox.toggled.connect(self.update_indicators)
        self.macd_checkbox.toggled.connect(self.update_indicators)
        self.volume_checkbox.toggled.connect(self.update_indicators)
    
    def update_chart(self):
        """Update the chart based on current settings."""
        timeframe = self.timeframe_combo.currentText()
        chart_type = self.chart_type_combo.currentText()
        
        # In real implementation, this would update the Plotly chart
        self.chart_placeholder.setText(f"Interactive {chart_type.title()} Chart\nTimeframe: {timeframe}")
    
    def update_indicators(self):
        """Update chart indicators."""
        active_indicators = []
        if self.ma_checkbox.isChecked():
            active_indicators.append("MA")
        if self.rsi_checkbox.isChecked():
            active_indicators.append("RSI")
        if self.macd_checkbox.isChecked():
            active_indicators.append("MACD")
        if self.volume_checkbox.isChecked():
            active_indicators.append("Volume")
        
        indicators_text = ", ".join(active_indicators) if active_indicators else "None"
        # Update chart with indicators
    
    def set_data(self, data: List[Dict[str, Any]]):
        """Set chart data."""
        self.data = data
        self.update_chart()


class ModernDataTable(QWidget):
    """
    Modern data table with search, sorting, and pagination.
    Inspired by Streamlit's dataframe display.
    """
    
    def __init__(self, columns: List[str], parent=None):
        super().__init__(parent)
        self.columns = columns
        self.data = []
        self.filtered_data = []
        self.current_page = 1
        self.page_size = 25
        
        self.setup_ui()
        self.apply_modern_styling()
    
    def setup_ui(self):
        """Set up the table UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Table controls
        controls_layout = QHBoxLayout()
        
        # Search
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search...")
        self.search_input.textChanged.connect(self.filter_data)
        controls_layout.addWidget(QLabel("Search:"))
        controls_layout.addWidget(self.search_input)
        
        controls_layout.addStretch()
        
        # Page size
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["10", "25", "50", "100"])
        self.page_size_combo.setCurrentText("25")
        self.page_size_combo.currentTextChanged.connect(self.change_page_size)
        controls_layout.addWidget(QLabel("Page size:"))
        controls_layout.addWidget(self.page_size_combo)
        
        layout.addLayout(controls_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(self.columns)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table)
        
        # Pagination
        pagination_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_page)
        pagination_layout.addWidget(self.prev_button)
        
        self.page_label = QLabel("Page 1 of 1")
        pagination_layout.addWidget(self.page_label)
        
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_page)
        pagination_layout.addWidget(self.next_button)
        
        pagination_layout.addStretch()
        
        self.rows_label = QLabel("Showing 0 of 0 rows")
        pagination_layout.addWidget(self.rows_label)
        
        layout.addLayout(pagination_layout)
    
    def apply_modern_styling(self):
        """Apply modern table styling."""
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #e1e5e9;
                background-color: white;
                alternate-background-color: #f8f9fa;
                selection-background-color: #0066cc;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e1e5e9;
            }
            QHeaderView::section {
                background-color: #f1f3f4;
                padding: 10px;
                border: none;
                border-bottom: 2px solid #e1e5e9;
                font-weight: 600;
                color: #202124;
            }
            QLineEdit {
                padding: 8px;
                border: 2px solid #e1e5e9;
                border-radius: 4px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #0066cc;
            }
            QPushButton {
                padding: 8px 16px;
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0052a3;
            }
            QPushButton:disabled {
                background-color: #e1e5e9;
                color: #6c757d;
            }
        """)
    
    def set_data(self, data: List[Dict[str, Any]]):
        """Set table data."""
        self.data = data
        self.filtered_data = data
        self.current_page = 1
        self.update_table()
    
    def filter_data(self, search_text: str):
        """Filter data based on search text."""
        if not search_text:
            self.filtered_data = self.data
        else:
            self.filtered_data = []
            for row in self.data:
                for value in row.values():
                    if search_text.lower() in str(value).lower():
                        self.filtered_data.append(row)
                        break
        
        self.current_page = 1
        self.update_table()
    
    def change_page_size(self, size_text: str):
        """Change page size."""
        self.page_size = int(size_text)
        self.current_page = 1
        self.update_table()
    
    def prev_page(self):
        """Go to previous page."""
        if self.current_page > 1:
            self.current_page -= 1
            self.update_table()
    
    def next_page(self):
        """Go to next page."""
        total_pages = (len(self.filtered_data) + self.page_size - 1) // self.page_size
        if self.current_page < total_pages:
            self.current_page += 1
            self.update_table()
    
    def update_table(self):
        """Update table display."""
        # Calculate pagination
        total_rows = len(self.filtered_data)
        total_pages = max(1, (total_rows + self.page_size - 1) // self.page_size)
        start_idx = (self.current_page - 1) * self.page_size
        end_idx = min(start_idx + self.page_size, total_rows)
        
        page_data = self.filtered_data[start_idx:end_idx]
        
        # Update table
        self.table.setRowCount(len(page_data))
        
        for row_idx, row_data in enumerate(page_data):
            for col_idx, column in enumerate(self.columns):
                value = row_data.get(column, "")
                item = QTableWidgetItem(str(value))
                self.table.setItem(row_idx, col_idx, item)
        
        # Update pagination controls
        self.page_label.setText(f"Page {self.current_page} of {total_pages}")
        self.rows_label.setText(f"Showing {start_idx + 1}-{end_idx} of {total_rows} rows")
        
        self.prev_button.setEnabled(self.current_page > 1)
        self.next_button.setEnabled(self.current_page < total_pages)


class ModernProgressIndicator(QWidget):
    """
    Modern progress indicator with multiple states.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the progress indicator UI."""
        layout = QVBoxLayout(self)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.apply_styling()
    
    def apply_styling(self):
        """Apply modern progress bar styling."""
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                text-align: center;
                font-weight: 600;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 6px;
            }
            QLabel {
                color: #6c757d;
                font-size: 12px;
                font-weight: 500;
            }
        """)
    
    def set_progress(self, value: int, status: str = ""):
        """Set progress value and status."""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)


class ResponsiveTradingDashboard(QMainWindow):
    """
    Main responsive trading dashboard using modern UI components.
    """
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.apply_modern_theme()
    
    def setup_ui(self):
        """Set up the dashboard UI."""
        self.setWindowTitle("Modern Trading Dashboard")
        self.setMinimumSize(1200, 800)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(16, 16, 16, 16)
        
        # Header with metrics cards
        self.create_metrics_section(main_layout)
        
        # Chart and table section
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # Left side - Charts
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.chart = InteractiveChart("candlestick")
        left_layout.addWidget(self.chart)
        
        content_layout.addWidget(left_widget, 2)  # 2/3 width
        
        # Right side - Data table
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        table_columns = ["Symbol", "Price", "Change", "Volume", "Signal"]
        self.data_table = ModernDataTable(table_columns)
        right_layout.addWidget(self.data_table)
        
        content_layout.addWidget(right_widget, 1)  # 1/3 width
        
        main_layout.addLayout(content_layout)
        
        # Progress indicator
        self.progress = ModernProgressIndicator()
        main_layout.addWidget(self.progress)
        
        # Load sample data
        self.load_sample_data()
    
    def create_metrics_section(self, layout):
        """Create the metrics cards section."""
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(12)
        
        # Portfolio value
        portfolio_card = ModernTradingCard(
            "Portfolio Value",
            "$125,432.50",
            "+2.34% (+$2,876)",
            "green"
        )
        metrics_layout.addWidget(portfolio_card)
        
        # Daily P&L
        pnl_card = ModernTradingCard(
            "Daily P&L",
            "+$1,245.67",
            "+0.98%",
            "green"
        )
        metrics_layout.addWidget(pnl_card)
        
        # Open positions
        positions_card = ModernTradingCard(
            "Open Positions",
            "8",
            "2 new today",
            "blue"
        )
        metrics_layout.addWidget(positions_card)
        
        # Win rate
        winrate_card = ModernTradingCard(
            "Win Rate",
            "73.2%",
            "+1.2% vs last week",
            "green"
        )
        metrics_layout.addWidget(winrate_card)
        
        layout.addLayout(metrics_layout)
    
    def load_sample_data(self):
        """Load sample data for demonstration."""
        sample_data = [
            {"Symbol": "AAPL", "Price": "$150.25", "Change": "+2.34%", "Volume": "45.2M", "Signal": "BUY"},
            {"Symbol": "GOOGL", "Price": "$2,745.50", "Change": "-0.52%", "Volume": "1.2M", "Signal": "HOLD"},
            {"Symbol": "MSFT", "Price": "$335.75", "Change": "+1.87%", "Volume": "25.8M", "Signal": "BUY"},
            {"Symbol": "TSLA", "Price": "$205.80", "Change": "+4.12%", "Volume": "78.5M", "Signal": "BUY"},
            {"Symbol": "NVDA", "Price": "$875.25", "Change": "+3.45%", "Volume": "35.7M", "Signal": "STRONG_BUY"},
        ]
        
        self.data_table.set_data(sample_data)
        self.progress.set_progress(85, "Market data updated")
    
    def apply_modern_theme(self):
        """Apply modern theme to the dashboard."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: transparent;
                font-family: "Segoe UI", Arial, sans-serif;
            }
        """)


# Example usage
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    dashboard = ResponsiveTradingDashboard()
    dashboard.show()
    
    sys.exit(app.exec())
