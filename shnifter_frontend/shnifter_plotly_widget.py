"""
Shnifterized Plotly Widget - PySide6 Qt Implementation
React Plotly component to native Qt with LLM integration
"""
import sys
from typing import Dict, Any, Optional, List
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QCheckBox, QTextEdit, QSplitter, QGroupBox,
    QToolBar, QMenuBar, QMenu, QFileDialog, QMessageBox,
    QSpinBox, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QIcon, QPixmap
import tempfile
import os

from core.events import EventLog
from llm_manager.llm_providers import OllamaProvider

class LLMChartAnalyzer(QThread):
    """Background thread for LLM chart analysis"""
    analysis_complete = Signal(str)
    
    def __init__(self, chart_data: Dict[str, Any], llm_provider: Optional[Any] = None):
        super().__init__()
        self.chart_data = chart_data
        self.llm_provider = llm_provider or OllamaProvider()
        
    def run(self):
        try:
            # Extract key chart metrics for LLM analysis
            analysis_prompt = self._build_analysis_prompt()
            
            # Get LLM analysis
            analysis = self.llm_provider.generate_response(
                analysis_prompt,
                context="You are an expert financial chart analyst."
            )
            
            self.analysis_complete.emit(analysis)
            
        except Exception as e:
            self.analysis_complete.emit(f"Analysis error: {str(e)}")
    
    def _build_analysis_prompt(self) -> str:
        """Build comprehensive prompt for chart analysis"""
        data_summary = {
            'data_points': len(self.chart_data.get('data', [])),
            'chart_type': self.chart_data.get('type', 'unknown'),
            'timeframe': self.chart_data.get('timeframe', 'unknown'),
            'symbol': self.chart_data.get('symbol', 'unknown')
        }
        
        return f"""
        Analyze this financial chart data:
        
        Symbol: {data_summary['symbol']}
        Chart Type: {data_summary['chart_type']}
        Data Points: {data_summary['data_points']}
        Timeframe: {data_summary['timeframe']}
        
        Please provide:
        1. Key trend analysis
        2. Support/resistance levels
        3. Pattern recognition
        4. Trading recommendations
        5. Risk assessment
        
        Keep response concise but actionable.
        """

class ShnifterPlotlyWidget(QWidget):
    """
    Advanced Plotly widget with LLM integration for trading analysis
    """
    chart_updated = Signal(dict)
    analysis_requested = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.chart_data = {}
        self.llm_provider = OllamaProvider()
        self.temp_html_file = None
        
        # Setup UI
        self._setup_ui()
        self._setup_connections()
        
        # Auto-analysis timer
        self.auto_analysis_timer = QTimer()
        self.auto_analysis_timer.timeout.connect(self._auto_analyze)
        
    def _setup_ui(self):
        """Setup the widget UI"""
        main_layout = QVBoxLayout()
        
        # Toolbar
        self._create_toolbar()
        main_layout.addWidget(self.toolbar)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Chart area
        chart_widget = self._create_chart_widget()
        content_splitter.addWidget(chart_widget)
        
        # Analysis panel
        analysis_widget = self._create_analysis_panel()
        content_splitter.addWidget(analysis_widget)
        
        # Set splitter ratios (70% chart, 30% analysis)
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter)
        self.setLayout(main_layout)
        
    def _create_toolbar(self):
        """Create chart toolbar with controls"""
        self.toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        
        # Chart type selector
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Candlestick", "OHLC", "Line", "Area", "Volume", "Bollinger Bands"
        ])
        toolbar_layout.addWidget(QLabel("Chart Type:"))
        toolbar_layout.addWidget(self.chart_type_combo)
        
        # Timeframe selector
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "1h", "4h", "1d", "1w"])
        toolbar_layout.addWidget(QLabel("Timeframe:"))
        toolbar_layout.addWidget(self.timeframe_combo)
        
        # Auto-analysis toggle
        self.auto_analysis_checkbox = QCheckBox("Auto LLM Analysis")
        self.auto_analysis_checkbox.setChecked(True)
        toolbar_layout.addWidget(self.auto_analysis_checkbox)
        
        # Manual analysis button
        self.analyze_btn = QPushButton("🤖 Analyze Chart")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        toolbar_layout.addWidget(self.analyze_btn)
        
        # Export button
        self.export_btn = QPushButton("💾 Export")
        toolbar_layout.addWidget(self.export_btn)
        
        toolbar_layout.addStretch()
        self.toolbar.setLayout(toolbar_layout)
        
    def _create_chart_widget(self):
        """Create the main chart display widget"""
        chart_group = QGroupBox("Interactive Chart")
        chart_layout = QVBoxLayout()
        
        # Web view for Plotly chart
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(400)
        chart_layout.addWidget(self.chart_view)
        
        chart_group.setLayout(chart_layout)
        return chart_group
        
    def _create_analysis_panel(self):
        """Create LLM analysis panel"""
        analysis_group = QGroupBox("🤖 LLM Analysis")
        analysis_layout = QVBoxLayout()
        
        # Analysis output
        self.analysis_text = QTextEdit()
        self.analysis_text.setPlaceholderText("LLM chart analysis will appear here...")
        self.analysis_text.setMaximumHeight(200)
        analysis_layout.addWidget(self.analysis_text)
        
        # Quick insights
        insights_group = QGroupBox("Quick Insights")
        insights_layout = QVBoxLayout()
        
        self.trend_label = QLabel("Trend: --")
        self.support_label = QLabel("Support: --")
        self.resistance_label = QLabel("Resistance: --")
        self.signal_label = QLabel("Signal: --")
        
        insights_layout.addWidget(self.trend_label)
        insights_layout.addWidget(self.support_label)
        insights_layout.addWidget(self.resistance_label)
        insights_layout.addWidget(self.signal_label)
        
        insights_group.setLayout(insights_layout)
        analysis_layout.addWidget(insights_group)
        
        analysis_group.setLayout(analysis_layout)
        return analysis_group
        
    def _setup_connections(self):
        """Setup signal connections"""
        self.chart_type_combo.currentTextChanged.connect(self._on_chart_type_changed)
        self.timeframe_combo.currentTextChanged.connect(self._on_timeframe_changed)
        self.auto_analysis_checkbox.stateChanged.connect(self._on_auto_analysis_toggled)
        self.analyze_btn.clicked.connect(self._manual_analyze)
        self.export_btn.clicked.connect(self._export_chart)
        
    def load_data(self, data: Dict[str, Any]):
        """Load and display chart data"""
        try:
            self.chart_data = data
            self._update_chart()
            
            # Trigger auto-analysis if enabled
            if self.auto_analysis_checkbox.isChecked():
                self._auto_analyze()
                
        except Exception as e:
            EventLog.log_event("ERROR", f"Failed to load chart data: {str(e)}")
            
    def _update_chart(self):
        """Update the chart display"""
        try:
            if not self.chart_data:
                return
                
            # Create Plotly figure based on chart type
            chart_type = self.chart_type_combo.currentText()
            fig = self._create_plotly_figure(chart_type)
            
            # Convert to HTML and display
            html_content = self._plotly_to_html(fig)
            self._display_html(html_content)
            
            self.chart_updated.emit(self.chart_data)
            
        except Exception as e:
            EventLog.log_event("ERROR", f"Failed to update chart: {str(e)}")
            
    def _create_plotly_figure(self, chart_type: str) -> go.Figure:
        """Create Plotly figure based on type"""
        # This would be implemented based on your data structure
        # For now, creating a sample candlestick chart
        
        if 'data' not in self.chart_data:
            # Create sample data for demonstration
            import yfinance as yf
            ticker = self.chart_data.get('symbol', 'AAPL')
            data = yf.download(ticker, period="1mo", interval="1d")
            
            if chart_type == "Candlestick":
                fig = go.Figure(data=go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                ))
            else:
                fig = go.Figure(data=go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price'
                ))
        else:
            # Use provided data
            fig = go.Figure()  # Implement based on your data structure
            
        # Styling
        fig.update_layout(
            title=f"{self.chart_data.get('symbol', 'Chart')} - {chart_type}",
            xaxis_title="Time",
            yaxis_title="Price",
            template="plotly_dark",
            height=500
        )
        
        return fig
        
    def _plotly_to_html(self, fig: go.Figure) -> str:
        """Convert Plotly figure to HTML"""
        return fig.to_html(include_plotlyjs='cdn', div_id="chart-div")
        
    def _display_html(self, html_content: str):
        """Display HTML content in web view"""
        # Create temporary HTML file
        if self.temp_html_file:
            try:
                os.remove(self.temp_html_file)
            except:
                pass
                
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_content)
            self.temp_html_file = f.name
            
        self.chart_view.load(f"file://{self.temp_html_file}")
        
    def _on_chart_type_changed(self, chart_type: str):
        """Handle chart type change"""
        self._update_chart()
        
    def _on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change"""
        # Update data for new timeframe
        if 'symbol' in self.chart_data:
            # Reload data with new timeframe
            pass
        self._update_chart()
        
    def _on_auto_analysis_toggled(self, checked: bool):
        """Handle auto-analysis toggle"""
        if checked:
            self.auto_analysis_timer.start(30000)  # 30 seconds
        else:
            self.auto_analysis_timer.stop()
            
    def _manual_analyze(self):
        """Trigger manual chart analysis"""
        self._analyze_chart()
        
    def _auto_analyze(self):
        """Automatic chart analysis"""
        if self.chart_data and self.auto_analysis_checkbox.isChecked():
            self._analyze_chart()
            
    def _analyze_chart(self):
        """Perform LLM chart analysis"""
        if not self.chart_data:
            return
            
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("🤖 Analyzing...")
        
        # Start analysis thread
        self.analysis_thread = LLMChartAnalyzer(self.chart_data, self.llm_provider)
        self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analysis_thread.start()
        
    def _on_analysis_complete(self, analysis: str):
        """Handle completed analysis"""
        self.analysis_text.setPlainText(analysis)
        
        # Extract insights for quick display
        self._extract_quick_insights(analysis)
        
        # Re-enable button
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🤖 Analyze Chart")
        
        EventLog.log_event("INFO", "Chart analysis completed")
        
    def _extract_quick_insights(self, analysis: str):
        """Extract key insights from analysis text"""
        # Simple extraction - could be enhanced with NLP
        lines = analysis.lower().split('\n')
        
        for line in lines:
            if 'trend' in line:
                self.trend_label.setText(f"Trend: {line.strip()}")
            elif 'support' in line:
                self.support_label.setText(f"Support: {line.strip()}")
            elif 'resistance' in line:
                self.resistance_label.setText(f"Resistance: {line.strip()}")
            elif 'signal' in line or 'recommendation' in line:
                self.signal_label.setText(f"Signal: {line.strip()}")
                
    def _export_chart(self):
        """Export chart and analysis"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Chart", f"shnifter_chart_{datetime.now():%Y%m%d_%H%M%S}.html",
                "HTML Files (*.html);;PNG Files (*.png);;PDF Files (*.pdf)"
            )
            
            if file_path:
                if file_path.endswith('.html'):
                    # Copy current HTML file
                    if self.temp_html_file:
                        import shutil
                        shutil.copy(self.temp_html_file, file_path)
                        
                # Add analysis as comment if HTML
                if file_path.endswith('.html'):
                    with open(file_path, 'a') as f:
                        f.write(f"\n<!-- LLM Analysis:\n{self.analysis_text.toPlainText()}\n-->")
                        
                QMessageBox.information(self, "Export Complete", f"Chart exported to:\n{file_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export chart:\n{str(e)}")
            
    def cleanup(self):
        """Cleanup resources"""
        if self.temp_html_file:
            try:
                os.remove(self.temp_html_file)
            except:
                pass
                
        self.auto_analysis_timer.stop()
        
    def update_figure(self, fig: go.Figure):
        """Update the chart with a new Plotly figure"""
        if fig is None:
            return
        
        try:
            # Convert figure to HTML and display
            html_content = self._plotly_to_html(fig)
            self._display_html(html_content)
            
            # Update internal chart data
            self.chart_data['figure'] = fig
            self.chart_updated.emit(self.chart_data)
            
            EventLog.emit("INFO", "Chart figure updated successfully")
            
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to update chart figure: {e}")
    
    def set_figure(self, fig: go.Figure):
        """Alias for update_figure for compatibility"""
        self.update_figure(fig)
        
    def clear_chart(self):
        """Clear the current chart"""
        try:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No Data Available",
                xaxis={'visible': False},
                yaxis={'visible': False},
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            self.update_figure(empty_fig)
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to clear chart: {e}")
