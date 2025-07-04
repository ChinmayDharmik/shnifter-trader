"""
Shnifterized Table Widget - PySide6 Qt Implementation
Native Qt with LLM integration
"""
import sys
from typing import Dict, Any, Optional, List, Union
import json
import pandas as pd
import numpy as np
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QCheckBox, QTextEdit, QSplitter, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QMenu, QFileDialog, QMessageBox, QAbstractItemView,
    QStyledItemDelegate, QApplication
)
from PySide6.QtCore import (
    Qt, Signal, QThread, QTimer, QSortFilterProxyModel, 
    QAbstractTableModel, QModelIndex
)
from PySide6.QtGui import QColor, QFont, QIcon, QAction

from core.events import EventLog
from llm_manager.llm_providers import OllamaProvider

class LLMTableAnalyzer(QThread):
    """Background thread for LLM table analysis"""
    analysis_complete = Signal(str)
    insights_complete = Signal(dict)
    
    def __init__(self, table_data: pd.DataFrame, analysis_type: str = "general", llm_provider: Optional[Any] = None):
        super().__init__()
        self.table_data = table_data
        self.analysis_type = analysis_type
        self.llm_provider = llm_provider or OllamaProvider()
        
    def run(self):
        try:
            # Build analysis prompt based on data
            analysis_prompt = self._build_analysis_prompt()
            
            # Get LLM analysis
            analysis = self.llm_provider.generate_response(
                analysis_prompt,
                context="You are an expert financial data analyst."
            )
            
            # Extract actionable insights
            insights = self._extract_insights(analysis)
            
            self.analysis_complete.emit(analysis)
            self.insights_complete.emit(insights)
            
        except Exception as e:
            self.analysis_complete.emit(f"Analysis error: {str(e)}")
            self.insights_complete.emit({})
    
    def _build_analysis_prompt(self) -> str:
        """Build comprehensive prompt for table analysis"""
        # Get data summary
        summary = {
            'rows': len(self.table_data),
            'columns': list(self.table_data.columns),
            'sample_data': self.table_data.head(3).to_dict('records') if not self.table_data.empty else [],
            'dtypes': self.table_data.dtypes.to_dict() if not self.table_data.empty else {}
        }
        
        return f"""
        Analyze this financial data table:
        
        Rows: {summary['rows']}
        Columns: {summary['columns']}
        Data Types: {summary['dtypes']}
        
        Sample Data:
        {json.dumps(summary['sample_data'], indent=2, default=str)}
        
        Analysis Type: {self.analysis_type}
        
        Please provide:
        1. Key patterns and trends
        2. Notable outliers or anomalies
        3. Performance insights
        4. Risk indicators
        5. Actionable recommendations
        
        Focus on financial metrics and trading implications.
        """
        
    def _extract_insights(self, analysis: str) -> Dict[str, Any]:
        """Extract structured insights from analysis"""
        insights = {
            'top_performers': [],
            'risk_alerts': [],
            'recommendations': [],
            'key_metrics': {}
        }
        
        # Simple extraction - could be enhanced with NLP
        lines = analysis.lower().split('\n')
        
        for line in lines:
            if 'top' in line or 'best' in line or 'highest' in line:
                insights['top_performers'].append(line.strip())
            elif 'risk' in line or 'alert' in line or 'warning' in line:
                insights['risk_alerts'].append(line.strip())
            elif 'recommend' in line or 'suggest' in line:
                insights['recommendations'].append(line.strip())
                
        return insights

class SmartTableModel(QAbstractTableModel):
    """Enhanced table model with LLM insights"""
    
    def __init__(self, data: pd.DataFrame = None):
        super().__init__()
        self._data = data if data is not None else pd.DataFrame()
        self._highlights = {}  # Cell highlighting based on LLM insights
        
    def rowCount(self, parent=QModelIndex()):
        return len(self._data)
        
    def columnCount(self, parent=QModelIndex()):
        return len(self._data.columns) if not self._data.empty else 0
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        value = self._data.iloc[index.row(), index.column()]
        
        if role == Qt.DisplayRole:
            if pd.isna(value):
                return ""
            elif isinstance(value, float):
                return f"{value:.4f}"
            else:
                return str(value)
                
        elif role == Qt.BackgroundRole:
            # Apply LLM-based highlighting
            key = (index.row(), index.column())
            if key in self._highlights:
                return QColor(self._highlights[key])
                
        elif role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float)):
                return Qt.AlignRight | Qt.AlignVCenter
                
        return None
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section]) if not self._data.empty else ""
            else:
                return str(section + 1)
        return None
        
    def update_data(self, data: pd.DataFrame):
        """Update table data"""
        self.beginResetModel()
        self._data = data
        self.endResetModel()
        
    def apply_highlights(self, highlights: Dict[tuple, str]):
        """Apply LLM-suggested highlights"""
        self._highlights = highlights
        self.dataChanged.emit(self.index(0, 0), 
                            self.index(self.rowCount()-1, self.columnCount()-1))

class ShnifterTableWidget(QWidget):
    """
    Advanced table widget with LLM integration for data analysis
    """
    data_updated = Signal(dict)
    analysis_requested = Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        
        # Initialize components
        self.table_data = pd.DataFrame()
        self.llm_provider = OllamaProvider()
        self.current_filter = ""
        
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
        
        # Table area
        table_widget = self._create_table_widget()
        content_splitter.addWidget(table_widget)
        
        # Analysis panel
        analysis_widget = self._create_analysis_panel()
        content_splitter.addWidget(analysis_widget)
        
        # Set splitter ratios (70% table, 30% analysis)
        content_splitter.setSizes([700, 300])
        
        main_layout.addWidget(content_splitter)
        self.setLayout(main_layout)
        
    def _create_toolbar(self):
        """Create table toolbar with controls"""
        self.toolbar = QWidget()
        toolbar_layout = QHBoxLayout()
        
        # Search/Filter
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("🔍 Filter data...")
        self.filter_input.setMaximumWidth(200)
        toolbar_layout.addWidget(QLabel("Filter:"))
        toolbar_layout.addWidget(self.filter_input)
        
        # Analysis type selector
        self.analysis_type_combo = QComboBox()
        self.analysis_type_combo.addItems([
            "General Analysis", "Performance Review", "Risk Assessment", 
            "Trend Analysis", "Anomaly Detection", "Portfolio Summary"
        ])
        toolbar_layout.addWidget(QLabel("Analysis:"))
        toolbar_layout.addWidget(self.analysis_type_combo)
        
        # Auto-analysis toggle
        self.auto_analysis_checkbox = QCheckBox("Auto LLM Analysis")
        self.auto_analysis_checkbox.setChecked(True)
        toolbar_layout.addWidget(self.auto_analysis_checkbox)
        
        # Manual analysis button
        self.analyze_btn = QPushButton("🤖 Analyze Data")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        toolbar_layout.addWidget(self.analyze_btn)
        
        # Highlight toggle
        self.highlight_checkbox = QCheckBox("LLM Highlights")
        self.highlight_checkbox.setChecked(True)
        toolbar_layout.addWidget(self.highlight_checkbox)
        
        # Export button
        self.export_btn = QPushButton("💾 Export")
        toolbar_layout.addWidget(self.export_btn)
        
        toolbar_layout.addStretch()
        self.toolbar.setLayout(toolbar_layout)
        
    def _create_table_widget(self):
        """Create the main table display widget"""
        table_group = QGroupBox("Smart Data Table")
        table_layout = QVBoxLayout()
        
        # Create table view with model
        self.table_model = SmartTableModel()
        self.table_view = QTableWidget()
        
        # Configure table
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Style the table
        self.table_view.setStyleSheet("""
            QTableWidget {
                gridline-color: #E0E0E0;
                background-color: white;
                alternate-background-color: #F5F5F5;
            }
            QTableWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
            QHeaderView::section {
                background-color: #37474F;
                color: white;
                padding: 8px;
                border: none;
                font-weight: bold;
            }
        """)
        
        table_layout.addWidget(self.table_view)
        
        # Table stats
        self.stats_label = QLabel("No data loaded")
        self.stats_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px;")
        table_layout.addWidget(self.stats_label)
        
        table_group.setLayout(table_layout)
        return table_group
        
    def _create_analysis_panel(self):
        """Create LLM analysis panel"""
        analysis_group = QGroupBox("🤖 LLM Data Analysis")
        analysis_layout = QVBoxLayout()
        
        # Analysis output
        self.analysis_text = QTextEdit()
        self.analysis_text.setPlaceholderText("LLM data analysis will appear here...")
        self.analysis_text.setMaximumHeight(200)
        analysis_layout.addWidget(self.analysis_text)
        
        # Quick insights panel
        insights_group = QGroupBox("Key Insights")
        insights_layout = QVBoxLayout()
        
        # Top performers
        self.top_performers_label = QLabel("🏆 Top Performers: --")
        self.top_performers_label.setWordWrap(True)
        insights_layout.addWidget(self.top_performers_label)
        
        # Risk alerts
        self.risk_alerts_label = QLabel("⚠️ Risk Alerts: --")
        self.risk_alerts_label.setWordWrap(True)
        self.risk_alerts_label.setStyleSheet("color: #FF5722;")
        insights_layout.addWidget(self.risk_alerts_label)
        
        # Recommendations
        self.recommendations_label = QLabel("💡 Recommendations: --")
        self.recommendations_label.setWordWrap(True)
        self.recommendations_label.setStyleSheet("color: #4CAF50;")
        insights_layout.addWidget(self.recommendations_label)
        
        insights_group.setLayout(insights_layout)
        analysis_layout.addWidget(insights_group)
        
        analysis_group.setLayout(analysis_layout)
        return analysis_group
        
    def _setup_connections(self):
        """Setup signal connections"""
        self.filter_input.textChanged.connect(self._apply_filter)
        self.analysis_type_combo.currentTextChanged.connect(self._on_analysis_type_changed)
        self.auto_analysis_checkbox.stateChanged.connect(self._on_auto_analysis_toggled)
        self.analyze_btn.clicked.connect(self._manual_analyze)
        self.highlight_checkbox.stateChanged.connect(self._on_highlight_toggled)
        self.export_btn.clicked.connect(self._export_table)
        
    def load_data(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict]]):
        """Load and display table data"""
        try:
            # Convert input to DataFrame
            if isinstance(data, dict):
                if 'data' in data and 'columns' in data:
                   
                    df = pd.DataFrame(data['data'], columns=data['columns'])
                else:
                    df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
                
            self.table_data = df
            self._update_table()
            
            # Update stats
            self._update_stats()
            
            # Trigger auto-analysis if enabled
            if self.auto_analysis_checkbox.isChecked():
                self._auto_analyze()
                
        except Exception as e:
            EventLog.log_event("ERROR", f"Failed to load table data: {str(e)}")
            
    def _update_table(self):
        """Update the table display"""
        try:
            if self.table_data.empty:
                return
                
            # Clear existing table
            self.table_view.clear()
            
            # Set dimensions
            self.table_view.setRowCount(len(self.table_data))
            self.table_view.setColumnCount(len(self.table_data.columns))
            
            # Set headers
            self.table_view.setHorizontalHeaderLabels(list(self.table_data.columns))
            
            # Populate data
            for row in range(len(self.table_data)):
                for col in range(len(self.table_data.columns)):
                    value = self.table_data.iloc[row, col]
                    
                    # Format value
                    if pd.isna(value):
                        display_value = ""
                    elif isinstance(value, float):
                        display_value = f"{value:.4f}"
                    else:
                        display_value = str(value)
                        
                    item = QTableWidgetItem(display_value)
                    
                    # Set alignment for numbers
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                        
                    self.table_view.setItem(row, col, item)
                    
            self.data_updated.emit({'rows': len(self.table_data), 'columns': len(self.table_data.columns)})
            
        except Exception as e:
            EventLog.log_event("ERROR", f"Failed to update table: {str(e)}")
            
    def _update_stats(self):
        """Update table statistics"""
        if self.table_data.empty:
            self.stats_label.setText("No data loaded")
        else:
            rows, cols = self.table_data.shape
            numeric_cols = len(self.table_data.select_dtypes(include=[np.number]).columns)
            self.stats_label.setText(f"📊 {rows} rows × {cols} columns | {numeric_cols} numeric columns")
            
    def _apply_filter(self, filter_text: str):
        """Apply text filter to table"""
        self.current_filter = filter_text.lower()
        
        if not filter_text:
            # Show all rows
            for row in range(self.table_view.rowCount()):
                self.table_view.setRowHidden(row, False)
        else:
            # Filter rows based on text
            for row in range(self.table_view.rowCount()):
                show_row = False
                for col in range(self.table_view.columnCount()):
                    item = self.table_view.item(row, col)
                    if item and filter_text in item.text().lower():
                        show_row = True
                        break
                self.table_view.setRowHidden(row, not show_row)
                
    def _on_analysis_type_changed(self, analysis_type: str):
        """Handle analysis type change"""
        # Could trigger re-analysis with new type
        pass
        
    def _on_auto_analysis_toggled(self, checked: bool):
        """Handle auto-analysis toggle"""
        if checked:
            self.auto_analysis_timer.start(45000)  # 45 seconds
        else:
            self.auto_analysis_timer.stop()
            
    def _on_highlight_toggled(self, checked: bool):
        """Handle highlight toggle"""
        if not checked:
            # Clear all highlights
            for row in range(self.table_view.rowCount()):
                for col in range(self.table_view.columnCount()):
                    item = self.table_view.item(row, col)
                    if item:
                        item.setBackground(QColor())
                        
    def _manual_analyze(self):
        """Trigger manual table analysis"""
        self._analyze_table()
        
    def _auto_analyze(self):
        """Automatic table analysis"""
        if not self.table_data.empty and self.auto_analysis_checkbox.isChecked():
            self._analyze_table()
            
    def _analyze_table(self):
        """Perform LLM table analysis"""
        if self.table_data.empty:
            return
            
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("🤖 Analyzing...")
        
        # Start analysis thread
        analysis_type = self.analysis_type_combo.currentText().lower()
        self.analysis_thread = LLMTableAnalyzer(self.table_data, analysis_type, self.llm_provider)
        self.analysis_thread.analysis_complete.connect(self._on_analysis_complete)
        self.analysis_thread.insights_complete.connect(self._on_insights_complete)
        self.analysis_thread.start()
        
    def _on_analysis_complete(self, analysis: str):
        """Handle completed analysis"""
        self.analysis_text.setPlainText(analysis)
        
        # Re-enable button
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("🤖 Analyze Data")
        
        EventLog.log_event("INFO", "Table analysis completed")
        
    def _on_insights_complete(self, insights: Dict[str, Any]):
        """Handle extracted insights"""
        # Update insight labels
        if insights.get('top_performers'):
            self.top_performers_label.setText(f"🏆 Top Performers: {', '.join(insights['top_performers'][:2])}")
        
        if insights.get('risk_alerts'):
            self.risk_alerts_label.setText(f"⚠️ Risk Alerts: {', '.join(insights['risk_alerts'][:2])}")
            
        if insights.get('recommendations'):
            self.recommendations_label.setText(f"💡 Recommendations: {', '.join(insights['recommendations'][:2])}")
            
        # Apply highlights if enabled
        if self.highlight_checkbox.isChecked():
            self._apply_llm_highlights(insights)
            
    def _apply_llm_highlights(self, insights: Dict[str, Any]):
        """Apply LLM-suggested highlights to table"""
        # This is a simplified version - in practice, you'd need more sophisticated
        # text matching to identify which cells to highlight
        
        # Example: highlight cells containing top performer names
        for performer in insights.get('top_performers', []):
            self._highlight_cells_containing(performer, '#C8E6C9')  # Light green
            
        # Highlight risk alerts in red
        for risk in insights.get('risk_alerts', []):
            self._highlight_cells_containing(risk, '#FFCDD2')  # Light red
            
    def _highlight_cells_containing(self, text: str, color: str):
        """Highlight cells containing specific text"""
        for row in range(self.table_view.rowCount()):
            for col in range(self.table_view.columnCount()):
                item = self.table_view.item(row, col)
                if item and text.lower() in item.text().lower():
                    item.setBackground(QColor(color))
                    
    def _export_table(self):
        """Export table and analysis"""
        try:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Table", f"shnifter_table_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "CSV Files (*.csv);;Excel Files (*.xlsx);;JSON Files (*.json)"
            )
            
            if file_path:
                if file_path.endswith('.csv'):
                    self.table_data.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    self.table_data.to_excel(file_path, index=False)
                elif file_path.endswith('.json'):
                    self.table_data.to_json(file_path, orient='records', indent=2)
                    
                # Export analysis as separate file
                analysis_path = file_path.rsplit('.', 1)[0] + '_analysis.txt'
                with open(analysis_path, 'w') as f:
                    f.write(f"Shnifter Table Analysis - {datetime.now()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(self.analysis_text.toPlainText())
                    
                QMessageBox.information(self, "Export Complete", 
                                      f"Table exported to:\n{file_path}\n\nAnalysis exported to:\n{analysis_path}")
                
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export table:\n{str(e)}")
            
    def get_selected_data(self) -> pd.DataFrame:
        """Get currently selected rows as DataFrame"""
        selected_rows = set()
        for item in self.table_view.selectedItems():
            selected_rows.add(item.row())
            
        if selected_rows:
            return self.table_data.iloc[list(selected_rows)]
        else:
            return self.table_data
