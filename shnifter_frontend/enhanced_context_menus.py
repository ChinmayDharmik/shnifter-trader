"""
Enhanced Context Menu System for Shnifter Trader
Provides intelligent right-click menus throughout the application
"""

from PySide6.QtWidgets import (QMenu, QAction, QWidget, QApplication, QTextEdit, 
                               QTableWidget, QTableWidgetItem, QHeaderView)
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QIcon, QPixmap, QClipboard
from typing import Dict, List, Callable, Optional, Any
import json
import csv
import os
from datetime import datetime
from core.events import EventLog

class ShnifterContextMenu(QMenu):
    """
    Base enhanced context menu with common functionality
    """
    
    action_triggered = Signal(str, dict)  # action_id, context_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context_data = {}
        self.setup_styling()
    
    def setup_styling(self):
        """Apply dark theme styling to context menu"""
        self.setStyleSheet("""
            QMenu {
                background-color: #2c3e50;
                color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 5px;
            }
            QMenu::item {
                padding: 8px 25px;
                margin: 2px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
            QMenu::item:disabled {
                color: #7f8c8d;
            }
            QMenu::separator {
                height: 2px;
                background: #34495e;
                margin: 5px 10px;
            }
            QMenu::indicator {
                width: 16px;
                height: 16px;
                margin-left: 5px;
            }
        """)
    
    def add_action_with_icon(self, text: str, action_id: str, icon_text: str = "", 
                            callback: Callable = None, enabled: bool = True):
        """Add action with emoji icon and callback"""
        display_text = f"{icon_text} {text}" if icon_text else text
        action = QAction(display_text, self)
        action.setEnabled(enabled)
        
        if callback:
            action.triggered.connect(callback)
        else:
            action.triggered.connect(lambda: self.action_triggered.emit(action_id, self.context_data))
        
        self.addAction(action)
        return action
    
    def add_submenu(self, text: str, icon_text: str = "") -> 'ShnifterContextMenu':
        """Add submenu and return it for further configuration"""
        display_text = f"{icon_text} {text}" if icon_text else text
        submenu = ShnifterContextMenu(self)
        submenu.setTitle(display_text)
        self.addMenu(submenu)
        return submenu
    
    def set_context_data(self, data: Dict[str, Any]):
        """Set context data for this menu"""
        self.context_data = data


class ShnifterTextContextMenu(ShnifterContextMenu):
    """
    Context menu for text widgets (QTextEdit, QLineEdit, etc.)
    """
    
    def __init__(self, text_widget: QTextEdit, parent=None):
        super().__init__(parent)
        self.text_widget = text_widget
        self.setup_text_menu()
    
    def setup_text_menu(self):
        """Setup text-specific menu items"""
        # Standard text operations
        self.add_action_with_icon("Cut", "cut", "✂️", self.cut_text)
        self.add_action_with_icon("Copy", "copy", "📋", self.copy_text)
        self.add_action_with_icon("Paste", "paste", "📌", self.paste_text)
        self.add_action_with_icon("Select All", "select_all", "🔲", self.select_all_text)
        
        self.addSeparator()
        
        # Advanced text operations
        self.add_action_with_icon("Clear All", "clear", "🗑️", self.clear_text)
        self.add_action_with_icon("Export to File", "export", "💾", self.export_text)
        
        self.addSeparator()
        
        # AI features
        ai_menu = self.add_submenu("AI Analysis", "🤖")
        ai_menu.add_action_with_icon("Analyze with LLM", "ai_analyze", "🧠", self.analyze_with_llm)
        ai_menu.add_action_with_icon("Summarize Content", "ai_summarize", "📄", self.summarize_content)
        ai_menu.add_action_with_icon("Extract Key Points", "ai_extract", "🔍", self.extract_key_points)
    
    def cut_text(self):
        self.text_widget.cut()
        EventLog.emit("INFO", "Text cut to clipboard")
    
    def copy_text(self):
        self.text_widget.copy()
        EventLog.emit("INFO", "Text copied to clipboard")
    
    def paste_text(self):
        self.text_widget.paste()
        EventLog.emit("INFO", "Text pasted from clipboard")
    
    def select_all_text(self):
        self.text_widget.selectAll()
        EventLog.emit("INFO", "All text selected")
    
    def clear_text(self):
        self.text_widget.clear()
        EventLog.emit("INFO", "Text cleared")
    
    def export_text(self):
        """Export text content to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shnifter_export_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.text_widget.toPlainText())
            
            EventLog.emit("INFO", f"Text exported to {filename}")
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to export text: {e}")
    
    def analyze_with_llm(self):
        """Analyze selected/all text with LLM"""
        text = self.text_widget.textCursor().selectedText() or self.text_widget.toPlainText()
        if text.strip():
            EventLog.emit("INFO", f"Analyzing {len(text)} characters with LLM")
            # Emit signal for parent to handle LLM analysis
            self.action_triggered.emit("llm_analyze", {"text": text})
    
    def summarize_content(self):
        """Summarize content with LLM"""
        text = self.text_widget.toPlainText()
        if text.strip():
            EventLog.emit("INFO", "Requesting content summary")
            self.action_triggered.emit("llm_summarize", {"text": text})
    
    def extract_key_points(self):
        """Extract key points with LLM"""
        text = self.text_widget.toPlainText()
        if text.strip():
            EventLog.emit("INFO", "Extracting key points")
            self.action_triggered.emit("llm_extract_keys", {"text": text})


class ShnifterTableContextMenu(ShnifterContextMenu):
    """
    Context menu for table widgets with data operations
    """
    
    def __init__(self, table_widget: QTableWidget, parent=None):
        super().__init__(parent)
        self.table_widget = table_widget
        self.setup_table_menu()
    
    def setup_table_menu(self):
        """Setup table-specific menu items"""
        # Cell operations
        self.add_action_with_icon("Copy Cell", "copy_cell", "📋", self.copy_cell)
        self.add_action_with_icon("Copy Row", "copy_row", "📊", self.copy_row)
        self.add_action_with_icon("Copy Column", "copy_column", "📈", self.copy_column)
        
        self.addSeparator()
        
        # Table operations
        self.add_action_with_icon("Export to CSV", "export_csv", "💾", self.export_to_csv)
        self.add_action_with_icon("Export to JSON", "export_json", "📄", self.export_to_json)
        
        self.addSeparator()
        
        # Analysis operations
        analysis_menu = self.add_submenu("Analysis", "📊")
        analysis_menu.add_action_with_icon("Calculate Statistics", "calc_stats", "🔢", self.calculate_statistics)
        analysis_menu.add_action_with_icon("Generate Chart", "generate_chart", "📈", self.generate_chart)
        analysis_menu.add_action_with_icon("AI Insights", "ai_insights", "🤖", self.ai_insights)
        
        # View operations
        view_menu = self.add_submenu("View", "👁️")
        view_menu.add_action_with_icon("Fit Columns", "fit_columns", "↔️", self.fit_columns)
        view_menu.add_action_with_icon("Sort Ascending", "sort_asc", "⬆️", self.sort_ascending)
        view_menu.add_action_with_icon("Sort Descending", "sort_desc", "⬇️", self.sort_descending)
    
    def copy_cell(self):
        """Copy current cell to clipboard"""
        current_item = self.table_widget.currentItem()
        if current_item:
            clipboard = QApplication.clipboard()
            clipboard.setText(current_item.text())
            EventLog.emit("INFO", "Cell copied to clipboard")
    
    def copy_row(self):
        """Copy current row to clipboard"""
        current_row = self.table_widget.currentRow()
        if current_row >= 0:
            row_data = []
            for col in range(self.table_widget.columnCount()):
                item = self.table_widget.item(current_row, col)
                row_data.append(item.text() if item else "")
            
            clipboard = QApplication.clipboard()
            clipboard.setText("\t".join(row_data))
            EventLog.emit("INFO", "Row copied to clipboard")
    
    def copy_column(self):
        """Copy current column to clipboard"""
        current_col = self.table_widget.currentColumn()
        if current_col >= 0:
            col_data = []
            for row in range(self.table_widget.rowCount()):
                item = self.table_widget.item(row, current_col)
                col_data.append(item.text() if item else "")
            
            clipboard = QApplication.clipboard()
            clipboard.setText("\n".join(col_data))
            EventLog.emit("INFO", "Column copied to clipboard")
    
    def export_to_csv(self):
        """Export table data to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shnifter_table_export_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write headers
                headers = []
                for col in range(self.table_widget.columnCount()):
                    header_item = self.table_widget.horizontalHeaderItem(col)
                    headers.append(header_item.text() if header_item else f"Column_{col}")
                writer.writerow(headers)
                
                # Write data
                for row in range(self.table_widget.rowCount()):
                    row_data = []
                    for col in range(self.table_widget.columnCount()):
                        item = self.table_widget.item(row, col)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)
            
            EventLog.emit("INFO", f"Table exported to {filename}")
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to export table: {e}")
    
    def export_to_json(self):
        """Export table data to JSON"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"shnifter_table_export_{timestamp}.json"
            
            # Get headers
            headers = []
            for col in range(self.table_widget.columnCount()):
                header_item = self.table_widget.horizontalHeaderItem(col)
                headers.append(header_item.text() if header_item else f"Column_{col}")
            
            # Get data
            data = []
            for row in range(self.table_widget.rowCount()):
                row_dict = {}
                for col in range(self.table_widget.columnCount()):
                    item = self.table_widget.item(row, col)
                    row_dict[headers[col]] = item.text() if item else ""
                data.append(row_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            EventLog.emit("INFO", f"Table exported to {filename}")
        except Exception as e:
            EventLog.emit("ERROR", f"Failed to export table: {e}")
    
    def calculate_statistics(self):
        """Calculate basic statistics for numeric columns"""
        EventLog.emit("INFO", "Calculating table statistics")
        self.action_triggered.emit("calculate_stats", {"table": self.table_widget})
    
    def generate_chart(self):
        """Generate chart from table data"""
        EventLog.emit("INFO", "Generating chart from table data")
        self.action_triggered.emit("generate_chart", {"table": self.table_widget})
    
    def ai_insights(self):
        """Get AI insights on table data"""
        EventLog.emit("INFO", "Requesting AI insights on table data")
        self.action_triggered.emit("ai_insights", {"table": self.table_widget})
    
    def fit_columns(self):
        """Auto-fit column widths"""
        self.table_widget.resizeColumnsToContents()
        EventLog.emit("INFO", "Columns auto-fitted")
    
    def sort_ascending(self):
        """Sort by current column ascending"""
        current_col = self.table_widget.currentColumn()
        if current_col >= 0:
            self.table_widget.sortItems(current_col, Qt.AscendingOrder)
            EventLog.emit("INFO", f"Sorted column {current_col} ascending")
    
    def sort_descending(self):
        """Sort by current column descending"""
        current_col = self.table_widget.currentColumn()
        if current_col >= 0:
            self.table_widget.sortItems(current_col, Qt.DescendingOrder)
            EventLog.emit("INFO", f"Sorted column {current_col} descending")


class ShnifterContextMenuManager:
    """
    Manages context menus throughout the application
    """
    
    def __init__(self):
        self.registered_widgets = {}
        self.llm_callback = None
    
    def register_widget(self, widget: QWidget, menu_type: str = "auto"):
        """Register a widget for context menu support"""
        if menu_type == "auto":
            if isinstance(widget, QTextEdit):
                menu_type = "text"
            elif isinstance(widget, QTableWidget):
                menu_type = "table"
            else:
                menu_type = "basic"
        
        self.registered_widgets[widget] = menu_type
        widget.setContextMenuPolicy(Qt.CustomContextMenu)
        widget.customContextMenuRequested.connect(
            lambda pos: self.show_context_menu(widget, pos)
        )
    
    def set_llm_callback(self, callback: Callable):
        """Set callback for LLM-related menu actions"""
        self.llm_callback = callback
    
    def show_context_menu(self, widget: QWidget, position: QPoint):
        """Show appropriate context menu for widget"""
        menu_type = self.registered_widgets.get(widget, "basic")
        
        if menu_type == "text" and isinstance(widget, QTextEdit):
            menu = ShnifterTextContextMenu(widget)
            if self.llm_callback:
                menu.action_triggered.connect(self.handle_llm_action)
        elif menu_type == "table" and isinstance(widget, QTableWidget):
            menu = ShnifterTableContextMenu(widget)
            menu.action_triggered.connect(self.handle_table_action)
        else:
            menu = ShnifterContextMenu()
            menu.add_action_with_icon("Properties", "properties", "ℹ️")
        
        # Show menu at cursor position
        global_pos = widget.mapToGlobal(position)
        menu.exec(global_pos)
    
    def handle_llm_action(self, action_id: str, context_data: Dict[str, Any]):
        """Handle LLM-related actions"""
        if self.llm_callback:
            self.llm_callback(action_id, context_data)
        else:
            EventLog.emit("WARNING", f"No LLM callback set for action: {action_id}")
    
    def handle_table_action(self, action_id: str, context_data: Dict[str, Any]):
        """Handle table-specific actions"""
        EventLog.emit("INFO", f"Table action triggered: {action_id}")
        # Add specific table action handling here


# Global context menu manager instance
shnifter_context_manager = ShnifterContextMenuManager()


# Usage example
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("Shnifter Context Menu Demo")
    window.resize(600, 400)
    
    layout = QVBoxLayout(window)
    
    # Text widget with context menu
    text_edit = QTextEdit()
    text_edit.setPlainText("Right-click me for enhanced context menu!")
    layout.addWidget(text_edit)
    
    # Table widget with context menu
    table = QTableWidget(5, 3)
    table.setHorizontalHeaderLabels(["Symbol", "Price", "Change"])
    for i in range(5):
        table.setItem(i, 0, QTableWidgetItem(f"STOCK{i}"))
        table.setItem(i, 1, QTableWidgetItem(f"${100 + i * 10}"))
        table.setItem(i, 2, QTableWidgetItem(f"+{i}%"))
    layout.addWidget(table)
    
    # Register widgets with context menu manager
    shnifter_context_manager.register_widget(text_edit, "text")
    shnifter_context_manager.register_widget(table, "table")
    
    window.show()
    sys.exit(app.exec())
