"""
Enhanced Dropdown Components for Shnifter Trader
Provides advanced dropdown functionality with autocomplete, grouping, and history
"""

from PySide6.QtWidgets import (QComboBox, QCompleter, QListWidget, QListWidgetItem, 
                               QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, 
                               QWidget, QLabel, QScrollArea, QFrame)
from PySide6.QtCore import Qt, Signal, QStringListModel, QTimer
from PySide6.QtGui import QIcon, QPixmap, QFont
from typing import List, Dict, Optional, Any
import json
import os

class ShnifterEnhancedDropdown(QComboBox):
    """
    Enhanced dropdown with autocomplete, grouping, and recent items history
    """
    
    selection_changed = Signal(str, dict)  # item_text, item_data
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.NoInsert)
        
        # History and grouping
        self.recent_items = []
        self.grouped_items = {}
        self.max_recent_items = 10
        self.history_file = None
        
        # Setup autocomplete
        self.completer = QCompleter()
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchContains)
        self.setCompleter(self.completer)
        
        # Connect signals
        self.currentTextChanged.connect(self._on_text_changed)
        self.activated.connect(self._on_activated)
        
        # Styling
        self.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 2px solid #3498db;
                border-radius: 5px;
                background: #2c3e50;
                color: white;
                min-height: 25px;
            }
            QComboBox:hover {
                border-color: #5dade2;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: url(none);
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid white;
                margin: 5px;
            }
            QComboBox QAbstractItemView {
                background: #34495e;
                color: white;
                selection-background-color: #3498db;
                border: 1px solid #3498db;
            }
        """)
    
    def set_history_file(self, filepath: str):
        """Set file path for storing dropdown history"""
        self.history_file = filepath
        self._load_history()
    
    def add_grouped_items(self, group_name: str, items: List[Dict[str, Any]]):
        """
        Add items under a group header
        items format: [{"text": "display", "data": any_data, "icon": optional_path}]
        """
        self.grouped_items[group_name] = items
        self._rebuild_dropdown()
    
    def add_recent_item(self, text: str, data: Any = None):
        """Add item to recent history"""
        item_data = {"text": text, "data": data}
        
        # Remove if already exists
        self.recent_items = [item for item in self.recent_items if item["text"] != text]
        
        # Add to front
        self.recent_items.insert(0, item_data)
        
        # Limit size
        if len(self.recent_items) > self.max_recent_items:
            self.recent_items = self.recent_items[:self.max_recent_items]
        
        self._rebuild_dropdown()
        self._save_history()
    
    def _rebuild_dropdown(self):
        """Rebuild dropdown with groups and recent items"""
        self.clear()
        all_items = []
        
        # Add recent items section
        if self.recent_items:
            for i, item in enumerate(self.recent_items):
                display_text = f"🕒 {item['text']}"
                self.addItem(display_text, item["data"])
                all_items.append(item["text"])
                
                if i == 0:  # Add separator after recent items
                    self.insertSeparator(self.count())
        
        # Add grouped items
        for group_name, items in self.grouped_items.items():
            # Add group header (disabled item)
            header_item = f"── {group_name} ──"
            self.addItem(header_item)
            self.model().item(self.count() - 1).setEnabled(False)
            
            # Add group items
            for item in items:
                display_text = item["text"]
                if "icon" in item and item["icon"]:
                    # Add icon if specified
                    icon = QIcon(item["icon"])
                    self.addItem(icon, display_text, item.get("data"))
                else:
                    self.addItem(display_text, item.get("data"))
                all_items.append(display_text)
        
        # Update completer
        if all_items:
            model = QStringListModel(all_items)
            self.completer.setModel(model)
    
    def _on_text_changed(self, text: str):
        """Handle text changes for autocomplete"""
        if text and len(text) >= 2:  # Start autocomplete after 2 characters
            self.completer.setCompletionPrefix(text)
            if self.completer.completionCount() > 0:
                self.completer.complete()
    
    def _on_activated(self, index: int):
        """Handle item selection"""
        text = self.itemText(index)
        data = self.itemData(index)
        
        # Skip headers and separators
        if text.startswith("──") or text == "":
            return
        
        # Clean recent item prefix
        clean_text = text.replace("🕒 ", "")
        
        # Add to recent items
        self.add_recent_item(clean_text, data)
        
        # Emit signal
        self.selection_changed.emit(clean_text, data or {})
    
    def _save_history(self):
        """Save recent items to file"""
        if self.history_file and self.recent_items:
            try:
                os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
                with open(self.history_file, 'w') as f:
                    json.dump(self.recent_items, f, indent=2)
            except Exception as e:
                print(f"Failed to save dropdown history: {e}")
    
    def _load_history(self):
        """Load recent items from file"""
        if self.history_file and os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.recent_items = json.load(f)
                self._rebuild_dropdown()
            except Exception as e:
                print(f"Failed to load dropdown history: {e}")


class ShnifterModelSelector(ShnifterEnhancedDropdown):
    """
    Specialized dropdown for LLM model selection with status indicators
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_history_file("data/model_history.json")
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_models)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds
    
    def add_ollama_models(self, models: List[str], statuses: Dict[str, str] = None):
        """Add Ollama models with status indicators"""
        if not statuses:
            statuses = {}
        
        model_items = []
        for model in models:
            status = statuses.get(model, "unknown")
            status_icon = {
                "running": "🟢",
                "available": "🟡", 
                "downloading": "🔄",
                "error": "🔴",
                "unknown": "⚪"
            }.get(status, "⚪")
            
            display_text = f"{status_icon} {model}"
            model_items.append({
                "text": display_text,
                "data": {"model": model, "status": status}
            })
        
        self.add_grouped_items("Ollama Models", model_items)
    
    def add_openai_models(self, models: List[str]):
        """Add OpenAI models"""
        model_items = []
        for model in models:
            model_items.append({
                "text": f"🤖 {model}",
                "data": {"model": model, "provider": "openai"}
            })
        
        self.add_grouped_items("OpenAI Models", model_items)
    
    def refresh_models(self):
        """Refresh available models"""
        # This would be called to update model availability
        # Implementation depends on your LLM provider setup
        pass
    
    def update_models(self, models: List[str]):
        """
        Update the dropdown with a fresh list of models, clearing any existing groups.
        """
        # Reset grouped items and rebuild
        self.grouped_items.clear()
        model_items = [{"text": m, "data": {"model": m}} for m in models]
        self.add_grouped_items("Models", model_items)


class ShnifterTickerSelector(ShnifterEnhancedDropdown):
    """
    Enhanced ticker selection with market data integration
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_history_file("data/ticker_history.json")
        self.setup_market_groups()
    
    def setup_market_groups(self):
        """Setup common market groups"""
        
        # Popular stocks
        popular_stocks = [
            {"text": "AAPL - Apple Inc.", "data": {"symbol": "AAPL", "type": "stock"}},
            {"text": "GOOGL - Alphabet Inc.", "data": {"symbol": "GOOGL", "type": "stock"}},
            {"text": "MSFT - Microsoft Corp.", "data": {"symbol": "MSFT", "type": "stock"}},
            {"text": "TSLA - Tesla Inc.", "data": {"symbol": "TSLA", "type": "stock"}},
            {"text": "NVDA - NVIDIA Corp.", "data": {"symbol": "NVDA", "type": "stock"}},
        ]
        
        # Major ETFs
        major_etfs = [
            {"text": "SPY - SPDR S&P 500 ETF", "data": {"symbol": "SPY", "type": "etf"}},
            {"text": "QQQ - Invesco QQQ Trust", "data": {"symbol": "QQQ", "type": "etf"}},
            {"text": "VTI - Vanguard Total Stock Market", "data": {"symbol": "VTI", "type": "etf"}},
        ]
        
        # Crypto
        crypto_symbols = [
            {"text": "BTC-USD - Bitcoin", "data": {"symbol": "BTC-USD", "type": "crypto"}},
            {"text": "ETH-USD - Ethereum", "data": {"symbol": "ETH-USD", "type": "crypto"}},
        ]
        
        self.add_grouped_items("Popular Stocks", popular_stocks)
        self.add_grouped_items("Major ETFs", major_etfs)
        self.add_grouped_items("Cryptocurrency", crypto_symbols)


class ShnifterProviderSelector(ShnifterEnhancedDropdown):
    """
    Data provider selection with capability indicators
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_history_file("data/provider_history.json")
        self.setup_providers()
    
    def setup_providers(self):
        """Setup available data providers"""
        
        free_providers = [
            {
                "text": "📈 YFinance - Yahoo Finance (Free)", 
                "data": {"provider": "yfinance", "cost": "free", "features": ["stocks", "etfs", "crypto"]}
            },
            {
                "text": "📊 Alpha Vantage (Free Tier)", 
                "data": {"provider": "alpha_vantage", "cost": "freemium", "features": ["stocks", "forex"]}
            }
        ]
        
        premium_providers = [
            {
                "text": "💼 Benzinga Pro", 
                "data": {"provider": "benzinga", "cost": "premium", "features": ["news", "options", "earnings"]}
            },
            {
                "text": "🏪 Financial Modeling Prep", 
                "data": {"provider": "fmp", "cost": "premium", "features": ["fundamentals", "analysis"]}
            },
            {
                "text": "🔍 Intrinio", 
                "data": {"provider": "intrinio", "cost": "premium", "features": ["institutional", "realtime"]}
            }
        ]
        
        self.add_grouped_items("Free Providers", free_providers)
        self.add_grouped_items("Premium Providers", premium_providers)
    
    def add_providers(self, providers: List[str]):
        """
        Update the dropdown with a custom list of providers.
        """
        # Reset grouped items
        self.grouped_items.clear()
        provider_items = [{"text": p, "data": {"provider": p}} for p in providers]
        self.add_grouped_items("Providers", provider_items)


# Usage example and testing
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget, QLabel
    import sys
    
    app = QApplication(sys.argv)
    
    window = QWidget()
    window.setWindowTitle("Shnifter Enhanced Dropdowns Demo")
    window.resize(500, 400)
    
    layout = QVBoxLayout(window)
    
    # Model selector
    layout.addWidget(QLabel("LLM Model Selector:"))
    model_selector = ShnifterModelSelector()
    model_selector.add_ollama_models(["llama3", "gemma", "mistral"], {"llama3": "running", "gemma": "available"})
    model_selector.add_openai_models(["gpt-4", "gpt-3.5-turbo"])
    layout.addWidget(model_selector)
    
    # Ticker selector
    layout.addWidget(QLabel("Ticker Selector:"))
    ticker_selector = ShnifterTickerSelector()
    layout.addWidget(ticker_selector)
    
    # Provider selector
    layout.addWidget(QLabel("Data Provider Selector:"))
    provider_selector = ShnifterProviderSelector()
    layout.addWidget(provider_selector)
    
    window.show()
    sys.exit(app.exec())
