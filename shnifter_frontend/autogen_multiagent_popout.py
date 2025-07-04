"""
AutoGen Multi-Agent Trading Popout
Real-time multi-agent discussions and consensus trading decisions for Shnifter Trader
"""

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QTextEdit, QGroupBox, QGridLayout, QComboBox, QProgressBar,
                             QTabWidget, QWidget, QListWidget, QListWidgetItem, QFrame,
                             QScrollArea, QSplitter)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, pyqtSignal
from PySide6.QtGui import QFont, QColor, QPalette
from core.events import EventLog
from core.shnifter_data_manager import data_manager
from shnifter_analysis_modules.chroma_knowledge_base import TradingKnowledgeBase
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any

class AutoGenAgentWidget(QFrame):
    """Individual agent display widget with real-time opinions"""
    
    def __init__(self, agent_name: str, agent_role: str, parent=None):
        super().__init__(parent)
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Agent header
        header_layout = QHBoxLayout()
        
        # Agent avatar/icon
        self.agent_icon = QLabel("🤖")
        self.agent_icon.setStyleSheet("font-size: 24px;")
        header_layout.addWidget(self.agent_icon)
        
        # Agent info
        info_layout = QVBoxLayout()
        self.name_label = QLabel(self.agent_name)
        self.name_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.role_label = QLabel(self.agent_role)
        self.role_label.setStyleSheet("color: #666; font-size: 9px;")
        
        info_layout.addWidget(self.name_label)
        info_layout.addWidget(self.role_label)
        header_layout.addLayout(info_layout)
        
        header_layout.addStretch()
        
        # Status indicator
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #4CAF50; font-size: 12px;")
        header_layout.addWidget(self.status_dot)
        
        layout.addLayout(header_layout)
        
        # Current opinion
        self.opinion_label = QLabel("Current Opinion:")
        self.opinion_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(self.opinion_label)
        
        self.opinion_text = QTextEdit()
        self.opinion_text.setMaximumHeight(80)
        self.opinion_text.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 4px;
                background-color: #f9f9f9;
                font-size: 9px;
            }
        """)
        layout.addWidget(self.opinion_text)
        
        # Decision and confidence
        decision_layout = QHBoxLayout()
        
        self.decision_label = QLabel("Decision:")
        self.decision_value = QLabel("HOLD")
        self.decision_value.setStyleSheet("font-weight: bold; color: #FF9800;")
        
        self.confidence_label = QLabel("Confidence:")
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setMaximum(100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setMaximumWidth(80)
        
        decision_layout.addWidget(self.decision_label)
        decision_layout.addWidget(self.decision_value)
        decision_layout.addStretch()
        decision_layout.addWidget(self.confidence_label)
        decision_layout.addWidget(self.confidence_bar)
        
        layout.addLayout(decision_layout)
        
        # Apply agent styling
        self.setStyleSheet("""
            AutoGenAgentWidget {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 4px;
            }
        """)
    
    def update_agent_state(self, opinion: str, decision: str, confidence: float, status: str = "active"):
        """Update agent's current state"""
        self.opinion_text.setPlainText(opinion)
        self.decision_value.setText(decision)
        
        # Update decision color
        colors = {"BUY": "#4CAF50", "SELL": "#F44336", "HOLD": "#FF9800"}
        self.decision_value.setStyleSheet(f"font-weight: bold; color: {colors.get(decision, '#666')};")
        
        # Update confidence
        self.confidence_bar.setValue(int(confidence * 100))
        
        # Update status indicator
        status_colors = {"active": "#4CAF50", "thinking": "#FF9800", "offline": "#F44336"}
        self.status_dot.setStyleSheet(f"color: {status_colors.get(status, '#666')}; font-size: 12px;")

class AutoGenMultiAgentPopout(QDialog):
    """
    Multi-Agent Trading Discussion Popout using AutoGen
    Shows real-time agent discussions and consensus building
    """
    
    consensus_reached = Signal(dict)  # Emitted when agents reach consensus
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 AutoGen Multi-Agent Trading Council")
        self.setMinimumSize(1000, 700)
        self.setWindowModality(Qt.NonModal)
        
        # Initialize agents configuration
        self.agents_config = {
            "TechnicalAnalyst": {
                "role": "Technical Analysis Expert",
                "icon": "📊",
                "focus": "Charts, indicators, patterns"
            },
            "FundamentalAnalyst": {
                "role": "Fundamental Analysis Expert", 
                "icon": "📈",
                "focus": "Company financials, earnings"
            },
            "RiskManager": {
                "role": "Risk Assessment Specialist",
                "icon": "🛡️",
                "focus": "Risk metrics, position sizing"
            },
            "MarketSentiment": {
                "role": "Sentiment Analysis Expert",
                "icon": "🎭",
                "focus": "News, social media, market mood"
            },
            "QuantAnalyst": {
                "role": "Quantitative Strategist",
                "icon": "🔢",
                "focus": "Statistical models, backtesting"
            }
        }
        
        # Discussion state
        self.current_discussion = None
        self.discussion_history = []
        self.agent_widgets = {}
        
        # Initialize Chroma knowledge base for storing decisions
        self.knowledge_base = TradingKnowledgeBase()
        
        self.setup_ui()
        self.setup_timers()
        
        # Register for real-time updates
        data_manager.register_autogen_callback(self.on_autogen_update)
        
        EventLog.emit("INFO", "AutoGen Multi-Agent Council initialized")
    
    def setup_ui(self):
        """Setup the multi-agent interface"""
        main_layout = QVBoxLayout()
        
        # Control panel
        control_group = QGroupBox("🎯 Multi-Agent Trading Council")
        control_layout = QHBoxLayout()
        
        # Ticker selection
        self.ticker_label = QLabel("Ticker:")
        self.ticker_combo = QComboBox()
        self.ticker_combo.setEditable(True)
        self.ticker_combo.addItems(["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META"])
        
        # Start discussion button
        self.start_discussion_btn = QPushButton("🗣️ Start Multi-Agent Discussion")
        self.start_discussion_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_discussion_btn.clicked.connect(self.start_discussion)
        
        # Stop discussion button
        self.stop_discussion_btn = QPushButton("⏹️ Stop Discussion")
        self.stop_discussion_btn.setEnabled(False)
        self.stop_discussion_btn.clicked.connect(self.stop_discussion)
        
        control_layout.addWidget(self.ticker_label)
        control_layout.addWidget(self.ticker_combo)
        control_layout.addStretch()
        control_layout.addWidget(self.start_discussion_btn)
        control_layout.addWidget(self.stop_discussion_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Agent panels
        agents_widget = self.create_agents_panel()
        splitter.addWidget(agents_widget)
        
        # Right side: Discussion and consensus
        discussion_widget = self.create_discussion_panel()
        splitter.addWidget(discussion_widget)
        
        # Set splitter proportions
        splitter.setStretchFactor(0, 2)  # Agents panel takes 2/3
        splitter.setStretchFactor(1, 1)  # Discussion panel takes 1/3
        
        main_layout.addWidget(splitter)
        
        self.setLayout(main_layout)
    
    def create_agents_panel(self) -> QWidget:
        """Create the agents display panel"""
        agents_widget = QWidget()
        agents_layout = QVBoxLayout(agents_widget)
        
        # Agents header
        agents_header = QLabel("🤖 Trading Agents Council")
        agents_header.setFont(QFont("Arial", 12, QFont.Bold))
        agents_layout.addWidget(agents_header)
        
        # Agents grid
        agents_scroll = QScrollArea()
        agents_scroll_widget = QWidget()
        agents_grid = QGridLayout(agents_scroll_widget)
        
        # Create agent widgets
        row, col = 0, 0
        for agent_name, config in self.agents_config.items():
            agent_widget = AutoGenAgentWidget(
                agent_name=f"{config['icon']} {agent_name}",
                agent_role=config['role']
            )
            self.agent_widgets[agent_name] = agent_widget
            
            agents_grid.addWidget(agent_widget, row, col)
            
            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1
        
        agents_scroll.setWidget(agents_scroll_widget)
        agents_scroll.setWidgetResizable(True)
        agents_layout.addWidget(agents_scroll)
        
        return agents_widget
    
    def create_discussion_panel(self) -> QWidget:
        """Create the discussion and consensus panel"""
        discussion_widget = QWidget()
        discussion_layout = QVBoxLayout(discussion_widget)
        
        # Discussion tab widget
        self.discussion_tabs = QTabWidget()
        
        # Live Discussion Tab
        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        
        # Discussion status
        self.discussion_status = QLabel("💬 Discussion Status: Idle")
        self.discussion_status.setStyleSheet("font-weight: bold; padding: 8px;")
        live_layout.addWidget(self.discussion_status)
        
        # Discussion log
        self.discussion_log = QTextEdit()
        self.discussion_log.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                font-family: 'Courier New', monospace;
                font-size: 10px;
            }
        """)
        live_layout.addWidget(self.discussion_log)
        
        self.discussion_tabs.addTab(live_tab, "💬 Live Discussion")
        
        # Consensus Tab
        consensus_tab = QWidget()
        consensus_layout = QVBoxLayout(consensus_tab)
        
        # Consensus result
        consensus_group = QGroupBox("🎯 Consensus Decision")
        consensus_grid = QGridLayout(consensus_group)
        
        # Consensus decision
        self.consensus_decision_label = QLabel("Decision:")
        self.consensus_decision_value = QLabel("--")
        self.consensus_decision_value.setFont(QFont("Arial", 16, QFont.Bold))
        
        # Consensus confidence
        self.consensus_confidence_label = QLabel("Confidence:")
        self.consensus_confidence_bar = QProgressBar()
        self.consensus_confidence_bar.setMaximum(100)
        
        # Consensus reasoning
        self.consensus_reasoning_label = QLabel("Reasoning:")
        self.consensus_reasoning = QTextEdit()
        self.consensus_reasoning.setMaximumHeight(100)
        
        consensus_grid.addWidget(self.consensus_decision_label, 0, 0)
        consensus_grid.addWidget(self.consensus_decision_value, 0, 1)
        consensus_grid.addWidget(self.consensus_confidence_label, 1, 0)
        consensus_grid.addWidget(self.consensus_confidence_bar, 1, 1)
        consensus_grid.addWidget(self.consensus_reasoning_label, 2, 0, 1, 2)
        consensus_grid.addWidget(self.consensus_reasoning, 3, 0, 1, 2)
        
        consensus_layout.addWidget(consensus_group)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.save_decision_btn = QPushButton("💾 Save to Knowledge Base")
        self.save_decision_btn.clicked.connect(self.save_consensus_decision)
        
        self.apply_decision_btn = QPushButton("⚡ Apply Decision")
        self.apply_decision_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.apply_decision_btn.clicked.connect(self.apply_consensus_decision)
        
        action_layout.addWidget(self.save_decision_btn)
        action_layout.addWidget(self.apply_decision_btn)
        consensus_layout.addLayout(action_layout)
        
        consensus_layout.addStretch()
        
        self.discussion_tabs.addTab(consensus_tab, "🎯 Consensus")
        
        # Historical Decisions Tab
        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)
        
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_historical_decision)
        history_layout.addWidget(self.history_list)
        
        self.discussion_tabs.addTab(history_tab, "📊 History")
        
        discussion_layout.addWidget(self.discussion_tabs)
        
        return discussion_widget
    
    def setup_timers(self):
        """Setup update timers"""
        # Discussion update timer
        self.discussion_timer = QTimer()
        self.discussion_timer.timeout.connect(self.update_discussion_display)
        
        # Agent status timer
        self.agent_timer = QTimer()
        self.agent_timer.timeout.connect(self.update_agent_status)
        self.agent_timer.start(2000)  # Update every 2 seconds
    
    def start_discussion(self):
        """Start multi-agent discussion for selected ticker"""
        ticker = self.ticker_combo.currentText().strip().upper()
        if not ticker:
            EventLog.emit("WARNING", "Please select a ticker symbol")
            return
        
        self.discussion_status.setText(f"💬 Discussion Status: Starting discussion for {ticker}...")
        self.discussion_log.clear()
        self.start_discussion_btn.setEnabled(False)
        self.stop_discussion_btn.setEnabled(True)
        
        # Log discussion start
        self.log_discussion_event(f"🚀 Starting multi-agent discussion for {ticker}")
        self.log_discussion_event(f"📊 Gathering market data and analysis...")
        
        # Start discussion timer
        self.discussion_timer.start(1000)  # Update every second
        
        # Trigger AutoGen discussion (mock for now)
        self.simulate_autogen_discussion(ticker)
        
        EventLog.emit("INFO", f"Started AutoGen discussion for {ticker}")
    
    def stop_discussion(self):
        """Stop the current discussion"""
        self.discussion_status.setText("💬 Discussion Status: Stopping discussion...")
        self.discussion_timer.stop()
        self.start_discussion_btn.setEnabled(True)
        self.stop_discussion_btn.setEnabled(False)
        
        self.log_discussion_event("⏹️ Discussion stopped by user")
        
        EventLog.emit("INFO", "AutoGen discussion stopped")
    
    def simulate_autogen_discussion(self, ticker: str):
        """Simulate AutoGen multi-agent discussion (replace with real AutoGen integration)"""
        # This is a mock implementation - in real usage, integrate with actual AutoGen
        
        import threading
        import time
        import random
        
        def discussion_thread():
            time.sleep(1)
            
            # Phase 1: Individual analysis
            self.log_discussion_event(f"🤖 TechnicalAnalyst: Analyzing {ticker} charts...")
            self.update_agent_mock("TechnicalAnalyst", "Analyzing RSI, MACD, support/resistance levels", "ANALYZING", 0.0)
            time.sleep(2)
            
            self.log_discussion_event(f"📈 FundamentalAnalyst: Reviewing {ticker} financials...")
            self.update_agent_mock("FundamentalAnalyst", "Reviewing P/E ratio, earnings growth, revenue trends", "ANALYZING", 0.0)
            time.sleep(2)
            
            self.log_discussion_event(f"🛡️ RiskManager: Assessing position sizing and risk metrics...")
            self.update_agent_mock("RiskManager", "Calculating volatility, maximum position size, stop-loss levels", "ANALYZING", 0.0)
            time.sleep(2)
            
            # Phase 2: Individual decisions
            decisions = {
                "TechnicalAnalyst": {"decision": "BUY", "confidence": 0.75, "reasoning": "Strong bullish pattern with RSI oversold bounce"},
                "FundamentalAnalyst": {"decision": "HOLD", "confidence": 0.60, "reasoning": "Fair valuation but limited upside potential"},
                "RiskManager": {"decision": "BUY", "confidence": 0.65, "reasoning": "Risk-reward ratio favorable with 2% position size"},
                "MarketSentiment": {"decision": "BUY", "confidence": 0.80, "reasoning": "Strong positive sentiment and momentum"},
                "QuantAnalyst": {"decision": "SELL", "confidence": 0.70, "reasoning": "Statistical models suggest overbought conditions"}
            }
            
            for agent, decision_data in decisions.items():
                self.log_discussion_event(f"🤖 {agent}: {decision_data['decision']} - {decision_data['reasoning']}")
                self.update_agent_mock(agent, decision_data['reasoning'], decision_data['decision'], decision_data['confidence'])
                time.sleep(1)
            
            # Phase 3: Discussion and consensus building
            time.sleep(1)
            self.log_discussion_event("💬 Starting consensus building discussion...")
            
            time.sleep(2)
            self.log_discussion_event("🤖 TechnicalAnalyst: The technical setup is very strong, I maintain BUY")
            
            time.sleep(2)
            self.log_discussion_event("🔢 QuantAnalyst: But my models show overbought conditions. Risk is elevated.")
            
            time.sleep(2)
            self.log_discussion_event("🛡️ RiskManager: We can manage risk with proper position sizing. I agree with BUY but small size.")
            
            time.sleep(2)
            self.log_discussion_event("📈 FundamentalAnalyst: Considering the technical and sentiment strength, I'll revise to BUY")
            
            # Final consensus
            time.sleep(2)
            consensus = {
                "decision": "BUY",
                "confidence": 0.72,
                "reasoning": "Strong technical setup and positive sentiment outweigh short-term overbought conditions. Recommend small position size for risk management."
            }
            
            self.log_discussion_event(f"🎯 CONSENSUS REACHED: {consensus['decision']} (Confidence: {consensus['confidence']:.0%})")
            self.log_discussion_event(f"📝 Reasoning: {consensus['reasoning']}")
            
            # Update consensus display
            QTimer.singleShot(0, lambda: self.update_consensus_display(consensus))
            
            # Mark discussion as complete
            QTimer.singleShot(0, lambda: self.discussion_status.setText("💬 Discussion Status: Complete"))
            QTimer.singleShot(0, lambda: self.stop_discussion_btn.setEnabled(False))
            QTimer.singleShot(0, lambda: self.start_discussion_btn.setEnabled(True))
        
        # Start discussion in background thread
        threading.Thread(target=discussion_thread, daemon=True).start()
    
    def update_agent_mock(self, agent_name: str, opinion: str, decision: str, confidence: float):
        """Update agent display (mock implementation)"""
        def update_ui():
            if agent_name in self.agent_widgets:
                self.agent_widgets[agent_name].update_agent_state(opinion, decision, confidence, "active")
        
        QTimer.singleShot(0, update_ui)
    
    def log_discussion_event(self, message: str):
        """Log an event to the discussion log"""
        def update_log():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.discussion_log.append(f"[{timestamp}] {message}")
            
            # Auto-scroll to bottom
            scrollbar = self.discussion_log.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        
        QTimer.singleShot(0, update_log)
    
    def update_consensus_display(self, consensus: Dict[str, Any]):
        """Update the consensus tab with results"""
        self.consensus_decision_value.setText(consensus["decision"])
        
        # Set decision color
        colors = {"BUY": "#4CAF50", "SELL": "#F44336", "HOLD": "#FF9800"}
        color = colors.get(consensus["decision"], "#666")
        self.consensus_decision_value.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # Update confidence
        self.consensus_confidence_bar.setValue(int(consensus["confidence"] * 100))
        
        # Update reasoning
        self.consensus_reasoning.setPlainText(consensus["reasoning"])
        
        # Switch to consensus tab
        self.discussion_tabs.setCurrentIndex(1)
        
        # Store current consensus
        self.current_consensus = consensus
    
    def save_consensus_decision(self):
        """Save consensus decision to knowledge base"""
        if not hasattr(self, 'current_consensus'):
            EventLog.emit("WARNING", "No consensus decision to save")
            return
        
        ticker = self.ticker_combo.currentText().strip().upper()
        
        # Create AutoGen result format for storage
        autogen_result = {
            "consensus_decision": self.current_consensus["decision"],
            "consensus_confidence": self.current_consensus["confidence"],
            "consensus_reasoning": self.current_consensus["reasoning"],
            "individual_decisions": {agent: {"decision": "MOCK", "confidence": 0.7} for agent in self.agents_config.keys()},
            "discussion_summary": "Multi-agent discussion completed with consensus"
        }
        
        # Store in knowledge base (async call)
        def store_decision():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.knowledge_base.store_autogen_decision(ticker, autogen_result))
            loop.close()
        
        import threading
        threading.Thread(target=store_decision, daemon=True).start()
        
        EventLog.emit("INFO", f"Saved AutoGen consensus for {ticker} to knowledge base")
    
    def apply_consensus_decision(self):
        """Apply the consensus decision to trading system"""
        if not hasattr(self, 'current_consensus'):
            EventLog.emit("WARNING", "No consensus decision to apply")
            return
        
        ticker = self.ticker_combo.currentText().strip().upper()
        decision = self.current_consensus["decision"]
        confidence = self.current_consensus["confidence"]
        
        # Emit consensus signal
        self.consensus_reached.emit({
            "ticker": ticker,
            "decision": decision,
            "confidence": confidence,
            "reasoning": self.current_consensus["reasoning"],
            "source": "autogen_consensus"
        })
        
        EventLog.emit("SUCCESS", f"Applied AutoGen consensus: {decision} {ticker} (Confidence: {confidence:.0%})")
    
    def update_discussion_display(self):
        """Update discussion display (called by timer)"""
        # This would be called during active discussions
        pass
    
    def update_agent_status(self):
        """Update agent status indicators"""
        # Update agent status indicators based on real data
        pass
    
    def on_autogen_update(self, update_data: Dict[str, Any]):
        """Handle real-time AutoGen updates from data manager"""
        # This would receive real AutoGen updates
        pass
    
    def load_historical_decision(self, item):
        """Load a historical decision from the list"""
        # Implementation for loading historical decisions
        pass


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Test the popout
    popout = AutoGenMultiAgentPopout()
    popout.show()
    
    sys.exit(app.exec())
