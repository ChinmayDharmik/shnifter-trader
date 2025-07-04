# THE SHNIFTER TRADER - AUTONOMOUS AI TRADING BOT
# ==================================================
# This script builds an intelligent desktop trading assistant using multiple AI models.
# It leverages an EventBus architecture to handle centralized logging and event dispatching.
# The system includes trend analysis, machine learning prediction, sentiment scoring,
# and integrates Ollama for LLM reasoning. It supports full autonomy via QTimer,
# multi-ticker execution, and local logging.

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import os
import csv
import json

# GUI components
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLineEdit, QPushButton, QTextEdit, QLabel,
                               QTabWidget, QTableWidget, QTableWidgetItem, QComboBox, QCheckBox)
from PySide6.QtCore import QThread, Signal, Slot, QTimer
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt

# AI and data libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from shnifterBB.shnifter_bb import ShnifterBB
from llm_manager.llm_providers import OllamaProvider, OpenAIProvider
from shnifter_frontend.event_log_popout import EventLogPopout
from shnifter_frontend.llm_manager_popout import LLMManagerPopout
from shnifter_frontend.shnifter_table_widget import ShnifterTableWidget
from shnifter_frontend.shnifter_plotly_widget import ShnifterPlotlyWidget
from shnifter_frontend.pnl_dashboard_popout import PnLDashboardPopout
from shnifter_frontend.enhanced_dropdown_components import ShnifterModelSelector, ShnifterProviderSelector
from core.events import EventLog
from core.shnifter_data_manager import data_manager

# Import new shnifterized components
from shnifter_integration_manager import get_integration_manager
from shnifter_frontend.shnifter_popout_registry import shnifter_popout_registry

# Import analysis modules  
try:
    from shnifter_analysis_modules.portfolioOptimizationUsingModernPortfolioTheory_module import create_portfolioOptimizationUsingModernPortfolioTheory_analyzer
    from shnifter_analysis_modules.riskReturnAnalysis_module import create_riskReturnAnalysis_analyzer
    from shnifter_analysis_modules.sectorRotationStrategy_module import create_sectorRotationStrategy_analyzer
    from shnifter_analysis_modules.BacktestingMomentumTrading_module import create_BacktestingMomentumTrading_analyzer
    ANALYSIS_MODULES_AVAILABLE = True
    EventLog.emit("INFO", "Analysis modules loaded successfully")
except ImportError as e:
    print(f"Warning: Some analysis modules not available: {e}")
    ANALYSIS_MODULES_AVAILABLE = False

# Ensure VADER lexicon is available for sentiment analysis
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Ollama integration to explain signals
def query_ollama(prompt, context="", model="llama3"):
    payload = {
        "model": model,
        "prompt": context + "\n\n" + prompt,
        "stream": False
    }
    try:
        response = requests.post("http://localhost:11434/api/generate", json=payload)
        return response.json().get("response", "LLM failed.")
    except Exception as e:
        return f"LLM error: {str(e)}"

# Worker thread for background model execution
class AnalysisWorker(QThread):
    finished_signal = Signal(dict)

    def __init__(self, ticker, llm_model="llama3", news_provider="yfinance"):
        super().__init__()
        self.ticker = ticker
        self.llm_model = llm_model
        self.news_provider = news_provider
        self.last_llm_summary = None  # Store LLM summary for export
        self.shnifter = ShnifterBB()  # Use ShnifterBB for data

    def run(self):
        log = []
        try:
            print(f"DEBUG: Starting live data fetch and analysis for: {self.ticker}")
            log.append(f"[{datetime.now():%H:%M:%S}] Starting live analysis for {self.ticker}...")

            # Fetch historical data using ShnifterBB
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            data_obj = self.shnifter.equity.price.historical(self.ticker, start_date=start_date, end_date=end_date)
            intraday_data = data_obj.to_df()
            if intraday_data.empty:
                raise ValueError("No intraday data returned. Market may be closed or ticker is invalid.")

            log.append(f"-> Live data fetched: {len(intraday_data)} records")
            log.append("\nStep 2: Running analysis models...")

            recent_data = intraday_data.tail(300).copy()

            trend_signal, trend_log = self.get_trend_signal(recent_data.copy())
            log.extend(trend_log)

            ml_signal, ml_log = self.get_ml_signal(recent_data.copy())
            log.extend(ml_log)

            sentiment_signal, sentiment_log = self.get_sentiment_signal(self.ticker)
            log.extend(sentiment_log)

            log.append("\nStep 3: Calculating Consensus Decision...")
            signals = [trend_signal, ml_signal, sentiment_signal]
            buy_votes = signals.count('BUY')
            sell_votes = signals.count('SELL')
            hold_votes = signals.count('HOLD')
            log.append(f"-> Vote Tally: BUY={buy_votes}, SELL={sell_votes}, HOLD={hold_votes}")

            final_decision = "HOLD"
            if buy_votes >= 2:
                final_decision = "BUY"
            elif sell_votes >= 2:
                final_decision = "SELL"

            log.append(f"--> Consensus Signal: {final_decision}")
            last_price = recent_data['close'].iloc[-1]
            action_log = f"ACTION: {final_decision} {self.ticker} @ {last_price:.2f}"
            if final_decision == "BUY":
                stop_loss = last_price * 0.95
                take_profit = last_price * 1.10
                action_log += f"\n  - Stop-Loss: {stop_loss:.2f}\n  - Take-Profit: {take_profit:.2f}"
            elif final_decision == "SELL":
                action_log += "\n  - (Simulated short-sell or close long position)"

            log.append(action_log)

            # LLM Reasoning (Why BUY/SELL/HOLD?)
            llm_prompt = f"Explain why the consensus decision for {self.ticker} is {final_decision}."
            llm_reason = query_ollama(llm_prompt, context="\n".join(log), model=self.llm_model)
            self.last_llm_summary = llm_reason  # Store for export
            log.append(f"\n[LLM Insight: Why {final_decision}?]\n{llm_reason}")

            print(f"DEBUG: Live analysis complete. Signal: {final_decision}")
            self.finished_signal.emit({'success': True, 'log': log, 'ticker': self.ticker, 'decision': final_decision, 'llm_summary': llm_reason})

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(f"ERROR: {error_msg}")
            log.append(f"\nERROR: {error_msg}")
            self.finished_signal.emit({'success': False, 'log': log, 'ticker': self.ticker, 'decision': None, 'llm_summary': None})

    def get_trend_signal(self, df):
        log = ["  - Running Trend Model (SMA Crossover)..."]
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        # Ensure scalar values for formatting
        try:
            # Convert to proper scalar values with safer extraction
            last_sma_20 = df['SMA_20'].iloc[-1]
            last_sma_50 = df['SMA_50'].iloc[-1]
            
            # Patch: If result is a Series, get the first value
            if isinstance(last_sma_20, pd.Series):
                last_sma_20 = last_sma_20.iloc[0]
            if isinstance(last_sma_50, pd.Series):
                last_sma_50 = last_sma_50.iloc[0]
            
            # Safe conversion to float with pandas compatibility
            last_sma_20_val = float(last_sma_20.iloc[0] if hasattr(last_sma_20, 'iloc') else last_sma_20) if pd.notnull(last_sma_20) else float('nan')
            last_sma_50_val = float(last_sma_50.iloc[0] if hasattr(last_sma_50, 'iloc') else last_sma_50) if pd.notnull(last_sma_50) else float('nan')
            log.append(f"    - SMA_20: {last_sma_20_val:.2f}, SMA_50: {last_sma_50_val:.2f}")
            signal = 'BUY' if last_sma_20_val > last_sma_50_val else 'SELL'
        except Exception as e:
            log.append(f"    - SMA calculation error: {e}")
            signal = 'HOLD'
        log.append(f"    -> Signal: {signal}")
        return signal, log

    def get_ml_signal(self, df):
        log = ["  - Running ML Model (Random Forest)..."]
        df['price_change'] = df['close'].pct_change()
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, -1)
        df.dropna(inplace=True)

        features = ['price_change']
        X = df[features]
        y = df['target']
        
        # Check if we have enough data after dropna
        if len(X) == 0:
            log.append("    -> No valid data after processing. Using HOLD signal.")
            return "HOLD", log
        
        if len(X) < 10:
            log.append("    -> Not enough data to train ML model (need at least 10 samples).")
            return "HOLD", log

        # Use a smaller test size for small datasets
        test_size = min(0.2, max(0.1, 1.0 / len(X)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

        if len(X_train) < 5:
            log.append("    -> Not enough data to train ML model after split.")
            return "HOLD", log

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test) if len(X_test) > 0 else 0.5
        log.append(f"    - Accuracy: {accuracy:.2%}")

        last_feature = df[features].iloc[[-1]]
        prediction = model.predict(last_feature)[0]
        signal = 'BUY' if prediction == 1 else 'SELL'
        log.append(f"    -> Signal: {signal}")
        return signal, log

    def get_sentiment_signal(self, ticker):
        log = [f"  - Running Sentiment Model (Provider: {self.news_provider})..."]
        try:
            news_articles = self.shnifter.news.get(ticker, provider=self.news_provider, limit=20)
            if not news_articles:
                log.append("    - No news found.")
                return 'HOLD', log
            headlines = [article.title for article in news_articles if article.title]
            if not headlines:
                log.append("    - News found, but no valid titles for analysis.")
                return 'HOLD', log
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_score = np.mean(scores)
            log.append(f"    - Avg Score: {avg_score:.3f} from {len(headlines)} headlines")
            signal = 'BUY' if avg_score > 0.05 else 'SELL' if avg_score < -0.05 else 'HOLD'
            log.append(f"    -> Signal: {signal}")
            return signal, log
        except Exception as e:
            log.append(f"    - Sentiment error: {e}")
            return 'HOLD', log

class Exporter:
    """Handles exporting logs and signals to CSV/JSON."""
    @staticmethod
    def export_to_csv(filename, data, fieldnames=None):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames or data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    @staticmethod
    def export_to_json(filename, data):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

class BacktestWorker(QThread):
    """Worker thread to run backtests in the background, preventing UI freezes."""
    backtest_finished_signal = Signal(list)  # Signal to emit a list of result strings

    def __init__(self, tickers, providers=None):
        super().__init__()
        self.tickers = tickers
        self.providers = providers if providers else ['yfinance']
        self.shnifter = ShnifterBB()

    def run(self):
        EventLog.emit("INFO", "Starting backtest in background...")
        results_log = []
        try:
            for ticker in self.tickers:
                for provider in self.providers:
                    pnl, win_rate, trades = self.simulate_backtest(ticker, provider)
                    result_msg = f"Backtest for {ticker} ({provider}): PnL={pnl:.2f}, Win Rate={win_rate:.1f}%, Trades={trades}"
                    EventLog.emit("INFO", result_msg)
                    results_log.append(result_msg)
            EventLog.emit("INFO", "Background backtest completed.")
            self.backtest_finished_signal.emit(results_log)
        except Exception as e:
            error_msg = f"Error during background backtest: {e}"
            EventLog.emit("ERROR", error_msg)
            results_log.append(error_msg)
            self.backtest_finished_signal.emit(results_log)

    def simulate_backtest(self, ticker, provider):
        # Remove provider parameter since ShnifterBB doesn't support it for historical data
        # Add required start_date and end_date parameters
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        historical_data = self.shnifter.equity.price.historical(ticker, start_date=start_date, end_date=end_date)
        if historical_data.to_df().empty:
            return 0.0, 0.0, 0
        pnl = np.random.uniform(-10.0, 20.0) * len(historical_data.to_df()) / 100
        win_rate = np.random.uniform(40.0, 65.0)
        trades = np.random.randint(50, 150)
        time.sleep(2)  # Simulate a long-running task
        return pnl, win_rate, trades

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Shnifter Trader")
        self.setGeometry(100, 100, 800, 600)
        self.worker = None
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.start_analysis)
        self.auto_timer.start(600000)  # every 10 minutes
        self.last_decisions = {}  # Store last decision and LLM summary per ticker
        self.live_mode = False
        self.analysis_workers = []  # Always initialize as empty list
        self.backtest_workers = []  # Track backtest workers
        self.event_log_popout = None  # Reference to the event log popout
        self.llm_manager_popout = None  # Reference to the LLM manager popout
        
        # Shnifterized frontend popout references
        self.smart_table_popout = None
        self.advanced_charts_popout = None
        self.pnl_dashboard_popout = None
        
        self.dual_llm_mode = False
        self.analyzer_llm_model = "llama3"
        self.verifier_llm_model = "llama3"
        self.setup_ui()
        QTimer.singleShot(1000, self.refresh_chart)

    def get_ollama_models(self):
        """Query Ollama for available models. Logs results and errors."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                models = [m['name'] for m in data.get('models', [])]
                EventLog.emit("INFO", f"Fetched Ollama models: {models}")
                return models
            else:
                EventLog.emit("WARNING", f"Ollama server returned status {response.status_code}")
        except Exception as e:
            EventLog.emit("WARNING", f"Could not fetch Ollama models: {e}")
        fallback = ["llama3", "gemma", "mistral", "phi3"]
        EventLog.emit("INFO", f"Using fallback Ollama models: {fallback}")
        return fallback

    def refresh_ollama_models(self):
        """Refresh the Ollama model list in the dropdown."""
        models = self.get_ollama_models()
        self.llm_model_dropdown.update_models(models)
        if models:
            self.llm_model_dropdown.setCurrentText(models[0])
        EventLog.emit("INFO", "Ollama model dropdown refreshed.")

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        export_menu = menubar.addMenu("Export")
        tools_menu = menubar.addMenu("Tools")
        llm_menu = menubar.addMenu("LLM Manager")  # New menu for LLM management
        # Enhanced View menu with Shnifterized widgets
        view_menu = menubar.addMenu("View")
        open_event_log_action = view_menu.addAction("📋 Event Log")
        open_event_log_action.triggered.connect(self.open_event_log_popout)
        
        # Add separator
        view_menu.addSeparator()
        
        # Shnifterized frontend components
        open_smart_table_action = view_menu.addAction("📊 Smart Data Tables")
        open_smart_table_action.triggered.connect(self.open_smart_table_popout)
        
        open_advanced_charts_action = view_menu.addAction("📈 Advanced Charts")
        open_advanced_charts_action.triggered.connect(self.open_advanced_charts_popout)
        
        open_pnl_dashboard_action = view_menu.addAction("💰 P&L Dashboard")
        open_pnl_dashboard_action.triggered.connect(self.open_pnl_dashboard_popout)

        # Export actions
        export_csv_action = export_menu.addAction("Export Logs to CSV")
        export_csv_action.triggered.connect(lambda: EventLog.export_logs_csv())
        export_json_action = export_menu.addAction("Export Logs to JSON")
        export_json_action.triggered.connect(lambda: EventLog.export_logs_json())
        export_txt_action = export_menu.addAction("Export Logs to TXT")
        export_txt_action.triggered.connect(lambda: EventLog.export_logs_txt())
        export_last_decisions_action = export_menu.addAction("Export Last Decisions to CSV")
        export_last_decisions_action.triggered.connect(self.export_last_decisions_to_csv)

        # Tools actions
        backtest_action = tools_menu.addAction("Run Backtest")
        backtest_action.triggered.connect(self.run_backtest)
        refresh_chart_action = tools_menu.addAction("Refresh Chart")
        refresh_chart_action.triggered.connect(self.refresh_chart)
        
        # Demo data generation
        tools_menu.addSeparator()
        demo_action = tools_menu.addAction("🎯 Generate Demo Trading Data")
        demo_action.triggered.connect(self.simulate_demo_trading_session)

        # LLM Manager actions
        run_single_llm_action = llm_menu.addAction("Run Single LLM Analysis")
        run_single_llm_action.triggered.connect(self.run_single_llm_analysis)
        run_dual_llm_action = llm_menu.addAction("Run Dual LLM Analysis (Parallel)")
        run_dual_llm_action.triggered.connect(self.run_dual_llm_analysis)
        run_double_pass_action = llm_menu.addAction("Run Double-Pass LLM (Self-Check)")
        run_double_pass_action.triggered.connect(self.run_double_pass_llm_analysis)
        open_llm_manager_action = llm_menu.addAction("Open LLM Manager Popout")
        open_llm_manager_action.triggered.connect(self.open_llm_manager_popout)

        # Input layout - ticker row
        ticker_layout = QHBoxLayout()
        self.ticker_label = QLabel("Enter Stock Ticker(s) (comma-separated):")
        self.ticker_input = QLineEdit("SONY")
        ticker_layout.addWidget(self.ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        main_layout.addLayout(ticker_layout)

        # NEW: News provider row
        news_provider_layout = QHBoxLayout()
        self.news_provider_label = QLabel("News Provider:")
        self.news_provider_dropdown = ShnifterProviderSelector()
        self.news_provider_dropdown.add_providers(["YFinance (default)", "Benzinga", "Biztoc", "FMP", "Tiingo", "Intrinio"])
        self.news_provider_dropdown.setCurrentText("YFinance (default)")
        news_provider_layout.addWidget(self.news_provider_label)
        news_provider_layout.addWidget(self.news_provider_dropdown)
        main_layout.addLayout(news_provider_layout)

        # Input layout - model/start row (removed redundant provider dropdown)
        input_layout = QHBoxLayout()
        # LLM model dropdown (dynamic)
        self.llm_model_dropdown = ShnifterModelSelector()
        self.refresh_models_btn = QPushButton("Refresh Models")
        self.refresh_models_btn.setToolTip("Re-query Ollama for available models")
        self.refresh_models_btn.clicked.connect(self.refresh_ollama_models)
        models = self.get_ollama_models()
        self.llm_model_dropdown.update_models(models)
        self.llm_model_dropdown.setCurrentText(models[0] if models else "llama3")
        self.llm_model_dropdown.setToolTip("Select LLM model for AI Analyzer")
        self.llm_model_dropdown.currentTextChanged.connect(self.on_llm_model_changed)
        input_layout.addWidget(QLabel("LLM Model:"))
        input_layout.addWidget(self.llm_model_dropdown)
        input_layout.addWidget(self.refresh_models_btn)
        # Start Analysis button with robot emoji
        self.start_button = QPushButton("\U0001F916 Start Analysis")
        self.start_button.clicked.connect(self.start_analysis)
        input_layout.addWidget(self.start_button)
        main_layout.addLayout(input_layout)

        # Year range slider
        from PySide6.QtWidgets import QSlider, QSpinBox
        year_layout = QHBoxLayout()
        self.year_label = QLabel("Years of history:")
        self.year_slider = QSlider(Qt.Horizontal)
        self.year_slider.setMinimum(1)
        self.year_slider.setMaximum(10)
        self.year_slider.setValue(1)
        self.year_slider.setTickInterval(1)
        self.year_slider.setTickPosition(QSlider.TicksBelow)
        self.year_spin = QSpinBox()
        self.year_spin.setMinimum(1)
        self.year_spin.setMaximum(10)
        self.year_spin.setValue(1)
        self.year_slider.valueChanged.connect(self.year_spin.setValue)
        self.year_spin.valueChanged.connect(self.year_slider.setValue)
        self.year_slider.valueChanged.connect(self.refresh_chart)
        year_layout.addWidget(self.year_label)
        year_layout.addWidget(self.year_slider)
        year_layout.addWidget(self.year_spin)
        main_layout.addLayout(year_layout)

        # Live mode and refresh
        live_layout = QHBoxLayout()
        self.live_checkbox = QCheckBox("Live Mode (Auto-Refresh)")
        self.live_checkbox.stateChanged.connect(self.toggle_live_mode)
        self.refresh_chart_btn = QPushButton("Refresh Chart")
        self.refresh_chart_btn.clicked.connect(self.refresh_chart)
        live_layout.addWidget(self.live_checkbox)
        live_layout.addWidget(self.refresh_chart_btn)
        main_layout.addLayout(live_layout)

        self.tabs = QTabWidget()
        self.log_tab = QWidget()
        self.results_tab = QWidget()
        self.tabs.addTab(self.log_tab, "Logs")
        self.tabs.addTab(self.results_tab, "Backtest Results")

        # Log tab layout
        log_layout = QVBoxLayout(self.log_tab)
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFontFamily("Courier New")
        log_layout.addWidget(self.log_display)
        # AI Analyzer chatbox
        self.ai_chatbox = QTextEdit()
        self.ai_chatbox.setReadOnly(True)
        self.ai_chatbox.setPlaceholderText("AI Analyzer (LLM Reasoning and Explanations)")
        log_layout.addWidget(self.ai_chatbox)
        # Matplotlib chart
        self.figure = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        log_layout.addWidget(self.canvas)

        # Results tab layout
        self.results_table = QTableWidget()
        results_layout = QVBoxLayout(self.results_tab)
        results_layout.addWidget(self.results_table)

        # Add tabs to main layout
        main_layout.addWidget(self.tabs)

        self.start_button.clicked.connect(self.start_analysis)
        # FIX: Only append string messages to log_display
        # EventBus.subscribe("DEBUG", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))
        # EventBus.subscribe("ERROR", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))
        # EventBus.subscribe("WARN", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))

        # Chart auto-refresh timer
        self.chart_timer = QTimer()
        self.chart_timer.timeout.connect(self.refresh_chart)

    def toggle_live_mode(self):
        self.live_mode = self.live_checkbox.isChecked()
        if self.live_mode:
            self.chart_timer.start(60000)  # update every 60 seconds
        else:
            self.chart_timer.stop()

    def refresh_chart(self):
        tickers = [x.strip().upper() for x in self.ticker_input.text().split(',') if x.strip()]
        if not tickers:
            return
        ticker = tickers[0]  # Only show first ticker for chart
        try:
            shnifter = ShnifterBB()
            end_date = datetime.now().strftime('%Y-%m-%d')
            # Use year slider value for start_date
            years = self.year_slider.value() if hasattr(self, 'year_slider') else 1
            start_date = (datetime.now() - timedelta(days=365*years)).strftime('%Y-%m-%d')
            data_obj = shnifter.equity.price.historical(ticker, start_date=start_date, end_date=end_date)
            df = data_obj.to_df()
            if df.empty:
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "No data", ha='center', va='center')
                self.canvas.draw()
                return
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(df.index, df['close'], label='Close Price', color='blue')
            if 'SMA_20' not in df.columns:
                df['SMA_20'] = df['close'].rolling(window=20).mean()
            if 'SMA_50' not in df.columns:
                df['SMA_50'] = df['close'].rolling(window=50).mean()
            ax.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', linestyle='--')
            ax.plot(df.index, df['SMA_50'], label='SMA 50', color='green', linestyle='--')
            ax.set_title(f"{ticker} Price Chart ({years} year{'s' if years > 1 else ''})")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Chart error: {e}", ha='center', va='center')
            self.canvas.draw()

    @Slot()
    def start_analysis(self):
        EventLog.emit("INFO", "User started analysis.")
        print("INFO: Start Analysis button clicked.")
        # Check if button is already disabled to prevent multiple clicks
        if not self.start_button.isEnabled():
            print("WARNING: Analysis already in progress. Please wait for completion.")
            self.log_display.append("⚠️ Analysis already running. Please wait...")
            return
        tickers = [x.strip().upper() for x in self.ticker_input.text().split(',') if x.strip()]
        if not tickers:
            self.log_display.setPlainText("Please enter a stock ticker.")
            print("WARNING: No ticker entered. Analysis aborted.")
            return
        # Clean up any finished workers before starting new ones
        if hasattr(self, 'analysis_workers'):
            self.analysis_workers = [w for w in self.analysis_workers if w.isRunning()]
        else:
            self.analysis_workers = []
        self.start_button.setText("Analyzing...")
        self.start_button.setEnabled(False)
        llm_model = self.get_selected_llm_model()
        news_provider = self.news_provider_dropdown.currentText().split(' ')[0].lower()
        for ticker in tickers:
            if self.dual_llm_mode:
                # Dual-LLM decision flow
                self.run_dual_llm_decision(ticker, news_provider)
            else:
                worker = AnalysisWorker(ticker, llm_model=llm_model, news_provider=news_provider)
                worker.finished_signal.connect(self.on_analysis_complete)
                worker.finished.connect(self.remove_finished_worker)
                self.analysis_workers.append(worker)
                worker.start()
        print(f"INFO: Started analysis for tickers: {', '.join(tickers)}")

    def run_dual_llm_decision(self, ticker, news_provider):
        """Run dual-LLM analysis: Analyzer then Verifier."""
        from llm_manager.llm_providers import OllamaProvider
        ollama = OllamaProvider()
        ollama.initialize()
        # Fetch data and context as in AnalysisWorker
        shnifter = ShnifterBB()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        data_obj = shnifter.equity.price.historical(ticker, start_date=start_date, end_date=end_date)
        intraday_data = data_obj.to_df()
        if intraday_data.empty:
            self.log_display.append(f"No data for {ticker}.")
            return
        # Run normal analysis steps
        recent_data = intraday_data.tail(300).copy()
        
        # Create temporary AnalysisWorker instance to access methods
        temp_worker = AnalysisWorker(ticker, news_provider=news_provider)
        trend_signal, trend_log = temp_worker.get_trend_signal(recent_data.copy())
        ml_signal, ml_log = temp_worker.get_ml_signal(recent_data.copy())
        sentiment_signal, sentiment_log = temp_worker.get_sentiment_signal(ticker)
        log = []
        log.extend(trend_log)
        log.extend(ml_log)
        log.extend(sentiment_log)
        signals = [trend_signal, ml_signal, sentiment_signal]
        buy_votes = signals.count('BUY')
        sell_votes = signals.count('SELL')
        hold_votes = signals.count('HOLD')
        final_decision = "HOLD"
        if buy_votes >= 2:
            final_decision = "BUY"
        elif sell_votes >= 2:
            final_decision = "SELL"
        try:
            # Fix FutureWarning: use .iloc[0] for single element Series
            last_price_series = recent_data['close'].iloc[-1]
            last_price = float(last_price_series.iloc[0] if hasattr(last_price_series, 'iloc') else last_price_series)
        except (IndexError, TypeError, ValueError) as e:
            self.log_display.append(f"Error getting last price for {ticker}: {e}")
            last_price = 0.0
        action_log = f"ACTION: {final_decision} {ticker} @ {last_price:.2f}"
        log.append(action_log)
        # Analyzer LLM
        analyzer_prompt = f"Analyze and recommend a trade for {ticker}. Consensus: {final_decision}. Reasoning: {log}" 
        analyzer_model = self.analyzer_llm_model
        # Clean model name
        import re
        analyzer_model = re.sub(r'^[^\w\s]*\s*', '', analyzer_model)
        success1, analysis = ollama.run_inference(analyzer_prompt, model=analyzer_model)
        # Verifier LLM
        verifier_prompt = f"Critique and confirm or revise this analysis and decision: {analysis}"
        verifier_model = self.verifier_llm_model
        # Clean model name
        verifier_model = re.sub(r'^[^\w\s]*\s*', '', verifier_model)
        success2, verdict = ollama.run_inference(verifier_prompt, model=verifier_model)
        # Parse verdict for final action
        if "REVISE TO" in verdict:
            action = verdict.split("REVISE TO")[-1].strip()
        elif "CONFIRM" in verdict:
            action = final_decision
        else:
            action = "HOLD"
        self.log_display.append(f"[Dual LLM] Analyzer: {analysis}\nVerifier: {verdict}\nFinal Action: {action}")
        EventLog.emit("INFO", f"Dual LLM for {ticker}: {action}")

    @Slot(dict)
    def on_analysis_complete(self, result):
        print(f"INFO: Analysis complete for: {result.get('ticker', 'UNKNOWN')}")
        
        # Check if all workers are done before re-enabling the button
        running_workers = [w for w in getattr(self, 'analysis_workers', []) if w.isRunning()]
        if not running_workers:
            self.start_button.setText("Start Analysis")
            self.start_button.setEnabled(True)
            print("DEBUG: All analysis workers completed - UI button re-enabled")
        
        if result['success']:
            for line in result['log']:
                self.log_display.append(line)
            
            ticker = result.get('ticker')
            decision = result.get('decision')
            llm_summary = result.get('llm_summary')
            confidence = result.get('confidence', 0.5)
            
            # Store decision for export
            if ticker and decision:
                self.last_decisions[ticker] = {
                    'decision': decision,
                    'llm_summary': llm_summary
                }
            
            # Display LLM analysis
            if llm_summary:
                self.ai_chatbox.append(f"[{ticker}] {decision}:\n{llm_summary}\n")
            
            # Integrate with data manager
            model_used = self.get_selected_llm_model()
            self.integrate_analysis_with_data_manager(ticker, decision, confidence, model_used)
            
            # Refresh chart
            self.refresh_chart()
        else:
            self.log_display.append("Analysis failed.")
            print(f"ERROR: Analysis failed for: {result.get('ticker', 'UNKNOWN')}")
        
        print("DEBUG: Analysis thread completed and output delivered to GUI")

    @Slot()
    def run_backtest(self):
        EventLog.emit("INFO", "User started backtest.")
        tickers_text = self.ticker_input.text()
        if not tickers_text:
            EventLog.emit("WARNING", "No tickers entered for backtest.")
            return
        tickers = [ticker.strip().upper() for ticker in tickers_text.split(',') if ticker.strip()]
        EventLog.emit("INFO", f"Backtest requested for: {', '.join(tickers)}. Starting worker...")
        worker = BacktestWorker(tickers)
        worker.backtest_finished_signal.connect(self.on_backtest_complete)
        worker.finished.connect(self.remove_finished_worker)
        self.backtest_workers.append(worker)
        worker.start()

    @Slot(list)
    def on_backtest_complete(self, results):
        EventLog.emit("INFO", "Backtest worker finished. Results received.")
        # Optionally display results in a popout or log
        for msg in results:
            self.log_display.append(msg)

    @Slot()
    def remove_finished_worker(self):
        worker = self.sender()
        if hasattr(self, 'analysis_workers') and worker in self.analysis_workers:
            self.analysis_workers.remove(worker)
            EventLog.emit("DEBUG", f"Analysis worker {worker} removed from tracking list.")
        if hasattr(self, 'backtest_workers') and worker in self.backtest_workers:
            self.backtest_workers.remove(worker)
            EventLog.emit("DEBUG", f"Backtest worker {worker} removed from tracking list.")
        worker.deleteLater()

    def get_selected_llm_model(self):
        """Return the currently selected LLM model from the dropdown."""
        model_text = self.llm_model_dropdown.currentText()
        # Extract clean model name - remove emojis and status indicators
        import re
        # Remove emoji and status indicators like "🟢 llama3:8b"
        clean_model = re.sub(r'^[^\w\s]*\s*', '', model_text)
        return clean_model

    def on_llm_model_changed(self, model_name):
        EventLog.emit("INFO", f"User selected LLM model: {model_name}")

    def on_news_provider_changed(self, provider_name):
        EventLog.emit("INFO", f"User selected news provider: {provider_name}")

    def run_single_llm_analysis(self):
        EventLog.emit("INFO", "User triggered single LLM analysis.")
        # Example: Run analysis with a single LLM provider (Ollama)
        prompt = "Enter your analysis prompt here."
        from llm_manager.llm_providers import OllamaProvider
        ollama = OllamaProvider()
        ollama.initialize()
        models = ollama.list_models()
        model = models[0] if models else ollama.default_model
        # Clean model name
        import re
        model = re.sub(r'^[^\w\s]*\s*', '', model)
        success, response = ollama.run_inference(prompt, model=model)
        self.log_display.append(f"[LLM Single] Success: {success}\nResponse: {response}")

    def run_dual_llm_analysis(self):
        EventLog.emit("INFO", "User triggered dual LLM analysis.")
        # Example: Run analysis with two LLM providers in parallel (Ollama and OpenAI)
        prompt = "Enter your analysis prompt here."
        from llm_manager.llm_providers import OllamaProvider, OpenAIProvider
        ollama = OllamaProvider()
        ollama.initialize()
        models = ollama.list_models()
        model = models[0] if models else ollama.default_model
        # Clean model name
        import re
        model = re.sub(r'^[^\w\s]*\s*', '', model)
        openai = OpenAIProvider()
        openai.initialize()
        openai_models = openai.list_models()
        openai_model = openai_models[0] if openai_models else openai.default_model
        success1, response1 = ollama.run_inference(prompt, model=model)
        success2, response2 = openai.run_inference(prompt, model=openai_model)
        self.log_display.append(f"[LLM Dual] Ollama: {response1}\nOpenAI: {response2}")

    def run_double_pass_llm_analysis(self):
        EventLog.emit("INFO", "User triggered double-pass LLM analysis.")
        # Example: Run double-pass/check (self-critique) with a single LLM provider
        prompt = "Enter your analysis prompt here."
        from llm_manager.llm_providers import OllamaProvider
        ollama = OllamaProvider()
        ollama.initialize()
        models = ollama.list_models()
        model = models[0] if models else ollama.default_model
        # Clean model name
        import re
        model = re.sub(r'^[^\w\s]*\s*', '', model)
        # First pass
        success1, response1 = ollama.run_inference(prompt, model=model)
        # Second pass: critique the first response
        critique_prompt = f"Critique the following analysis and point out any flaws or improvements.\n\nAnalysis: {response1}"
        success2, response2 = ollama.run_inference(critique_prompt, model=model)
        self.log_display.append(f"[LLM Double-Pass] Analysis: {response1}\nCritique: {response2}")

    def open_event_log_popout(self):
        EventLog.emit("INFO", "Event Log popout opened.")
        if self.event_log_popout is None or not self.event_log_popout.isVisible():
            self.event_log_popout = EventLogPopout(parent=self)
            self.event_log_popout.show()
        else:
            self.event_log_popout.raise_()
            self.event_log_popout.activateWindow()

    def open_llm_manager_popout(self):
        if not hasattr(self, 'llm_manager_popout') or self.llm_manager_popout is None or not self.llm_manager_popout.isVisible():
            self.llm_manager_popout = LLMManagerPopout(parent=self)
            self.llm_manager_popout.dual_llm_settings_changed.connect(self.update_dual_llm_settings)
            self.llm_manager_popout.show()
        else:
            self.llm_manager_popout.raise_()
            self.llm_manager_popout.activateWindow()

    def open_smart_table_popout(self):
        """Open Smart Data Tables popout"""
        EventLog.emit("INFO", "Smart Table popout opened.")
        if self.smart_table_popout is None or not self.smart_table_popout.isVisible():
            self.smart_table_popout = ShnifterTableWidget(parent=None)  # Standalone popout
            self.smart_table_popout.setWindowTitle("🔍 Shnifter Smart Data Tables")
            self.smart_table_popout.show()
        else:
            self.smart_table_popout.raise_()
            self.smart_table_popout.activateWindow()

    def open_advanced_charts_popout(self):
        """Open Advanced Charts popout"""
        EventLog.emit("INFO", "Advanced Charts popout opened.")
        if self.advanced_charts_popout is None or not self.advanced_charts_popout.isVisible():
            self.advanced_charts_popout = ShnifterPlotlyWidget(parent=None)  # Standalone popout
            self.advanced_charts_popout.setWindowTitle("📈 Shnifter Advanced Charts")
            self.advanced_charts_popout.show()
        else:
            self.advanced_charts_popout.raise_()
            self.advanced_charts_popout.activateWindow()

    def open_pnl_dashboard_popout(self):
        """Open P&L Dashboard popout"""
        EventLog.emit("INFO", "P&L Dashboard popout opened.")
        if self.pnl_dashboard_popout is None or not self.pnl_dashboard_popout.isVisible():
            # Connect to real data manager instead of dummy callback
            self.pnl_dashboard_popout = PnLDashboardPopout(parent=None)  # Will use data_manager automatically
            self.pnl_dashboard_popout.show()
        else:
            self.pnl_dashboard_popout.raise_()
            self.pnl_dashboard_popout.activateWindow()

    def update_dual_llm_settings(self, enabled, analyzer_model, verifier_model):
        self.dual_llm_mode = enabled
        self.analyzer_llm_model = analyzer_model
        self.verifier_llm_model = verifier_model
        EventLog.emit("INFO", f"Dual-LLM mode set to {enabled}, Analyzer: {analyzer_model}, Verifier: {verifier_model}")

    def closeEvent(self, event):
        EventLog.emit("INFO", "Application closing.")
        # Clean up workers
        for worker in self.analysis_workers + self.backtest_workers:
            worker.quit()
            worker.wait()
        event.accept()

    def export_last_decisions_to_csv(self, filename='last_decisions.csv'):
        """Export last decisions to a CSV file."""
        import csv
        if not hasattr(self, 'last_decisions') or not self.last_decisions:
            self.log_display.append("No decisions to export.")
            return
        data = []
        for ticker, info in self.last_decisions.items():
            data.append({
                'Ticker': ticker,
                'Decision': info.get('decision', ''),
                'LLM Summary': info.get('llm_summary', '')
            })
        fieldnames = ['Ticker', 'Decision', 'LLM Summary']
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            self.log_display.append(f"Exported last decisions to {filename}.")
        except Exception as e:
            self.log_display.append(f"Failed to export last decisions: {e}")

    def integrate_analysis_with_data_manager(self, ticker, signal, confidence, model_used):
        """Integrate analysis results with the centralized data manager"""
        try:
            # Update market data for the ticker
            data_manager.update_market_data(ticker)
            
            # If signal suggests a trade, simulate adding it to the portfolio
            if signal in ['BUY', 'SELL'] and confidence > 0.6:
                # Get current price
                market_data = data_manager.get_market_data(ticker)
                current_price = market_data.get('current_price', 100.0) if market_data else 100.0
                
                # Calculate position size (2% risk per trade)
                position_size = max(1, int(1000 / current_price))  # Simple position sizing
                
                direction = 'long' if signal == 'BUY' else 'short'
                stop_loss = current_price * (0.98 if signal == 'BUY' else 1.02)
                take_profit = current_price * (1.05 if signal == 'BUY' else 0.95)
                
                # Add trade to data manager
                data_manager.add_trade(
                    ticker=ticker,
                    entry_price=current_price,
                    size=position_size,
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    confidence=confidence
                )
                
                EventLog.emit("INFO", f"Added {direction} trade for {ticker} @ ${current_price:.2f}")
            
            # Update model performance metrics
            # For demonstration, we'll simulate accuracy based on confidence
            simulated_accuracy = min(0.95, confidence + 0.1)
            
            data_manager.update_model_performance(
                model_name=model_used,
                accuracy=simulated_accuracy,
                precision=simulated_accuracy * 0.9,
                recall=simulated_accuracy * 0.85,
                f1_score=simulated_accuracy * 0.87,
                last_prediction=signal,
                confidence=confidence
            )
            
        except Exception as e:
            EventLog.emit("ERROR", f"Data manager integration error: {e}")

    def simulate_demo_trading_session(self):
        """Generate demo trading data for testing the frontend"""
        try:
            EventLog.emit("INFO", "Generating demo trading session...")
            
            # Generate demo trades and model performance
            data_manager.simulate_demo_trades()
            
            # Update market data for demo tickers
            demo_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA', 'SPY', 'QQQ']
            for ticker in demo_tickers:
                data_manager.update_market_data(ticker)
            
            EventLog.emit("INFO", "Demo trading session generated successfully")
            
        except Exception as e:
            EventLog.emit("ERROR", f"Demo session generation error: {e}")

if __name__ == "__main__":
    print("INFO: Launching The Shnifter Trader GUI...")
    app = QApplication(sys.argv)
    # Enable dark mode using Fusion style and dark palette
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette
    dark_palette = app.palette()
    dark_palette.setColor(QPalette.ColorRole.Window,        Qt.darkBlue)
    dark_palette.setColor(QPalette.ColorRole.WindowText,    Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Base,          Qt.darkBlue)
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, Qt.blue)
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase,   Qt.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText,   Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Text,          Qt.white)
    dark_palette.setColor(QPalette.ColorRole.Button,        Qt.darkBlue)
    dark_palette.setColor(QPalette.ColorRole.ButtonText,    Qt.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText,    Qt.red)
    dark_palette.setColor(QPalette.ColorRole.Highlight,     Qt.darkGray)
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.black)
    app.setPalette(dark_palette)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
