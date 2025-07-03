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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt

# EventBus system for centralized logging and messaging
class EventBus:
    _subscribers = {}
    log_level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    min_log_level = "DEBUG"

    @classmethod
    def subscribe(cls, event_type, handler):
        cls._subscribers.setdefault(event_type, []).append(handler)

    @classmethod
    def publish(cls, event_type, payload):
        # Only publish if event_type >= min_log_level
        if event_type in cls.log_level_order:
            min_idx = cls.log_level_order.index(cls.min_log_level)
            evt_idx = cls.log_level_order.index(event_type)
            if evt_idx < min_idx:
                return
        for handler in cls._subscribers.get(event_type, []):
            handler(payload)

    @classmethod
    def set_min_log_level(cls, level):
        if level in cls.log_level_order:
            cls.min_log_level = level

# EventLog wraps EventBus and prints logs with timestamps
class EventLog:
    logs = []  # Store all log events as dicts for export

    @staticmethod
    def emit(level, message, extra=None):
        event = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'level': level,
            'message': message,
        }
        if extra:
            event.update(extra)
        EventLog.logs.append(event)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        EventBus.publish(level, event)  # Pass structured event
        with open("shnifter_log.txt", "a") as f:
            f.write(json.dumps(event) + "\n")

    @staticmethod
    def export_logs_txt(filename='shnifter_log.txt'):
        with open(filename, 'w', encoding='utf-8') as f:
            for event in EventLog.logs:
                f.write(f"[{event['timestamp']}] [{event['level']}] {event['message']}\n")

    @staticmethod
    def export_logs_json(filename='shnifter_log.json'):
        if EventLog.logs:
            Exporter.export_to_json(filename, EventLog.logs)

    @staticmethod
    def export_logs_csv(filename='shnifter_log.csv'):
        if EventLog.logs:
            Exporter.export_to_csv(filename, EventLog.logs)

# AI and data libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from shnifter_bb import ShnifterBB

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

    def __init__(self, ticker, llm_model="llama3"):
        super().__init__()
        self.ticker = ticker
        self.llm_model = llm_model
        self.last_llm_summary = None  # Store LLM summary for export
        self.shnifter = ShnifterBB()  # Use ShnifterBB for data

    def run(self):
        log = []
        try:
            EventLog.emit("DEBUG", f"Starting live data fetch and analysis for: {self.ticker}")
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

            EventLog.emit("DEBUG", f"Live analysis complete. Signal: {final_decision}")
            self.finished_signal.emit({'success': True, 'log': log, 'ticker': self.ticker, 'decision': final_decision, 'llm_summary': llm_reason})

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            EventLog.emit("ERROR", error_msg)
            log.append(f"\nERROR: {error_msg}")
            self.finished_signal.emit({'success': False, 'log': log, 'ticker': self.ticker, 'decision': None, 'llm_summary': None})

    def get_trend_signal(self, df):
        log = ["  - Running Trend Model (SMA Crossover)..."]
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        # Ensure scalar values for formatting
        try:
            last_sma_20 = df['SMA_20'].iloc[-1]
            last_sma_50 = df['SMA_50'].iloc[-1]
            # Patch: If result is a Series, get the first value
            if isinstance(last_sma_20, pd.Series):
                last_sma_20 = last_sma_20.iloc[0]
            if isinstance(last_sma_50, pd.Series):
                last_sma_50 = last_sma_50.iloc[0]
            last_sma_20_val = float(last_sma_20) if pd.notnull(last_sma_20) else float('nan')
            last_sma_50_val = float(last_sma_50) if pd.notnull(last_sma_50) else float('nan')
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
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        if len(X_train) < 50:
            log.append("    -> Not enough data to train ML model.")
            return "HOLD", log

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        log.append(f"    - Accuracy: {accuracy:.2%}")

        last_feature = df[features].iloc[[-1]]
        prediction = model.predict(last_feature)[0]
        signal = 'BUY' if prediction == 1 else 'SELL'
        log.append(f"    -> Signal: {signal}")
        return signal, log

    def get_sentiment_signal(self, ticker):
        log = ["  - Running Sentiment Model (News Headlines)..."]
        try:
            # Use ShnifterBB for news
            news_df = self.shnifter._providers['yfinance'].get_news(ticker, limit=20)
            if news_df.empty:
                log.append("    - No news found.")
                return 'HOLD', log

            headlines = news_df['title'].dropna().tolist()
            scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
            avg_score = np.mean(scores)
            log.append(f"    - Avg Score: {avg_score:.3f}")
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
        self.setup_ui()
        # Auto-refresh chart 1 second after window opens
        QTimer.singleShot(1000, self.refresh_chart)

    def get_ollama_models(self):
        """Query Ollama for available models."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                data = response.json()
                # Ollama returns models under 'models', each with a 'name'
                return [m['name'] for m in data.get('models', [])]
        except Exception as e:
            EventLog.emit("WARN", f"Could not fetch Ollama models: {e}")
        # Fallback to hardcoded list if Ollama is not running
        return ["llama3", "gemma", "mistral", "phi3"]

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        export_menu = menubar.addMenu("Export")
        tools_menu = menubar.addMenu("Tools")

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

        # Input layout - ticker row
        ticker_layout = QHBoxLayout()
        self.ticker_label = QLabel("Enter Stock Ticker(s) (comma-separated):")
        self.ticker_input = QLineEdit("SONY")
        ticker_layout.addWidget(self.ticker_label)
        ticker_layout.addWidget(self.ticker_input)
        main_layout.addLayout(ticker_layout)

        # Input layout - model/provider/start row
        input_layout = QHBoxLayout()
        # Provider dropdown
        self.provider_dropdown = QComboBox()
        self.provider_dropdown.addItems(["YFinance (default)"])
        self.provider_dropdown.setEnabled(False)  # Only one provider for now
        # LLM model dropdown (dynamic)
        self.llm_model_dropdown = QComboBox()
        models = self.get_ollama_models()
        self.llm_model_dropdown.addItems(models)
        self.llm_model_dropdown.setCurrentText(models[0] if models else "llama3")
        self.llm_model_dropdown.setToolTip("Select LLM model for AI Analyzer")
        input_layout.addWidget(QLabel("LLM Model:"))
        input_layout.addWidget(self.llm_model_dropdown)
        input_layout.addWidget(QLabel("Provider:"))
        input_layout.addWidget(self.provider_dropdown)
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
        EventBus.subscribe("DEBUG", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))
        EventBus.subscribe("ERROR", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))
        EventBus.subscribe("WARN", lambda event: self.log_display.append(f"[{event['level']}] {event['message']}"))

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
        EventLog.emit("INFO", "Start Analysis button clicked.")
        
        # Prevent multiple simultaneous analyses
        if hasattr(self, 'analysis_workers') and any(w.isRunning() for w in self.analysis_workers):
            EventLog.emit("WARNING", "Analysis already in progress. Please wait for completion.")
            self.log_display.append("⚠️ Analysis already running. Please wait...")
            return
            
        tickers = [x.strip().upper() for x in self.ticker_input.text().split(',') if x.strip()]
        if not tickers:
            self.log_display.setPlainText("Please enter a stock ticker.")
            EventLog.emit("WARNING", "No ticker entered. Analysis aborted.")
            return

        # Clean up any finished workers before starting new ones
        if hasattr(self, 'analysis_workers'):
            self.analysis_workers = [w for w in self.analysis_workers if w.isRunning()]
        else:
            self.analysis_workers = []
            
        self.start_button.setText("Analyzing...")
        self.start_button.setEnabled(False)
        llm_model = self.get_selected_llm_model()
        EventLog.emit("INFO", f"Selected LLM model: {llm_model}")
        for ticker in tickers:
            EventLog.emit("DEBUG", f"User triggered analysis for: {ticker}")
            worker = AnalysisWorker(ticker, llm_model=llm_model)
            worker.finished_signal.connect(self.on_analysis_complete)
            worker.finished.connect(self.remove_finished_worker)
            self.analysis_workers.append(worker)
            worker.start()
        EventLog.emit("INFO", f"Started analysis for tickers: {', '.join(tickers)}")

    @Slot(dict)
    def on_analysis_complete(self, result):
        EventLog.emit("INFO", f"Analysis complete for: {result.get('ticker', 'UNKNOWN')}")
        self.start_button.setText("Start Analysis")
        self.start_button.setEnabled(True)
        if result['success']:
            for line in result['log']:
                self.log_display.append(line)
            ticker = result.get('ticker')
            decision = result.get('decision')
            llm_summary = result.get('llm_summary')
            if ticker and decision:
                self.last_decisions[ticker] = {
                    'decision': decision,
                    'llm_summary': llm_summary
                }
            if llm_summary:
                self.ai_chatbox.append(f"[{ticker}] {decision}:\n{llm_summary}\n")
            self.refresh_chart()
        else:
            self.log_display.append("Analysis failed.")
            EventLog.emit("ERROR", f"Analysis failed for: {result.get('ticker', 'UNKNOWN')}")
        EventLog.emit("DEBUG", "Analysis thread completed and output delivered to GUI")

    def run_backtest(self, providers=None):
        EventLog.emit("INFO", "Backtest started.")
        tickers = [x.strip().upper() for x in self.ticker_input.text().split(',') if x.strip()]
        # Defensive: ensure providers is a list
        if not isinstance(providers, list):
            EventLog.emit("WARNING", f"Invalid providers argument ({providers}), using default provider list.")
            providers = ["YFinance (default)"]
        results = []
        for provider in providers:
            for ticker in tickers:
                pnl, win_rate, trades = self.simple_backtest(ticker, provider)
                EventLog.emit("INFO", f"Backtest for {ticker} ({provider}): PnL={pnl:.2f}, Win Rate={win_rate:.1%}, Trades={trades}")
                results.append({
                    'Ticker': ticker,
                    'Provider': provider,
                    'PnL': f"{pnl:.2f}",
                    'Win Rate': f"{win_rate:.1%}",
                    'Trades': trades
                })
        self.display_backtest_results(results)
        EventLog.emit("INFO", "Backtest completed.")

    def simple_backtest(self, ticker, provider):
        # Simulate a simple backtest using random data and your models
        import numpy as np
        import pandas as pd
        df = pd.DataFrame({'close': np.linspace(100, 120, 100) + np.random.normal(0, 1, 100)})
        worker = AnalysisWorker(ticker)
        signals = []
        for i in range(50, len(df)):
            sub_df = df.iloc[:i+1].copy()
            trend_signal, _ = worker.get_trend_signal(sub_df)
            ml_signal, _ = worker.get_ml_signal(sub_df)
            sentiment_signal, _ = worker.get_sentiment_signal(ticker)
            votes = [trend_signal, ml_signal, sentiment_signal]
            if votes.count('BUY') >= 2:
                signals.append('BUY')
            elif votes.count('SELL') >= 2:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        # Calculate PnL and win rate (dummy logic)
        pnl = np.random.uniform(-10, 20)
        win_rate = np.random.uniform(0.4, 0.7)
        trades = len([s for s in signals if s != 'HOLD'])
        return pnl, win_rate, trades

    def display_backtest_results(self, results):
        # Support multiple providers in results
        self.results_table.setRowCount(len(results))
        if results and 'Provider' in results[0]:
            self.results_table.setColumnCount(5)
            self.results_table.setHorizontalHeaderLabels(["Ticker", "Provider", "PnL", "Win Rate", "Trades"])
        else:
            self.results_table.setColumnCount(4)
            self.results_table.setHorizontalHeaderLabels(["Ticker", "PnL", "Win Rate", "Trades"])
        for row, res in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(res['Ticker']))
            col = 1
            if 'Provider' in res:
                self.results_table.setItem(row, col, QTableWidgetItem(res['Provider']))
                col += 1
            self.results_table.setItem(row, col, QTableWidgetItem(res['PnL']))
            self.results_table.setItem(row, col+1, QTableWidgetItem(res['Win Rate']))
            self.results_table.setItem(row, col+2, QTableWidgetItem(str(res['Trades'])))

    def set_update_interval(self, seconds):
        # Set the auto-timer interval for analysis updates (1-120 seconds)
        seconds = max(1, min(120, int(seconds)))
        self.auto_timer.setInterval(seconds * 1000)
        EventLog.emit("INFO", f"Auto-update interval set to {seconds} seconds.")

    def export_last_decisions_to_csv(self, filename='last_decisions.csv'):
        EventLog.emit("INFO", "Exporting last decisions to CSV.")
        if not hasattr(self, 'last_decisions') or not self.last_decisions:
            self.log_display.append("No decisions to export.")
            EventLog.emit("WARNING", "No decisions to export.")
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
            EventLog.emit("INFO", f"Exported last decisions to {filename}.")
        except Exception as e:
            self.log_display.append(f"Failed to export last decisions: {e}")
            EventLog.emit("ERROR", f"Failed to export last decisions: {e}")

    def closeEvent(self, event):
        EventLog.emit("INFO", "Application closing. Attempting to stop all analysis threads.")
        if hasattr(self, 'analysis_workers'):
            for worker in self.analysis_workers:
                if worker.isRunning():
                    EventLog.emit("DEBUG", f"Stopping thread for worker: {getattr(worker, 'ticker', 'unknown')}")
                    worker.quit()
                    # Add timeout to prevent hanging
                    if not worker.wait(3000):  # Wait max 3 seconds
                        EventLog.emit("WARNING", f"Thread did not stop gracefully, terminating: {getattr(worker, 'ticker', 'unknown')}")
                        worker.terminate()
                        worker.wait(1000)  # Wait 1 more second for termination
        EventLog.emit("INFO", "All analysis threads stopped. Application closed.")
        event.accept()

    def get_selected_llm_model(self):
        return self.llm_model_dropdown.currentText()

    def remove_finished_worker(self):
        # Remove finished workers from the list and clean up properly
        if hasattr(self, 'analysis_workers'):
            finished_workers = [w for w in self.analysis_workers if not w.isRunning()]
            self.analysis_workers = [w for w in self.analysis_workers if w.isRunning()]
            
            # Clean up finished workers
            for worker in finished_workers:
                try:
                    worker.deleteLater()
                except:
                    pass
            
            # Re-enable button if no workers are running
            if not self.analysis_workers:
                self.start_button.setText("Start Analysis")
                self.start_button.setEnabled(True)
                EventLog.emit("DEBUG", "All analysis workers completed. Button re-enabled.")

if __name__ == "__main__":
    EventLog.emit("INFO", "Launching The Shnifter Trader GUI...")
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
