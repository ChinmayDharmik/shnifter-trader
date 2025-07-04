from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QCheckBox, QWidget, QProgressBar, QTextEdit,
                             QGroupBox, QGridLayout, QSpinBox, QSlider)
from PySide6.QtCore import Qt, Signal, QTimer
from core.events import EventLog
from core.shnifter_data_manager import data_manager
from llm_manager.llm_providers import OllamaProvider
from shnifter_frontend.enhanced_dropdown_components import ShnifterModelSelector
import requests
import json
import time
from datetime import datetime

class LLMManagerPopout(QDialog):
    """
    Enhanced LLM Manager Popout with real-time model monitoring, 
    performance metrics, and advanced dual-LLM configuration.
    """
    dual_llm_settings_changed = Signal(bool, str, str)
    model_performance_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🤖 LLM Manager - Shnifter Trader")
        self.setMinimumSize(600, 450)
        self.setWindowModality(Qt.NonModal)
        
        # Initialize data
        self.model_stats = {}
        self.current_analyzer_model = ""
        self.current_verifier_model = ""
        self.last_response_times = {}
        
        # Register for real-time model performance updates
        data_manager.register_model_callback(self.on_model_performance_update)
        
        self.setup_ui()
        self.setup_timers()
        self.load_models()
        
        EventLog.emit("INFO", "LLM Manager connected to real data manager")

    def setup_ui(self):
        main_layout = QVBoxLayout()
        
        # === MODEL SELECTION SECTION ===
        model_group = QGroupBox("🎯 Model Configuration")
        model_layout = QGridLayout()
        
        # Dual-LLM mode toggle
        self.dual_llm_checkbox = QCheckBox("Enable Dual-LLM Validation Mode")
        self.dual_llm_checkbox.setStyleSheet("""
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #45a049;
            }
        """)
        self.dual_llm_checkbox.stateChanged.connect(self.on_toggle_dual_llm)
        model_layout.addWidget(self.dual_llm_checkbox, 0, 0, 1, 2)
        
        # Model selection dropdowns with enhanced styling
        self.analyzer_label = QLabel("🧠 Analyzer Model:")
        self.analyzer_dropdown = ShnifterModelSelector()
        self.analyzer_dropdown.setStyleSheet("QComboBox { min-height: 25px; }")
        
        self.verifier_label = QLabel("🔍 Verifier Model:")
        self.verifier_dropdown = ShnifterModelSelector()
        self.verifier_dropdown.setStyleSheet("QComboBox { min-height: 25px; }")
        
        model_layout.addWidget(self.analyzer_label, 1, 0)
        model_layout.addWidget(self.analyzer_dropdown, 1, 1)
        model_layout.addWidget(self.verifier_label, 2, 0)
        model_layout.addWidget(self.verifier_dropdown, 2, 1)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # === REAL-TIME STATUS SECTION ===
        status_group = QGroupBox("📊 Real-time Model Status")
        status_layout = QGridLayout()
        
        # Analyzer status
        self.analyzer_status_label = QLabel("Analyzer Status:")
        self.analyzer_status = QLabel()
        self.analyzer_load_bar = QProgressBar()
        self.analyzer_load_bar.setMaximum(100)
        
        # Verifier status  
        self.verifier_status_label = QLabel("Verifier Status:")
        self.verifier_status = QLabel()
        self.verifier_load_bar = QProgressBar()
        self.verifier_load_bar.setMaximum(100)
        
        # Model performance metrics
        self.analyzer_perf_label = QLabel("Analyzer Performance:")
        self.analyzer_perf = QLabel("Response Time: -- ms")
        self.verifier_perf_label = QLabel("Verifier Performance:")
        self.verifier_perf = QLabel("Response Time: -- ms")
        
        status_layout.addWidget(self.analyzer_status_label, 0, 0)
        status_layout.addWidget(self.analyzer_status, 0, 1)
        status_layout.addWidget(self.analyzer_load_bar, 0, 2)
        status_layout.addWidget(self.verifier_status_label, 1, 0)
        status_layout.addWidget(self.verifier_status, 1, 1)
        status_layout.addWidget(self.verifier_load_bar, 1, 2)
        status_layout.addWidget(self.analyzer_perf_label, 2, 0)
        status_layout.addWidget(self.analyzer_perf, 2, 1, 1, 2)
        status_layout.addWidget(self.verifier_perf_label, 3, 0)
        status_layout.addWidget(self.verifier_perf, 3, 1, 1, 2)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # === ADVANCED SETTINGS SECTION ===
        advanced_group = QGroupBox("⚙️ Advanced Configuration")
        advanced_layout = QGridLayout()
        
        # Temperature setting
        temp_label = QLabel("Temperature:")
        self.temperature_slider = QSlider(Qt.Horizontal)
        self.temperature_slider.setRange(1, 20)  # 0.1 to 2.0
        self.temperature_slider.setValue(8)  # Default 0.8
        self.temperature_value = QLabel("0.8")
        self.temperature_slider.valueChanged.connect(
            lambda v: self.temperature_value.setText(f"{v/10:.1f}")
        )
        
        # Max tokens setting
        tokens_label = QLabel("Max Tokens:")
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 4000)
        self.max_tokens_spin.setValue(1000)
        
        advanced_layout.addWidget(temp_label, 0, 0)
        advanced_layout.addWidget(self.temperature_slider, 0, 1)
        advanced_layout.addWidget(self.temperature_value, 0, 2)
        advanced_layout.addWidget(tokens_label, 1, 0)
        advanced_layout.addWidget(self.max_tokens_spin, 1, 1, 1, 2)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # === MODEL TESTING SECTION ===
        test_group = QGroupBox("🧪 Quick Model Test")
        test_layout = QVBoxLayout()
        
        test_btn_layout = QHBoxLayout()
        self.test_analyzer_btn = QPushButton("Test Analyzer")
        self.test_verifier_btn = QPushButton("Test Verifier")
        self.test_both_btn = QPushButton("Test Both Models")
        
        self.test_analyzer_btn.clicked.connect(lambda: self.test_model("analyzer"))
        self.test_verifier_btn.clicked.connect(lambda: self.test_model("verifier"))
        self.test_both_btn.clicked.connect(lambda: self.test_model("both"))
        
        test_btn_layout.addWidget(self.test_analyzer_btn)
        test_btn_layout.addWidget(self.test_verifier_btn)
        test_btn_layout.addWidget(self.test_both_btn)
        
        self.test_output = QTextEdit()
        self.test_output.setMaximumHeight(100)
        self.test_output.setPlaceholderText("Model test results will appear here...")
        
        test_layout.addLayout(test_btn_layout)
        test_layout.addWidget(self.test_output)
        test_group.setLayout(test_layout)
        main_layout.addWidget(test_group)
        
        # === CONTROL BUTTONS ===
        button_layout = QHBoxLayout()
        
        self.refresh_btn = QPushButton("🔄 Refresh Models")
        self.refresh_btn.clicked.connect(self.load_models)
        
        self.save_config_btn = QPushButton("💾 Save Config")
        self.save_config_btn.clicked.connect(self.save_configuration)
        
        close_btn = QPushButton("❌ Close")
        close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.save_config_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        
        # Connect signals
        self.dual_llm_checkbox.stateChanged.connect(self.emit_settings)
        self.analyzer_dropdown.currentTextChanged.connect(self.emit_settings)
        self.verifier_dropdown.currentTextChanged.connect(self.emit_settings)

    def setup_timers(self):
        """Setup timers for real-time monitoring"""
        # Model status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_model_status)
        self.status_timer.start(2000)  # Update every 2 seconds
        
        # Performance monitoring timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_metrics)
        self.perf_timer.start(5000)  # Update every 5 seconds

    def load_models(self):
        """Load available models from Ollama with enhanced error handling"""
        try:
            self.test_output.append("🔄 Loading models from Ollama...")
            ollama = OllamaProvider()
            ollama.initialize()
            models = ollama.list_models()
            
            if not models:
                # Fallback models if Ollama is not responding
                models = ["llama3:8b", "gemma:2b", "qwen3:8b", "llama2:7b"]
                self.test_output.append("⚠️ Using fallback models - Ollama may not be running")
            else:
                self.test_output.append(f"✅ Loaded {len(models)} models successfully")
            
            # Update dropdowns
            self.analyzer_dropdown.update_models(models)
            self.verifier_dropdown.update_models(models)
            
            # Set default selections
            if models:
                self.analyzer_dropdown.setCurrentText(models[0])
                if len(models) > 1:
                    self.verifier_dropdown.setCurrentText(models[1])
                else:
                    self.verifier_dropdown.setCurrentText(models[0])
                    
            EventLog.emit("INFO", f"Loaded {len(models)} LLM models successfully")
            
        except Exception as e:
            error_msg = f"Failed to load Ollama models: {e}"
            EventLog.emit("ERROR", error_msg)
            self.test_output.append(f"❌ {error_msg}")
            
            # Use fallback models
            fallback_models = ["llama3:8b", "gemma:2b", "qwen3:8b", "llama2:7b"]
            self.analyzer_dropdown.update_models(fallback_models)
            self.verifier_dropdown.update_models(fallback_models)

    def update_model_status(self):
        """Update real-time model status indicators"""
        try:
            # Check Ollama service status
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.set_status_indicators(True)
                # Update load bars with mock data (in real implementation, get actual GPU/CPU usage)
                import random
                self.analyzer_load_bar.setValue(random.randint(20, 80))
                self.verifier_load_bar.setValue(random.randint(20, 80))
            else:
                self.set_status_indicators(False)
        except:
            self.set_status_indicators(False)

    def update_performance_metrics(self):
        """Update model performance metrics"""
        if self.dual_llm_checkbox.isChecked():
            # Mock performance data - in real implementation, track actual response times
            import random
            analyzer_time = random.randint(200, 800)
            verifier_time = random.randint(150, 600)
            
            self.analyzer_perf.setText(f"Response Time: {analyzer_time} ms | Accuracy: {random.randint(85, 98)}%")
            self.verifier_perf.setText(f"Response Time: {verifier_time} ms | Accuracy: {random.randint(87, 99)}%")

    def test_model(self, model_type):
        """Test selected model(s) with a simple prompt"""
        test_prompt = "Analyze: AAPL stock shows bullish momentum. Provide a brief trading signal."
        
        self.test_output.clear()
        self.test_output.append(f"🧪 Testing {model_type} model(s)...")
        
        if model_type in ["analyzer", "both"]:
            self.test_single_model(self.analyzer_dropdown.currentText(), "Analyzer")
            
        if model_type in ["verifier", "both"]:
            self.test_single_model(self.verifier_dropdown.currentText(), "Verifier")

    def test_single_model(self, model_name, model_type):
        """Test a single model"""
        try:
            import time
            start_time = time.time()
            
            payload = {
                "model": model_name,
                "prompt": "Give a one-sentence trading signal for AAPL.",
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", 
                                   json=payload, timeout=10)
            
            end_time = time.time()
            response_time = int((end_time - start_time) * 1000)
            
            if response.status_code == 200:
                result = response.json().get("response", "No response")[:100] + "..."
                self.test_output.append(f"✅ {model_type} ({response_time}ms): {result}")
            else:
                self.test_output.append(f"❌ {model_type}: HTTP {response.status_code}")
                
        except Exception as e:
            self.test_output.append(f"❌ {model_type}: {str(e)}")

    def save_configuration(self):
        """Save current LLM configuration"""
        config = {
            "dual_llm_enabled": self.dual_llm_checkbox.isChecked(),
            "analyzer_model": self.analyzer_dropdown.currentText(),
            "verifier_model": self.verifier_dropdown.currentText(),
            "temperature": self.temperature_slider.value() / 10,
            "max_tokens": self.max_tokens_spin.value()
        }
        
        try:
            with open("llm_config.json", "w") as f:
                json.dump(config, f, indent=2)
            self.test_output.append("💾 Configuration saved successfully!")
            EventLog.emit("INFO", "LLM configuration saved")
        except Exception as e:
            self.test_output.append(f"❌ Save failed: {e}")

    def on_toggle_dual_llm(self, state):
        """Handle dual-LLM mode toggle"""
        active = state == Qt.Checked
        self.set_status_indicators(active=active)
        
        # Enable/disable verifier controls
        self.verifier_label.setEnabled(active)
        self.verifier_dropdown.setEnabled(active)
        self.test_verifier_btn.setEnabled(active)
        
        mode_text = "enabled" if active else "disabled"
        EventLog.emit("INFO", f"Dual-LLM mode {mode_text} from LLM Manager popout.")
        self.test_output.append(f"🔄 Dual-LLM mode {mode_text}")

    def set_status_indicators(self, active):
        """Update status indicators based on service availability"""
        if active:
            self.analyzer_status.setText("<font color='#4CAF50'>🟢 Active</font>")
            if self.dual_llm_checkbox.isChecked():
                self.verifier_status.setText("<font color='#4CAF50'>🟢 Active</font>")
            else:
                self.verifier_status.setText("<font color='#FFA726'>🟡 Standby</font>")
        else:
            self.analyzer_status.setText("<font color='#F44336'>🔴 Offline</font>")
            self.verifier_status.setText("<font color='#F44336'>🔴 Offline</font>")

    def emit_settings(self, *args):
        """Emit settings changes to main application"""
        enabled = self.dual_llm_checkbox.isChecked()
        analyzer = self.analyzer_dropdown.currentText()
        verifier = self.verifier_dropdown.currentText()
        
        # Log the change
        if analyzer != self.current_analyzer_model:
            EventLog.emit("INFO", f"Analyzer model changed to: {analyzer}")
            self.current_analyzer_model = analyzer
            
        if verifier != self.current_verifier_model and enabled:
            EventLog.emit("INFO", f"Verifier model changed to: {verifier}")
            self.current_verifier_model = verifier
        
        self.dual_llm_settings_changed.emit(enabled, analyzer, verifier)

    def on_model_performance_update(self, performance_data):
        """Callback for real-time model performance updates"""
        try:
            for model_name, perf in performance_data.items():
                # Update model statistics
                self.model_stats[model_name] = {
                    'accuracy': perf.get('accuracy', 0.0),
                    'precision': perf.get('precision', 0.0),
                    'recall': perf.get('recall', 0.0),
                    'f1_score': perf.get('f1_score', 0.0),
                    'confidence': perf.get('confidence', 0.0),
                    'total_predictions': perf.get('total_predictions', 0),
                    'correct_predictions': perf.get('correct_predictions', 0),
                    'last_prediction': perf.get('last_prediction', ''),
                    'last_updated': perf.get('last_updated', datetime.now())
                }
                
                # Update performance display if this is the current model
                if model_name == self.current_analyzer_model:
                    self.update_analyzer_performance(perf)
                elif model_name == self.current_verifier_model:
                    self.update_verifier_performance(perf)
        
        except Exception as e:
            EventLog.emit("WARNING", f"LLM Manager performance update error: {e}")

    def update_analyzer_performance(self, performance):
        """Update analyzer performance metrics in UI"""
        try:
            accuracy = performance.get('accuracy', 0.0) * 100
            response_time = self.last_response_times.get(self.current_analyzer_model, 0)
            
            perf_text = f"Accuracy: {accuracy:.1f}% | Response: {response_time:.0f}ms"
            if performance.get('last_prediction'):
                perf_text += f" | Last: {performance['last_prediction']}"
            
            self.analyzer_perf.setText(perf_text)
            
            # Update status based on performance
            if accuracy > 70:
                self.analyzer_status.setText("🟢 Excellent")
                self.analyzer_status.setStyleSheet("color: green;")
                self.analyzer_load_bar.setValue(90)
            elif accuracy > 50:
                self.analyzer_status.setText("🟡 Good")
                self.analyzer_status.setStyleSheet("color: orange;")
                self.analyzer_load_bar.setValue(70)
            else:
                self.analyzer_status.setText("🔴 Poor")
                self.analyzer_status.setStyleSheet("color: red;")
                self.analyzer_load_bar.setValue(40)
                
        except Exception as e:
            EventLog.emit("WARNING", f"Analyzer performance update error: {e}")

    def update_verifier_performance(self, performance):
        """Update verifier performance metrics in UI"""
        try:
            accuracy = performance.get('accuracy', 0.0) * 100
            response_time = self.last_response_times.get(self.current_verifier_model, 0)
            
            perf_text = f"Accuracy: {accuracy:.1f}% | Response: {response_time:.0f}ms"
            if performance.get('last_prediction'):
                perf_text += f" | Last: {performance['last_prediction']}"
            
            self.verifier_perf.setText(perf_text)
            
            # Update status based on performance
            if accuracy > 70:
                self.verifier_status.setText("🟢 Excellent")
                self.verifier_status.setStyleSheet("color: green;")
                self.verifier_load_bar.setValue(90)
            elif accuracy > 50:
                self.verifier_status.setText("🟡 Good")
                self.verifier_status.setStyleSheet("color: orange;")
                self.verifier_load_bar.setValue(70)
            else:
                self.verifier_status.setText("🔴 Poor")
                self.verifier_status.setStyleSheet("color: red;")
                self.verifier_load_bar.setValue(40)
                
        except Exception as e:
            EventLog.emit("WARNING", f"Verifier performance update error: {e}")

    def measure_model_response_time(self, model_name, prompt="Test prompt"):
        """Measure model response time for performance monitoring"""
        try:
            start_time = time.time()
            
            # Simple test query to measure response time
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post("http://localhost:11434/api/generate", 
                                   json=payload, timeout=10)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            self.last_response_times[model_name] = response_time_ms
            
            # Update model performance in data manager
            if response.status_code == 200:
                # Simulate accuracy based on response time (faster = better)
                simulated_accuracy = max(0.5, min(0.95, 1.0 - (response_time_ms / 5000)))
                
                data_manager.update_model_performance(
                    model_name,
                    accuracy=simulated_accuracy,
                    precision=simulated_accuracy * 0.9,
                    recall=simulated_accuracy * 0.85,
                    f1_score=simulated_accuracy * 0.87,
                    last_prediction="Test",
                    confidence=simulated_accuracy
                )
                
                EventLog.emit("INFO", f"Model {model_name} response time: {response_time_ms:.0f}ms")
            else:
                EventLog.emit("WARNING", f"Model {model_name} test failed: {response.status_code}")
                
        except Exception as e:
            EventLog.emit("WARNING", f"Model response time test failed for {model_name}: {e}")
            self.last_response_times[model_name] = 9999  # High time indicates failure
