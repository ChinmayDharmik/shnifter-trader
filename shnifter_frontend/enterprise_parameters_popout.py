"""
Enterprise Trading Parameters Configuration Popout
Provides precise date/time controls and live data settings for professional trading analysis
"""

import sys
from datetime import datetime, timedelta, time
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QDateTimeEdit, QSpinBox, QCheckBox, QGroupBox, QGridLayout,
    QComboBox, QLineEdit, QSlider, QFormLayout, QTabWidget,
    QWidget, QDoubleSpinBox, QFrame, QTextEdit
)
from PySide6.QtCore import Qt, QDateTime, QTimer, Signal
from PySide6.QtGui import QFont, QIcon


class EnterpriseParametersPopout(QDialog):
    """Enterprise-grade parameters configuration dialog"""
    
    # Signals for parameter changes
    parameters_changed = Signal(dict)
    live_update_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("🏢 Enterprise Trading Parameters")
        self.setModal(False)  # Allow interaction with main window
        self.resize(800, 700)
        
        # Default parameters
        self.current_params = {
            'start_date': datetime.now() - timedelta(days=365),
            'end_date': datetime.now(),
            'live_mode': True,
            'update_interval': 10,  # seconds
            'data_granularity': '1min',
            'premarket_hours': True,
            'extended_hours': True,
            'realtime_quotes': True,
            'auto_refresh': True,
            'max_history_days': 365,
            'streaming_enabled': False,
            'websocket_enabled': False
        }
        
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Setup the enterprise UI layout"""
        main_layout = QVBoxLayout(self)
        
        # Title section
        title_frame = QFrame()
        title_frame.setFrameStyle(QFrame.StyledPanel)
        title_layout = QHBoxLayout(title_frame)
        title_label = QLabel("🏢 Enterprise Trading Parameters")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_layout.addWidget(title_label)
        main_layout.addWidget(title_frame)
        
        # Create tabs for organized parameter groups
        self.tabs = QTabWidget()
        
        # Tab 1: Date & Time Controls
        self.datetime_tab = self.create_datetime_tab()
        self.tabs.addTab(self.datetime_tab, "📅 Date & Time")
        
        # Tab 2: Live Data Settings
        self.live_data_tab = self.create_live_data_tab()
        self.tabs.addTab(self.live_data_tab, "🔴 Live Data")
        
        # Tab 3: Performance Settings
        self.performance_tab = self.create_performance_tab()
        self.tabs.addTab(self.performance_tab, "⚡ Performance")
        
        # Tab 4: Advanced Settings
        self.advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "🔧 Advanced")
        
        main_layout.addWidget(self.tabs)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("✅ Apply Changes")
        self.apply_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.apply_btn.clicked.connect(self.apply_parameters)
        
        self.reset_btn = QPushButton("🔄 Reset to Defaults")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        
        self.live_update_btn = QPushButton("🔴 Live Update Now")
        self.live_update_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        self.live_update_btn.clicked.connect(self.trigger_live_update)
        
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.apply_btn)
        button_layout.addWidget(self.reset_btn)
        button_layout.addWidget(self.live_update_btn)
        button_layout.addStretch()
        button_layout.addWidget(self.close_btn)
        
        main_layout.addLayout(button_layout)
        
        # Status bar
        self.status_label = QLabel("Ready - Configure parameters above")
        self.status_label.setStyleSheet("QLabel { background-color: #e3f2fd; padding: 5px; border-radius: 3px; }")
        main_layout.addWidget(self.status_label)
        
    def create_datetime_tab(self):
        """Create precise date and time controls"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Date Range Group
        date_group = QGroupBox("📅 Precise Date Range Selection")
        date_layout = QFormLayout(date_group)
        
        # Start date/time
        self.start_datetime = QDateTimeEdit()
        self.start_datetime.setCalendarPopup(True)
        self.start_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(self.current_params['start_date'].timestamp())))
        self.start_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        date_layout.addRow("Start Date & Time:", self.start_datetime)
        
        # End date/time
        self.end_datetime = QDateTimeEdit()
        self.end_datetime.setCalendarPopup(True)
        self.end_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(self.current_params['end_date'].timestamp())))
        self.end_datetime.setDisplayFormat("yyyy-MM-dd hh:mm:ss")
        date_layout.addRow("End Date & Time:", self.end_datetime)
        
        layout.addWidget(date_group)
        
        # Quick Date Presets
        presets_group = QGroupBox("⚡ Quick Date Presets")
        presets_layout = QGridLayout(presets_group)
        
        preset_buttons = [
            ("Last Hour", lambda: self.set_date_range(hours=1)),
            ("Last Day", lambda: self.set_date_range(days=1)),
            ("Last Week", lambda: self.set_date_range(days=7)),
            ("Last Month", lambda: self.set_date_range(days=30)),
            ("Last 3 Months", lambda: self.set_date_range(days=90)),
            ("Last Year", lambda: self.set_date_range(days=365)),
            ("YTD", self.set_year_to_date),
            ("Market Open Today", self.set_market_open_today)
        ]
        
        for i, (text, func) in enumerate(preset_buttons):
            btn = QPushButton(text)
            btn.clicked.connect(func)
            presets_layout.addWidget(btn, i // 4, i % 4)
            
        layout.addWidget(presets_group)
        
        # Time Zone Settings
        tz_group = QGroupBox("🌍 Time Zone & Market Hours")
        tz_layout = QFormLayout(tz_group)
        
        self.timezone_combo = QComboBox()
        self.timezone_combo.addItems([
            "US/Eastern (NYSE/NASDAQ)",
            "US/Central", 
            "US/Mountain",
            "US/Pacific",
            "Europe/London (LSE)",
            "Asia/Tokyo (TSE)",
            "UTC"
        ])
        tz_layout.addRow("Time Zone:", self.timezone_combo)
        
        self.premarket_cb = QCheckBox("Include Pre-Market Hours (4:00-9:30 AM EST)")
        self.premarket_cb.setChecked(self.current_params['premarket_hours'])
        tz_layout.addRow(self.premarket_cb)
        
        self.extended_cb = QCheckBox("Include Extended Hours (4:00 PM-8:00 PM EST)")
        self.extended_cb.setChecked(self.current_params['extended_hours'])
        tz_layout.addRow(self.extended_cb)
        
        layout.addWidget(tz_group)
        
        return tab
        
    def create_live_data_tab(self):
        """Create live data and real-time settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Live Mode Settings
        live_group = QGroupBox("🔴 Live Data Configuration")
        live_layout = QFormLayout(live_group)
        
        self.live_mode_cb = QCheckBox("Enable Live Mode")
        self.live_mode_cb.setChecked(self.current_params['live_mode'])
        live_layout.addRow(self.live_mode_cb)
        
        # Update interval with millisecond precision
        interval_layout = QHBoxLayout()
        self.update_interval_spin = QDoubleSpinBox()
        self.update_interval_spin.setRange(0.1, 3600)  # 100ms to 1 hour
        self.update_interval_spin.setValue(self.current_params['update_interval'])
        self.update_interval_spin.setSuffix(" seconds")
        self.update_interval_spin.setDecimals(1)
        
        # Quick interval buttons
        quick_intervals = [
            ("10s", 10), ("30s", 30), ("1m", 60), ("5m", 300), ("15m", 900)
        ]
        for text, value in quick_intervals:
            btn = QPushButton(text)
            btn.clicked.connect(lambda checked, v=value: self.update_interval_spin.setValue(v))
            interval_layout.addWidget(btn)
            
        live_layout.addRow("Update Interval:", self.update_interval_spin)
        live_layout.addRow("Quick Intervals:", interval_layout)
        
        # Data granularity
        self.granularity_combo = QComboBox()
        self.granularity_combo.addItems([
            "1s", "5s", "10s", "30s", "1min", "2min", "5min", 
            "15min", "30min", "1h", "4h", "1d"
        ])
        self.granularity_combo.setCurrentText(self.current_params['data_granularity'])
        live_layout.addRow("Data Granularity:", self.granularity_combo)
        
        layout.addWidget(live_group)
        
        # Real-time Features
        realtime_group = QGroupBox("⚡ Real-time Features")
        realtime_layout = QFormLayout(realtime_group)
        
        self.realtime_quotes_cb = QCheckBox("Real-time Quotes")
        self.realtime_quotes_cb.setChecked(self.current_params['realtime_quotes'])
        realtime_layout.addRow(self.realtime_quotes_cb)
        
        self.streaming_cb = QCheckBox("Streaming Data (WebSocket)")
        self.streaming_cb.setChecked(self.current_params['streaming_enabled'])
        realtime_layout.addRow(self.streaming_cb)
        
        self.auto_refresh_cb = QCheckBox("Auto-refresh Charts")
        self.auto_refresh_cb.setChecked(self.current_params['auto_refresh'])
        realtime_layout.addRow(self.auto_refresh_cb)
        
        layout.addWidget(realtime_group)
        
        # Live Status Display
        status_group = QGroupBox("📊 Live Data Status")
        status_layout = QVBoxLayout(status_group)
        
        self.live_status_text = QTextEdit()
        self.live_status_text.setMaximumHeight(100)
        self.live_status_text.setReadOnly(True)
        self.live_status_text.setText("Live data status will appear here...")
        status_layout.addWidget(self.live_status_text)
        
        layout.addWidget(status_group)
        
        return tab
        
    def create_performance_tab(self):
        """Create performance optimization settings"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Data Management
        data_group = QGroupBox("💾 Data Management")
        data_layout = QFormLayout(data_group)
        
        self.max_history_spin = QSpinBox()
        self.max_history_spin.setRange(1, 3650)  # 1 day to 10 years
        self.max_history_spin.setValue(self.current_params['max_history_days'])
        self.max_history_spin.setSuffix(" days")
        data_layout.addRow("Max History to Keep:", self.max_history_spin)
        
        self.cache_cb = QCheckBox("Enable Data Caching")
        self.cache_cb.setChecked(True)
        data_layout.addRow(self.cache_cb)
        
        self.compression_cb = QCheckBox("Compress Historical Data")
        self.compression_cb.setChecked(True)
        data_layout.addRow(self.compression_cb)
        
        layout.addWidget(data_group)
        
        # Performance Tuning
        perf_group = QGroupBox("⚡ Performance Tuning")
        perf_layout = QFormLayout(perf_group)
        
        self.parallel_requests_spin = QSpinBox()
        self.parallel_requests_spin.setRange(1, 20)
        self.parallel_requests_spin.setValue(5)
        perf_layout.addRow("Parallel API Requests:", self.parallel_requests_spin)
        
        self.chart_buffer_spin = QSpinBox()
        self.chart_buffer_spin.setRange(100, 10000)
        self.chart_buffer_spin.setValue(1000)
        self.chart_buffer_spin.setSuffix(" data points")
        perf_layout.addRow("Chart Buffer Size:", self.chart_buffer_spin)
        
        layout.addWidget(perf_group)
        
        return tab
        
    def create_advanced_tab(self):
        """Create advanced configuration options"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # API Settings
        api_group = QGroupBox("🔧 API Configuration")
        api_layout = QFormLayout(api_group)
        
        self.api_timeout_spin = QDoubleSpinBox()
        self.api_timeout_spin.setRange(1.0, 60.0)
        self.api_timeout_spin.setValue(30.0)
        self.api_timeout_spin.setSuffix(" seconds")
        api_layout.addRow("API Timeout:", self.api_timeout_spin)
        
        self.retry_attempts_spin = QSpinBox()
        self.retry_attempts_spin.setRange(1, 10)
        self.retry_attempts_spin.setValue(3)
        api_layout.addRow("Retry Attempts:", self.retry_attempts_spin)
        
        layout.addWidget(api_group)
        
        # Debug Settings
        debug_group = QGroupBox("🐛 Debug & Logging")
        debug_layout = QFormLayout(debug_group)
        
        self.debug_mode_cb = QCheckBox("Debug Mode")
        debug_layout.addRow(self.debug_mode_cb)
        
        self.verbose_logging_cb = QCheckBox("Verbose Logging")
        debug_layout.addRow(self.verbose_logging_cb)
        
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level_combo.setCurrentText("INFO")
        debug_layout.addRow("Log Level:", self.log_level_combo)
        
        layout.addWidget(debug_group)
        
        return tab
        
    def connect_signals(self):
        """Connect all signals and slots"""
        self.start_datetime.dateTimeChanged.connect(self.on_parameters_changed)
        self.end_datetime.dateTimeChanged.connect(self.on_parameters_changed)
        self.live_mode_cb.stateChanged.connect(self.on_parameters_changed)
        self.update_interval_spin.valueChanged.connect(self.on_parameters_changed)
        
    def set_date_range(self, days=None, hours=None):
        """Set date range based on relative time"""
        end_time = datetime.now()
        if days:
            start_time = end_time - timedelta(days=days)
        elif hours:
            start_time = end_time - timedelta(hours=hours)
        else:
            return
            
        self.start_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(start_time.timestamp())))
        self.end_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(end_time.timestamp())))
        
    def set_year_to_date(self):
        """Set range to year-to-date"""
        now = datetime.now()
        start_of_year = datetime(now.year, 1, 1)
        self.start_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(start_of_year.timestamp())))
        self.end_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(now.timestamp())))
        
    def set_market_open_today(self):
        """Set range to market open today"""
        now = datetime.now()
        market_open = datetime.combine(now.date(), time(9, 30))  # 9:30 AM
        self.start_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(market_open.timestamp())))
        self.end_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(now.timestamp())))
        
    def on_parameters_changed(self):
        """Handle parameter changes"""
        self.status_label.setText("Parameters modified - Click 'Apply Changes' to update")
        self.status_label.setStyleSheet("QLabel { background-color: #fff3cd; padding: 5px; border-radius: 3px; }")
        
    def apply_parameters(self):
        """Apply current parameters and emit signal"""
        # Collect all parameters
        params = {
            'start_date': datetime.fromtimestamp(self.start_datetime.dateTime().toSecsSinceEpoch()),
            'end_date': datetime.fromtimestamp(self.end_datetime.dateTime().toSecsSinceEpoch()),
            'live_mode': self.live_mode_cb.isChecked(),
            'update_interval': self.update_interval_spin.value(),
            'data_granularity': self.granularity_combo.currentText(),
            'premarket_hours': self.premarket_cb.isChecked(),
            'extended_hours': self.extended_cb.isChecked(),
            'realtime_quotes': self.realtime_quotes_cb.isChecked(),
            'auto_refresh': self.auto_refresh_cb.isChecked(),
            'max_history_days': self.max_history_spin.value(),
            'streaming_enabled': self.streaming_cb.isChecked(),
            'timezone': self.timezone_combo.currentText()
        }
        
        self.current_params.update(params)
        self.parameters_changed.emit(params)
        
        self.status_label.setText("✅ Parameters applied successfully")
        self.status_label.setStyleSheet("QLabel { background-color: #d4edda; padding: 5px; border-radius: 3px; }")
        
    def reset_to_defaults(self):
        """Reset all parameters to defaults"""
        now = datetime.now()
        
        self.start_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int((now - timedelta(days=365)).timestamp())))
        self.end_datetime.setDateTime(QDateTime.fromSecsSinceEpoch(int(now.timestamp())))
        self.live_mode_cb.setChecked(True)
        self.update_interval_spin.setValue(10)
        self.granularity_combo.setCurrentText("1min")
        
        self.status_label.setText("🔄 Reset to default parameters")
        self.status_label.setStyleSheet("QLabel { background-color: #e3f2fd; padding: 5px; border-radius: 3px; }")
        
    def trigger_live_update(self):
        """Trigger immediate live update"""
        self.live_update_requested.emit()
        self.live_status_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] Live update requested")
        
    def get_current_parameters(self):
        """Get current parameter values"""
        return self.current_params.copy()


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = EnterpriseParametersPopout()
    dialog.show()
    sys.exit(app.exec())
