from shnifter_frontend.popout import PopoutWindow
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QVBoxLayout, QTextEdit
from core.events import EventBus

class EventLogPopout(PopoutWindow):
    """Popout window for real-time event log display."""
    def __init__(self, parent=None):
        super().__init__(title="Event Log", content="", parent=parent)
        self.setMinimumSize(500, 300)
        self.setWindowModality(Qt.NonModal)
        # Replace label with QTextEdit for scrolling logs
        self.layout().removeWidget(self.label)
        self.label.deleteLater()
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.layout().insertWidget(0, self.log_display)
        # Subscribe to all log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            EventBus.subscribe(level, self.handle_event)
        # Optionally, poll for new logs if needed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.scroll_to_end)
        self.timer.start(1000)

    def handle_event(self, event):
        msg = f"[{event['timestamp']}] [{event['level']}] {event['message']}"
        self.log_display.append(msg)

    def scroll_to_end(self):
        from PySide6.QtGui import QTextCursor
        self.log_display.moveCursor(QTextCursor.MoveOperation.End)
