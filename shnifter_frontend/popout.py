# mock popout window for shnifter frontend (template)
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

class PopoutWindow(QDialog):
    """A simple template for a popout window in the Shnifter frontend."""
    def __init__(self, title: str = "Popout", content: str = "Hello, Shnifter!", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.NonModal)
        self.setMinimumSize(300, 150)  # Allow resizing, set a minimum size
        self.resize(400, 200)
        layout = QVBoxLayout()
        self.label = QLabel(content)
        layout.addWidget(self.label)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)
        self.setLayout(layout)
        self.setAttribute(Qt.WA_DeleteOnClose, True)  # Ensure window is deleted on close

    def set_content(self, content: str):
        self.label.setText(content)

# Example usage for respawning:
# In your main window or manager, always create a new PopoutWindow instance when needed:
# def show_popout(self):
#     popout = PopoutWindow(title="Info", content="Dynamic content here")
#     popout.show()
#     # Optionally keep a reference if you want to manage multiple popouts
