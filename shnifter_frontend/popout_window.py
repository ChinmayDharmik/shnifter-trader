"""
PopoutWindow - Custom window class for Shnifter Trader popout windows

This class extends QMainWindow to provide proper lifecycle management for popout windows,
ensuring they are properly removed from the parent's tracking dictionary when closed.
"""

import logging
from PySide6.QtWidgets import QMainWindow, QWidget

class PopoutWindow(QMainWindow):
    """
    Custom window class for Shnifter Trader popout windows with proper lifecycle management.
    
    This class ensures that when a popout window is closed, it is properly removed from
    the parent's tracking dictionary, preventing stale references and allowing windows
    to be reopened after being closed.
    """
    
    def __init__(self, parent=None, title="Popout Window", parent_dict=None, key=None):
        """
        Initialize the popout window.
        
        Args:
            parent: Parent widget
            title: Window title
            parent_dict: Dictionary in the parent that tracks this window
            key: Key in the parent_dict that refers to this window
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.parent_dict = parent_dict
        self.dict_key = key
        self.logger = logging.getLogger(__name__)
        
    def closeEvent(self, event):
        """
        Handle window close event.
        
        This method ensures the window is properly removed from the parent's tracking
        dictionary when closed, allowing it to be reopened later.
        
        Args:
            event: Close event
        """
        try:
            # Remove this window from the parent's tracking dictionary
            if self.parent_dict is not None and self.dict_key is not None:
                if self.dict_key in self.parent_dict:
                    self.logger.info(f"Removing window '{self.dict_key}' from tracking dictionary")
                    del self.parent_dict[self.dict_key]
                else:
                    self.logger.warning(f"Window key '{self.dict_key}' not found in tracking dictionary")
        except Exception as e:
            self.logger.error(f"Error in closeEvent: {str(e)}")
        
        # Accept the close event
        event.accept()
