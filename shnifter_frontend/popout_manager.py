"""
PopoutManager - Manages popout windows for Shnifter Trader

This class provides centralized management of popout windows, ensuring that
only one instance of each window type is open at a time, and that windows can
be properly reopened after being closed.
"""

import logging
from typing import Dict, Any, Type, Optional
from PySide6.QtWidgets import QWidget, QMainWindow
from shnifter_frontend.popout_window import PopoutWindow

class PopoutManager:
    """
    Manages popout windows for Shnifter Trader.
    
    This class ensures that each window type has only one instance open at a time,
    and that windows can be properly reopened after being closed.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the popout manager for Shnifter Trader.
        
        Args:
            parent: Parent widget that owns this manager
        """
        self.parent = parent
        self.windows: Dict[str, QMainWindow] = {}
        self.logger = logging.getLogger(__name__)
        
    def open_window(self, title: str, widget_class: Type[QWidget], use_popout_window: bool = True) -> Optional[QMainWindow]:
        """
        Open a popout window with the specified widget for Shnifter Trader.
        
        If a window with the given title already exists, it will be brought to the front
        instead of creating a new instance.
        
        Args:
            title: Window title
            widget_class: Widget class to instantiate
            use_popout_window: Whether to use the PopoutWindow class (if available)
            
        Returns:
            The window instance, or None if an error occurred
        """
        # Check if window already exists
        if title in self.windows:
            window = self.windows[title]
            try:
                if window and window.isVisible():
                    # Window exists and is visible, bring to front
                    self.logger.info(f"Window '{title}' already exists, bringing to front")
                    window.raise_()
                    window.activateWindow()
                    return window
                else:
                    # Window exists but not visible, remove stale reference
                    self.logger.info(f"Window '{title}' exists but not visible, removing stale reference")
                    del self.windows[title]
            except RuntimeError:
                # Window was destroyed, remove stale reference
                self.logger.info(f"Window '{title}' was destroyed, removing stale reference")
                del self.windows[title]
        
        try:
            # Create new popout window
            if use_popout_window:
                window = PopoutWindow(
                    parent=self.parent,
                    title=f"Shnifter Trader - {title}",
                    parent_dict=self.windows,
                    key=title
                )
            else:
                # Fallback to QMainWindow
                window = QMainWindow(self.parent)
                window.setWindowTitle(f"Shnifter Trader - {title}")
                
                # Connect destroyed signal to cleanup reference
                window.destroyed.connect(lambda _, t=title: self.windows.pop(t, None))
            
            # Set window geometry
            window.setGeometry(200, 200, 800, 600)
            
            # Create widget instance
            widget = widget_class()
            window.setCentralWidget(widget)
            
            # Store reference
            self.windows[title] = window
            
            # Show window
            window.show()
            
            self.logger.info(f"Opened window '{title}'")
            return window
            
        except Exception as e:
            self.logger.error(f"Error opening window '{title}': {str(e)}")
            return None
            
    def close_all_windows(self):
        """
        Close all open windows for Shnifter Trader.
        """
        for title, window in list(self.windows.items()):
            try:
                window.close()
            except Exception as e:
                self.logger.error(f"Error closing window '{title}': {str(e)}")
        
        # Clear the windows dictionary
        self.windows.clear()
        
    def get_window(self, title: str) -> Optional[QMainWindow]:
        """
        Get a window by title for Shnifter Trader.
        
        Args:
            title: Window title
            
        Returns:
            The window instance, or None if not found
        """
        return self.windows.get(title)

    def register_widget_type(self, key: str, widget_class: Type[QWidget]):
        """
        Register a widget type with a string key for later instantiation.
        """
        if not hasattr(self, '_widget_types'):
            self._widget_types = {}
        self._widget_types[key] = widget_class
        self.logger.info(f"Registered widget type '{key}'")

    def open_by_type(self, key: str, title: str = None, use_popout_window: bool = True, callback: Any = None) -> Optional[QMainWindow]:
        """
        Open a window by registered widget type key.
        Optionally pass a callback to the widget if it supports it.
        """
        if not hasattr(self, '_widget_types') or key not in self._widget_types:
            self.logger.error(f"Widget type '{key}' not registered.")
            return None
        widget_class = self._widget_types[key]
        # Optionally pass callback to widget if supported
        def widget_factory():
            widget = widget_class()
            if callback and hasattr(widget, 'set_callback'):
                widget.set_callback(callback)
            return widget
        return self.open_window(title or key, widget_factory, use_popout_window)

    def close_windows_by_type(self, key: str):
        """
        Close all windows of a given registered widget type key.
        """
        to_close = [title for title, window in self.windows.items()
                    if hasattr(window.centralWidget(), '__class__') and window.centralWidget().__class__ == self._widget_types.get(key)]
        for title in to_close:
            self.windows[title].close()
            del self.windows[title]
        self.logger.info(f"Closed all windows of type '{key}'")

    def bring_all_to_front(self):
        """
        Bring all managed windows to the front.
        """
        for window in self.windows.values():
            if window.isVisible():
                window.raise_()
                window.activateWindow()
        self.logger.info("Brought all windows to front.")

    def save_window_states(self):
        """
        Save geometry/state for all windows (stub for extension).
        """
        # Implement as needed (e.g., QSettings)
        pass

    def restore_window_states(self):
        """
        Restore geometry/state for all windows (stub for extension).
        """
        # Implement as needed (e.g., QSettings)
        pass
