"""
Shnifter Popout Registry - Complete Widget Integration
Auto-generated registry for all converted components
"""
from typing import Dict, Any, Callable

class ShnifterPopoutRegistry:
    """Registry for all Shnifter popout widgets"""
    
    def __init__(self):
        self.registered_widgets = {}
        
    def register_widget(self, widget_id: str, widget_class, config: Dict[str, Any]):
        """Register a widget with the registry"""
        self.registered_widgets[widget_id] = {
            "class": widget_class,
            "config": config
        }
        
    def get_registered_widgets(self):
        """Get all registered widgets"""
        return self.registered_widgets
        
    def create_widget(self, widget_id: str, parent=None):
        """Create a widget instance"""
        if widget_id in self.registered_widgets:
            widget_class = self.registered_widgets[widget_id]["class"]
            return widget_class(parent=parent)
        return None

# Global registry instance
shnifter_popout_registry = ShnifterPopoutRegistry()
