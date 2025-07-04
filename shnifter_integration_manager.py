# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+
"""
Shnifter Integration Manager
Central manager for all Shnifter components and future extensions or intergrations.
Handles initialization, event routing, and widget lifecycle
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from PySide6.QtCore import QObject, Signal
from core.events import EventLog, EventBus
from core.config import shnifter_config
from shnifter_frontend.popout_manager import PopoutManager
from shnifter_frontend.shnifter_popout_registry import shnifter_popout_registry

class ShnifterIntegrationManager(QObject):
    """Central manager for all Shnifter components"""
    
    # Signals for application-wide communication
    component_loaded = Signal(str, dict)
    analysis_complete = Signal(str, dict)
    error_occurred = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.event_log = EventLog()
        self.popout_manager = PopoutManager.get_instance()
        self.active_components = {}
        self.analysis_modules = {}
        
        # Initialize integration
        self._setup_event_subscriptions()
        self._load_analysis_modules()
        
        self.event_log.publish("integration_manager_init", {
            "timestamp": datetime.now().isoformat(),
            "components_available": len(self.active_components)
        })
        
    def _setup_event_subscriptions(self):
        """Setup event system subscriptions"""
        EventBus.subscribe("widget_created", self._on_widget_created)
        EventBus.subscribe("widget_destroyed", self._on_widget_destroyed)
        EventBus.subscribe("analysis_request", self._on_analysis_request)
        EventBus.subscribe("data_update", self._on_data_update)
        
    def _load_analysis_modules(self):
        """Load all converted analysis modules"""
        try:
            # Import analysis modules (dynamically discover)
            import importlib
            import pkgutil
            
            # Discover analysis modules
            import shnifter_analysis_modules
            
            for importer, modname, ispkg in pkgutil.iter_modules(shnifter_analysis_modules.__path__):
                try:
                    module = importlib.import_module(f'shnifter_analysis_modules.{modname}')
                    self.analysis_modules[modname] = module
                    self.event_log.publish("module_loaded", {"module": modname})
                except Exception as e:
                    self.event_log.publish("module_load_error", {"module": modname, "error": str(e)})
                    
        except ImportError:
            self.event_log.publish("analysis_modules_not_found", {})
            
    def create_widget(self, widget_type: str, parent=None, config=None) -> Optional[Any]:
        """Create widget instance through popout manager"""
        try:
            widget = self.popout_manager.create_widget(widget_type, parent, config)
            if widget:
                widget_id = getattr(widget, 'widget_id', f"{widget_type}_{id(widget)}")
                self.active_components[widget_id] = widget
                self.component_loaded.emit(widget_type, {"widget_id": widget_id})
                
            return widget
            
        except Exception as e:
            self.error_occurred.emit(widget_type, str(e))
            return None
            
    def run_analysis(self, analysis_type: str, data: Any, **kwargs) -> Optional[Dict[str, Any]]:
        """Run analysis using converted modules"""
        if analysis_type not in self.analysis_modules:
            self.error_occurred.emit(analysis_type, "Analysis module not found")
            return None
            
        try:
            module = self.analysis_modules[analysis_type]
            
            # Look for analyzer factory function
            factory_name = f"create_{analysis_type}_analyzer"
            if hasattr(module, factory_name):
                analyzer = getattr(module, factory_name)()
                result = analyzer.run_analysis(data, **kwargs)
                
                self.analysis_complete.emit(analysis_type, result)
                return result
                
        except Exception as e:
            self.error_occurred.emit(analysis_type, str(e))
            
        return None
        
    def get_available_widgets(self) -> List[str]:
        """Get list of available widget types"""
        return list(self.popout_manager.get_registered_widgets().keys())
        
    def get_available_analyses(self) -> List[str]:
        """Get list of available analysis modules"""
        return list(self.analysis_modules.keys())
        
    def shutdown(self):
        """Cleanup on application shutdown"""
        self.event_log.publish("integration_manager_shutdown", {
            "active_components": len(self.active_components),
            "timestamp": datetime.now().isoformat()
        })
        
        # Cleanup active components
        for component in self.active_components.values():
            if hasattr(component, 'close'):
                component.close()
                
        self.active_components.clear()
        
    # Event handlers
    def _on_widget_created(self, event_data):
        """Handle widget creation events"""
        widget_id = event_data.get("widget_id")
        widget_type = event_data.get("widget_type")
        
        self.event_log.publish("widget_registered", {
            "widget_id": widget_id,
            "widget_type": widget_type
        })
        
    def _on_widget_destroyed(self, event_data):
        """Handle widget destruction events"""
        widget_id = event_data.get("widget_id")
        
        if widget_id in self.active_components:
            del self.active_components[widget_id]
            
        self.event_log.publish("widget_unregistered", {"widget_id": widget_id})
        
    def _on_analysis_request(self, event_data):
        """Handle analysis requests"""
        analysis_type = event_data.get("analysis_type")
        data = event_data.get("data")
        params = event_data.get("params", {})
        
        self.run_analysis(analysis_type, data, **params)
        
    def _on_data_update(self, event_data):
        """Handle data updates and route to relevant components"""
        target_widgets = event_data.get("target_widgets", [])
        
        for widget_id in target_widgets:
            if widget_id in self.active_components:
                widget = self.active_components[widget_id]
                if hasattr(widget, 'data_updated'):
                    widget.data_updated.emit(event_data.get("data", {}))

# Global integration manager instance
shnifter_integration_manager = None

def get_integration_manager() -> ShnifterIntegrationManager:
    """Get global integration manager instance"""
    global shnifter_integration_manager
    if shnifter_integration_manager is None:
        shnifter_integration_manager = ShnifterIntegrationManager()
    return shnifter_integration_manager
