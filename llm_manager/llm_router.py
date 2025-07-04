"""
LLM Router - Natural Language to EventBus Command Translation

This module provides the LLM Router component that translates natural language
commands from the LLM into structured EventBus events for system components.
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CommandMapping:
    """Mapping between natural language patterns and EventBus commands."""
    pattern: str
    event_topic: str
    event_data_template: Dict[str, Any]
    description: str
    priority: int = 0

class LLMRouter:
    """
    LLM Router for translating natural language commands to EventBus events.
    
    This component analyzes natural language input and converts it into
    structured EventBus events that can be processed by system components.
    """
    
    def __init__(self, event_bus=None):
        """
        Initialize LLM Router.
        
        Args:
            event_bus: EventBus instance for publishing events
        """
        self.event_bus = event_bus
        self.command_mappings = []
        self.context_history = []
        self.max_history = 100
        
        # Initialize default command mappings
        self._initialize_default_mappings()
        
        logger.info("LLM Router initialized")
    
    def _initialize_default_mappings(self):
        """Initialize default command mappings."""
        
        # Computer Control Commands
        self.add_command_mapping(
            pattern=r"(open|launch|start)\s+(.*?)\s*(application|app|program|software)",
            event_topic="computer.control.launch_application",
            event_data_template={"application": "{1}"},
            description="Launch application"
        )
        
        self.add_command_mapping(
            pattern=r"(click|press)\s+(.*?)\s*(button|link)",
            event_topic="computer.control.click",
            event_data_template={"target": "{1}"},
            description="Click UI element"
        )
        
        self.add_command_mapping(
            pattern=r"type\s+(.*)",
            event_topic="computer.control.type_text",
            event_data_template={"text": "{0}"},
            description="Type text"
        )
        
        # Vehicle/Automotive Commands
        self.add_command_mapping(
            pattern=r"(connect|scan|diagnose)\s+(ecu|vehicle|car)",
            event_topic="vehicle.diagnostics.scan",
            event_data_template={"action": "scan"},
            description="Scan vehicle ECU"
        )
        
        self.add_command_mapping(
            pattern=r"(flash|program|update)\s+(ecu|firmware)",
            event_topic="vehicle.ecu.flash",
            event_data_template={"action": "flash"},
            description="Flash ECU firmware"
        )
        
        self.add_command_mapping(
            pattern=r"(read|get|retrieve)\s+(dtc|codes|faults)",
            event_topic="vehicle.diagnostics.read_dtc",
            event_data_template={"action": "read_dtc"},
            description="Read diagnostic trouble codes"
        )
        
        # Research Commands
        self.add_command_mapping(
            pattern=r"(research|investigate|analyze)\s+(.*)",
            event_topic="research.deep_research.start",
            event_data_template={"topic": "{1}", "depth": "comprehensive"},
            description="Start deep research"
        )
        
        self.add_command_mapping(
            pattern=r"(generate|create|write)\s+(document|report|analysis)",
            event_topic="document.generator.create",
            event_data_template={"type": "document", "content": "{0}"},
            description="Generate document"
        )
        
        # System Commands
        self.add_command_mapping(
            pattern=r"(monitor|check|status)\s+(system|performance)",
            event_topic="system.monitor.status",
            event_data_template={"action": "status_check"},
            description="Check system status"
        )
        
        self.add_command_mapping(
            pattern=r"(open|show|display)\s+(.*?)\s*(popout|window|interface)",
            event_topic="ui.popout.open",
            event_data_template={"component": "{1}"},
            description="Open popout window"
        )
        
        # LLM/AI Commands
        self.add_command_mapping(
            pattern=r"(switch|change|use)\s+(.*?)\s*(model|llm|ai)",
            event_topic="llm.model.switch",
            event_data_template={"model": "{1}"},
            description="Switch AI model"
        )
        
        self.add_command_mapping(
            pattern=r"(configure|set|adjust)\s+(.*?)\s*(parameters|settings)",
            event_topic="llm.config.update",
            event_data_template={"parameters": "{1}"},
            description="Configure LLM parameters"
        )
    
    def add_command_mapping(self, pattern: str, event_topic: str, 
                          event_data_template: Dict[str, Any], 
                          description: str, priority: int = 0):
        """
        Add a command mapping.
        
        Args:
            pattern: Regex pattern to match natural language
            event_topic: EventBus topic to publish to
            event_data_template: Template for event data
            description: Human-readable description
            priority: Priority (higher = processed first)
        """
        mapping = CommandMapping(
            pattern=pattern,
            event_topic=event_topic,
            event_data_template=event_data_template,
            description=description,
            priority=priority
        )
        
        self.command_mappings.append(mapping)
        
        # Sort by priority (highest first)
        self.command_mappings.sort(key=lambda x: x.priority, reverse=True)
        
        logger.debug(f"Added command mapping: {description}")
    
    def route_command(self, natural_language_input: str, 
                     context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Route natural language input to EventBus events.
        
        Args:
            natural_language_input: Natural language command
            context: Optional context information
            
        Returns:
            List of events to publish
        """
        events = []
        input_lower = natural_language_input.lower().strip()
        
        # Add to context history
        self.context_history.append({
            "input": natural_language_input,
            "context": context,
            "timestamp": self._get_timestamp()
        })
        
        # Trim history if too long
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]
        
        # Try to match command patterns
        for mapping in self.command_mappings:
            match = re.search(mapping.pattern, input_lower, re.IGNORECASE)
            if match:
                try:
                    # Extract matched groups
                    groups = match.groups()
                    
                    # Build event data from template
                    event_data = {}
                    for key, template in mapping.event_data_template.items():
                        if isinstance(template, str) and "{" in template:
                            # Replace placeholders with matched groups
                            value = template
                            for i, group in enumerate(groups):
                                value = value.replace(f"{{{i}}}", group or "")
                            event_data[key] = value.strip()
                        else:
                            event_data[key] = template
                    
                    # Add context if provided
                    if context:
                        event_data["context"] = context
                    
                    # Add original input
                    event_data["original_input"] = natural_language_input
                    
                    # Create event
                    event = {
                        "topic": mapping.event_topic,
                        "data": event_data,
                        "description": mapping.description,
                        "confidence": self._calculate_confidence(match, mapping)
                    }
                    
                    events.append(event)
                    
                    logger.info(f"Routed command: '{natural_language_input}' -> {mapping.event_topic}")
                    
                except Exception as e:
                    logger.error(f"Error processing command mapping: {e}")
        
        # If no specific mappings found, create a general command event
        if not events:
            events.append({
                "topic": "llm.command.general",
                "data": {
                    "input": natural_language_input,
                    "context": context,
                    "requires_interpretation": True
                },
                "description": "General command requiring interpretation",
                "confidence": 0.5
            })
        
        return events
    
    def publish_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Publish events to EventBus.
        
        Args:
            events: List of events to publish
            
        Returns:
            True if all events published successfully
        """
        if not self.event_bus:
            logger.warning("No EventBus available for publishing events")
            return False
        
        success = True
        for event in events:
            try:
                self.event_bus.emit(
                    topic=event["topic"],
                    data=event["data"]
                )
                logger.debug(f"Published event: {event['topic']}")
            except Exception as e:
                logger.error(f"Failed to publish event {event['topic']}: {e}")
                success = False
        
        return success
    
    def process_command(self, natural_language_input: str, 
                       context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process natural language command end-to-end.
        
        Args:
            natural_language_input: Natural language command
            context: Optional context information
            
        Returns:
            True if command processed successfully
        """
        try:
            # Route command to events
            events = self.route_command(natural_language_input, context)
            
            # Publish events
            return self.publish_events(events)
            
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            return False
    
    def _calculate_confidence(self, match, mapping: CommandMapping) -> float:
        """
        Calculate confidence score for a command match.
        
        Args:
            match: Regex match object
            mapping: Command mapping
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence from match quality
        confidence = 0.7
        
        # Adjust based on number of captured groups
        if match.groups():
            confidence += 0.1 * len([g for g in match.groups() if g])
        
        # Adjust based on mapping priority
        confidence += mapping.priority * 0.05
        
        # Cap at 1.0
        return min(confidence, 1.0)
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def get_available_commands(self) -> List[Dict[str, str]]:
        """
        Get list of available commands.
        
        Returns:
            List of command descriptions
        """
        return [
            {
                "pattern": mapping.pattern,
                "description": mapping.description,
                "topic": mapping.event_topic
            }
            for mapping in self.command_mappings
        ]
    
    def get_command_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent command history.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of recent commands
        """
        return self.context_history[-limit:] if self.context_history else []


# Global instance
_llm_router_instance = None

def get_llm_router(event_bus=None):
    """
    Get global LLM Router instance.
    
    Args:
        event_bus: EventBus instance
        
    Returns:
        LLM Router instance
    """
    global _llm_router_instance
    
    if _llm_router_instance is None:
        _llm_router_instance = LLMRouter(event_bus)
    elif event_bus and not _llm_router_instance.event_bus:
        _llm_router_instance.event_bus = event_bus
    
    return _llm_router_instance

