"""
Event system for Shnifter Trader:

- EventBus: simple publish-subscribe mechanism with log-level filtering and custom events.
- EventLog: centralized logging to memory, file, and EventBus; provides specialized emitters for analysis and trade signals.
- ShnifterEventEmitter: Qt signal bridge for EventBus events, enabling GUI components to react via PySide6 signals.
"""

from datetime import datetime, timedelta
import json
import threading
import queue
import time
from collections import defaultdict
from typing import Dict, Any, Callable, List
from PySide6.QtCore import QObject, Signal

class Event:
    """A structured data class for events passing through the bus."""
    def __init__(self, event_type: str, payload: Dict = None, source: str = None):
        self.event_type = event_type
        self.payload = payload or {}
        self.source = source
        self.timestamp = datetime.now()

    def __repr__(self):
        return f"Event(type={self.event_type}, source={self.source}, payload={self.payload})"

class EventBusMetrics:
    """A thread-safe singleton for tracking EventBus performance metrics."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.events_published = 0
        self.events_processed = 0
        self.events_failed = 0
        self.processing_times = []
        self._lock = threading.Lock()
        self._initialized = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = EventBusMetrics()
        return cls._instance

    def record_publish(self):
        with self._lock:
            self.events_published += 1

    def record_processed(self, processing_time: float):
        with self._lock:
            self.events_processed += 1
            self.processing_times.append(processing_time)
            # Keep last 10k measurements for stats
            if len(self.processing_times) > 10000:
                self.processing_times = self.processing_times[-10000:]

    def record_failure(self):
        with self._lock:
            self.events_failed += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_time = sum(self.processing_times)
            avg_time = total_time / len(self.processing_times) if self.processing_times else 0
            throughput = self.events_processed / total_time if total_time > 0 else 0
            return {
                "published": self.events_published,
                "processed": self.events_processed,
                "failed": self.events_failed,
                "queue_size": EventBus.get_queue_size(),
                "avg_processing_time_ms": avg_time * 1000,
                "throughput_events_per_sec": throughput
            }

class EventBus:
    """
    Enhanced, thread-safe, asynchronous event bus inspired by KingAI AGI.
    This is a singleton class that manages event publication and subscription
    using a background thread and a queue for high throughput and non-blocking operation.

    Features:
    - Asynchronous processing via a message queue.
    - Circuit breaker pattern for unreliable subscribers to prevent cascading failures.
    - Integrated performance metrics.
    - Structured event objects for clarity and consistency.
    - Compatibility with the previous class-based API.
    """
    _instance = None
    _lock = threading.Lock()

    # Circuit Breaker configuration
    CB_MAX_FAILURES = 3
    CB_COOLDOWN_SECONDS = 60

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._subscribers = defaultdict(list)
        self._event_queue = queue.Queue()
        self._metrics = EventBusMetrics.get_instance()
        self._circuit_breakers = defaultdict(lambda: {'failures': 0, 'state': 'CLOSED', 'opens_at': None})
        
        self._processing_thread = threading.Thread(target=self._process_event_queue, daemon=True)
        self._processing_thread.start()
        self._initialized = True

    @classmethod
    def _get_instance(cls):
        """Access the singleton instance, creating it if it doesn't exist."""
        return cls()

    def _process_event_queue(self):
        """The core loop of the event bus, running in a background thread."""
        while True:
            event = self._event_queue.get()
            if event is None:  # Sentinel for graceful shutdown
                break
            
            start_time = time.perf_counter()
            
            subscribers = self._subscribers.get(event.event_type, [])
            for handler in subscribers:
                self._dispatch(event, handler)

            processing_time = time.perf_counter() - start_time
            self._metrics.record_processed(processing_time)
            self._event_queue.task_done()

    def _dispatch(self, event: Event, handler: Callable):
        """
        Dispatches an event to a single handler, wrapping the call with
        circuit breaker logic to handle failures gracefully.
        """
        handler_id = id(handler)
        cb = self._circuit_breakers[handler_id]

        if cb['state'] == 'OPEN':
            if datetime.now() > cb['opens_at']:
                cb['state'] = 'HALF-OPEN'
            else:
                return  # Skip handler, it's in cooldown

        try:
            handler(event.payload)
            if cb['state'] == 'HALF-OPEN':
                # Handler succeeded, close the circuit
                cb['state'] = 'CLOSED'
                cb['failures'] = 0
                print(f"EventBus: Circuit breaker for handler {handler.__name__} is now CLOSED.")
        except Exception as e:
            self._metrics.record_failure()
            cb['failures'] += 1
            print(f"EventBus: Handler {getattr(handler, '__name__', 'unknown')} failed for event {event.event_type}. Failure {cb['failures']}/{self.CB_MAX_FAILURES}. Error: {e}")

            if cb['state'] == 'HALF-OPEN' or cb['failures'] >= self.CB_MAX_FAILURES:
                cb['state'] = 'OPEN'
                cb['opens_at'] = datetime.now() + timedelta(seconds=self.CB_COOLDOWN_SECONDS)
                print(f"EventBus: Circuit breaker for handler {getattr(handler, '__name__', 'unknown')} is now OPEN. Will reopen at {cb['opens_at']}.")
            
            if "wrapped C/C++ object" in str(e):
                self.unsubscribe(event.event_type, handler)
                print(f"EventBus: Removed dead Qt widget handler for {event.event_type}")

    @classmethod
    def subscribe(cls, event_type: str, handler: Callable):
        """Class method to subscribe a handler to an event type."""
        instance = cls._get_instance()
        instance._subscribers[event_type].append(handler)

    @classmethod
    def unsubscribe(cls, event_type: str, handler: Callable):
        """Class method to unsubscribe a handler from an event type."""
        instance = cls._get_instance()
        if event_type in instance._subscribers:
            try:
                instance._subscribers[event_type].remove(handler)
            except ValueError:
                # Handler already removed, ignore.
                pass

    @classmethod
    def publish(cls, event_type: str, payload: Dict = None, source: str = "Unknown"):
        """Class method to publish an event asynchronously."""
        instance = cls._get_instance()
        event = Event(event_type, payload, source)
        instance._event_queue.put(event)
        instance._metrics.record_publish()

    @classmethod
    def publish_custom(cls, event_type: str, data: Dict[str, Any]):
        """Alias for publish for compatibility with old code."""
        cls.publish(event_type, data, source="publish_custom")

    @classmethod
    def set_min_log_level(cls, level: str):
        """
        This method is now deprecated in the EventBus.
        Log level filtering is handled by the EventLog before publishing.
        """
        print(f"EventBus.set_min_log_level('{level}') is deprecated. Set this on the EventLog class instead.")
        EventLog.set_min_log_level(level)

    @classmethod
    def get_subscribers(cls, event_type: str = None) -> Dict | List:
        """Get current subscribers for debugging."""
        instance = cls._get_instance()
        if event_type:
            return instance._subscribers.get(event_type, [])
        return instance._subscribers

    @classmethod
    def clear_subscribers(cls, event_type: str = None):
        """Clear subscribers (useful for testing)."""
        instance = cls._get_instance()
        if event_type:
            if event_type in instance._subscribers:
                del instance._subscribers[event_type]
        else:
            instance._subscribers.clear()

    @classmethod
    def get_queue_size(cls) -> int:
        """Get the current number of events in the queue."""
        instance = cls._get_instance()
        return instance._event_queue.qsize()

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Get the latest performance metrics."""
        instance = cls._get_instance()
        return instance._metrics.get_stats()

class EventLog:
    """Centralized logging system.
    Stores log events in memory with rotation, writes to a log file, and broadcasts events via EventBus.
    Provides helper methods to emit specialized events (analysis, trade, data update, LLM activity) and export logs.
    """
    logs = []  # Store all log events as dicts for export
    max_logs = 10000  # Prevent memory issues
    log_level_order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    min_log_level = "DEBUG"

    @classmethod
    def set_min_log_level(cls, level: str):
        """Set the minimum log level to process and publish."""
        if level in cls.log_level_order:
            cls.min_log_level = level

    @staticmethod
    def emit(level: str, message: str, extra: Dict = None):
        """Enhanced emit with better event broadcasting and pre-filtering."""
        # Log level filtering is now done at the source (here)
        try:
            min_idx = EventLog.log_level_order.index(EventLog.min_log_level)
            evt_idx = EventLog.log_level_order.index(level)
            if evt_idx < min_idx:
                return
        except ValueError:
            # If level is not a standard log level, process it anyway
            pass

        event = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'level': level,
            'message': message,
        }
        if extra:
            event.update(extra)
        
        # Add to logs with rotation
        EventLog.logs.append(event)
        if len(EventLog.logs) > EventLog.max_logs:
            EventLog.logs = EventLog.logs[-EventLog.max_logs:]
        
        # Console output
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        
        # Broadcast to EventBus subscribers.
        # Publish to the specific level channel (e.g., "INFO")
        EventBus.publish(level, event, source="EventLog")
        # Also publish to a generic channel for anyone listening to all logs.
        EventBus.publish("LOG_EVENT", event, source="EventLog")
        
        # File logging (non-blocking)
        try:
            with open("shnifter_log.txt", "a", encoding='utf-8') as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to write log to file: {e}")

    @staticmethod
    def log_event(level: str, message: str, **kwargs):
        """Alias for emit with keyword args"""
        EventLog.emit(level, message, kwargs)

    @staticmethod
    def emit_analysis_result(ticker: str, decision: str, confidence: float, llm_summary: str):
        """Emit specialized analysis result event"""
        event_data = {
            'ticker': ticker,
            'decision': decision,
            'confidence': confidence,
            'llm_summary': llm_summary,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the analysis
        EventLog.emit("INFO", f"Analysis complete for {ticker}: {decision} (confidence: {confidence:.2f})")
        
        # Broadcast specialized event
        EventBus.publish("ANALYSIS_COMPLETE", event_data, source="emit_analysis_result")

    @staticmethod
    def emit_trade_signal(ticker: str, signal: str, price: float, reasoning: str):
        """Emit trade signal event"""
        event_data = {
            'ticker': ticker,
            'signal': signal,
            'price': price,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
        
        EventLog.emit("INFO", f"Trade signal: {signal} {ticker} @ ${price:.2f}")
        EventBus.publish("TRADE_SIGNAL", event_data, source="emit_trade_signal")

    @staticmethod
    def emit_data_update(source: str, ticker: str, data_type: str, records: int):
        """Emit data update event"""
        event_data = {
            'source': source,
            'ticker': ticker,
            'data_type': data_type,
            'records': records,
            'timestamp': datetime.now().isoformat()
        }
        
        EventLog.emit("DEBUG", f"Data updated: {source} - {ticker} ({records} {data_type} records)")
        EventBus.publish("DATA_UPDATE", event_data, source="emit_data_update")

    @staticmethod
    def emit_llm_activity(model: str, prompt_type: str, response_time: float, success: bool):
        """Emit LLM activity event"""
        event_data = {
            'model': model,
            'prompt_type': prompt_type,
            'response_time': response_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        status = "success" if success else "failed"
        EventLog.emit("INFO", f"LLM {model} {prompt_type} {status} ({response_time:.2f}s)")
        EventBus.publish("LLM_ACTIVITY", event_data, source="emit_llm_activity")

    @staticmethod
    def get_recent_logs(level: str = None, limit: int = 100) -> List[Dict]:
        """Get recent logs, optionally filtered by level"""
        if level:
            filtered_logs = [log for log in EventLog.logs if log.get('level') == level]
            return filtered_logs[-limit:]
        return EventLog.logs[-limit:]

    @staticmethod
    def clear_logs():
        """Clear all logs"""
        EventLog.logs = []
        EventLog.emit("INFO", "Event logs cleared")

    @staticmethod
    def export_logs_txt(filename='shnifter_log.txt'):
        with open(filename, 'w', encoding='utf-8') as f:
            for event in EventLog.logs:
                f.write(f"[{event['timestamp']}] [{event['level']}] {event['message']}\n")

    @staticmethod
    def export_logs_json(filename='shnifter_log.json'):
        if EventLog.logs:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(EventLog.logs, f, indent=2, ensure_ascii=False)

    @staticmethod
    def export_logs_csv(filename='shnifter_log.csv'):
        import csv
        if EventLog.logs:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=EventLog.logs[0].keys())
                writer.writeheader()
                writer.writerows(EventLog.logs)

class ShnifterEventEmitter(QObject):
    """Qt signal emitter for GUI integration.
    Bridges EventBus events into PySide6 signals for log messages, analysis results, trade signals, data updates, and LLM activity.
    Implemented as a singleton to ensure consistent subscriptions across the application.
    """
    
    # Log signals
    log_message = Signal(dict)  # Emit log events
    error_occurred = Signal(str)  # Emit errors
    
    # Trading signals
    analysis_complete = Signal(dict)  # Analysis results
    trade_signal = Signal(dict)  # Trade signals
    data_updated = Signal(dict)  # Data updates
    
    # LLM signals
    llm_activity = Signal(dict)  # LLM activity
    llm_response = Signal(str, str)  # model, response
    
    # UI signals
    status_changed = Signal(str)  # Status updates
    progress_updated = Signal(int)  # Progress updates
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True
            self._setup_event_subscriptions()
    
    def _setup_event_subscriptions(self):
        """Wire EventBus to Qt signals"""
        # Subscribe to all log events
        EventBus.subscribe("LOG_EVENT", self._on_log_event)
        EventBus.subscribe("ANALYSIS_COMPLETE", self._on_analysis_complete)
        EventBus.subscribe("TRADE_SIGNAL", self._on_trade_signal)
        EventBus.subscribe("DATA_UPDATE", self._on_data_update)
        EventBus.subscribe("LLM_ACTIVITY", self._on_llm_activity)
        
        # Subscribe to specific log levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            EventBus.subscribe(level, self._on_log_message)
    
    def _on_log_event(self, event_data):
        """Handle general log events"""
        self.log_message.emit(event_data)
        
        if event_data.get('level') == 'ERROR':
            self.error_occurred.emit(event_data.get('message', 'Unknown error'))
    
    def _on_log_message(self, event_data):
        """Handle specific log level messages"""
        self.log_message.emit(event_data)
    
    def _on_analysis_complete(self, event_data):
        """Handle analysis completion"""
        self.analysis_complete.emit(event_data)
    
    def _on_trade_signal(self, event_data):
        """Handle trade signals"""
        self.trade_signal.emit(event_data)
    
    def _on_data_update(self, event_data):
        """Handle data updates"""
        self.data_updated.emit(event_data)
    
    def _on_llm_activity(self, event_data):
        """Handle LLM activity"""
        self.llm_activity.emit(event_data)
        
        if event_data.get('success'):
            self.llm_response.emit(
                event_data.get('model', 'unknown'),
                f"Completed {event_data.get('prompt_type', 'request')} in {event_data.get('response_time', 0):.2f}s"
            )

# Global singleton instance
event_emitter = ShnifterEventEmitter()
