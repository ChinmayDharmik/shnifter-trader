"""
Offline LLM Manager for KingAI AGI

This module provides the Offline LLM Manager for efficient model inference
across all agents in the system, with support for llama.cpp, Ollama, and
other local LLM backends.

Features:
- Support for multiple LLM backends (llama.cpp, Ollama)
- Synchronous and asynchronous inference
- Fallback logic for reliability
- Confidence scoring for responses
- MemSys integration for caching and logging
- EventBus integration for event-driven processing
"""

import os
import logging
import uuid
import json
import datetime
import threading
import queue
import subprocess
import requests
import time
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OfflineLLMManager")

class OfflineLLMManager:
    """
    Offline LLM Manager for efficient model inference across all agents.
    
    This manager provides a unified interface for interacting with various
    local LLM backends, including llama.cpp and Ollama.
    """
    
    def __init__(self, event_bus=None, memsys=None, config: Dict = None):
        """
        Initialize the Offline LLM Manager.
        
        Args:
            event_bus: EventBus instance for event-driven processing
            memsys: MemSys instance for caching and logging
            config: Configuration options
        """
        self.event_bus = event_bus
        self.memsys = memsys
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            "default_backend": "ollama",  # Default backend to use
            "backends": {
                "ollama": {
                    "enabled": True,
                    "api_base": "http://localhost:11434/api",
                    "default_model": "llama2",
                    "timeout": 60,
                    "max_tokens": 2048
                },
                "llama_cpp": {
                    "enabled": True,
                    "model_path": "/home/ubuntu/models/llama-2-7b-chat.gguf",
                    "context_size": 4096,
                    "threads": 4,
                    "max_tokens": 2048
                }
            },
            "cache_enabled": True,
            "cache_ttl": 3600,  # 1 hour
            "fallback_enabled": True,
            "async_enabled": True,
            "max_workers": 4
        }
        
        # Merge provided config with defaults
        self._merge_config()
        
        # Initialize backends
        self._backends = {}
        self._initialize_backends()
        
        # Async processing
        self._async_queue = queue.Queue()
        self._async_workers = []
        self._async_running = False
        
        # Start async workers if enabled
        if self.config.get("async_enabled", True):
            self._start_async_workers()
        
        # Register event handlers if event_bus is provided
        if self.event_bus:
            self._register_event_handlers()
            logger.info("OfflineLLMManager registered with EventBus")
    
    def _merge_config(self):
        """Merge provided config with defaults."""
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Merge nested dictionaries
                for nested_key, nested_value in value.items():
                    if nested_key not in self.config[key]:
                        self.config[key][nested_key] = nested_value
    
    def _initialize_backends(self):
        """Initialize LLM backends based on configuration."""
        backends_config = self.config.get("backends", {})
        
        # Initialize Ollama backend
        if backends_config.get("ollama", {}).get("enabled", False):
            try:
                from .backends.ollama import OllamaBackend
                self._backends["ollama"] = OllamaBackend(backends_config.get("ollama", {}))
                logger.info("Ollama backend initialized")
            except ImportError:
                logger.warning("Failed to import OllamaBackend, creating stub")
                self._backends["ollama"] = self._create_backend_stub("ollama")
        
        # Initialize llama.cpp backend
        if backends_config.get("llama_cpp", {}).get("enabled", False):
            try:
                from .backends.llama_cpp import LlamaCppBackend
                self._backends["llama_cpp"] = LlamaCppBackend(backends_config.get("llama_cpp", {}))
                logger.info("llama.cpp backend initialized")
            except ImportError:
                logger.warning("Failed to import LlamaCppBackend, creating stub")
                self._backends["llama_cpp"] = self._create_backend_stub("llama_cpp")
    
    def _create_backend_stub(self, backend_name: str):
        """
        Create a stub backend for testing or when actual backend is not available.
        
        Args:
            backend_name: Name of the backend
            
        Returns:
            Dict: Backend stub with required methods
        """
        return {
            "name": backend_name,
            "generate": lambda prompt, options=None: {
                "text": f"Stub response from {backend_name} backend",
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10},
                "model": f"{backend_name}_stub"
            },
            "generate_async": lambda prompt, callback, options=None: callback({
                "text": f"Async stub response from {backend_name} backend",
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": 10, "total_tokens": len(prompt.split()) + 10},
                "model": f"{backend_name}_stub"
            }),
            "is_available": lambda: True
        }
    
    def _register_event_handlers(self):
        """Register event handlers with EventBus."""
        if not self.event_bus:
            return
        
        # Register for LLM events
        self.event_bus.subscribe("llm.generate", self.handle_generate_event)
        self.event_bus.subscribe("llm.generate_async", self.handle_generate_async_event)
        self.event_bus.subscribe("llm.status", self.handle_status_event)
        self.event_bus.subscribe("llm.list_models", self.handle_list_models_event)
    
    def handle_generate_event(self, data):
        """
        Handle generate event.
        
        Args:
            data: Event data containing prompt and options
        """
        if not isinstance(data, dict):
            logger.error("Invalid event data format")
            return
        
        prompt = data.get("prompt")
        if not prompt:
            logger.error("No prompt provided")
            return
        
        options = data.get("options", {})
        
        try:
            result = self.generate_text(prompt, options)
            
            # Emit result event
            if self.event_bus:
                self.event_bus.emit("llm.generate_result", {
                    "success": True,
                    "result": result,
                    "request_id": data.get("request_id")
                })
        
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            
            # Emit error event
            if self.event_bus:
                self.event_bus.emit("llm.generate_result", {
                    "success": False,
                    "error": str(e),
                    "request_id": data.get("request_id")
                })
    
    def handle_generate_async_event(self, data):
        """
        Handle generate async event.
        
        Args:
            data: Event data containing prompt and options
        """
        if not isinstance(data, dict):
            logger.error("Invalid event data format")
            return
        
        prompt = data.get("prompt")
        if not prompt:
            logger.error("No prompt provided")
            return
        
        options = data.get("options", {})
        request_id = data.get("request_id")
        
        # Define callback function
        def callback(result):
            if self.event_bus:
                self.event_bus.emit("llm.generate_async_result", {
                    "success": True,
                    "result": result,
                    "request_id": request_id
                })
        
        # Define error callback function
        def error_callback(error):
            if self.event_bus:
                self.event_bus.emit("llm.generate_async_result", {
                    "success": False,
                    "error": str(error),
                    "request_id": request_id
                })
        
        try:
            self.generate_text_async(prompt, callback, error_callback, options)
            
            # Emit acknowledgment event
            if self.event_bus:
                self.event_bus.emit("llm.generate_async_ack", {
                    "success": True,
                    "message": "Async generation started",
                    "request_id": request_id
                })
        
        except Exception as e:
            logger.error(f"Error starting async generation: {str(e)}")
            
            # Emit error event
            if self.event_bus:
                self.event_bus.emit("llm.generate_async_ack", {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                })
    
    def handle_status_event(self, data):
        """
        Handle status event.
        
        Args:
            data: Event data
        """
        try:
            status = self.get_status()
            
            # Emit result event
            if self.event_bus:
                self.event_bus.emit("llm.status_result", {
                    "success": True,
                    "status": status,
                    "request_id": data.get("request_id") if isinstance(data, dict) else None
                })
        
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            
            # Emit error event
            if self.event_bus:
                self.event_bus.emit("llm.status_result", {
                    "success": False,
                    "error": str(e),
                    "request_id": data.get("request_id") if isinstance(data, dict) else None
                })
    
    def handle_list_models_event(self, data):
        """
        Handle list models event.
        
        Args:
            data: Event data
        """
        try:
            models = self.list_models()
            
            # Emit result event
            if self.event_bus:
                self.event_bus.emit("llm.list_models_result", {
                    "success": True,
                    "models": models,
                    "request_id": data.get("request_id") if isinstance(data, dict) else None
                })
        
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            
            # Emit error event
            if self.event_bus:
                self.event_bus.emit("llm.list_models_result", {
                    "success": False,
                    "error": str(e),
                    "request_id": data.get("request_id") if isinstance(data, dict) else None
                })
    
    def generate_text(self, prompt: str, options: Dict = None) -> Dict:
        """
        Generate text using the configured LLM backend.
        
        Args:
            prompt: Input prompt
            options: Generation options
            
        Returns:
            Dict: Generated text and metadata
        """
        if options is None:
            options = {}
        
        # Check cache if enabled
        if self.config.get("cache_enabled", True):
            cache_result = self._check_cache(prompt, options)
            if cache_result:
                logger.info("Cache hit for prompt")
                return cache_result
        
        # Get backend
        backend_name = options.get("backend", self.config.get("default_backend", "ollama"))
        backend = self._get_backend(backend_name)
        
        try:
            # Generate text
            result = backend["generate"](prompt, options)
            
            # Add metadata
            result["backend"] = backend_name
            result["timestamp"] = datetime.datetime.now().isoformat()
            result["prompt"] = prompt
            result["options"] = options
            
            # Calculate confidence score
            result["confidence"] = self._calculate_confidence(result)
            
            # Store in cache if enabled
            if self.config.get("cache_enabled", True):
                self._store_in_cache(prompt, options, result)
            
            # Log to MemSys if available
            if self.memsys:
                self._log_to_memsys(prompt, options, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating text with {backend_name} backend: {str(e)}")
            
            # Try fallback if enabled
            if self.config.get("fallback_enabled", True):
                return self._fallback_generate(prompt, options, backend_name)
            
            raise
    
    def generate_text_async(self, prompt: str, callback: Callable, 
                          error_callback: Callable = None, options: Dict = None):
        """
        Generate text asynchronously using the configured LLM backend.
        
        Args:
            prompt: Input prompt
            callback: Callback function to call with the result
            error_callback: Callback function to call on error
            options: Generation options
        """
        if options is None:
            options = {}
        
        if not self.config.get("async_enabled", True):
            raise ValueError("Async generation is disabled")
        
        # Check cache if enabled
        if self.config.get("cache_enabled", True):
            cache_result = self._check_cache(prompt, options)
            if cache_result:
                logger.info("Cache hit for prompt (async)")
                callback(cache_result)
                return
        
        # Add to async queue
        self._async_queue.put({
            "prompt": prompt,
            "options": options,
            "callback": callback,
            "error_callback": error_callback
        })
    
    def _start_async_workers(self):
        """Start async worker threads."""
        if self._async_running:
            return
        
        self._async_running = True
        max_workers = self.config.get("max_workers", 4)
        
        for i in range(max_workers):
            worker = threading.Thread(target=self._async_worker, daemon=True)
            worker.start()
            self._async_workers.append(worker)
        
        logger.info(f"Started {max_workers} async workers")
    
    def _async_worker(self):
        """Async worker thread function."""
        while self._async_running:
            try:
                # Get task from queue
                task = self._async_queue.get(timeout=1)
                
                # Process task
                prompt = task["prompt"]
                options = task["options"]
                callback = task["callback"]
                error_callback = task["error_callback"]
                
                try:
                    # Generate text
                    result = self.generate_text(prompt, options)
                    
                    # Call callback
                    callback(result)
                
                except Exception as e:
                    logger.error(f"Error in async worker: {str(e)}")
                    
                    # Call error callback if provided
                    if error_callback:
                        error_callback(e)
                
                finally:
                    # Mark task as done
                    self._async_queue.task_done()
            
            except queue.Empty:
                # Queue is empty, continue
                pass
            
            except Exception as e:
                logger.error(f"Error in async worker loop: {str(e)}")
    
    def _get_backend(self, backend_name: str) -> Dict:
        """
        Get the specified backend.
        
        Args:
            backend_name: Backend name
            
        Returns:
            Dict: Backend object
        """
        if backend_name not in self._backends:
            raise ValueError(f"Backend not found: {backend_name}")
        
        backend = self._backends[backend_name]
        
        # Check if backend is available
        if not backend["is_available"]():
            raise ValueError(f"Backend not available: {backend_name}")
        
        return backend
    
    def _fallback_generate(self, prompt: str, options: Dict, failed_backend: str) -> Dict:
        """
        Try to generate text using fallback backends.
        
        Args:
            prompt: Input prompt
            options: Generation options
            failed_backend: Name of the failed backend
            
        Returns:
            Dict: Generated text and metadata
        """
        # Get available backends excluding the failed one
        available_backends = [name for name, backend in self._backends.items() 
                             if name != failed_backend and backend["is_available"]()]
        
        if not available_backends:
            raise ValueError("No fallback backends available")
        
        # Try each available backend
        last_error = None
        for backend_name in available_backends:
            try:
                # Update options with new backend
                fallback_options = options.copy()
                fallback_options["backend"] = backend_name
                
                # Generate text
                result = self.generate_text(prompt, fallback_options)
                
                # Add fallback metadata
                result["fallback"] = True
                result["original_backend"] = failed_backend
                
                logger.info(f"Fallback to {backend_name} successful")
                return result
            
            except Exception as e:
                logger.error(f"Fallback to {backend_name} failed: {str(e)}")
                last_error = e
        
        # All fallbacks failed
        if last_error:
            raise ValueError(f"All fallback backends failed: {str(last_error)}")
        else:
            raise ValueError("All fallback backends failed")
    
    def _calculate_confidence(self, result: Dict) -> float:
        """
        Calculate confidence score for a generation result.
        
        Args:
            result: Generation result
            
        Returns:
            float: Confidence score (0.0 to 1.0)
        """
        # Base confidence
        confidence = 0.7
        
        # Adjust based on text length
        text = result.get("text", "")
        if len(text) < 10:
            confidence *= 0.5
        elif len(text) > 1000:
            confidence *= 0.9
        
        # Adjust based on token usage
        usage = result.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        if completion_tokens < 10:
            confidence *= 0.6
        elif completion_tokens > 100:
            confidence *= 0.9
        
        # Cap at 1.0
        return min(1.0, confidence)
    
    def _check_cache(self, prompt: str, options: Dict) -> Optional[Dict]:
        """
        Check if a prompt is in the cache.
        
        Args:
            prompt: Input prompt
            options: Generation options
            
        Returns:
            Optional[Dict]: Cached result or None
        """
        if not self.memsys:
            return None
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, options)
        
        try:
            # Check if key exists in MemSys
            metadata = self.memsys.get_metadata(cache_key)
            if not metadata:
                return None
            
            # Check TTL
            timestamp = metadata.get("timestamp", 0)
            ttl = self.config.get("cache_ttl", 3600)
            if time.time() - timestamp > ttl:
                # Cache expired
                return None
            
            # Get cached result
            result = self.memsys.get_context(cache_key)
            if not result:
                return None
            
            # Add cache metadata
            result["cached"] = True
            result["cache_age"] = time.time() - timestamp
            
            return result
        
        except Exception as e:
            logger.error(f"Error checking cache: {str(e)}")
            return None
    
    def _store_in_cache(self, prompt: str, options: Dict, result: Dict):
        """
        Store a result in the cache.
        
        Args:
            prompt: Input prompt
            options: Generation options
            result: Generation result
        """
        if not self.memsys:
            return
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, options)
        
        try:
            # Prepare metadata
            metadata = {
                "type": "llm_cache",
                "timestamp": time.time(),
                "backend": result.get("backend", "unknown"),
                "model": result.get("model", "unknown")
            }
            
            # Store in MemSys
            self.memsys.store_context(
                cache_key,
                result,
                compression_ratio=0.7,
                metadata=metadata
            )
            
            logger.info(f"Stored result in cache: {cache_key}")
        
        except Exception as e:
            logger.error(f"Error storing in cache: {str(e)}")
    
    def _generate_cache_key(self, prompt: str, options: Dict) -> str:
        """
        Generate a cache key for a prompt and options.
        
        Args:
            prompt: Input prompt
            options: Generation options
            
        Returns:
            str: Cache key
        """
        # Create a string representation of the options
        # Exclude non-deterministic options like temperature
        cache_options = options.copy()
        for key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
            if key in cache_options:
                del cache_options[key]
        
        # Create a string to hash
        cache_str = f"{prompt}|{json.dumps(cache_options, sort_keys=True)}"
        
        # Generate hash
        cache_hash = hashlib.md5(cache_str.encode()).hexdigest()
        
        return f"llm_cache_{cache_hash}"
    
    def _log_to_memsys(self, prompt: str, options: Dict, result: Dict):
        """
        Log a generation result to MemSys.
        
        Args:
            prompt: Input prompt
            options: Generation options
            result: Generation result
        """
        if not self.memsys:
            return
        
        try:
            # Generate a unique log ID
            log_id = f"llm_log_{uuid.uuid4()}"
            
            # Prepare metadata
            metadata = {
                "type": "llm_log",
                "timestamp": time.time(),
                "backend": result.get("backend", "unknown"),
                "model": result.get("model", "unknown"),
                "confidence": result.get("confidence", 0.0)
            }
            
            # Store in MemSys
            self.memsys.store_context(
                log_id,
                {
                    "prompt": prompt,
                    "options": options,
                    "result": result
                },
                compression_ratio=0.5,
                metadata=metadata
            )
            
            logger.info(f"Logged generation to MemSys: {log_id}")
        
        except Exception as e:
            logger.error(f"Error logging to MemSys: {str(e)}")
    
    def get_status(self) -> Dict:
        """
        Get the status of the LLM manager and backends.
        
        Returns:
            Dict: Status information
        """
        # Check backend status
        backend_status = {}
        for name, backend in self._backends.items():
            try:
                available = backend["is_available"]()
                backend_status[name] = {
                    "available": available,
                    "error": None
                }
            except Exception as e:
                backend_status[name] = {
                    "available": False,
                    "error": str(e)
                }
        
        # Get async queue status
        async_status = {
            "enabled": self.config.get("async_enabled", True),
            "running": self._async_running,
            "workers": len(self._async_workers),
            "queue_size": self._async_queue.qsize()
        }
        
        return {
            "backends": backend_status,
            "async": async_status,
            "cache_enabled": self.config.get("cache_enabled", True),
            "fallback_enabled": self.config.get("fallback_enabled", True),
            "default_backend": self.config.get("default_backend", "ollama")
        }
    
    def list_models(self) -> List[Dict]:
        """
        List available models across all backends.
        
        Returns:
            List[Dict]: List of available models
        """
        models = []
        
        for backend_name, backend in self._backends.items():
            try:
                if not backend["is_available"]():
                    continue
                
                # Get models from backend
                if hasattr(backend, "list_models") and callable(backend["list_models"]):
                    backend_models = backend["list_models"]()
                    
                    # Add backend name to each model
                    for model in backend_models:
                        model["backend"] = backend_name
                    
                    models.extend(backend_models)
                else:
                    # Fallback for backends without list_models
                    backend_config = self.config.get("backends", {}).get(backend_name, {})
                    default_model = backend_config.get("default_model", f"{backend_name}_default")
                    
                    models.append({
                        "id": default_model,
                        "name": default_model,
                        "backend": backend_name,
                        "default": True
                    })
            
            except Exception as e:
                logger.error(f"Error listing models for {backend_name}: {str(e)}")
        
        return models
    
    def stop(self):
        """Stop the LLM manager and release resources."""
        # Stop async workers
        self._async_running = False
        
        # Wait for workers to finish
        for worker in self._async_workers:
            if worker.is_alive():
                worker.join(timeout=1)
        
        self._async_workers = []
        
        # Stop backends
        for backend_name, backend in self._backends.items():
            try:
                if hasattr(backend, "stop") and callable(backend["stop"]):
                    backend["stop"]()
            except Exception as e:
                logger.error(f"Error stopping {backend_name} backend: {str(e)}")
        
        logger.info("LLM manager stopped")


# Singleton instance
_offline_llm_manager_instance = None

def get_offline_llm_manager(event_bus=None, memsys=None, config: Dict = None):
    """
    Get or create the singleton OfflineLLMManager instance.
    
    Args:
        event_bus: EventBus instance
        memsys: MemSys instance
        config: Configuration options
        
    Returns:
        OfflineLLMManager: Singleton instance
    """
    global _offline_llm_manager_instance
    
    if _offline_llm_manager_instance is None:
        _offline_llm_manager_instance = OfflineLLMManager(event_bus, memsys, config)
    
    return _offline_llm_manager_instance
