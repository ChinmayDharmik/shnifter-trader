"""
LLM Manager

This module provides a unified interface for interacting with various LLM providers,
including BitNet, Ollama, OpenAI, and others. It handles provider registration,
fallback logic, error handling, model selection, async processing, and caching.
"""

import os
import sys
import time
import json
import yaml
import logging
import threading
import queue
import asyncio
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Import component registry (fallback if not available)
try:
    from src.core.component_registry import ComponentInterface, register_component
except ImportError:
    # Fallback for standalone usage
    class ComponentInterface:
        def __init__(self):
            self._initialized = False
        def get_status(self):
            return {"status": "active"}
        def emit_event(self, event_type, data):
            pass
    def register_component(name, dependencies=None, auto_initialize=False):
        def decorator(cls):
            return cls
        return decorator

logger = logging.getLogger(__name__)

@register_component("llm_manager", dependencies=[], auto_initialize=True)
class LLMManager(ComponentInterface):
    """
    LLM Manager for Shnifter Trader
    
    Provides a unified interface for interacting with various LLM providers,
    including BitNet, Ollama, OpenAI, and others. Features async processing,
    caching, fallback logic, and worker threads.
    """
    
    def __init__(self):
        """Initialize the LLM Manager."""
        super().__init__()
        self._component_id = "llm_manager"
        self._config = None
        self._providers = {}
        self._provider_order = []
        self._default_provider = None
        self._status = "Created"
        self._lock = threading.RLock()
        self._retry_count = 3
        self._retry_delay = 1.0  # seconds
        
        # Async processing
        self._async_queue = queue.Queue()
        self._async_workers = []
        self._async_running = False
        self._max_workers = 4
        
        # Caching
        self._cache = {}
        self._cache_enabled = True
        self._cache_ttl = 3600  # 1 hour
        self._cache_lock = threading.RLock()
        
        logger.info("LLMManager created")
    
    def initialize(self) -> bool:
        """
        Initialize the LLM Manager.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Load configuration
            self._config = self._load_config()
            
            # Set retry parameters
            self._retry_count = self._config.get("retry_count", 3)
            self._retry_delay = self._config.get("retry_delay", 1.0)
            
            # Set provider order
            self._provider_order = self._config.get("provider_order", ["bitnet", "ollama", "openai"])
            
            # Set default provider
            self._default_provider = self._config.get("default_provider", "bitnet")
            
            # Initialize built-in providers
            self._initialize_builtin_providers()
            
            # Initialize async workers
            self._initialize_async_workers()
            
            self._initialized = True
            self._status = "Initialized"
            logger.info("LLMManager initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing LLMManager: {e}")
            self._status = f"Initialization failed: {e}"
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load LLM Manager configuration.
        
        Returns:
            Dict[str, Any]: Configuration dictionary.
        """
        try:
            # Determine config path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            src_dir = os.path.dirname(os.path.dirname(current_dir))  # src directory
            project_root = os.path.dirname(src_dir)  # project root
            config_path = os.path.join(project_root, "config", "api_keys.yaml")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            llm_config = config.get("llm_manager", {})
            logger.info(f"Loaded LLM Manager configuration from {config_path}")
            return llm_config
        except Exception as e:
            logger.error(f"Error loading LLM Manager configuration: {e}")
            return {}
    
    def _initialize_builtin_providers(self):
        """Initialize built-in LLM providers."""
        try:
            # Initialize OpenAI provider if configured
            if "openai" in self._provider_order and self._config.get("openai", {}).get("enabled", False):
                self._initialize_openai_provider()
            
            # Initialize Ollama provider if configured
            if "ollama" in self._provider_order and self._config.get("ollama", {}).get("enabled", False):
                self._initialize_ollama_provider()
            
            # BitNet provider is expected to be registered externally by the BitNetOptimizer component
            
            logger.info(f"Initialized built-in providers: {list(self._providers.keys())}")
        except Exception as e:
            logger.error(f"Error initializing built-in providers: {e}")
    
    def _initialize_openai_provider(self):
        """Initialize OpenAI provider."""
        try:
            # Import OpenAI module
            from src.core.llm.openai_utils import generate_openai_response
            
            # Register provider
            self.register_provider("openai", lambda prompt, **kwargs: 
                self._openai_inference(prompt, **kwargs))
            
            logger.info("Initialized OpenAI provider")
        except ImportError:
            logger.warning("OpenAI utils not available, skipping provider initialization")
        except Exception as e:
            logger.error(f"Error initializing OpenAI provider: {e}")
    
    def _initialize_ollama_provider(self):
        """Initialize Ollama provider."""
        try:
            # Import Ollama module
            from src.core.llm.ollama_provider import OllamaProvider
            
            # Create provider instance
            ollama_config = self._config.get("ollama", {})
            ollama_provider = OllamaProvider(ollama_config)
            
            # Register provider
            self.register_provider("ollama", lambda prompt, **kwargs: 
                self._ollama_inference(prompt, ollama_provider, **kwargs))
            
            logger.info("Initialized Ollama provider")
        except ImportError:
            logger.warning("Ollama provider not available, creating fallback implementation")
            self._create_ollama_fallback()
        except Exception as e:
            logger.error(f"Error initializing Ollama provider: {e}")
            self._create_ollama_fallback()
    
    def _create_ollama_fallback(self):
        """Create fallback Ollama provider implementation."""
        try:
            # Register fallback provider
            self.register_provider("ollama", lambda prompt, **kwargs: 
                self._ollama_fallback_inference(prompt, **kwargs))
            
            logger.info("Created fallback Ollama provider")
        except Exception as e:
            logger.error(f"Error creating fallback Ollama provider: {e}")
    
    def _openai_inference(self, prompt: str, **kwargs) -> Tuple[bool, str]:
        """
        Run inference with OpenAI.
        
        Args:
            prompt (str): Prompt to process.
            **kwargs: Additional parameters.
                
        Returns:
            Tuple[bool, str]: Success flag and response text.
        """
        try:
            # Import OpenAI module
            from src.core.llm.openai_utils import generate_openai_response
            
            # Get parameters
            model = kwargs.get("model", self._config.get("openai", {}).get("model", "gpt-3.5-turbo"))
            temperature = kwargs.get("temperature", self._config.get("openai", {}).get("temperature", 0.7))
            max_tokens = kwargs.get("max_tokens", self._config.get("openai", {}).get("max_tokens", 1000))
            
            # Run inference
            logger.info(f"Running OpenAI inference with model {model}")
            response = generate_openai_response(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            
            return True, response
        except Exception as e:
            logger.error(f"Error running OpenAI inference: {e}")
            return False, f"Error running OpenAI inference: {e}"
    
    def _ollama_inference(self, prompt: str, provider, **kwargs) -> Tuple[bool, str]:
        """
        Run inference with Ollama.
        
        Args:
            prompt (str): Prompt to process.
            provider: Ollama provider instance.
            **kwargs: Additional parameters.
                
        Returns:
            Tuple[bool, str]: Success flag and response text.
        """
        try:
            # Get parameters
            model = kwargs.get("model", self._config.get("ollama", {}).get("model", "llama2"))
            
            # Run inference
            logger.info(f"Running Ollama inference with model {model}")
            return provider.run_inference(prompt, model=model, **kwargs)
        except Exception as e:
            logger.error(f"Error running Ollama inference: {e}")
            return False, f"Error running Ollama inference: {e}"
    
    def _ollama_fallback_inference(self, prompt: str, **kwargs) -> Tuple[bool, str]:
        """
        Run fallback inference for Ollama when the provider is not available.
        
        Args:
            prompt (str): Prompt to process.
            **kwargs: Additional parameters.
                
        Returns:
            Tuple[bool, str]: Success flag and response text.
        """
        try:
            # Try to use HTTP API directly
            import requests
            
            # Get parameters
            model = kwargs.get("model", self._config.get("ollama", {}).get("model", "llama2"))
            temperature = kwargs.get("temperature", self._config.get("ollama", {}).get("temperature", 0.7))
            max_tokens = kwargs.get("max_tokens", self._config.get("ollama", {}).get("max_tokens", 1000))
            
            # Get API endpoint
            endpoint = self._config.get("ollama", {}).get("endpoint", "http://localhost:11434/api/generate")
            
            # Prepare request
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # Send request
            logger.info(f"Running Ollama fallback inference with model {model}")
            response = requests.post(endpoint, json=payload)
            
            # Check response
            if response.status_code == 200:
                result = response.json()
                return True, result.get("response", "")
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return False, f"Ollama API error: {response.status_code}"
        except Exception as e:
            logger.error(f"Error running Ollama fallback inference: {e}")
            return False, f"Error running Ollama fallback inference: {e}"
    
    def register_provider(self, provider_name: str, inference_func: Callable) -> bool:
        """
        Register an LLM provider.
        
        Args:
            provider_name (str): Name of the provider.
            inference_func (Callable): Function to call for inference.
                
        Returns:
            bool: True if registration was successful, False otherwise.
        """
        with self._lock:
            try:
                self._providers[provider_name] = inference_func
                logger.info(f"Registered LLM provider: {provider_name}")
                return True
            except Exception as e:
                logger.error(f"Error registering LLM provider {provider_name}: {e}")
                return False
    
    def unregister_provider(self, provider_name: str) -> bool:
        """
        Unregister an LLM provider.
        
        Args:
            provider_name (str): Name of the provider to unregister.
                
        Returns:
            bool: True if unregistration was successful, False otherwise.
        """
        with self._lock:
            try:
                if provider_name in self._providers:
                    del self._providers[provider_name]
                    logger.info(f"Unregistered LLM provider: {provider_name}")
                    return True
                else:
                    logger.warning(f"Provider {provider_name} not found")
                    return False
            except Exception as e:
                logger.error(f"Error unregistering LLM provider {provider_name}: {e}")
                return False
    
    def list_providers(self) -> List[str]:
        """
        List registered LLM providers.
        
        Returns:
            List[str]: List of provider names.
        """
        with self._lock:
            return list(self._providers.keys())
    
    def set_provider_order(self, provider_order: List[str]) -> bool:
        """
        Set the order of providers for fallback.
        
        Args:
            provider_order (List[str]): List of provider names in fallback order.
                
        Returns:
            bool: True if successful, False otherwise.
        """
        with self._lock:
            try:
                # Validate providers
                for provider in provider_order:
                    if provider not in self._providers:
                        logger.warning(f"Provider {provider} not registered")
                
                # Set order
                self._provider_order = provider_order
                logger.info(f"Set provider order: {provider_order}")
                return True
            except Exception as e:
                logger.error(f"Error setting provider order: {e}")
                return False
    
    def set_default_provider(self, provider_name: str) -> bool:
        """
        Set the default LLM provider.
        
        Args:
            provider_name (str): Name of the provider to set as default.
                
        Returns:
            bool: True if successful, False otherwise.
        """
        with self._lock:
            try:
                if provider_name not in self._providers:
                    logger.warning(f"Provider {provider_name} not registered")
                    return False
                
                self._default_provider = provider_name
                logger.info(f"Set default provider: {provider_name}")
                return True
            except Exception as e:
                logger.error(f"Error setting default provider: {e}")
                return False
    
    def get_default_provider(self) -> str:
        """
        Get the default LLM provider.
        
        Returns:
            str: Name of the default provider.
        """
        return self._default_provider
    
    def generate_response(self, prompt: str, provider: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response using the specified provider or fallback chain.
        
        Args:
            prompt (str): Prompt to process.
            provider (str, optional): Provider to use. If not specified, uses default provider.
            **kwargs: Additional parameters for the provider.
                
        Returns:
            str: Generated response.
        """
        try:
            # Check if initialized
            if not self._initialized:
                logger.error("LLMManager not initialized")
                return "Error: LLMManager not initialized"
            
            # Check cache
            if self._cache_enabled:
                cache_key = self._generate_cache_key(prompt, provider, **kwargs)
                cached_response = self._check_cache(cache_key)
                if cached_response:
                    logger.info(f"Cache hit for prompt: {prompt[:50]}")
                    return cached_response
            
            # Determine provider
            selected_provider = provider or self._default_provider
            
            # Check if provider exists
            if selected_provider not in self._providers:
                logger.warning(f"Provider {selected_provider} not found, using fallback chain")
                return self._generate_with_fallback(prompt, **kwargs)
            
            # Try with selected provider
            success, response = self._generate_with_retries(selected_provider, prompt, **kwargs)
            if success:
                # Update cache
                if self._cache_enabled:
                    self._store_in_cache(cache_key, response)
                return response
            
            # If failed, use fallback chain
            logger.warning(f"Provider {selected_provider} failed, using fallback chain")
            return self._generate_with_fallback(prompt, exclude=[selected_provider], **kwargs)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {e}"
    
    def _start_async_workers(self):
        """Start async worker threads for background processing"""
        if self._async_running:
            return
            
        self._async_running = True
        for i in range(self._max_workers):
            worker = threading.Thread(target=self._async_worker, daemon=True)
            worker.start()
            self._async_workers.append(worker)
            
        logger.info(f"Started {self._max_workers} async workers")
    
    def _async_worker(self):
        """Async worker thread function"""
        while self._async_running:
            try:
                # Get task from queue
                task = self._async_queue.get(timeout=1)
                
                prompt = task.get("prompt")
                provider = task.get("provider")
                callback = task.get("callback")
                error_callback = task.get("error_callback")
                kwargs = task.get("kwargs", {})
                
                try:
                    # Generate response
                    response = self.generate_response(prompt, provider, **kwargs)
                    
                    # Call success callback
                    if callback:
                        callback(response)
                        
                except Exception as e:
                    logger.error(f"Error in async worker: {e}")
                    if error_callback:
                        error_callback(e)
                        
                finally:
                    self._async_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in async worker loop: {e}")
    
    def generate_response_async(self, prompt: str, callback: Callable, 
                              error_callback: Callable = None, 
                              provider: Optional[str] = None, **kwargs):
        """Generate response asynchronously"""
        # Start workers if not running
        if not self._async_running:
            self._start_async_workers()
            
        # Add task to queue
        task = {
            "prompt": prompt,
            "provider": provider,
            "callback": callback,
            "error_callback": error_callback,
            "kwargs": kwargs
        }
        self._async_queue.put(task)
    
    def _generate_cache_key(self, prompt: str, provider: str, **kwargs) -> str:
        """Generate cache key for prompt and settings"""
        # Create deterministic key excluding non-deterministic params
        cache_data = {
            "prompt": prompt,
            "provider": provider,
        }
        # Add only deterministic kwargs
        for key, value in kwargs.items():
            if key not in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                cache_data[key] = value
                
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """Check if response is in cache"""
        if not self._cache_enabled:
            return None
            
        with self._cache_lock:
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                # Check TTL
                if time.time() - entry["timestamp"] < self._cache_ttl:
                    logger.info("Cache hit")
                    return entry["response"]
                else:
                    # Expired, remove from cache
                    del self._cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, response: str):
        """Store response in cache"""
        if not self._cache_enabled:
            return
            
        with self._cache_lock:
            self._cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            # Cleanup old entries (simple LRU)
            if len(self._cache) > 1000:  # Max 1000 entries
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k]["timestamp"])
                del self._cache[oldest_key]
    
    def clear_cache(self):
        """Clear the response cache"""
        with self._cache_lock:
            self._cache.clear()
        logger.info("Cache cleared")
    
    def set_cache_enabled(self, enabled: bool):
        """Enable or disable caching"""
        self._cache_enabled = enabled
        logger.info(f"Cache {'enabled' if enabled else 'disabled'}")
    
    def stop_async_workers(self):
        """Stop all async workers"""
        self._async_running = False
        for worker in self._async_workers:
            if worker.is_alive():
                worker.join(timeout=1)
        self._async_workers.clear()
        logger.info("Async workers stopped")
    
    def _initialize_async_workers(self):
        """Initialize async worker configuration"""
        # Set async configuration from config
        self._max_workers = self._config.get("max_workers", 4)
        self._cache_enabled = self._config.get("cache_enabled", True) 
        self._cache_ttl = self._config.get("cache_ttl", 3600)
        
        # Start async workers if enabled
        if self._config.get("async_enabled", True):
            self._start_async_workers()
            
        logger.info("Async workers initialized")
