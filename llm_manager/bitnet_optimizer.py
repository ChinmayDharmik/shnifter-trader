"""
BitNet Optimization Module

This module enhances the BitNet integration with the LLMManager,
providing improved performance, fallback logic, and model selection.
"""

import os
import sys
import json
import yaml
import logging
import subprocess
from typing import Dict, List, Any, Optional, Union, Tuple

# Import component registry
from src.core.component_registry import ComponentInterface, register_component

logger = logging.getLogger(__name__)

@register_component("bitnet_optimizer", dependencies=["llm_manager"], auto_initialize=False)
class BitNetOptimizer(ComponentInterface):
    """
    BitNet Optimizer
    
    Enhances BitNet integration with improved performance, fallback logic,
    and model selection capabilities.
    """
    
    def __init__(self, llm_manager=None):
        """
        Initialize the BitNet Optimizer.
        
        Args:
            llm_manager: LLMManager instance (injected by component registry).
        """
        super().__init__()
        self._component_id = "bitnet_optimizer"
        self._llm_manager = llm_manager
        self._config = None
        self._models = {}
        self._current_model = None
        self._quantization_options = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0"]
        self._status = "Created"
        
        logger.info("BitNetOptimizer created")
    
    def initialize(self) -> bool:
        """
        Initialize the BitNet Optimizer.
        
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Load configuration
            self._config = self._load_config()
            
            # Discover available models
            self._discover_models()
            
            # Set default model
            default_model = self._config.get("default_model")
            if default_model and default_model in self._models:
                self._current_model = default_model
            elif self._models:
                self._current_model = list(self._models.keys())[0]
            
            # Register with LLMManager if available
            if self._llm_manager:
                self._register_with_llm_manager()
            
            self._initialized = True
            self._status = "Initialized"
            logger.info("BitNetOptimizer initialized")
            return True
        except Exception as e:
            logger.error(f"Error initializing BitNetOptimizer: {e}")
            self._status = f"Initialization failed: {e}"
            return False
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load BitNet configuration.
        
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
            
            bitnet_config = config.get("bitnet", {})
            logger.info(f"Loaded BitNet configuration from {config_path}")
            return bitnet_config
        except Exception as e:
            logger.error(f"Error loading BitNet configuration: {e}")
            return {}
    
    def _discover_models(self):
        """Discover available BitNet models."""
        try:
            # Get model directory from config
            model_dir = self._config.get("model_dir")
            if not model_dir:
                # Try to find models in standard locations
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.dirname(os.path.dirname(current_dir))  # src directory
                project_root = os.path.dirname(src_dir)  # project root
                model_dir = os.path.join(project_root, "models")
            
            # Check if directory exists
            if not os.path.isdir(model_dir):
                logger.warning(f"Model directory {model_dir} not found")
                return
            
            # Find GGUF models
            for filename in os.listdir(model_dir):
                if filename.endswith(".gguf"):
                    model_path = os.path.join(model_dir, filename)
                    model_name = os.path.splitext(filename)[0]
                    self._models[model_name] = {
                        "path": model_path,
                        "name": model_name,
                        "size": os.path.getsize(model_path)
                    }
            
            logger.info(f"Discovered {len(self._models)} BitNet models")
        except Exception as e:
            logger.error(f"Error discovering BitNet models: {e}")
    
    def _register_with_llm_manager(self):
        """Register BitNet with LLMManager."""
        try:
            # Check if LLMManager has register_provider method
            if hasattr(self._llm_manager, "register_provider") and callable(self._llm_manager.register_provider):
                self._llm_manager.register_provider("bitnet", self.run_inference)
                logger.info("Registered BitNet with LLMManager")
            else:
                logger.warning("LLMManager does not support provider registration")
        except Exception as e:
            logger.error(f"Error registering BitNet with LLMManager: {e}")
    
    def run_inference(self, prompt: str, **kwargs) -> Tuple[bool, str]:
        """
        Run inference with BitNet.
        
        Args:
            prompt (str): Prompt to process.
            **kwargs: Additional parameters.
                
        Returns:
            Tuple[bool, str]: Success flag and response text.
        """
        try:
            # Check if initialized
            if not self._initialized:
                logger.error("BitNetOptimizer not initialized")
                return False, "BitNetOptimizer not initialized"
            
            # Check if model is selected
            if not self._current_model:
                logger.error("No BitNet model selected")
                return False, "No BitNet model selected"
            
            # Get model info
            model_info = self._models.get(self._current_model)
            if not model_info:
                logger.error(f"Model {self._current_model} not found")
                return False, f"Model {self._current_model} not found"
            
            # Get parameters
            model_path = model_info["path"]
            threads = kwargs.get("threads", self._config.get("threads", 4))
            context_size = kwargs.get("context_size", self._config.get("context_size", 2048))
            temperature = kwargs.get("temperature", self._config.get("temperature", 0.7))
            
            # Determine binary path
            binary_path = self._config.get("cli_binary")
            if not binary_path:
                # Try to find binary in standard locations
                current_dir = os.path.dirname(os.path.abspath(__file__))
                src_dir = os.path.dirname(os.path.dirname(current_dir))  # src directory
                project_root = os.path.dirname(src_dir)  # project root
                binary_path = os.path.join(project_root, "bin", "bitnet-cli")
            
            # Check if binary exists
            if not os.path.isfile(binary_path):
                logger.error(f"BitNet CLI binary {binary_path} not found")
                return False, f"BitNet CLI binary {binary_path} not found"
            
            # Prepare command
            cmd = [
                binary_path,
                "-m", model_path,
                "-t", str(threads),
                "-c", str(context_size),
                "--temp", str(temperature),
                "-p", prompt
            ]
            
            # Run command
            logger.info(f"Running BitNet inference with model {self._current_model}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Check result
            if result.returncode != 0:
                logger.error(f"BitNet inference failed: {result.stderr}")
                return False, f"BitNet inference failed: {result.stderr}"
            
            # Return output
            return True, result.stdout.strip()
        except Exception as e:
            logger.error(f"Error running BitNet inference: {e}")
            return False, f"Error running BitNet inference: {e}"
    
    def run_inference_python(self, prompt: str, **kwargs) -> Tuple[bool, str]:
        """
        Run inference with BitNet using Python-based approach.
        
        Args:
            prompt (str): Prompt to process.
            **kwargs: Additional parameters.
                
        Returns:
            Tuple[bool, str]: Success flag and response text.
        """
        try:
            # Check if initialized
            if not self._initialized:
                logger.error("BitNetOptimizer not initialized")
                return False, "BitNetOptimizer not initialized"
            
            # Check if model is selected
            if not self._current_model:
                logger.error("No BitNet model selected")
                return False, "No BitNet model selected"
            
            # Get model info
            model_info = self._models.get(self._current_model)
            if not model_info:
                logger.error(f"Model {self._current_model} not found")
                return False, f"Model {self._current_model} not found"
            
            # Get parameters
            model_path = model_info["path"]
            threads = kwargs.get("threads", self._config.get("threads", 4))
            context_size = kwargs.get("context_size", self._config.get("context_size", 2048))
            temperature = kwargs.get("temperature", self._config.get("temperature", 0.7))
            
            # Import llama_cpp
            try:
                from llama_cpp import Llama
            except ImportError:
                logger.error("llama_cpp not installed")
                return False, "llama_cpp not installed"
            
            # Load model
            logger.info(f"Loading BitNet model {self._current_model}")
            llm = Llama(
                model_path=model_path,
                n_ctx=context_size,
                n_threads=threads
            )
            
            # Run inference
            logger.info(f"Running BitNet inference with model {self._current_model}")
            output = llm(
                prompt,
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens", 1000),
                stop=kwargs.get("stop", [])
            )
            
            # Extract text
            if isinstance(output, dict) and "choices" in output:
                text = output["choices"][0]["text"]
            else:
                text = str(output)
            
            return True, text.strip()
        except Exception as e:
            logger.error(f"Error running BitNet Python inference: {e}")
            return False, f"Error running BitNet Python inference: {e}"
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available BitNet models.
        
        Returns:
            List[Dict[str, Any]]: List of model information dictionaries.
        """
        return list(self._models.values())
    
    def select_model(self, model_name: str) -> bool:
        """
        Select a BitNet model.
        
        Args:
            model_name (str): Name of the model to select.
                
        Returns:
            bool: True if selection was successful, False otherwise.
        """
        if model_name not in self._models:
            logger.error(f"Model {model_name} not found")
            return False
        
        self._current_model = model_name
        logger.info(f"Selected BitNet model {model_name}")
        return True
    
    def get_current_model(self) -> Optional[str]:
        """
        Get the currently selected model.
        
        Returns:
            Optional[str]: Name of the currently selected model, or None if no model is selected.
        """
        return self._current_model
    
    def get_quantization_options(self) -> List[str]:
        """
        Get available quantization options.
        
        Returns:
            List[str]: List of quantization options.
        """
        return self._quantization_options
    
    def optimize_parameters(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize parameters for a model.
        
        Args:
            model_name (str, optional): Name of the model to optimize.
                If not provided, uses the currently selected model.
                
        Returns:
            Dict[str, Any]: Optimized parameters.
        """
        model = model_name or self._current_model
        if not model or model not in self._models:
            logger.error(f"Model {model} not found")
            return {}
        
        # Get model info
        model_info = self._models[model]
        
        # Determine optimal parameters based on model size
        size_mb = model_info["size"] / (1024 * 1024)
        
        if size_mb < 500:
            # Small model
            return {
                "threads": min(4, os.cpu_count() or 4),
                "context_size": 2048,
                "temperature": 0.7
            }
        elif size_mb < 2000:
            # Medium model
            return {
                "threads": min(6, os.cpu_count() or 4),
                "context_size": 4096,
                "temperature": 0.8
            }
        else:
            # Large model
            return {
                "threads": min(8, os.cpu_count() or 4),
                "context_size": 8192,
                "temperature": 0.9
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get component status.
        
        Returns:
            Dict[str, Any]: Dictionary with status information.
        """
        status = super().get_status()
        status.update({
            "models_count": len(self._models),
            "current_model": self._current_model,
            "config_loaded": self._config is not None
        })
        return status
    
    def handle_event(self, event_type: str, event_data: Any) -> bool:
        """
        Handle an event.
        
        Args:
            event_type (str): Type of event.
            event_data (Any): Event data.
                
        Returns:
            bool: True if event was handled successfully, False otherwise.
        """
        if event_type == "llm_fallback_requested":
            # Handle fallback request
            prompt = event_data.get("prompt")
            if prompt:
                success, response = self.run_inference(prompt)
                if success:
                    self.emit_event("llm_fallback_response", {
                        "success": True,
                        "response": response,
                        "provider": "bitnet",
                        "model": self._current_model
                    })
                    return True
            
            self.emit_event("llm_fallback_response", {
                "success": False,
                "provider": "bitnet"
            })
            return False
        
        return True
