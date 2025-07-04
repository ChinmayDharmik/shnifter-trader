"""
llm_providers.py - Unified LLM Provider Interface for Shnifter Trader

This module defines a common interface for all LLM providers (Ollama, OpenAI, BitNet, etc.)
and provides concrete implementations for Ollama and OpenAI.
"""

import asyncio
import logging
import os
import random
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

import aiolimiter
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

class BaseLLMProvider:
    def initialize(self):
        raise NotImplementedError
    def list_models(self) -> List[str]:
        raise NotImplementedError
    def run_inference(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[bool, str]:
        raise NotImplementedError
    async def run_inference_async(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[bool, str]:
        """Async inference - default implementation falls back to sync"""
        return self.run_inference(prompt, model, **kwargs)
    async def batch_inference(self, prompts: List[str], model: Optional[str] = None, **kwargs) -> List[str]:
        """Batch inference - default implementation processes sequentially"""
        results = []
        for prompt in prompts:
            success, response = await self.run_inference_async(prompt, model, **kwargs)
            results.append(response if success else "")
        return results
    def get_status(self) -> Dict[str, Any]:
        raise NotImplementedError

class OllamaProvider(BaseLLMProvider):
    def __init__(self, config=None):
        self.config = config or {}
        self.endpoint = self.config.get("endpoint", "http://localhost:11434")
        self.api_base = f"{self.endpoint}/api"
        self.available_models = []
        self.default_model = self.config.get("model", "llama3")
        self.initialized = False
        self.status = "Created"
    def initialize(self):
        try:
            response = requests.get(f"{self.api_base}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                self.available_models = [m.get("name") for m in data.get("models", [])]
                self.initialized = True
                self.status = "Initialized"
            else:
                self.status = f"API error: {response.status_code}"
        except Exception as e:
            self.status = f"Initialization failed: {e}"
    def list_models(self) -> List[str]:
        return self.available_models
    def run_inference(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[bool, str]:
        if not self.initialized:
            return False, "OllamaProvider not initialized"
        model = model or self.default_model
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        payload.update(kwargs)
        try:
            response = requests.post(f"{self.api_base}/generate", json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                return True, result.get("response", "")
            else:
                return False, f"Ollama API error: {response.status_code} - {response.text}"
        except Exception as e:
            return False, f"Ollama error: {e}"
    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "status": self.status,
            "endpoint": self.endpoint,
            "models_count": len(self.available_models),
            "default_model": self.default_model
        }

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config=None):
        self.config = config or {}
        self.api_key = os.environ.get("OPENAI_API_KEY", self.config.get("api_key"))
        self.organization = os.environ.get("OPENAI_ORGANIZATION", self.config.get("organization", ""))
        self.default_model = self.config.get("model", "gpt-3.5-turbo")
        self.initialized = False
        self.status = "Created"
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
        
    def initialize(self):
        try:
            import openai
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set when using OpenAI API.")
            # Use new OpenAI v1.0+ client
            self.client = openai.OpenAI(api_key=self.api_key, organization=self.organization)
            self.initialized = True
            self.status = "Initialized"
        except Exception as e:
            self.status = f"Initialization failed: {e}"
            
    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        """Retry a function with exponential backoff."""
        import openai
        
        num_retries = 0
        delay = self.retry_delay
        
        while True:
            try:
                return func(*args, **kwargs)
            except (openai.RateLimitError, openai.APIError) as e:
                num_retries += 1
                if num_retries > self.max_retries:
                    raise Exception(f"Maximum number of retries ({self.max_retries}) exceeded.")
                
                # Exponential backoff with jitter
                delay *= 2 * (1 + 0.1 * random.random())
                logger.warning(f"Retrying in {delay:.2f} seconds. Error: {e}")
                time.sleep(delay)
            except Exception as e:
                raise e
                
    def list_models(self) -> List[str]:
        try:
            if not self.initialized:
                self.initialize()
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return [self.default_model]
            
    def run_inference(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[bool, str]:
        try:
            if not self.initialized:
                self.initialize()
                
            model = model or self.default_model
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            top_p = kwargs.get("top_p", 1.0)
            stop_token = kwargs.get("stop_token")
            
            # Prepare messages
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt  # Assume it's already in message format
                
            def _create_completion():
                return self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=[stop_token] if stop_token else None,
                )
            
            response = self._retry_with_exponential_backoff(_create_completion)
            answer = response.choices[0].message.content
            return True, answer
            
        except Exception as e:
            return False, f"OpenAI error: {e}"
            
    async def run_inference_async(self, prompt: str, model: Optional[str] = None, **kwargs) -> Tuple[bool, str]:
        """Async inference with rate limiting"""
        try:
            if not self.initialized:
                self.initialize()
                
            model = model or self.default_model
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            top_p = kwargs.get("top_p", 1.0)
            
            # Prepare messages
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = prompt
                
            # Async completion with retry logic
            for attempt in range(3):
                try:
                    # Use new OpenAI v1.0+ async client
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    answer = response.choices[0].message.content
                    return True, answer
                    
                except Exception as e:
                    if "rate limit" in str(e).lower():
                        logger.warning("OpenAI API rate limit exceeded. Sleeping for 10 seconds.")
                        await asyncio.sleep(10)
                    else:
                        logger.warning(f"OpenAI API error: {e}")
                        break
                    
            return False, "OpenAI async inference failed after retries"
            
        except Exception as e:
            return False, f"OpenAI async error: {e}"
            
    async def batch_inference(self, prompts: List[str], model: Optional[str] = None, 
                            requests_per_minute: int = 300, **kwargs) -> List[str]:
        """Batch inference with rate limiting"""
        try:
            if not self.initialized:
                self.initialize()
                
            model = model or self.default_model
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1000)
            top_p = kwargs.get("top_p", 1.0)
            
            # Create rate limiter (fallback if aiolimiter not available)
            try:
                limiter = aiolimiter.AsyncLimiter(requests_per_minute)
            except NameError:
                # Fallback: simple delay-based rate limiting
                limiter = None
                
            async def _process_prompt(prompt):
                if limiter:
                    async with limiter:
                        return await self.run_inference_async(prompt, model, **kwargs)
                else:
                    # Simple rate limiting
                    await asyncio.sleep(60.0 / requests_per_minute)
                    return await self.run_inference_async(prompt, model, **kwargs)
            
            # Process all prompts
            if 'tqdm_asyncio' in globals():
                results = await tqdm_asyncio.gather(*[_process_prompt(p) for p in prompts])
            else:
                results = await asyncio.gather(*[_process_prompt(p) for p in prompts])
                
            return [result[1] if result[0] else "" for result in results]
            
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return [""] * len(prompts)
            
    def get_status(self) -> Dict[str, Any]:
        return {
            "initialized": self.initialized,
            "status": self.status,
            "default_model": self.default_model,
            "api_key_set": bool(self.api_key),
            "max_retries": self.max_retries,
            "features": ["text_generation", "async_inference", "batch_processing", "retry_logic"]
        }
