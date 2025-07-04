"""
Enhanced Model Switching for Shnifter Trader

This module provides utilities for managing and switching between different AI/ML models
and providers in the Shnifter Trader application. It is designed to be modular and extensible,
matching the Shnifter architecture and removing any legacy or unrelated code.
"""

from typing import List, Dict, Any

class ModelSwitcher:
    """
    Handles switching between different models and providers for Shnifter Trader.
    """
    def __init__(self, available_models: List[str], available_providers: List[str]):
        self.available_models = available_models
        self.available_providers = available_providers
        self.current_model = available_models[0] if available_models else None
        self.current_provider = available_providers[0] if available_providers else None

    def set_model(self, model_name: str):
        if model_name in self.available_models:
            self.current_model = model_name
            return True
        return False

    def set_provider(self, provider_name: str):
        if provider_name in self.available_providers:
            self.current_provider = provider_name
            return True
        return False

    def get_current(self) -> Dict[str, Any]:
        return {
            'model': self.current_model,
            'provider': self.current_provider
        }

    def get_available_models(self) -> List[str]:
        return self.available_models

    def get_available_providers(self) -> List[str]:
        return self.available_providers

