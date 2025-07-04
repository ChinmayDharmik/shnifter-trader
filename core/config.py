"""
Shnifter Configuration Management
Centralized settings for the trading bot
"""
import json
import os
from typing import Dict, Any
from datetime import datetime

class ShnifterConfig:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: str = "shnifter_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self.load_config()
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Default configuration values"""
        return {
            "trading": {
                "default_ticker": "SONY",
                "max_concurrent_analysis": 5,
                "auto_analysis_interval": 600000,  # 10 minutes
                "stop_loss_percentage": 0.05,
                "take_profit_percentage": 0.10
            },
            "llm": {
                "default_model": "llama3",
                "ollama_url": "http://localhost:11434",
                "max_retries": 3,
                "timeout": 30,
                "dual_llm_mode": False
            },
            "ui": {
                "theme": "dark",
                "window_size": [800, 600],
                "auto_refresh": True,
                "popout_positions": {}
            },
            "data": {
                "default_provider": "yfinance",
                "cache_duration": 300,  # 5 minutes
                "max_history_years": 10
            },
            "logging": {
                "level": "INFO",
                "max_log_size": 10000,
                "export_format": "csv"
            }
        }
        
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self._merge_config(saved_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
                
    def save_config(self):
        """Save current configuration to file"""
        try:
            self.config['meta'] = {
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config file: {e}")
            
    def _merge_config(self, saved_config: Dict[str, Any]):
        """Merge saved config with defaults"""
        for section, values in saved_config.items():
            if section in self.config and isinstance(values, dict):
                self.config[section].update(values)
            else:
                self.config[section] = values
                
    def get(self, key_path: str, default=None):
        """Get config value using dot notation: 'trading.default_ticker'"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
        
    def set(self, key_path: str, value: Any):
        """Set config value using dot notation"""
        keys = key_path.split('.')
        config_section = self.config
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        config_section[keys[-1]] = value
        self.save_config()

# Global config instance
shnifter_config = ShnifterConfig()
