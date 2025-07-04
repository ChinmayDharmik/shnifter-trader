"""Credentials model and its utilities (ShnifterBB version)."""

import json
from pathlib import Path
from typing import Dict, Optional

from core.constants import USER_SETTINGS_PATH

class CredentialsLoader:
    """Handles loading and saving credentials for ShnifterBB."""

    @staticmethod
    def load_credentials() -> Optional[Dict[str, str]]:
        if USER_SETTINGS_PATH.exists():
            with open(USER_SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f).get('credentials', {})
        return None

    @staticmethod
    def save_credentials(credentials: Dict[str, str]) -> None:
        USER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {'credentials': credentials}
        with open(USER_SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
