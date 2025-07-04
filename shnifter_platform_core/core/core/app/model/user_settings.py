# Generated: 2025-07-04T09:50:40.452666
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""User settings model."""

import json
import os
import warnings

from shnifter_core.app.constants import USER_SETTINGS_PATH
from shnifter_core.app.model.abstract.tagged import Tagged
from shnifter_core.app.model.credentials import Credentials
from shnifter_core.app.model.defaults import Defaults
from shnifter_core.app.model.preferences import Preferences
from shnifter_core.app.model.profile import Profile
from pydantic import Field


class UserSettings(Tagged):
    """User settings."""

    profile: Profile = Field(default_factory=Profile)
    credentials: Credentials = Field(default_factory=Credentials)
    preferences: Preferences = Field(default_factory=Preferences)
    defaults: Defaults = Field(default_factory=Defaults)

    def __init__(self, **kwargs):
        """Initialize user settings by loading directly from file if it exists."""
        # Check if user settings file exists and load from it
        if os.path.exists(USER_SETTINGS_PATH):
            try:
                with open(USER_SETTINGS_PATH) as f:
                    file_settings = json.load(f)
                # Initialize with settings from file
                super().__init__(**{k: v for k, v in file_settings.items() if v})
            except (json.JSONDecodeError, OSError) as e:
                warnings.warn(
                    f"Error loading user settings from file: {e}",
                    stacklevel=2,
                    category=UserWarning,
                )
                # Fall back to defaults if file can't be read
                super().__init__(**kwargs)
        else:
            # Use defaults if file doesn't exist
            super().__init__(**kwargs)

    def __repr__(self) -> str:
        """Human readable representation of the object."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )
