# Generated: 2025-07-04T09:50:40.489684
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Logging settings."""

from pathlib import Path
from typing import List, Optional

from shnifter_core.app.logs.utils.utils import get_app_id, get_log_dir, get_session_id
from shnifter_core.app.model.system_settings import SystemSettings
from shnifter_core.app.model.user_settings import UserSettings


# pylint: disable=too-many-instance-attributes
class LoggingSettings:
    """Logging settings."""

    def __init__(
        self,
        user_settings: Optional[UserSettings] = None,
        system_settings: Optional[SystemSettings] = None,
    ):
        """Initialize the logging settings."""
        user_settings = user_settings if user_settings is not None else UserSettings()
        system_settings = (
            system_settings if system_settings is not None else SystemSettings()
        )
        user_data_directory = (
            str(Path.home() / "ShnifterUserData")
            if not user_settings.preferences
            else user_settings.preferences.data_directory
        )
        core_session = (
            user_settings.profile.core_session if user_settings.profile else None
        )
        if core_session:
            user_id = core_session.user_uuid
            user_email = core_session.email
            user_primary_usage = core_session.primary_usage
        else:
            user_id, user_email, user_primary_usage = None, None, None

        # System
        self.app_name: str = system_settings.logging_app_name
        self.sub_app_name: str = system_settings.logging_sub_app
        self.app_id: str = get_app_id(user_data_directory)
        self.session_id: str = get_session_id()
        self.frequency: str = system_settings.logging_frequency
        self.handler_list: List[str] = system_settings.logging_handlers
        self.rolling_clock: bool = system_settings.logging_rolling_clock
        self.verbosity: int = system_settings.logging_verbosity
        self.engine: str = system_settings.engine
        self.python_version: str = system_settings.python_version
        self.engine_version: str = system_settings.version
        self.logging_suppress: bool = system_settings.logging_suppress
        # User
        self.user_id: Optional[str] = user_id
        self.user_logs_directory: Path = get_log_dir(user_data_directory)
        self.user_email: Optional[str] = user_email
        self.user_primary_usage: Optional[str] = user_primary_usage
