# Generated: 2025-07-04T09:50:40.440537
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Command Context."""

from shnifter_core.app.model.system_settings import SystemSettings
from shnifter_core.app.model.user_settings import UserSettings
from pydantic import BaseModel, Field


class CommandContext(BaseModel):
    """Command Context."""

    user_settings: UserSettings = Field(default_factory=UserSettings)
    system_settings: SystemSettings = Field(default_factory=SystemSettings)
