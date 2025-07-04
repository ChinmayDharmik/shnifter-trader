"""Constants for the ShnifterBB Platform."""

from pathlib import Path

HOME_DIRECTORY = Path.home()
SHNIFTER_DIRECTORY = Path(HOME_DIRECTORY, ".shnifter_platform")
USER_SETTINGS_PATH = Path(SHNIFTER_DIRECTORY, "user_settings.json")
SYSTEM_SETTINGS_PATH = Path(SHNIFTER_DIRECTORY, "system_settings.json")
