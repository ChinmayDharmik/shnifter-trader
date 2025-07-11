# Generated: 2025-07-04T09:50:40.461160
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""App factory."""

from typing import Dict, Optional, Type, TypeVar

from shnifter_core.app.command_runner import CommandRunner
from shnifter_core.app.model.system_settings import SystemSettings
from shnifter_core.app.model.user_settings import UserSettings
from shnifter_core.app.static.account import Account
from shnifter_core.app.static.container import Container
from shnifter_core.app.static.coverage import Coverage
from shnifter_core.app.static.reference_loader import ReferenceLoader
from shnifter_core.app.version import VERSION

E = TypeVar("E", bound=Type[Container])
BASE_DOC = f"""Shnifter Engine v{VERSION}

Utilities:
    /account
    /user
    /system
    /coverage
"""


class BaseApp:
    """Base app."""

    def __init__(self, command_runner: CommandRunner):
        """Initialize the app."""
        command_runner.init_logging_service()
        self._command_runner = command_runner
        self._account = Account(self)
        self._coverage = Coverage(self)
        self._reference = ReferenceLoader().reference

    @property
    def account(self) -> Account:
        """Account menu."""
        return self._account

    @property
    def user(self) -> UserSettings:
        """User settings."""
        return self._command_runner.user_settings

    @property
    def system(self) -> SystemSettings:
        """System settings."""
        return self._command_runner.system_settings

    @property
    def coverage(self) -> Coverage:
        """Coverage menu."""
        return self._coverage

    @property
    def reference(self) -> Dict[str, Dict]:
        """Return reference data."""
        return self._reference


def create_app(extensions: Optional[E] = None) -> Type[BaseApp]:
    """Create the app."""

    class App(BaseApp, extensions or object):  # type: ignore[misc]
        def __repr__(self) -> str:
            # pylint: disable=E1101
            ext_doc = extensions.__doc__ if extensions else ""
            return BASE_DOC + (ext_doc or "")

    return App(command_runner=CommandRunner())
