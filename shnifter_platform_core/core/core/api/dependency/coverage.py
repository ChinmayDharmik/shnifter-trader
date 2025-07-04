# Generated: 2025-07-04T09:50:40.495270
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Coverage dependency."""

from fastapi import Depends
from shnifter_core.app.provider_interface import ProviderInterface
from shnifter_core.app.router import CommandMap
from shnifter_core.app.service.auth_service import AuthService
from typing_extensions import Annotated


async def get_command_map(
    _: Annotated[None, Depends(AuthService().auth_hook)]
) -> CommandMap:
    """Get command map."""
    return CommandMap()


async def get_provider_interface(
    _: Annotated[None, Depends(AuthService().auth_hook)]
) -> ProviderInterface:
    """Get provider interface."""
    return ProviderInterface()
