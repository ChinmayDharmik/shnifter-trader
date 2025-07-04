# Generated: 2025-07-04T09:50:40.496144
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""System dependency."""

from fastapi import Depends
from shnifter_core.app.model.system_settings import SystemSettings
from shnifter_core.app.service.auth_service import AuthService
from shnifter_core.app.service.system_service import SystemService
from typing_extensions import Annotated


async def get_system_service() -> SystemService:
    """Get system service."""
    return SystemService()


async def get_system_settings(
    _: Annotated[None, Depends(AuthService().auth_hook)],
    system_service: Annotated[SystemService, Depends(get_system_service)],
) -> SystemSettings:
    """Get system settings."""
    return system_service.system_settings
