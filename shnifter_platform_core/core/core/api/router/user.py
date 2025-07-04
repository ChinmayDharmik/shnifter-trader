# Generated: 2025-07-04T09:50:40.501999
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Engine API Account Router."""

from fastapi import APIRouter, Depends
from shnifter_core.api.auth.user import authenticate_user, get_user_settings
from shnifter_core.app.model.user_settings import UserSettings
from typing_extensions import Annotated

router = APIRouter(prefix="/user", tags=["User"])
auth_hook = authenticate_user
user_settings_hook = get_user_settings


@router.get("/me")
async def read_user_settings(
    user_settings: Annotated[UserSettings, Depends(get_user_settings)]
):
    """Read current user settings."""
    return user_settings
