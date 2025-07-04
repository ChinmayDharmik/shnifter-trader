# Generated: 2025-07-04T09:50:40.500957
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""System router."""

from fastapi import APIRouter, Depends
from shnifter_core.api.dependency.system import get_system_settings
from shnifter_core.app.model.system_settings import SystemSettings
from typing_extensions import Annotated

router = APIRouter(prefix="/system", tags=["System"])


@router.get("")
async def get_system_model(
    system_settings: Annotated[SystemSettings, Depends(get_system_settings)],
):
    """Get system model."""
    return system_settings
