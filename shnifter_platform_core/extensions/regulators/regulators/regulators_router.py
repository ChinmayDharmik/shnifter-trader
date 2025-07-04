# Generated: 2025-07-04T09:50:40.089822
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

# pylint: disable=import-outside-toplevel
# pylint: disable=unused-import
# ruff: noqa: F401
"""Regulators Router."""


from shnifter_core.app.router import Router

from .cftc.cftc_router import (
    router as cftc_router,
)
from .sec.sec_router import router as sec_router

router = Router(prefix="", description="Financial market regulators data.")
router.include_router(sec_router)
router.include_router(cftc_router)
