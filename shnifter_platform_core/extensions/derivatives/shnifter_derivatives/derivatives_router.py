# Generated: 2025-07-04T09:50:40.190113
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Derivatives Router."""

from shnifter_core.app.router import Router

from shnifter_derivatives.futures.futures_router import router as futures_router
from shnifter_derivatives.options.options_router import router as options_router

router = Router(prefix="", description="Derivatives market data.")
router.include_router(options_router)
router.include_router(futures_router)
