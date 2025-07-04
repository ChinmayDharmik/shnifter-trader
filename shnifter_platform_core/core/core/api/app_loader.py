# Generated: 2025-07-04T09:50:40.222290
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""App loader module."""

from typing import List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.exceptions import ResponseValidationError
from shnifter_core.api.exception_handlers import ExceptionHandlers
from shnifter_core.app.model.abstract.error import ShnifterError
from shnifter_core.app.router import RouterLoader
from shnifter_core.provider.utils.errors import EmptyDataError, UnauthorizedError
from pydantic import ValidationError


class AppLoader:
    """App loader."""

    @staticmethod
    def add_routers(app: FastAPI, routers: List[Optional[APIRouter]], prefix: str):
        """Add routers."""
        for router in routers:
            if router:
                app.include_router(router=router, prefix=prefix)

    @staticmethod
    def add_openapi_tags(app: FastAPI):
        """Add openapi tags."""
        main_router = RouterLoader.from_extensions()
        # Add tag data for each router in the main router
        app.openapi_tags = [
            {
                "name": r,
                "description": main_router.get_attr(r, "description"),
            }
            for r in main_router.routers
        ]

    @staticmethod
    def add_exception_handlers(app: FastAPI):
        """Add exception handlers."""
        app.exception_handlers[Exception] = ExceptionHandlers.exception
        app.exception_handlers[ValidationError] = ExceptionHandlers.validation
        app.exception_handlers[ResponseValidationError] = ExceptionHandlers.validation
        app.exception_handlers[ShnifterError] = ExceptionHandlers.shnifter
        app.exception_handlers[EmptyDataError] = ExceptionHandlers.empty_data
        app.exception_handlers[UnauthorizedError] = ExceptionHandlers.unauthorized
