# Generated: 2025-07-04T09:50:40.430143
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Custom exceptions for the provider."""

from typing import Union

from shnifter_core.app.model.abstract.error import ShnifterError


class EmptyDataError(ShnifterError):
    """Exception raised for empty data."""

    def __init__(
        self, message: str = "No results found. Try adjusting the query parameters."
    ):
        """Initialize the exception."""
        self.message = message
        super().__init__(self.message)


class UnauthorizedError(ShnifterError):
    """Exception raised for an unauthorized provider request response."""

    def __init__(
        self,
        message: Union[str, tuple[str]] = (
            "Unauthorized <provider name> API request."
            " Please check your <provider name> credentials and subscription access.",
        ),
        provider_name: str = "<provider name>",
    ):
        """Initialize the exception."""
        if provider_name and provider_name != "<provider name>":
            msg = message
            if isinstance(msg, tuple):
                msg = msg[0].replace("<provider name>", provider_name)
            elif isinstance(msg, str):
                msg = msg.replace("<provider name>", provider_name)
            message = msg
        self.message = message
        super().__init__(str(self.message))
