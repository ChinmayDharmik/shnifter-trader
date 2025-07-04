# Generated: 2025-07-04T09:50:39.474126
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""TradingEconomics Helpers."""

from typing import Union

from shnifter_core.app.model.abstract.error import ShnifterError
from shnifter_core.provider.utils.errors import EmptyDataError, UnauthorizedError


async def response_callback(response, _) -> Union[dict, list[dict]]:
    """Return the response."""
    if response.status != 200:
        message = await response.text()

        if "credentials" in message or "unauthorized" in message.lower():
            raise UnauthorizedError(
                f"Unauthorized TradingEconomics request -> {message}"
            )

        raise ShnifterError(f"{response.status} -> {message}")

    results = await response.json()

    if not results:
        raise EmptyDataError("The request was returned empty.")

    return results
