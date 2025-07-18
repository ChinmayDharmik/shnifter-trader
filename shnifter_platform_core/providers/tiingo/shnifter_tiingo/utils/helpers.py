# Generated: 2025-07-04T09:50:39.526670
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Tiingo Helpers Module."""

from typing import Optional, Union


async def get_data(url: str) -> Union[dict, list]:
    """Get data from Tiingo endpoint and parse the JSON response."""
    # pylint: disable=import-outside-toplevel
    from json import JSONDecodeError  # noqa
    from shnifter_core.app.model.abstract.error import ShnifterError
    from shnifter_core.provider.utils.errors import EmptyDataError, UnauthorizedError
    from shnifter_core.provider.utils.helpers import amake_request

    response: Optional[Union[dict, list]] = None

    try:
        response = await amake_request(url)
        if (
            response
            and isinstance(response, dict)
            and len(list(response.keys())) == 1
            and response.get("detail")
        ):
            if (
                "token" in response["detail"]
                or "access" in response["detail"]
                or "permission" in response["detail"]
                or "authorized" in response["detail"]
            ):
                raise UnauthorizedError(
                    f"Unauthorized Tiingo request -> {response['detail']}"
                )
            raise ShnifterError(response["detail"])

        if not response:
            raise EmptyDataError("The response is empty")

        return response

    except JSONDecodeError as e:
        raise ShnifterError(f"Failed to parse JSON response -> {e}") from e
