# Generated: 2025-07-04T09:50:39.687716
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""IMF Helper Utilities."""


async def get_data(url: str) -> list[dict]:
    """Get data from the IMF API."""
    # pylint: disable=import-outside-toplevel

    from aiohttp.interfaceent_exceptions import ContentTypeError  # noqa
    from json.decoder import JSONDecodeError
    from shnifter_core.provider.utils.helpers import amake_request
    from shnifter_core.app.model.abstract.error import ShnifterError

    try:
        response = await amake_request(url, timeout=20)
    except (JSONDecodeError, ContentTypeError) as e:
        raise ShnifterError(
            "Error fetching data; This might be rate-limiting. Try again later."
        ) from e

    if "ErrorDetails" in response:
        raise ShnifterError(
            f"{response['ErrorDetails'].get('Code')} -> {response['ErrorDetails'].get('Message')}"  # type: ignore
        )

    series = response.get("CompactData", {}).get("DataSet", {}).pop("Series", {})  # type: ignore

    if not series:
        raise ShnifterError(f"No time series data found -> {response}")

    # If there is only one series, they ruturn a dict instead of a list.
    if series and isinstance(series, dict):
        series = [series]

    return series
