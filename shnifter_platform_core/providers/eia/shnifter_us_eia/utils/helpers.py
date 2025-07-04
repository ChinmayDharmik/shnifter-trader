# Generated: 2025-07-04T09:50:39.856027
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter EIA Provider Module Helpers."""

from typing import TYPE_CHECKING

from async_lru import alru_cache
from shnifter_core.app.model.abstract.error import ShnifterError

if TYPE_CHECKING:
    from pandas import ExcelFile


async def response_callback(response, _):
    """Response callback to read the response."""
    if response.status == 403:
        res = await response.json()
        code = res.get("error", {}).get("code", "")
        msg = res.get("error", {}).get("message", "An invalid api_key was supplied.")
        raise ShnifterError(f"{code} -> {msg}")
    return await response.json()


@alru_cache(maxsize=14)
async def download_excel_file(url: str, use_cache: bool = True) -> "ExcelFile":
    """Download the excel file from the URL. Set use_cache to False to invalidate the ALRU cache."""
    # pylint: disable=import-outside-toplevel
    from io import BytesIO  # noqa
    from shnifter_core.provider.utils.helpers import amake_request
    from pandas import ExcelFile

    async def callback(response, _):
        """Read the response and return the ExcelFile object."""
        res = await response.read()
        file = ExcelFile(BytesIO(res))
        return file

    # Clear the cache to download the file again.
    if use_cache is False:
        download_excel_file.cache_invalidate(url)

    try:
        return await amake_request(url, response_callback=callback)
    except Exception as e:  # pylint: disable=broad-except
        raise ShnifterError(f"Error downloading the file from the EIA site -> {e}") from e
