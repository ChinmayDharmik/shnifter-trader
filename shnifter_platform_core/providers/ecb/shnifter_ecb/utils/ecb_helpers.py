# Generated: 2025-07-04T09:50:39.883828
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

""" ECB helpers"""

from typing import List


async def get_series_data(series_id: str, start_date: str = "", end_date: str = ""):
    """Get ECB data

    Parameters
    ----------
    series_id: str
        ECB ID of data
    start_date: Optional[str]
        Start date, formatted YYYY-MM-DD
    end_date: Optional[str]
        End date, formatted YYYY-MM-DD
    """
    # pylint: disable=import-outside-toplevel
    import json  # noqa
    from shnifter_core.app.model.abstract.error import ShnifterError  # noqa
    from shnifter_core.provider.utils.helpers import amake_request  # noqa

    start_date = start_date.replace("-", "")
    end_date = end_date.replace("-", "")
    url = f"https://data.ecb.europa.eu/data-detail-api/{series_id}"
    data: List = []  # type: ignore
    try:
        data = await amake_request(  # type: ignore
            url=url,
            params={"startPeriod": start_date, "endPeriod": end_date},
        )
    except KeyboardInterrupt as interrupt:
        raise interrupt
    except json.JSONDecodeError as exc:
        raise ShnifterError("Invalid JSON response from ECB") from exc

    if start_date:
        data = [item for item in data if item["PERIOD"][0] >= start_date]
    if end_date:
        data = [item for item in data if item["PERIOD"][0] <= end_date]

    return data
