# Generated: 2025-07-04T09:50:40.117396
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

# pylint: disable=import-outside-toplevel, W0613:unused-argument
"""News Router."""

from shnifter_core.app.model.command_context import CommandContext
from shnifter_core.app.model.example import APIEx
from shnifter_core.app.model.shnifterject import Shnifterject
from shnifter_core.app.provider_interface import (
    ExtraParams,
    ProviderChoices,
    StandardParams,
)
from shnifter_core.app.query import Query
from shnifter_core.app.router import Router

router = Router(prefix="", description="Financial market news data.")


@router.command(
    model="WorldNews",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(parameters={"limit": 100, "provider": "intrinio"}),
        APIEx(
            description="Get news on the specified dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "intrinio",
            },
        ),
        APIEx(
            description="Display the headlines of the news.",
            parameters={"display": "headline", "provider": "benzinga"},
        ),
        APIEx(
            description="Get news by topics.",
            parameters={"topics": "finance", "provider": "benzinga"},
        ),
        APIEx(
            description="Get news by source using 'tingo' as provider.",
            parameters={"provider": "tiingo", "source": "bloomberg"},
        ),
        APIEx(
            description="Filter aticles by term using 'biztoc' as provider.",
            parameters={"provider": "biztoc", "term": "apple"},
        ),
    ],
)
async def world(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """World News. Global news data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CompanyNews",
    examples=[
        APIEx(parameters={"provider": "benzinga"}),
        APIEx(parameters={"limit": 100, "provider": "benzinga"}),
        APIEx(
            description="Get news on the specified dates.",
            parameters={
                "symbol": "AAPL",
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "intrinio",
            },
        ),
        APIEx(
            description="Display the headlines of the news.",
            parameters={
                "symbol": "AAPL",
                "display": "headline",
                "provider": "benzinga",
            },
        ),
        APIEx(
            description="Get news for multiple symbols.",
            parameters={"symbol": "aapl,tsla", "provider": "fmp"},
        ),
        APIEx(
            description="Get news company's ISIN.",
            parameters={
                "symbol": "NVDA",
                "isin": "US0378331005",
                "provider": "benzinga",
            },
        ),
    ],
)
async def company(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Company News. Get news for one or more companies."""
    return await Shnifterject.from_query(Query(**locals()))
