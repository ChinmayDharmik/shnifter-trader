# Generated: 2025-07-04T09:50:40.155536
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Calendar Router."""

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

router = Router(prefix="/calendar")

# pylint: disable=unused-argument


@router.command(
    model="CalendarIpo",
    examples=[
        APIEx(parameters={"provider": "intrinio"}),
        APIEx(parameters={"limit": 100, "provider": "nasdaq"}),
        APIEx(
            description="Get all IPOs available.", parameters={"provider": "intrinio"}
        ),
        APIEx(
            description="Get IPOs for specific dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "nasdaq",
            },
        ),
    ],
)
async def ipo(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical and upcoming initial public offerings (IPOs)."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CalendarDividend",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(
            description="Get dividend calendar for specific dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "nasdaq",
            },
        ),
    ],
)
async def dividend(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical and upcoming dividend payments. Includes dividend amount, ex-dividend and payment dates."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CalendarSplits",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(
            description="Get stock splits calendar for specific dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "fmp",
            },
        ),
    ],
)
async def splits(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical and upcoming stock split operations."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CalendarEvents",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(
            description="Get company events calendar for specific dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "fmp",
            },
        ),
    ],
)
async def events(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical and upcoming company events, such as Investor Day, Conference Call, Earnings Release."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CalendarEarnings",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(
            description="Get earnings calendar for specific dates.",
            parameters={
                "start_date": "2024-02-01",
                "end_date": "2024-02-07",
                "provider": "fmp",
            },
        ),
    ],
)
async def earnings(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical and upcoming company earnings releases. Includes earnings per share (EPS) and revenue data."""
    return await Shnifterject.from_query(Query(**locals()))
