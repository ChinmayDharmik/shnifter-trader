# Generated: 2025-07-04T09:50:40.195843
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Options Router."""

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

router = Router(prefix="/options")

# pylint: disable=unused-argument


@router.command(
    model="OptionsChains",
    examples=[
        APIEx(parameters={"symbol": "AAPL", "provider": "intrinio"}),
        APIEx(
            description='Use the "date" parameter to get the end-of-day-data for a specific date, where supported.',
            parameters={"symbol": "AAPL", "date": "2023-01-25", "provider": "intrinio"},
        ),
    ],
)
async def chains(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the complete options chain for a ticker."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="OptionsUnusual",
    examples=[
        APIEx(parameters={"symbol": "TSLA", "provider": "intrinio"}),
        APIEx(
            description="Use the 'symbol' parameter to get the most recent activity for a specific symbol.",
            parameters={"symbol": "TSLA", "provider": "intrinio"},
        ),
    ],
)
async def unusual(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the complete options chain for a ticker."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="OptionsSnapshots",
    examples=[
        APIEx(
            parameters={"provider": "intrinio"},
        ),
    ],
)
async def snapshots(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get a snapshot of the options market universe."""
    return await Shnifterject.from_query(Query(**locals()))
