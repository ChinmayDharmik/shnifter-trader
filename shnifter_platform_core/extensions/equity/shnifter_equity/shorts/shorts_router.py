# Generated: 2025-07-04T09:50:40.171592
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shorts Router."""

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

router = Router(prefix="/shorts")

# pylint: disable=unused-argument


@router.command(
    model="EquityFTD",
    examples=[APIEx(parameters={"symbol": "AAPL", "provider": "sec"})],
)
async def fails_to_deliver(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get reported Fail-to-deliver (FTD) data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ShortVolume",
    examples=[APIEx(parameters={"symbol": "AAPL", "provider": "stockgrid"})],
)
async def short_volume(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get reported Fail-to-deliver (FTD) data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityShortInterest",
    examples=[APIEx(parameters={"symbol": "AAPL", "provider": "finra"})],
)
async def short_interest(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get reported short volume and days to cover data."""
    return await Shnifterject.from_query(Query(**locals()))
