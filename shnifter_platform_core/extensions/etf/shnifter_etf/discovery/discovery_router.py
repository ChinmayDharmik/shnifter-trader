# Generated: 2025-07-04T09:50:40.149670
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Disc router for ETFs."""

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

router = Router(prefix="/discovery")

# pylint: disable=unused-argument


@router.command(
    model="ETFGainers",
    operation_id="etf_gainers",
    examples=[
        APIEx(description="Get the top ETF gainers.", parameters={"provider": "wsj"}),
    ],
)
async def gainers(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the top ETF gainers."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ETFLosers",
    operation_id="etf_losers",
    examples=[
        APIEx(description="Get the top ETF losers.", parameters={"provider": "wsj"}),
    ],
)
async def losers(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the top ETF losers."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ETFActive",
    operation_id="etf_active",
    examples=[
        APIEx(description="Get the most active ETFs.", parameters={"provider": "wsj"}),
    ],
)
async def active(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the most active ETFs."""
    return await Shnifterject.from_query(Query(**locals()))
