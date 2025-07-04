# Generated: 2025-07-04T09:50:40.180354
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Economy shipping router."""

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

router = Router(prefix="/shipping")

# pylint: disable=unused-argument


@router.command(
    model="PortInfo",
    examples=[
        APIEx(parameters={"provider": "imf"}),
        APIEx(parameters={"provider": "imf", "continent": "asia_pacific"}),
    ],
)
async def port_info(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get general metadata and statistics for all ports from a given provider."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="PortVolume",
    examples=[
        APIEx(
            description="Get average dwelling times and TEU volumes from the top ports.",
            parameters={"provider": "econdb"},
        ),
        APIEx(
            description="Get daily port calls and estimated trading volumes for specific ports"
            + " Get the list of available ports with `shnifter shipping port_info`",
            parameters={
                "provider": "imf",
                "port_code": "rotterdam,singapore",
            },
        ),
        APIEx(
            description="Get data for all ports in a specific country. Use the 3-letter ISO country code.",
            parameters={
                "provider": "imf",
                "country": "GBR",
            },
        ),
    ],
)
async def port_volume(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Daily port calls and estimates of trading volumes for ports around the world."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="MaritimeChokePointInfo",
    examples=[
        APIEx(parameters={"provider": "imf"}),
    ],
)
async def chokepoint_info(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get general metadata and statistics for all maritime chokepoint locations from a given provider."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="MaritimeChokePointVolume",
    examples=[
        APIEx(parameters={"provider": "imf"}),
        APIEx(
            parameters={
                "provider": "imf",
                "chokepoint": "suez_canal,panama_canal",
            }
        ),
    ],
)
async def chokepoint_volume(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Daily transit calls and estimates of transit trade volumes for shipping lane chokepoints around the world."""
    return await Shnifterject.from_query(Query(**locals()))
