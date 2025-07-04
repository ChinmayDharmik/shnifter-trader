# Generated: 2025-07-04T09:50:40.179291
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Economy GDP Router."""

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

router = Router(prefix="/gdp")

# pylint: disable=unused-argument


@router.command(
    model="GdpForecast",
    examples=[
        APIEx(parameters={"provider": "oecd"}),
        APIEx(
            parameters={
                "country": "united_states,germany,france",
                "frequency": "annual",
                "units": "capita",
                "provider": "oecd",
            }
        ),
    ],
)
async def forecast(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get Forecasted GDP Data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="GdpNominal",
    examples=[
        APIEx(parameters={"provider": "oecd"}),
        APIEx(
            parameters={
                "units": "capita",
                "country": "all",
                "frequency": "annual",
                "provider": "oecd",
            }
        ),
    ],
)
async def nominal(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get Nominal GDP Data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="GdpReal",
    examples=[
        APIEx(parameters={"provider": "oecd"}),
        APIEx(
            parameters={"country": "united_states,germany,japan", "provider": "econdb"}
        ),
    ],
)
async def real(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get Real GDP Data."""
    return await Shnifterject.from_query(Query(**locals()))
