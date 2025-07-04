# Generated: 2025-07-04T09:50:40.127713
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Index Router."""

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

from shnifter_index.price.price_router import router as price_router

router = Router(prefix="", description="Indices data.")
router.include_router(price_router)

# pylint: disable=unused-argument


@router.command(
    model="IndexConstituents",
    examples=[
        APIEx(parameters={"symbol": "dowjones", "provider": "fmp"}),
        APIEx(
            description="Providers other than FMP will use the ticker symbol.",
            parameters={"symbol": "BEP50P", "provider": "cboe"},
        ),
    ],
)
async def constituents(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get Index Constituents."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="IndexSnapshots",
    examples=[
        APIEx(parameters={"provider": "tmx"}),
        APIEx(parameters={"region": "us", "provider": "cboe"}),
    ],
)
async def snapshots(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Index Snapshots. Current levels for all indices from a provider, grouped by `region`."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="AvailableIndices",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(parameters={"provider": "yfinance"}),
    ],
)
async def available(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """All indices available from a given provider."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="IndexSearch",
    examples=[
        APIEx(parameters={"provider": "cboe"}),
        APIEx(parameters={"query": "SPX", "provider": "cboe"}),
    ],
)
async def search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Filter indices for rows containing the query."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SP500Multiples",
    examples=[
        APIEx(parameters={"provider": "multpl"}),
        APIEx(parameters={"series_name": "shiller_pe_year", "provider": "multpl"}),
    ],
)
async def sp500_multiples(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical S&P 500 multiples and Shiller PE ratios."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="IndexSectors",
    examples=[APIEx(parameters={"symbol": "^TX60", "provider": "tmx"})],
)
async def sectors(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get Index Sectors. Sector weighting of an index."""
    return await Shnifterject.from_query(Query(**locals()))
