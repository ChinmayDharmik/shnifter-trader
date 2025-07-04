# Generated: 2025-07-04T09:50:40.152521
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Equity Router."""

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

from shnifter_equity.calendar.calendar_router import router as calendar_router
from shnifter_equity.compare.compare_router import router as compare_router
from shnifter_equity.darkpool.darkpool_router import router as darkpool_router
from shnifter_equity.discovery.discovery_router import router as discovery_router
from shnifter_equity.estimates.estimates_router import router as estimates_router
from shnifter_equity.fundamental.fundamental_router import router as fundamental_router
from shnifter_equity.ownership.ownership_router import router as ownership_router
from shnifter_equity.price.price_router import router as price_router
from shnifter_equity.shorts.shorts_router import router as shorts_router

router = Router(prefix="", description="Equity market data.")
router.include_router(calendar_router)
router.include_router(compare_router)
router.include_router(estimates_router)
router.include_router(darkpool_router)
router.include_router(discovery_router)
router.include_router(fundamental_router)
router.include_router(ownership_router)
router.include_router(price_router)
router.include_router(shorts_router)

# pylint: disable=import-outside-toplevel, W0613:unused-argument


@router.command(
    model="EquitySearch",
    examples=[
        APIEx(parameters={"provider": "intrinio"}),
        APIEx(
            parameters={
                "query": "AAPL",
                "is_symbol": False,
                "use_cache": True,
                "provider": "nasdaq",
            }
        ),
    ],
)
async def search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Search for stock symbol, CIK, LEI, or company name."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityScreener", examples=[APIEx(parameters={"provider": "fmp"})]
)
async def screener(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Screen for companies meeting various criteria.

    These criteria include market cap, price, beta, volume, and dividend yield.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityInfo",
    examples=[APIEx(parameters={"symbol": "AAPL", "provider": "fmp"})],
)
async def profile(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get general information about a company. This includes company name, industry, sector and price data."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="MarketSnapshots", examples=[APIEx(parameters={"provider": "fmp"})]
)
async def market_snapshots(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get an updated equity market snapshot. This includes price data for thousands of stocks."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="HistoricalMarketCap",
    examples=[APIEx(parameters={"provider": "fmp", "symbol": "AAPL"})],
)
async def historical_market_cap(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the historical market cap of a ticker symbol."""
    return await Shnifterject.from_query(Query(**locals()))
