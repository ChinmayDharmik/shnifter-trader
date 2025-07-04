# Generated: 2025-07-04T09:50:40.145551
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ETF Router."""

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

from shnifter_etf.discovery.discovery_router import router as discovery_router

router = Router(prefix="", description="Exchange Traded Funds market data.")
router.include_router(discovery_router)

# pylint: disable=unused-argument


@router.command(
    model="EtfSearch",
    examples=[
        APIEx(
            description="An empty query returns the full list of ETFs from the provider.",
            parameters={"provider": "fmp"},
        ),
        APIEx(
            description="The query will return results from text-based fields containing the term.",
            parameters={"query": "commercial real estate", "provider": "fmp"},
        ),
    ],
)
async def search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Search for ETFs.

    An empty query returns the full list of ETFs from the provider.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfHistorical",
    operation_id="etf_historical",
    examples=[
        APIEx(parameters={"symbol": "SPY", "provider": "fmp"}),
        APIEx(parameters={"symbol": "SPY", "provider": "yfinance"}),
        APIEx(
            description="This function accepts multiple tickers.",
            parameters={"symbol": "SPY,IWM,QQQ,DJIA", "provider": "yfinance"},
        ),
    ],
)
async def historical(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """ETF Historical Market Price."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfInfo",
    examples=[
        APIEx(parameters={"symbol": "SPY", "provider": "fmp"}),
        APIEx(
            description="This function accepts multiple tickers.",
            parameters={"symbol": "SPY,IWM,QQQ,DJIA", "provider": "fmp"},
        ),
    ],
)
async def info(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """ETF Information Overview."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfSectors",
    examples=[APIEx(parameters={"symbol": "SPY", "provider": "fmp"})],
)
async def sectors(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """ETF Sector weighting."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfCountries",
    examples=[APIEx(parameters={"symbol": "VT", "provider": "fmp"})],
)
async def countries(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """ETF Country weighting."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfPricePerformance",
    examples=[
        APIEx(parameters={"symbol": "QQQ", "provider": "fmp"}),
        APIEx(parameters={"symbol": "SPY,QQQ,IWM,DJIA", "provider": "fmp"}),
    ],
)
async def price_performance(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Price performance as a return, over different periods."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfHoldings",
    examples=[
        APIEx(parameters={"symbol": "XLK", "provider": "fmp"}),
        APIEx(
            description="Including a date (FMP, SEC) will return the holdings as per NPORT-P filings.",
            parameters={"symbol": "XLK", "date": "2022-03-31", "provider": "fmp"},
        ),
        APIEx(
            description="The same data can be returned from the SEC directly.",
            parameters={"symbol": "XLK", "date": "2022-03-31", "provider": "sec"},
        ),
    ],
)
async def holdings(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the holdings for an individual ETF."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfHoldingsDate",
    examples=[APIEx(parameters={"symbol": "XLK", "provider": "fmp"})],
)
async def holdings_date(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Use this function to get the holdings dates, if available."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EtfEquityExposure",
    examples=[
        APIEx(parameters={"symbol": "MSFT", "provider": "fmp"}),
        APIEx(
            description="This function accepts multiple tickers.",
            parameters={"symbol": "MSFT,AAPL", "provider": "fmp"},
        ),
    ],
)
async def equity_exposure(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the exposure to ETFs for a specific stock."""
    return await Shnifterject.from_query(Query(**locals()))
