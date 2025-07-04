# Generated: 2025-07-04T09:50:40.161308
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Disc router for Equities."""

# pylint: disable=unused-argument
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


@router.command(
    model="EquityGainers",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def gainers(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the top price gainers in the stock market."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityLosers",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def losers(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the top price losers in the stock market."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityActive",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def active(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the most actively traded stocks based on volume."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityUndervaluedLargeCaps",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def undervalued_large_caps(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get potentially undervalued large cap stocks."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityUndervaluedGrowth",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def undervalued_growth(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get potentially undervalued growth stocks."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="EquityAggressiveSmallCaps",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def aggressive_small_caps(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get top small cap stocks based on earnings growth."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="GrowthTechEquities",
    examples=[
        APIEx(parameters={"provider": "yfinance"}),
        APIEx(parameters={"sort": "desc", "provider": "yfinance"}),
    ],
)
async def growth_tech(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get top tech stocks based on revenue and earnings growth."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="TopRetail",
    examples=[APIEx(parameters={"provider": "nasdaq"})],
)
async def top_retail(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Track over $30B USD/day of individual investors trades.

    It gives a daily view into retail activity and sentiment for over 9,500 US traded stocks,
    ADRs, and ETPs.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="DiscoveryFilings",
    examples=[
        APIEx(parameters={"provider": "fmp"}),
        APIEx(
            description="Get filings for the year 2023, limited to 100 results",
            parameters={
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "limit": 100,
                "provider": "fmp",
            },
        ),
    ],
)
async def filings(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the URLs to SEC filings reported to EDGAR database, such as 10-K, 10-Q, 8-K, and more.

    SEC filings include Form 10-K, Form 10-Q, Form 8-K, the proxy statement, Forms 3, 4, and 5, Schedule 13, Form 114,
    Foreign Investment Disclosures and others. The annual 10-K report is required to be
    filed annually and includes the company's financial statements, management discussion and analysis,
    and audited financial statements.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="LatestFinancialReports",
    examples=[
        APIEx(parameters={"provider": "sec"}),
        APIEx(parameters={"provider": "sec", "date": "2024-09-30"}),
    ],
)
async def latest_financial_reports(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the newest quarterly, annual, and current reports for all companies."""
    return await Shnifterject.from_query(Query(**locals()))
