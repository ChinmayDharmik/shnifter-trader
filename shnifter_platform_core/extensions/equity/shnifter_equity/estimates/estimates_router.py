# Generated: 2025-07-04T09:50:40.163443
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Estimates Router."""

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

router = Router(prefix="/estimates")

# pylint: disable=unused-argument


@router.command(
    model="PriceTarget",
    examples=[
        APIEx(parameters={"provider": "benzinga"}),
        APIEx(
            description="Get price targets for Microsoft using 'benzinga' as provider.",
            parameters={
                "start_date": "2020-01-01",
                "end_date": "2024-02-16",
                "limit": 10,
                "symbol": "msft",
                "provider": "benzinga",
                "action": "downgrades",
            },
        ),
    ],
)
async def price_target(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get analyst price targets by company."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="AnalystEstimates",
    examples=[
        APIEx(parameters={"symbol": "AAPL", "provider": "fmp"}),
    ],
)
async def historical(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical analyst estimates for earnings and revenue."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="PriceTargetConsensus",
    examples=[
        APIEx(parameters={"symbol": "AAPL", "provider": "fmp"}),
        APIEx(parameters={"symbol": "AAPL,MSFT", "provider": "yfinance"}),
    ],
)
async def consensus(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get consensus price target and recommendation."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="AnalystSearch",
    examples=[
        APIEx(parameters={"provider": "benzinga"}),
        APIEx(parameters={"firm_name": "Wedbush", "provider": "benzinga"}),
    ],
)
async def analyst_search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Search for specific analysts and get their forecast track record."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ForwardSalesEstimates",
    examples=[
        APIEx(parameters={"symbol": "AAPL", "provider": "intrinio"}),
        APIEx(
            parameters={
                "fiscal_year": 2025,
                "fiscal_period": "fy",
                "provider": "intrinio",
            }
        ),
    ],
)
async def forward_sales(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get forward sales estimates."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ForwardEbitdaEstimates",
    examples=[
        APIEx(parameters={"provider": "intrinio"}),
        APIEx(
            parameters={
                "symbol": "AAPL",
                "fiscal_period": "annual",
                "provider": "intrinio",
            }
        ),
        APIEx(
            parameters={
                "symbol": "AAPL,MSFT",
                "fiscal_period": "quarter",
                "provider": "fmp",
            }
        ),
    ],
)
async def forward_ebitda(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get forward EBITDA estimates."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ForwardEpsEstimates",
    examples=[
        APIEx(parameters={"symbol": "AAPL", "provider": "intrinio"}),
        APIEx(
            parameters={
                "fiscal_year": 2025,
                "fiscal_period": "fy",
                "provider": "intrinio",
            }
        ),
    ],
)
async def forward_eps(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get forward EPS estimates."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="ForwardPeEstimates",
    examples=[
        APIEx(parameters={"provider": "intrinio"}),
        APIEx(
            parameters={
                "symbol": "AAPL,MSFT,GOOG",
                "provider": "intrinio",
            }
        ),
    ],
)
async def forward_pe(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get forward PE estimates."""
    return await Shnifterject.from_query(Query(**locals()))
