# Generated: 2025-07-04T09:50:40.205831
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

# pylint: disable=W0613:unused-argument
"""Crypto Price Router."""

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

router = Router(prefix="/price")


# pylint: disable=unused-argument,line-too-long
@router.command(
    model="CryptoHistorical",
    examples=[
        APIEx(parameters={"symbol": "BTCUSD", "provider": "fmp"}),
        APIEx(
            parameters={
                "symbol": "BTCUSD",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "provider": "fmp",
            },
        ),
        APIEx(
            parameters={
                "symbol": "BTCUSD,ETHUSD",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "provider": "polygon",
            },
        ),
        APIEx(
            description="Get monthly historical prices from Yahoo Finance for Ethereum.",
            parameters={
                "symbol": "ETH-USD",
                "interval": "1m",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "provider": "yfinance",
            },
        ),
    ],
)
async def historical(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get historical price data for cryptocurrency pair(s) within a provider."""
    return await Shnifterject.from_query(Query(**locals()))
