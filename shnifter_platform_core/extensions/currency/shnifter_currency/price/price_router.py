# Generated: 2025-07-04T09:50:40.200943
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Price router for Currency."""

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

router = Router(prefix="/price")


# pylint: disable=unused-argument
@router.command(
    model="CurrencyHistorical",
    examples=[
        APIEx(parameters={"symbol": "EURUSD", "provider": "fmp"}),
        APIEx(
            description="Filter historical data with specific start and end date.",
            parameters={
                "symbol": "EURUSD",
                "start_date": "2023-01-01",
                "end_date": "2023-12-31",
                "provider": "fmp",
            },
        ),
        APIEx(
            description="Get data with different granularity.",
            parameters={"symbol": "EURUSD", "provider": "polygon", "interval": "15m"},
        ),
    ],
)
async def historical(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """
    Currency Historical Price. Currency historical data.

    Currency historical prices refer to the past exchange rates of one currency against
    another over a specific period.
    This data provides insight into the fluctuations and trends in the foreign exchange market,
    helping analysts, traders, and economists understand currency performance,
    evaluate economic health, and make predictions about future movements.
    """
    return await Shnifterject.from_query(Query(**locals()))
