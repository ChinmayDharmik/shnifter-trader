# Generated: 2025-07-04T09:50:40.159269
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Dark Pool Router."""

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

router = Router(prefix="/darkpool")

# pylint: disable=unused-argument


@router.command(
    model="OTCAggregate",
    examples=[
        APIEx(parameters={"provider": "finra"}),
        APIEx(
            description="Get OTC data for a symbol",
            parameters={"symbol": "AAPL", "provider": "finra"},
        ),
    ],
)
async def otc(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the weekly aggregate trade data for Over The Counter deals.

    ATS and non-ATS trading data for each ATS/firm
    with trade reporting obligations under FINRA rules.
    """
    return await Shnifterject.from_query(Query(**locals()))
