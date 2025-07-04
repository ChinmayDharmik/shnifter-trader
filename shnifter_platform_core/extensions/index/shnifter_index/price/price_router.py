# Generated: 2025-07-04T09:50:40.130398
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Price Router."""

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
    model="IndexHistorical",
    examples=[
        APIEx(parameters={"symbol": "^GSPC", "provider": "fmp"}),
        APIEx(
            description="Not all providers have the same symbols.",
            parameters={"symbol": "SPX", "provider": "intrinio"},
        ),
    ],
)
async def historical(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Historical Index Levels."""
    return await Shnifterject.from_query(Query(**locals()))
