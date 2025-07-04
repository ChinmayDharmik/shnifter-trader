# Generated: 2025-07-04T09:50:40.142442
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Fixed Income Corporate Router."""

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

router = Router(prefix="/spreads")

# pylint: disable=unused-argument


@router.command(
    model="TreasuryConstantMaturity",
    examples=[
        APIEx(parameters={"provider": "fred"}),
        APIEx(parameters={"maturity": "2y", "provider": "fred"}),
    ],
)
async def tcm(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Treasury Constant Maturity.

    Get data for 10-Year Treasury Constant Maturity Minus Selected Treasury Constant Maturity.
    Constant maturity is the theoretical value of a U.S. Treasury that is based on recent values of auctioned U.S.
    Treasuries. The value is obtained by the U.S. Treasury on a daily basis through interpolation of the Treasury
    yield curve which, in turn, is based on closing bid-yields of actively-traded Treasury securities.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SelectedTreasuryConstantMaturity",
    examples=[
        APIEx(parameters={"provider": "fred"}),
        APIEx(parameters={"maturity": "10y", "provider": "fred"}),
    ],
)
async def tcm_effr(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Select Treasury Constant Maturity.

    Get data for Selected Treasury Constant Maturity Minus Federal Funds Rate
    Constant maturity is the theoretical value of a U.S. Treasury that is based on recent values of auctioned U.S.
    Treasuries. The value is obtained by the U.S. Treasury on a daily basis through interpolation of the Treasury
    yield curve which, in turn, is based on closing bid-yields of actively-traded Treasury securities.
    """
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SelectedTreasuryBill",
    examples=[
        APIEx(parameters={"provider": "fred"}),
        APIEx(parameters={"maturity": "6m", "provider": "fred"}),
    ],
)
async def treasury_effr(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Select Treasury Bill.

    Get Selected Treasury Bill Minus Federal Funds Rate.
    Constant maturity is the theoretical value of a U.S. Treasury that is based on recent values of
    auctioned U.S. Treasuries.
    The value is obtained by the U.S. Treasury on a daily basis through interpolation of the Treasury
    yield curve which, in turn, is based on closing bid-yields of actively-traded Treasury securities.
    """
    return await Shnifterject.from_query(Query(**locals()))
