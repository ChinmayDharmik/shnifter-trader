# Generated: 2025-07-04T09:50:39.832592
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Federal Reserve provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_federal_reserve.models.central_bank_holdings import (
    FederalReserveCentralBankHoldingsFetcher,
)
from shnifter_federal_reserve.models.federal_funds_rate import (
    FederalReserveFederalFundsRateFetcher,
)
from shnifter_federal_reserve.models.fomc_documents import (
    FederalReserveFomcDocumentsFetcher,
)
from shnifter_federal_reserve.models.money_measures import (
    FederalReserveMoneyMeasuresFetcher,
)
from shnifter_federal_reserve.models.overnight_bank_funding_rate import (
    FederalReserveOvernightBankFundingRateFetcher,
)
from shnifter_federal_reserve.models.primary_dealer_fails import (
    FederalReservePrimaryDealerFailsFetcher,
)
from shnifter_federal_reserve.models.primary_dealer_positioning import (
    FederalReservePrimaryDealerPositioningFetcher,
)
from shnifter_federal_reserve.models.sofr import FederalReserveSOFRFetcher
from shnifter_federal_reserve.models.treasury_rates import (
    FederalReserveTreasuryRatesFetcher,
)
from shnifter_federal_reserve.models.yield_curve import FederalReserveYieldCurveFetcher

federal_reserve_provider = Provider(
    name="federal_reserve",
    website="https://www.federalreserve.gov/data.htm",  #  Not a typo, it's really .htm
    description="""Access data provided by the Federal Reserve System, the Central Bank of the United States.""",
    fetcher_dict={
        "CentralBankHoldings": FederalReserveCentralBankHoldingsFetcher,
        "FederalFundsRate": FederalReserveFederalFundsRateFetcher,
        "FomcDocuments": FederalReserveFomcDocumentsFetcher,
        "MoneyMeasures": FederalReserveMoneyMeasuresFetcher,
        "OvernightBankFundingRate": FederalReserveOvernightBankFundingRateFetcher,
        "PrimaryDealerFails": FederalReservePrimaryDealerFailsFetcher,
        "PrimaryDealerPositioning": FederalReservePrimaryDealerPositioningFetcher,
        "SOFR": FederalReserveSOFRFetcher,
        "TreasuryRates": FederalReserveTreasuryRatesFetcher,
        "YieldCurve": FederalReserveYieldCurveFetcher,
    },
    repr_name="Federal Reserve (FED)",
)
