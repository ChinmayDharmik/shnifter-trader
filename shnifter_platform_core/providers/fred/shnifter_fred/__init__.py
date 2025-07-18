# Generated: 2025-07-04T09:50:39.701823
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FRED provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_fred.models.ameribor import FredAmeriborFetcher
from shnifter_fred.models.balance_of_payments import FredBalanceOfPaymentsFetcher
from shnifter_fred.models.bond_indices import FredBondIndicesFetcher
from shnifter_fred.models.commercial_paper import FREDCommercialPaperFetcher
from shnifter_fred.models.commodity_spot_prices import FredCommoditySpotPricesFetcher
from shnifter_fred.models.consumer_price_index import FREDConsumerPriceIndexFetcher
from shnifter_fred.models.dwpcr_rates import FREDDiscountWindowPrimaryCreditRateFetcher
from shnifter_fred.models.ecb_interest_rates import (
    FREDEuropeanCentralBankInterestRatesFetcher,
)
from shnifter_fred.models.euro_short_term_rate import FredEuroShortTermRateFetcher
from shnifter_fred.models.fed_projections import FREDPROJECTIONFetcher
from shnifter_fred.models.federal_funds_rate import FredFederalFundsRateFetcher
from shnifter_fred.models.ffrmc import FREDSelectedTreasuryConstantMaturityFetcher
from shnifter_fred.models.high_quality_market import (
    FredHighQualityMarketCorporateBondFetcher,
)
from shnifter_fred.models.ice_bofa import FREDICEBofAFetcher
from shnifter_fred.models.iorb_rates import FREDIORBFetcher
from shnifter_fred.models.manufacturing_outlook_ny import (
    FredManufacturingOutlookNYFetcher,
)
from shnifter_fred.models.manufacturing_outlook_texas import (
    FredManufacturingOutlookTexasFetcher,
)
from shnifter_fred.models.moody import FREDMoodyCorporateBondIndexFetcher
from shnifter_fred.models.mortgage_indices import FredMortgageIndicesFetcher
from shnifter_fred.models.non_farm_payrolls import FredNonFarmPayrollsFetcher
from shnifter_fred.models.overnight_bank_funding_rate import (
    FredOvernightBankFundingRateFetcher,
)
from shnifter_fred.models.personal_consumption_expenditures import (
    FredPersonalConsumptionExpendituresFetcher,
)
from shnifter_fred.models.regional import FredRegionalDataFetcher
from shnifter_fred.models.release_table import FredReleaseTableFetcher
from shnifter_fred.models.retail_prices import FredRetailPricesFetcher
from shnifter_fred.models.search import (
    FredSearchFetcher,
)
from shnifter_fred.models.senior_loan_officer_survey import (
    FredSeniorLoanOfficerSurveyFetcher,
)
from shnifter_fred.models.series import FredSeriesFetcher
from shnifter_fred.models.sofr import FREDSOFRFetcher
from shnifter_fred.models.sonia_rates import FREDSONIAFetcher
from shnifter_fred.models.spot import FREDSpotRateFetcher
from shnifter_fred.models.survey_of_economic_conditions_chicago import (
    FredSurveyOfEconomicConditionsChicagoFetcher,
)
from shnifter_fred.models.tbffr import FREDSelectedTreasuryBillFetcher
from shnifter_fred.models.tips_yields import FredTipsYieldsFetcher
from shnifter_fred.models.tmc import FREDTreasuryConstantMaturityFetcher
from shnifter_fred.models.university_of_michigan import FredUofMichiganFetcher
from shnifter_fred.models.yield_curve import FREDYieldCurveFetcher

fred_provider = Provider(
    name="fred",
    website="https://fred.stlouisfed.org",
    description="""Federal Reserve Economic Data is a database maintained by the
Research division of the Federal Reserve Bank of St. Louis that has more than
816,000 economic time series from various sources.""",
    credentials=["api_key"],
    fetcher_dict={
        "BalanceOfPayments": FredBalanceOfPaymentsFetcher,
        "BondIndices": FredBondIndicesFetcher,
        "CommoditySpotPrices": FredCommoditySpotPricesFetcher,
        "ConsumerPriceIndex": FREDConsumerPriceIndexFetcher,
        "SOFR": FREDSOFRFetcher,
        "EuroShortTermRate": FredEuroShortTermRateFetcher,
        "SONIA": FREDSONIAFetcher,
        "Ameribor": FredAmeriborFetcher,
        "FederalFundsRate": FredFederalFundsRateFetcher,
        "PROJECTIONS": FREDPROJECTIONFetcher,
        "IORB": FREDIORBFetcher,
        "DiscountWindowPrimaryCreditRate": FREDDiscountWindowPrimaryCreditRateFetcher,
        "EuropeanCentralBankInterestRates": FREDEuropeanCentralBankInterestRatesFetcher,
        "ICEBofA": FREDICEBofAFetcher,
        "ManufacturingOutlookNY": FredManufacturingOutlookNYFetcher,
        "ManufacturingOutlookTexas": FredManufacturingOutlookTexasFetcher,
        "MoodyCorporateBondIndex": FREDMoodyCorporateBondIndexFetcher,
        "MortgageIndices": FredMortgageIndicesFetcher,
        "NonFarmPayrolls": FredNonFarmPayrollsFetcher,
        "OvernightBankFundingRate": FredOvernightBankFundingRateFetcher,
        "PersonalConsumptionExpenditures": FredPersonalConsumptionExpendituresFetcher,
        "CommercialPaper": FREDCommercialPaperFetcher,
        "FredReleaseTable": FredReleaseTableFetcher,
        "FredSearch": FredSearchFetcher,
        "FredSeries": FredSeriesFetcher,
        "FredRegional": FredRegionalDataFetcher,
        "RetailPrices": FredRetailPricesFetcher,
        "SeniorLoanOfficerSurvey": FredSeniorLoanOfficerSurveyFetcher,
        "SpotRate": FREDSpotRateFetcher,
        "HighQualityMarketCorporateBond": FredHighQualityMarketCorporateBondFetcher,
        "TreasuryConstantMaturity": FREDTreasuryConstantMaturityFetcher,
        "SelectedTreasuryConstantMaturity": FREDSelectedTreasuryConstantMaturityFetcher,
        "SelectedTreasuryBill": FREDSelectedTreasuryBillFetcher,
        "SurveyOfEconomicConditionsChicago": FredSurveyOfEconomicConditionsChicagoFetcher,
        "TipsYields": FredTipsYieldsFetcher,
        "UniversityOfMichigan": FredUofMichiganFetcher,
        "YieldCurve": FREDYieldCurveFetcher,
    },
    repr_name="Federal Reserve Economic Data | St. Louis FED (FRED)",
    deprecated_credentials={"API_FRED_KEY": "fred_api_key"},
    instructions='Go to: https://fred.stlouisfed.org\n\n![FRED](https://user-images.gitcoreusercontent.com/46355364/207827137-d143ba4c-72cb-467d-a7f4-5cc27c597aec.png)\n\nClick on, "My Account", create a new account or sign in with Google:\n\n![FRED](https://user-images.gitcoreusercontent.com/46355364/207827011-65cdd501-27e3-436f-bd9d-b0d8381d46a7.png)\n\nAfter completing the sign-up, go to "My Account", and select "API Keys". Then, interfaceck on, "Request API Key".\n\n![FRED](https://user-images.gitcoreusercontent.com/46355364/207827577-c869f989-4ef4-4949-ab57-6f3931f2ae9d.png)\n\nFill in the box for information about the use-case for FRED, and by interfacecking, "Request API key", at the bottom of the page, the API key will be issued.\n\n![FRED](https://user-images.gitcoreusercontent.com/46355364/207828032-0a32d3b8-1378-4db2-9064-aa1eb2111632.png)',  # noqa: E501  pylint: disable=line-too-long
)
