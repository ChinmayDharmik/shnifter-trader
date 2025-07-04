# Generated: 2025-07-04T09:50:39.587466
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""OECD provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_oecd.models.composite_leading_indicator import (
    OECDCompositeLeadingIndicatorFetcher,
)
from shnifter_oecd.models.consumer_price_index import OECDCPIFetcher
from shnifter_oecd.models.country_interest_rates import OecdCountryInterestRatesFetcher
from shnifter_oecd.models.gdp_forecast import OECDGdpForecastFetcher
from shnifter_oecd.models.gdp_nominal import OECDGdpNominalFetcher
from shnifter_oecd.models.gdp_real import OECDGdpRealFetcher
from shnifter_oecd.models.house_price_index import OECDHousePriceIndexFetcher
from shnifter_oecd.models.immediate_interest_rate import OECDImmediateInterestRateFetcher
from shnifter_oecd.models.long_term_interest_rate import OECDLTIRFetcher
from shnifter_oecd.models.share_price_index import OECDSharePriceIndexFetcher
from shnifter_oecd.models.short_term_interest_rate import OECDSTIRFetcher
from shnifter_oecd.models.unemployment import OECDUnemploymentFetcher

oecd_provider = Provider(
    name="oecd",
    website="https://data-explorer.oecd.org/",
    description="""OECD Data Explorer includes data and metadata for OECD countries and selected
non-member economies.""",
    fetcher_dict={
        "CompositeLeadingIndicator": OECDCompositeLeadingIndicatorFetcher,
        "ConsumerPriceIndex": OECDCPIFetcher,
        "CountryInterestRates": OecdCountryInterestRatesFetcher,
        "GdpNominal": OECDGdpNominalFetcher,
        "GdpReal": OECDGdpRealFetcher,
        "GdpForecast": OECDGdpForecastFetcher,
        "HousePriceIndex": OECDHousePriceIndexFetcher,
        "SharePriceIndex": OECDSharePriceIndexFetcher,
        "Unemployment": OECDUnemploymentFetcher,
        "ImmediateInterestRate": OECDImmediateInterestRateFetcher,  # TODO: deprecated
        "STIR": OECDSTIRFetcher,  # TODO: deprecated
        "LTIR": OECDLTIRFetcher,  # TODO: deprecated
    },
    repr_name="Organization for Economic Co-operation and Development (OECD)",
)
