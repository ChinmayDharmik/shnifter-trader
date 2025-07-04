# Generated: 2025-07-04T09:50:39.875952
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""ECB provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_ecb.models.balance_of_payments import ECBBalanceOfPaymentsFetcher
from shnifter_ecb.models.currency_reference_rates import ECBCurrencyReferenceRatesFetcher
from shnifter_ecb.models.yield_curve import ECBYieldCurveFetcher

ecb_provider = Provider(
    name="ECB",
    website="https://data.ecb.europa.eu",
    description="""The ECB Data Portal provides access to all official ECB statistics.
The portal also provides options to download data and comprehensive metadata for each dataset.
Statistical publications and dashboards offer a compilation of key data on selected topics.""",
    fetcher_dict={
        "BalanceOfPayments": ECBBalanceOfPaymentsFetcher,
        "CurrencyReferenceRates": ECBCurrencyReferenceRatesFetcher,
        "YieldCurve": ECBYieldCurveFetcher,
    },
    repr_name="European Central Bank (ECB)",
)
