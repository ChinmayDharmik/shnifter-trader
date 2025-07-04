# Generated: 2025-07-04T09:50:39.694804
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Government US provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_government_us.models.treasury_auctions import (
    GovernmentUSTreasuryAuctionsFetcher,
)
from shnifter_government_us.models.treasury_prices import (
    GovernmentUSTreasuryPricesFetcher,
)

government_us_provider = Provider(
    name="government_us",
    website="https://data.gov",
    description="""Data.gov is the United States government's open data website.
It provides access to datasets published by agencies across the federal government.
Data.gov is intended to provide access to government open data to the public, achieve
agency missions, drive innovation, fuel economic activity, and uphold the ideals of
an open and transparent government.""",
    fetcher_dict={
        "TreasuryAuctions": GovernmentUSTreasuryAuctionsFetcher,
        "TreasuryPrices": GovernmentUSTreasuryPricesFetcher,
    },
    repr_name="Data.gov | United States Government",
)
