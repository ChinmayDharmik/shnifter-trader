# Generated: 2025-07-04T09:50:39.941206
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Benzinga provider module."""

from shnifter_benzinga.models.analyst_search import BenzingaAnalystSearchFetcher
from shnifter_benzinga.models.company_news import BenzingaCompanyNewsFetcher
from shnifter_benzinga.models.price_target import BenzingaPriceTargetFetcher
from shnifter_benzinga.models.world_news import BenzingaWorldNewsFetcher
from shnifter_core.provider.abstract.provider import Provider

benzinga_provider = Provider(
    name="benzinga",
    website="https://www.benzinga.com",
    description="""Benzinga is a financial data provider that offers an API
focused on information that moves the market.""",
    credentials=["api_key"],
    fetcher_dict={
        "AnalystSearch": BenzingaAnalystSearchFetcher,
        "CompanyNews": BenzingaCompanyNewsFetcher,
        "WorldNews": BenzingaWorldNewsFetcher,
        "PriceTarget": BenzingaPriceTargetFetcher,
    },
    repr_name="Benzinga",
)
