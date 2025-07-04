# Generated: 2025-07-04T09:50:39.811688
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Finviz provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_finviz.models.compare_groups import FinvizCompareGroupsFetcher
from shnifter_finviz.models.equity_profile import FinvizEquityProfileFetcher
from shnifter_finviz.models.equity_screener import FinvizEquityScreenerFetcher
from shnifter_finviz.models.key_metrics import FinvizKeyMetricsFetcher
from shnifter_finviz.models.price_performance import FinvizPricePerformanceFetcher
from shnifter_finviz.models.price_target import FinvizPriceTargetFetcher

finviz_provider = Provider(
    name="finviz",
    website="https://finviz.com",
    description="Unofficial Finviz API - https://gitcore.com/lit26/finvizfinance/releases",
    credentials=None,
    fetcher_dict={
        "CompareGroups": FinvizCompareGroupsFetcher,
        "EtfPricePerformance": FinvizPricePerformanceFetcher,
        "EquityInfo": FinvizEquityProfileFetcher,
        "EquityScreener": FinvizEquityScreenerFetcher,
        "KeyMetrics": FinvizKeyMetricsFetcher,
        "PricePerformance": FinvizPricePerformanceFetcher,
        "PriceTarget": FinvizPriceTargetFetcher,
    },
    repr_name="FinViz",
)
