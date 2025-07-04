# Generated: 2025-07-04T09:50:39.529200
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""stockgrid provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_stockgrid.models.short_volume import StockgridShortVolumeFetcher

stockgrid_provider = Provider(
    name="stockgrid",
    website="https://www.stockgrid.io",
    description="""Stockgrid gives you a detailed view of what smart money is doing.
Get in depth data about large option blocks being traded, including
the sentiment score, size, volume and order type. Stop guessing and
build a strategy around the number 1 factor moving the market: money.""",
    fetcher_dict={
        "ShortVolume": StockgridShortVolumeFetcher,
    },
    repr_name="Stockgrid",
)
