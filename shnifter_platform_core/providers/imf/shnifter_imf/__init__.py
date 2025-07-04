# Generated: 2025-07-04T09:50:39.673413
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter IMF Provider Module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_imf.models.available_indicators import ImfAvailableIndicatorsFetcher
from shnifter_imf.models.direction_of_trade import ImfDirectionOfTradeFetcher
from shnifter_imf.models.economic_indicators import ImfEconomicIndicatorsFetcher
from shnifter_imf.models.maritime_chokepoint_info import ImfMaritimeChokePointInfoFetcher
from shnifter_imf.models.maritime_chokepoint_volume import (
    ImfMaritimeChokePointVolumeFetcher,
)
from shnifter_imf.models.port_info import ImfPortInfoFetcher
from shnifter_imf.models.port_volume import ImfPortVolumeFetcher

imf_provider = Provider(
    name="imf",
    website="https://datahelp.imf.org/knowledgebase/articles/667681-using-json-restful-web-service",
    description="Access International Monetary Fund (IMF) data APIs.",
    fetcher_dict={
        "AvailableIndicators": ImfAvailableIndicatorsFetcher,
        "DirectionOfTrade": ImfDirectionOfTradeFetcher,
        "EconomicIndicators": ImfEconomicIndicatorsFetcher,
        "MaritimeChokePointInfo": ImfMaritimeChokePointInfoFetcher,
        "MaritimeChokePointVolume": ImfMaritimeChokePointVolumeFetcher,
        "PortInfo": ImfPortInfoFetcher,
        "PortVolume": ImfPortVolumeFetcher,
    },
    repr_name="International Monetary Fund (IMF) Data APIs",
)
