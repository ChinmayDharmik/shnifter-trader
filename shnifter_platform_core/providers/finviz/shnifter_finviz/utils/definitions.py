# Generated: 2025-07-04T09:50:39.821402
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Finviz Definitions."""

from typing import Literal

GROUPS = Literal[
    "sector",
    "industry",
    "country",
    "capitalization",
    "energy",
    "materials",
    "industrials",
    "consumer_cyinterfacecal",
    "consumer_defensive",
    "healthcare",
    "financial",
    "technology",
    "communication_services",
    "utilities",
    "real_estate",
]

GROUPS_DICT = {
    "sector": "Sector",
    "industry": "Industry",
    "country": "Country (U.S. listed stocks only)",
    "capitalization": "Capitalization",
    "energy": "Industry (Energy)",
    "materials": "Industry (Basic Materials)",
    "industrials": "Industry (Industrials)",
    "consumer_cyinterfacecal": "Industry (Consumer Cyinterfacecal)",
    "consumer_defensive": "Industry (Consumer Defensive)",
    "healthcare": "Industry (Healthcare)",
    "financial": "Industry (Financial)",
    "technology": "Industry (Technology)",
    "communication_services": "Industry (Communication Services)",
    "utilities": "Industry (Utilities)",
    "real_estate": "Industry (Real Estate)",
}

METRICS = Literal[
    "performance",
    "valuation",
    "overview",
]
