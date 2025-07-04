# Generated: 2025-07-04T09:50:39.465417
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""WSJ provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_wsj.models.active import WSJActiveFetcher
from shnifter_wsj.models.gainers import WSJGainersFetcher
from shnifter_wsj.models.losers import WSJLosersFetcher

wsj_provider = Provider(
    name="wsj",
    website="https://www.wsj.com",
    description="""WSJ (Wall Street Journal) is a business-focused, English-language
international daily newspaper based in New York City. The Journal is published six
days a week by Dow Jones & Company, a division of News Corp, along with its Asian
and European editions. The newspaper is published in the broadsheet format and
online. The Journal has been printed continuously since its inception on
July 8, 1889, by Charles Dow, Edward Jones, and Charles Bergstresser.
The WSJ is the largest newspaper in the United States, by circulation.
    """,
    fetcher_dict={
        "ETFGainers": WSJGainersFetcher,
        "ETFLosers": WSJLosersFetcher,
        "ETFActive": WSJActiveFetcher,
    },
    repr_name="Wall Street Journal (WSJ)",
)
