# Generated: 2025-07-04T09:50:40.093971
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

# pylint: disable=W0613:unused-argument
"""SEC Router."""

from shnifter_core.app.model.command_context import CommandContext
from shnifter_core.app.model.example import APIEx, PythonEx
from shnifter_core.app.model.shnifterject import Shnifterject
from shnifter_core.app.provider_interface import (
    ExtraParams,
    ProviderChoices,
    StandardParams,
)
from shnifter_core.app.query import Query
from shnifter_core.app.router import Router

router = Router(prefix="/sec")


@router.command(
    model="SecFiling",
    examples=[
        APIEx(
            parameters={
                "url": "https://www.sec.gov/Archives/edgar/data/317540/000119312524076556/d645509ddef14a.htm",
                "provider": "sec",
            }
        )
    ],
    openapi_extra={
        "widget_config": {
            "description": "Get a list of all the documents associated with a filing, and their direct URLs.",
            "gridData": {
                "w": 30,
                "h": 10,
            },
            "refetchInterval": False,
            "data": {"dataKey": "results.document_urls"},
        }
    },
)
async def filing_headers(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Download the index headers, and cover page if available, for any SEC filing."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SecHtmFile",
    examples=[
        APIEx(
            parameters={
                "url": "https://www.sec.gov/Archives/edgar/data/1723690/000119312525030074/d866336dex991.htm",
                "provider": "sec",
            }
        )
    ],
    openapi_extra={
        "widget_config": {
            "name": "Open HTML",
            "description": "Open a HTM/HTML document from the SEC website.",
            "gridData": {
                "w": 40,
                "h": 25,
            },
            "refetchInterval": False,
            "type": "markdown",
            "data": {
                "dataKey": "results.content",
            },
        }
    },
)
async def htm_file(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Download a raw HTML object from the SEC website."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="CikMap",
    examples=[APIEx(parameters={"symbol": "MSFT", "provider": "sec"})],
)
async def cik_map(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Map a ticker symbol to a CIK number."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="InstitutionsSearch",
    examples=[
        APIEx(parameters={"provider": "sec"}),
        APIEx(parameters={"query": "blackstone real estate", "provider": "sec"}),
    ],
)
async def institutions_search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Search SEC-regulated institutions by name and return a list of results with CIK numbers."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SchemaFiles",
    examples=[
        APIEx(parameters={"provider": "sec"}),
        PythonEx(
            description="Get a list of schema files.",
            code=[
                "data = shnifter.regulators.sec.schema_files().results",
                "data.files[0]",
                "'https://xbrl.fasb.org/us-gaap/'",
                "# The directory structure can be navigated by constructing a URL from the 'results' list.",
                "url = data.files[0]+data.files[-1]",
                "# The URL base will always be the 0 position in the list, feed  the URL back in as a parameter.",
                "shnifter.regulators.sec.schema_files(url=url).results.files",
                "['https://xbrl.fasb.org/us-gaap/2024/'",
                "'USGAAP2024FileList.xml'",
                "'dis/'",
                "'dqcrules/'",
                "'ebp/'",
                "'elts/'",
                "'entire/'",
                "'meta/'",
                "'stm/'",
                "'us-gaap-2024.zip']",
            ],
        ),
    ],
)
async def schema_files(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Use tool for navigating the directory of SEC XML schema files by year."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SymbolMap",
    examples=[APIEx(parameters={"query": "0000789019", "provider": "sec"})],
)
async def symbol_map(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Map a CIK number to a ticker symbol, leading 0s can be omitted or included."""
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="RssLitigation",
    examples=[APIEx(parameters={"provider": "sec"})],
)
async def rss_litigation(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Get the RSS feed that provides links to litigation releases concerning civil lawsuits brought by the Commission in federal court."""  # noqa: E501 pylint: disable=C0301
    return await Shnifterject.from_query(Query(**locals()))


@router.command(
    model="SicSearch",
    examples=[
        APIEx(parameters={"provider": "sec"}),
        APIEx(parameters={"query": "real estate investment trusts", "provider": "sec"}),
    ],
)
async def sic_search(
    cc: CommandContext,
    provider_choices: ProviderChoices,
    standard_params: StandardParams,
    extra_params: ExtraParams,
) -> Shnifterject:
    """Search for Industry Titles, Reporting Office, and SIC Codes. An empty query string returns all results."""
    return await Shnifterject.from_query(Query(**locals()))
