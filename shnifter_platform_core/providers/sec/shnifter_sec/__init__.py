# Generated: 2025-07-04T09:50:39.540362
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""SEC provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_sec.models.cik_map import SecCikMapFetcher
from shnifter_sec.models.company_filings import SecCompanyFilingsFetcher
from shnifter_sec.models.compare_company_facts import SecCompareCompanyFactsFetcher
from shnifter_sec.models.equity_ftd import SecEquityFtdFetcher
from shnifter_sec.models.equity_search import SecEquitySearchFetcher
from shnifter_sec.models.etf_holdings import SecEtfHoldingsFetcher
from shnifter_sec.models.form_13FHR import SecForm13FHRFetcher
from shnifter_sec.models.htm_file import SecHtmFileFetcher
from shnifter_sec.models.insider_trading import SecInsiderTradingFetcher
from shnifter_sec.models.institutions_search import SecInstitutionsSearchFetcher
from shnifter_sec.models.latest_financial_reports import SecLatestFinancialReportsFetcher
from shnifter_sec.models.management_discussion_analysis import (
    SecManagementDiscussionAnalysisFetcher,
)
from shnifter_sec.models.rss_litigation import SecRssLitigationFetcher
from shnifter_sec.models.schema_files import SecSchemaFilesFetcher
from shnifter_sec.models.sec_filing import SecFilingFetcher
from shnifter_sec.models.sic_search import SecSicSearchFetcher
from shnifter_sec.models.symbol_map import SecSymbolMapFetcher

sec_provider = Provider(
    name="sec",
    website="https://www.sec.gov/data",
    description="SEC is the public listings regulatory body for the United States.",
    credentials=None,
    fetcher_dict={
        "CikMap": SecCikMapFetcher,
        "CompanyFilings": SecCompanyFilingsFetcher,
        "CompareCompanyFacts": SecCompareCompanyFactsFetcher,
        "EquityFTD": SecEquityFtdFetcher,
        "EquitySearch": SecEquitySearchFetcher,
        "EtfHoldings": SecEtfHoldingsFetcher,
        "Filings": SecCompanyFilingsFetcher,
        "Form13FHR": SecForm13FHRFetcher,
        "SecHtmFile": SecHtmFileFetcher,
        "InsiderTrading": SecInsiderTradingFetcher,
        "InstitutionsSearch": SecInstitutionsSearchFetcher,
        "LatestFinancialReports": SecLatestFinancialReportsFetcher,
        "ManagementDiscussionAnalysis": SecManagementDiscussionAnalysisFetcher,
        "RssLitigation": SecRssLitigationFetcher,
        "SchemaFiles": SecSchemaFilesFetcher,
        "SecFiling": SecFilingFetcher,
        "SicSearch": SecSicSearchFetcher,
        "SymbolMap": SecSymbolMapFetcher,
    },
    repr_name="Securities and Exchange Commission (SEC)",
)
