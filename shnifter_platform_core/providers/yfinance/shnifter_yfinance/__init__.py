# Generated: 2025-07-04T09:50:39.431336
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Yahoo Finance provider module."""

from shnifter_core.provider.abstract.provider import Provider
from shnifter_yfinance.models.active import YFActiveFetcher
from shnifter_yfinance.models.aggressive_small_caps import YFAggressiveSmallCapsFetcher
from shnifter_yfinance.models.available_indices import YFinanceAvailableIndicesFetcher
from shnifter_yfinance.models.balance_sheet import YFinanceBalanceSheetFetcher
from shnifter_yfinance.models.cash_flow import YFinanceCashFlowStatementFetcher
from shnifter_yfinance.models.company_news import YFinanceCompanyNewsFetcher
from shnifter_yfinance.models.crypto_historical import YFinanceCryptoHistoricalFetcher
from shnifter_yfinance.models.currency_historical import YFinanceCurrencyHistoricalFetcher
from shnifter_yfinance.models.equity_historical import YFinanceEquityHistoricalFetcher
from shnifter_yfinance.models.equity_profile import YFinanceEquityProfileFetcher
from shnifter_yfinance.models.equity_quote import YFinanceEquityQuoteFetcher
from shnifter_yfinance.models.equity_screener import YFinanceEquityScreenerFetcher
from shnifter_yfinance.models.etf_info import YFinanceEtfInfoFetcher
from shnifter_yfinance.models.futures_curve import YFinanceFuturesCurveFetcher
from shnifter_yfinance.models.futures_historical import YFinanceFuturesHistoricalFetcher
from shnifter_yfinance.models.gainers import YFGainersFetcher
from shnifter_yfinance.models.growth_tech_equities import YFGrowthTechEquitiesFetcher
from shnifter_yfinance.models.historical_dividends import (
    YFinanceHistoricalDividendsFetcher,
)
from shnifter_yfinance.models.income_statement import YFinanceIncomeStatementFetcher
from shnifter_yfinance.models.index_historical import (
    YFinanceIndexHistoricalFetcher,
)
from shnifter_yfinance.models.key_executives import YFinanceKeyExecutivesFetcher
from shnifter_yfinance.models.key_metrics import YFinanceKeyMetricsFetcher
from shnifter_yfinance.models.losers import YFLosersFetcher
from shnifter_yfinance.models.options_chains import YFinanceOptionsChainsFetcher
from shnifter_yfinance.models.price_target_consensus import (
    YFinancePriceTargetConsensusFetcher,
)
from shnifter_yfinance.models.share_statistics import YFinanceShareStatisticsFetcher
from shnifter_yfinance.models.undervalued_growth_equities import (
    YFUndervaluedGrowthEquitiesFetcher,
)
from shnifter_yfinance.models.undervalued_large_caps import YFUndervaluedLargeCapsFetcher

yfinance_provider = Provider(
    name="yfinance",
    website="https://finance.yahoo.com",
    description="""Yahoo! Finance is a web-based engine that offers financial news,
data, and tools for investors and individuals interested in tracking and analyzing
financial markets and assets.""",
    fetcher_dict={
        "AvailableIndices": YFinanceAvailableIndicesFetcher,
        "BalanceSheet": YFinanceBalanceSheetFetcher,
        "CashFlowStatement": YFinanceCashFlowStatementFetcher,
        "CompanyNews": YFinanceCompanyNewsFetcher,
        "CryptoHistorical": YFinanceCryptoHistoricalFetcher,
        "CurrencyHistorical": YFinanceCurrencyHistoricalFetcher,
        "EquityActive": YFActiveFetcher,
        "EquityAggressiveSmallCaps": YFAggressiveSmallCapsFetcher,
        "EquityGainers": YFGainersFetcher,
        "EquityHistorical": YFinanceEquityHistoricalFetcher,
        "EquityInfo": YFinanceEquityProfileFetcher,
        "EquityLosers": YFLosersFetcher,
        "EquityQuote": YFinanceEquityQuoteFetcher,
        "EquityScreener": YFinanceEquityScreenerFetcher,
        "EquityUndervaluedGrowth": YFUndervaluedGrowthEquitiesFetcher,
        "EquityUndervaluedLargeCaps": YFUndervaluedLargeCapsFetcher,
        "EtfHistorical": YFinanceEquityHistoricalFetcher,
        "EtfInfo": YFinanceEtfInfoFetcher,
        "FuturesCurve": YFinanceFuturesCurveFetcher,
        "FuturesHistorical": YFinanceFuturesHistoricalFetcher,
        "GrowthTechEquities": YFGrowthTechEquitiesFetcher,
        "HistoricalDividends": YFinanceHistoricalDividendsFetcher,
        "IncomeStatement": YFinanceIncomeStatementFetcher,
        "IndexHistorical": YFinanceIndexHistoricalFetcher,
        "KeyExecutives": YFinanceKeyExecutivesFetcher,
        "KeyMetrics": YFinanceKeyMetricsFetcher,
        "OptionsChains": YFinanceOptionsChainsFetcher,
        "PriceTargetConsensus": YFinancePriceTargetConsensusFetcher,
        "ShareStatistics": YFinanceShareStatisticsFetcher,
    },
    repr_name="Yahoo Finance",
)
