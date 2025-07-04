# Generated: 2025-07-04T09:50:39.962542
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

### THIS FILE IS AUTO-GENERATED. DO NOT EDIT. ###

import datetime
from typing import Literal, Optional, Union

from shnifter_core.app.model.field import ShnifterField
from shnifter_core.app.model.shnifterject import Shnifterject
from shnifter_core.app.static.container import Container
from shnifter_core.app.static.utils.decorators import exception_handler, validate
from shnifter_core.app.static.utils.filters import filter_inputs
from typing_extensions import Annotated


class ROUTER_crypto_price(Container):
    """/crypto/price
    historical
    """

    def __repr__(self) -> str:
        return self.__doc__ or ""

    @exception_handler
    @validate
    def historical(
        self,
        symbol: Annotated[
            Union[str, list[str]],
            ShnifterField(
                description="Symbol to get data for. Can use CURR1-CURR2 or CURR1CURR2 format. Multiple comma separated items allowed for provider(s): fmp, polygon, tiingo, yfinance."
            ),
        ],
        start_date: Annotated[
            Union[datetime.date, None, str],
            ShnifterField(description="Start date of the data, in YYYY-MM-DD format."),
        ] = None,
        end_date: Annotated[
            Union[datetime.date, None, str],
            ShnifterField(description="End date of the data, in YYYY-MM-DD format."),
        ] = None,
        provider: Annotated[
            Optional[Literal["fmp", "polygon", "tiingo", "yfinance"]],
            ShnifterField(
                description="The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp, polygon, tiingo, yfinance."
            ),
        ] = None,
        **kwargs
    ) -> Shnifterject:
        """Get historical price data for cryptocurrency pair(s) within a provider.

        Parameters
        ----------
        provider : str
            The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp, polygon, tiingo, yfinance.
        symbol : Union[str, list[str]]
            Symbol to get data for. Can use CURR1-CURR2 or CURR1CURR2 format. Multiple comma separated items allowed for provider(s): fmp, polygon, tiingo, yfinance.
        start_date : Union[date, None, str]
            Start date of the data, in YYYY-MM-DD format.
        end_date : Union[date, None, str]
            End date of the data, in YYYY-MM-DD format.
        interval : str
            Time interval of the data to return. (provider: fmp, polygon, tiingo, yfinance)
            Choices for fmp: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
            Choices for yfinance: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1W', '1M', '1Q'
        sort : Literal['asc', 'desc']
            Sort order of the data. This impacts the results in combination with the 'limit' parameter. The results are always returned in ascending order by date. (provider: polygon)
        limit : int
            The number of data entries to return. (provider: polygon)
        exchanges : Union[list[str], str, None]
            To limit the query to a subset of exchanges e.g. ['POLONIEX', 'GDAX'] Multiple comma separated items allowed. (provider: tiingo)

        Returns
        -------
        Shnifterject
            results : list[CryptoHistorical]
                Serializable results.
            provider : Optional[str]
                Provider name.
            warnings : Optional[list[Warning_]]
                list of warnings.
            chart : Optional[Chart]
                Chart object.
            extra : Dict[str, Any]
                Extra info.

        CryptoHistorical
        ----------------
        date : Union[date, datetime]
            The date of the data.
        open : float
            The open price.
        high : float
            The high price.
        low : float
            The low price.
        close : float
            The close price.
        volume : Optional[float]
            The trading volume.
        vwap : Optional[float]
            Volume Weighted Average Price over the period.
        adj_close : Optional[float]
            The adjusted close price. (provider: fmp)
        change : Optional[float]
            Change in the price from the previous close. (provider: fmp)
        change_percent : Optional[float]
            Change in the price from the previous close, as a normalized percent. (provider: fmp)
        transactions : Optional[Union[Annotated[int, Gt(gt=0)], int]]
            Number of transactions for the symbol in the time period. (provider: polygon, tiingo)
        volume_notional : Optional[float]
            The last size done for the asset on the specific date in the quote currency. The volume of the asset on the specific date in the quote currency. (provider: tiingo)

        Examples
        --------
        >>> from shnifter import shnifter
        >>> shnifter.crypto.price.historical(symbol='BTCUSD', provider='fmp')
        >>> shnifter.crypto.price.historical(symbol='BTCUSD', start_date='2024-01-01', end_date='2024-01-31', provider='fmp')
        >>> shnifter.crypto.price.historical(symbol='BTCUSD,ETHUSD', start_date='2024-01-01', end_date='2024-01-31', provider='polygon')
        >>> # Get monthly historical prices from Yahoo Finance for Ethereum.
        >>> shnifter.crypto.price.historical(symbol='ETH-USD', interval='1m', start_date='2024-01-01', end_date='2024-12-31', provider='yfinance')
        """  # noqa: E501

        return self._run(
            "/crypto/price/historical",
            **filter_inputs(
                provider_choices={
                    "provider": self._get_provider(
                        provider,
                        "crypto.price.historical",
                        ("fmp", "polygon", "tiingo", "yfinance"),
                    )
                },
                standard_params={
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                },
                extra_params=kwargs,
                info={
                    "symbol": {
                        "fmp": {"multiple_items_allowed": True, "choices": None},
                        "polygon": {"multiple_items_allowed": True, "choices": None},
                        "tiingo": {"multiple_items_allowed": True, "choices": None},
                        "yfinance": {"multiple_items_allowed": True, "choices": None},
                    },
                    "interval": {
                        "fmp": {
                            "multiple_items_allowed": False,
                            "choices": ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                        },
                        "tiingo": {
                            "multiple_items_allowed": False,
                            "choices": [
                                "1m",
                                "5m",
                                "15m",
                                "30m",
                                "90m",
                                "1h",
                                "2h",
                                "4h",
                                "1d",
                                "7d",
                                "30d",
                            ],
                        },
                        "yfinance": {
                            "multiple_items_allowed": False,
                            "choices": [
                                "1m",
                                "2m",
                                "5m",
                                "15m",
                                "30m",
                                "60m",
                                "90m",
                                "1h",
                                "1d",
                                "5d",
                                "1W",
                                "1M",
                                "1Q",
                            ],
                        },
                    },
                    "exchanges": {
                        "tiingo": {"multiple_items_allowed": True, "choices": None}
                    },
                },
            )
        )
