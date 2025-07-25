# Generated: 2025-07-04T09:50:40.032650
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

### THIS FILE IS AUTO-GENERATED. DO NOT EDIT. ###

from typing import Literal, Optional

from shnifter_core.app.model.field import ShnifterField
from shnifter_core.app.model.shnifterject import Shnifterject
from shnifter_core.app.static.container import Container
from shnifter_core.app.static.utils.decorators import exception_handler, validate
from shnifter_core.app.static.utils.filters import filter_inputs
from typing_extensions import Annotated


class ROUTER_index(Container):
    """/index
    available
    constituents
    /price
    """

    def __repr__(self) -> str:
        return self.__doc__ or ""

    @exception_handler
    @validate
    def available(
        self,
        provider: Annotated[
            Optional[Literal["fmp", "yfinance"]],
            ShnifterField(
                description="The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp, yfinance."
            ),
        ] = None,
        **kwargs
    ) -> Shnifterject:
        """All indices available from a given provider.

        Parameters
        ----------
        provider : str
            The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp, yfinance.

        Returns
        -------
        Shnifterject
            results : list[AvailableIndices]
                Serializable results.
            provider : Optional[str]
                Provider name.
            warnings : Optional[list[Warning_]]
                list of warnings.
            chart : Optional[Chart]
                Chart object.
            extra : Dict[str, Any]
                Extra info.

        AvailableIndices
        ----------------
        name : Optional[str]
            Name of the index.
        currency : Optional[str]
            Currency the index is traded in.
        stock_exchange : Optional[str]
            Stock exchange where the index is listed. (provider: fmp)
        exchange_short_name : Optional[str]
            Short name of the stock exchange where the index is listed. (provider: fmp)
        code : Optional[str]
            ID code for keying the index in the Shnifter Trader. (provider: yfinance)
        symbol : Optional[str]
            Symbol for the index. (provider: yfinance)

        Examples
        --------
        >>> from shnifter import shnifter
        >>> shnifter.index.available(provider='fmp')
        >>> shnifter.index.available(provider='yfinance')
        """  # noqa: E501

        return self._run(
            "/index/available",
            **filter_inputs(
                provider_choices={
                    "provider": self._get_provider(
                        provider,
                        "index.available",
                        ("fmp", "yfinance"),
                    )
                },
                standard_params={},
                extra_params=kwargs,
            )
        )

    @exception_handler
    @validate
    def constituents(
        self,
        symbol: Annotated[str, ShnifterField(description="Symbol to get data for.")],
        provider: Annotated[
            Optional[Literal["fmp"]],
            ShnifterField(
                description="The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp."
            ),
        ] = None,
        **kwargs
    ) -> Shnifterject:
        """Get Index Constituents.

        Parameters
        ----------
        provider : str
            The provider to use, by default None. If None, the priority list configured in the settings is used. Default priority: fmp.
        symbol : str
            Symbol to get data for.

        Returns
        -------
        Shnifterject
            results : list[IndexConstituents]
                Serializable results.
            provider : Optional[str]
                Provider name.
            warnings : Optional[list[Warning_]]
                list of warnings.
            chart : Optional[Chart]
                Chart object.
            extra : Dict[str, Any]
                Extra info.

        IndexConstituents
        -----------------
        symbol : str
            Symbol representing the entity requested in the data.
        name : Optional[str]
            Name of the constituent company in the index.
        sector : Optional[str]
            Sector the constituent company in the index belongs to. (provider: fmp)
        sub_sector : Optional[str]
            Sub-sector the constituent company in the index belongs to. (provider: fmp)
        headquarter : Optional[str]
            Location of the headquarter of the constituent company in the index. (provider: fmp)
        date_first_added : Optional[Union[date, str]]
            Date the constituent company was added to the index. (provider: fmp)
        cik : Optional[int]
            Central Index Key (CIK) for the requested entity. (provider: fmp)
        founded : Optional[Union[date, str]]
            Founding year of the constituent company in the index. (provider: fmp)

        Examples
        --------
        >>> from shnifter import shnifter
        >>> shnifter.index.constituents(symbol='dowjones', provider='fmp')
        """  # noqa: E501

        return self._run(
            "/index/constituents",
            **filter_inputs(
                provider_choices={
                    "provider": self._get_provider(
                        provider,
                        "index.constituents",
                        ("fmp",),
                    )
                },
                standard_params={
                    "symbol": symbol,
                },
                extra_params=kwargs,
            )
        )

    @property
    def price(self):
        # pylint: disable=import-outside-toplevel
        from . import index_price

        return index_price.ROUTER_index_price(command_runner=self._command_runner)
