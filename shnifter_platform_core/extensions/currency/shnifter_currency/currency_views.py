# Generated: 2025-07-04T09:50:40.198957
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Views for the Currency Extension."""

from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from shnifter_charting.core.shnifter_figure import (
        ShnifterFigure,
    )


class CurrencyViews:
    """Currency Views."""

    @staticmethod
    def currency_price_historical(  # noqa: PLR0912
        **kwargs,
    ) -> Tuple["ShnifterFigure", Dict[str, Any]]:
        """Currency Price Historical Chart."""
        # pylint: disable=import-outside-toplevel
        from shnifter_charting.charts.price_historical import price_historical

        return price_historical(**kwargs)
