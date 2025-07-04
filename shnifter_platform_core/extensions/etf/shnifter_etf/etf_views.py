# Generated: 2025-07-04T09:50:40.146651
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Views for the ETF Extension."""

# pylint: disable=unused-argument

from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

if TYPE_CHECKING:
    from shnifter_charting.core.shnifter_figure import (
        ShnifterFigure,
    )
    from plotly.graph_objs import Figure


class EtfViews:
    """Etf Views."""

    @staticmethod
    def etf_historical(
        **kwargs,
    ) -> Tuple["ShnifterFigure", Dict[str, Any]]:
        """Etf Price Historical Chart."""
        # pylint: disable=import-outside-toplevel

        from shnifter_charting.charts.price_historical import price_historical

        return price_historical(**kwargs)

    @staticmethod
    def etf_price_performance(
        **kwargs,
    ) -> Tuple["ShnifterFigure", Dict[str, Any]]:
        """Etf Price Performance Chart."""
        # pylint: disable=import-outside-toplevel
        from shnifter_charting.charts.price_performance import price_performance

        return price_performance(**kwargs)

    @staticmethod
    def etf_holdings(
        **kwargs,
    ) -> Tuple[Union["ShnifterFigure", "Figure"], Dict[str, Any]]:
        """Equity Compare Groups Chart."""
        # pylint: disable=import-outside-toplevel
        from pandas import DataFrame  # noqa
        from shnifter_core.app.utils import basemodel_to_df  # noqa
        from shnifter_core.app.model.abstract.error import ShnifterError  # noqa
        from shnifter_charting.charts.generic_charts import bar_chart  # noqa

        if "data" in kwargs and isinstance(kwargs["data"], DataFrame):
            data = kwargs["data"]
        elif "data" in kwargs and isinstance(kwargs["data"], list):
            data = basemodel_to_df(kwargs["data"], index=None)  # type: ignore
        else:
            data = basemodel_to_df(kwargs["shnifterject_item"], index=None)  # type: ignore

        if "weight" not in data.columns:
            raise ShnifterError("No 'weight' column found in the data.")

        orientation = kwargs.get("orientation", "h")
        limit = kwargs.get("limit", 20)
        symbol = kwargs["standard_params"].get("symbol")  # type: ignore
        title = kwargs.get("title", f"Top {limit} {symbol} Holdings")
        layout_kwargs = kwargs.get("layout_kwargs", {})

        data = data.sort_values("weight", ascending=False)
        limit = min(limit, len(data))  # type: ignore
        target = data.head(limit)[["symbol", "weight"]].set_index("symbol")
        target = target.multiply(100)
        axis_title = "Weight (%)"

        fig = bar_chart(
            target.reset_index(),
            "symbol",
            ["weight"],
            title=title,  # type: ignore
            xtitle=axis_title if orientation == "h" else None,
            ytitle=axis_title if orientation == "v" else None,
            orientation=orientation,  # type: ignore
        )

        fig.update_layout(
            hovermode="x" if orientation == "v" else "y",
            margin=dict(r=0, l=50) if orientation == "h" else None,
        )

        fig.update_traces(
            hovertemplate=(
                "%{y:.3f}%<extra></extra>"
                if orientation == "v"
                else "%{x:.3f}%<extra></extra>"
            )
        )

        if layout_kwargs:
            fig.update_layout(**layout_kwargs)  # type: ignore

        content = fig.show(external=True).to_plotly_json()  # type: ignore

        return fig, content
