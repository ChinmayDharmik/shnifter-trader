# Generated: 2025-07-04T09:50:40.185510
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Views for the Econometrics Extension."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shnifter_charting.core.shnifter_figure import (
        ShnifterFigure,
    )


class EconometricsViews:
    """Econometrics Views."""

    @staticmethod
    def econometrics_correlation_matrix(  # noqa: PLR0912
        **kwargs,
    ) -> tuple["ShnifterFigure", dict[str, Any]]:
        """Correlation Matrix Chart.

        Parameters
        ----------
        data : Union[list[Data], DataFrame]
            Input dataset.
        method : Literal["pearson", "kendall", "spearman"]
            Method to use for correlation calculation. Default is "pearson".
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
        colorscale : str
            Plotly colorscale to use for the heatmap. Default is "RdBu".
        title : str
            Title of the chart. Default is "Asset Correlation Matrix".
        layout_kwargs : Dict[str, Any]
            Additional keyword arguments to apply with figure.update_layout(), by default None.
        """
        # pylint: disable=import-outside-toplevel
        from shnifter_charting.charts.correlation_matrix import correlation_matrix

        return correlation_matrix(**kwargs)
