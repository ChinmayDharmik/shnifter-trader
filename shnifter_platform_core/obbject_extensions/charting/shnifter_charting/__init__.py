# Generated: 2025-07-04T09:50:40.043314
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Shnifterject extension for charting."""

import warnings

from shnifter_core.app.model.extension import Extension

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="shnifter_core.app.model.extension",
)


def get_charting_module():
    """Get the Charting module."""
    # pylint: disable=import-outside-toplevel
    import importlib

    _Charting = importlib.import_module("shnifter_charting.charting").Charting
    return _Charting


ext = Extension(name="charting", description="Create custom charts from Shnifterject data.")

Charting = ext.shnifterject_accessor(get_charting_module())
