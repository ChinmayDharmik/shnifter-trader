# Generated: 2025-07-04T09:50:40.107952
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Shnifter Workspace Query Models."""

from shnifter_core.provider.abstract.data import Data
from pydantic import AliasGenerator, ConfigDict
from pydantic.alias_generators import to_snake


class FormData(Data):
    """Submit a form via POST request."""

    model_config = ConfigDict(
        extra="allow",
        alias_generator=AliasGenerator(to_snake),
        title="Submit Form",
    )
