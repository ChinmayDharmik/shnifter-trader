# Generated: 2025-07-04T09:50:40.448566
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Preferences for the Shnifter engine."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class Preferences(BaseModel):
    """Preferences for the Shnifter engine."""

    cache_directory: str = str(Path.home() / "ShnifterUserData" / "cache")
    chart_style: Literal["dark", "light"] = "dark"
    data_directory: str = str(Path.home() / "ShnifterUserData")
    export_directory: str = str(Path.home() / "ShnifterUserData" / "exports")
    metadata: bool = True
    output_type: Literal[
        "Shnifterject", "dataframe", "polars", "numpy", "dict", "chart", "llm"
    ] = Field(
        default="Shnifterject",
        description="Python default output type.",
        validate_default=True,
    )
    request_timeout: PositiveInt = 60
    show_warnings: bool = False
    table_style: Literal["dark", "light"] = "dark"
    user_styles_directory: str = str(Path.home() / "ShnifterUserData" / "styles" / "user")

    model_config = ConfigDict(validate_assignment=True)

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )
