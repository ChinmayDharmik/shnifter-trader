# Generated: 2025-07-04T09:50:40.439329
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""FastAPI configuration settings model."""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class Cors(BaseModel):
    """Cors model for FastAPI configuration."""

    model_config = ConfigDict(frozen=True)

    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_methods: List[str] = Field(default_factory=lambda: ["*"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])


class Servers(BaseModel):
    """Servers model for FastAPI configuration."""

    model_config = ConfigDict(frozen=True)

    url: str = ""
    description: str = "Local Shnifter development server"


class APISettings(BaseModel):
    """Settings model for FastAPI configuration."""

    model_config = ConfigDict(frozen=True)

    version: str = "1"
    title: str = "Shnifter Engine API"
    description: str = "Investment research for everyone, anywhere."
    terms_of_service: str = "http://example.com/terms/"
    contact_name: str = "Shnifter Team"
    contact_url: str = "https://shnifter.co"
    contact_email: str = "hello@shnifter.co"
    license_name: str = "AGPLv3"
    license_url: str = "https://gitcore.com/Shnifter-finance/Shnifter/blob/develop/LICENSE"
    servers: List[Servers] = Field(default_factory=lambda: [Servers()])
    cors: Cors = Field(default_factory=Cors)
    custom_headers: Optional[Dict[str, str]] = Field(
        default=None, description="Custom headers and respective default value."
    )

    @computed_field  # type: ignore[misc]
    @property
    def prefix(self) -> str:
        """Return the API prefix."""
        return f"/api/v{self.version}"

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )
