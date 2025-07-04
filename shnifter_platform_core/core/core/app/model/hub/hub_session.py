# Generated: 2025-07-04T09:50:40.483121
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Model for CoreSession."""

from typing import Optional

from pydantic import BaseModel, SecretStr, field_serializer


class CoreSession(BaseModel):
    """Model for CoreSession."""

    username: Optional[str] = None
    email: str
    primary_usage: str
    user_uuid: str
    token_type: str
    access_token: SecretStr

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )

    @field_serializer("access_token", when_used="json-unless-none")
    def _dump_secret(self, v):
        """Dump secret."""
        return v.get_secret_value()
