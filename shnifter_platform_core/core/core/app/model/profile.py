# Generated: 2025-07-04T09:50:40.449593
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Profile model."""

from typing import Optional

from shnifter_core.app.model.core.core_session import CoreSession
from pydantic import BaseModel, ConfigDict, Field


class Profile(BaseModel):
    """Profile."""

    core_session: Optional[CoreSession] = Field(default=None)
    model_config = ConfigDict(validate_assignment=True)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}\n\n" + "\n".join(
            f"{k}: {v}" for k, v in self.model_dump().items()
        )
