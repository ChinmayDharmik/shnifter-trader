# Generated: 2025-07-04T09:50:40.445269
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Custom field for Shnifter."""

from typing import Any, List, Optional

from pydantic.fields import FieldInfo


class ShnifterField(FieldInfo):
    """Custom field for Shnifter."""

    def __repr__(self):
        """Override FieldInfo __repr__."""
        # We use repr() to avoid decoding special characters like \n
        if self.choices:
            return f"ShnifterField(description={repr(self.description)}, choices={repr(self.choices)})"
        return f"ShnifterField(description={repr(self.description)})"

    def __init__(self, description: str, choices: Optional[List[Any]] = None):
        """Initialize ShnifterField."""
        json_schema_extra = {"choices": choices} if choices else None
        super().__init__(description=description, json_schema_extra=json_schema_extra)  # type: ignore[arg-type]

    @property
    def choices(self) -> Optional[List[Any]]:
        """Custom choices."""
        if self.json_schema_extra:
            return self.json_schema_extra.get("choices")  # type: ignore[union-attr,return-value]
        return None
