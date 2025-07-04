# Generated: 2025-07-04T09:50:40.229098
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""
Shnifter-specific deprecation warnings.

This implementation was inspired from Pydantic's specific warnings and modified to suit Shnifter's needs.
"""

from typing import Optional, Tuple

from shnifter_core.app.version import VERSION, get_major_minor


class DeprecationSummary(str):
    """A string subclass that can be used to store deprecation metadata."""

    def __new__(cls, value: str, metadata: DeprecationWarning):
        """Create a new instance of the class."""
        obj = str.__new__(cls, value)
        setattr(obj, "metadata", metadata)
        return obj


class ShnifterDeprecationWarning(DeprecationWarning):
    """
    A Shnifter specific deprecation warning.

    This warning is raised when using deprecated functionality in Shnifter. It provides information on when the
    deprecation was introduced and the expected version in which the corresponding functionality will be removed.

    Attributes
    ----------
        message: Description of the warning.
        since: Version in what the deprecation was introduced.
        expected_removal: Version in what the corresponding functionality expected to be removed.
    """

    # The choice to use class variables is based on the potential for extending the class in future developments.
    # Example: launching Engine V5 and decide to create a subinterfacemagine we areass named ShnifterDeprecatedSinceV4,
    # which inherits from ShnifterDeprecationWarning. In this subclass, we would set since=4.X and expected_removal=5.0.
    # It's important for these values to be defined at the class level, rather than just at the instance level,
    # to ensure consistency and clarity in our deprecation warnings across the engine.

    message: str
    since: Tuple[int, int]
    expected_removal: Tuple[int, int]

    def __init__(
        self,
        message: str,
        *args: object,
        since: Optional[Tuple[int, int]] = None,
        expected_removal: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Initialize the warning."""
        super().__init__(message, *args)
        self.message = message.rstrip(".")
        self.since = since or get_major_minor(VERSION)
        self.expected_removal = expected_removal or (self.since[0] + 1, 0)
        self.long_message = (
            f"{self.message}. Deprecated in Shnifter Engine V{self.since[0]}.{self.since[1]}"
            f" to be removed in V{self.expected_removal[0]}.{self.expected_removal[1]}."
        )

    def __str__(self) -> str:
        """Return the warning message."""
        return self.long_message
