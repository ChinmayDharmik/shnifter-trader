# Generated: 2025-07-04T09:50:40.444417
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Extension class for Shnifterject extensions."""

import warnings
from typing import Callable, List, Optional


class Extension:
    """
    Serves as Shnifterject extension entry point and must be created by each extension package.

    See https://docs.shnifter.co/engine/development/developer-guidelines/shnifterject_extensions.
    """

    def __init__(
        self,
        name: str,
        credentials: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """Initialize the extension.

        Parameters
        ----------
        name : str
            Name of the extension.
        credentials : Optional[List[str]], optional
            List of required credentials, by default None
        description: Optional[str]
            Extension description.
        """
        self.name = name
        self.credentials = credentials or []
        self.description = description

    @property
    def shnifterject_accessor(self) -> Callable:
        """Extend an Shnifterject, inspired by pandas."""
        # pylint: disable=import-outside-toplevel
        # Avoid circular imports

        from shnifter_core.app.model.shnifterject import Shnifterject

        return self.register_accessor(self.name, Shnifterject)

    @staticmethod
    def register_accessor(name, cls) -> Callable:
        """Register a custom accessor."""

        def decorator(accessor):
            if hasattr(cls, name):
                warnings.warn(
                    f"registration of accessor '{repr(accessor)}' under name "
                    f"'{repr(name)}' for type '{repr(cls)}' is overriding a preexisting "
                    f"attribute with the same name.",
                    UserWarning,
                )
            setattr(cls, name, CachedAccessor(name, accessor))
            # pylint: disable=protected-access
            cls.accessors.add(name)
            return accessor

        return decorator


class CachedAccessor:
    """CachedAccessor."""

    def __init__(self, name: str, accessor) -> None:
        """Initialize the cached accessor."""
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        """Get the cached accessor."""
        if obj is None:
            return self._accessor
        accessor_obj = self._accessor(obj)
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj
