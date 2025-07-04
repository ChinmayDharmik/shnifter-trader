# Generated: 2025-07-04T09:50:40.477822
# Target: Shnifter Trader Platform with PySide6 and LLM integration
# Python Version: 3.13.2+

"""Singleton metaclass implementation."""

from typing import Dict, Generic, TypeVar

T = TypeVar("T")


class SingletonMeta(type, Generic[T]):
    """Singleton metaclass."""

    # TODO : check if we want to update this to be thread safe
    _instances: Dict[T, T] = {}

    def __call__(cls: "SingletonMeta", *args, **kwargs):
        """Singleton pattern implementation."""
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]
