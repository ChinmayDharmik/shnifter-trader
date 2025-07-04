import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the shnifterject registry."""

from shnifter_interface.argparse_translator.shnifterject_registry import Registry
from shnifter_core.app.model.shnifterject import Shnifterject

# pylint: disable=unused-variable
# ruff: noqa: disable=F841


def test_registry_operations():
    """Test the registry operations."""
    registry = Registry()
    shnifterject1 = Shnifterject(
        id="1", results=True, extra={"register_key": "key1", "command": "cmd1"}
    )
    shnifterject2 = Shnifterject(
        id="2", results=True, extra={"register_key": "key2", "command": "cmd2"}
    )
    shnifterject3 = Shnifterject(  # noqa: F841
        id="3", results=True, extra={"register_key": "key3", "command": "cmd3"}
    )

    # Add shnifterjects to the registry
    assert registry.register(shnifterject1) is True
    assert registry.register(shnifterject2) is True
    # Attempt to add the same object again
    assert registry.register(shnifterject1) is False
    # Ensure the registry size is correct
    assert len(registry.shnifterjects) == 2

    # Get by index
    assert registry.get(0) == shnifterject2
    assert registry.get(1) == shnifterject1
    # Get by key
    assert registry.get("key1") == shnifterject1
    assert registry.get("key2") == shnifterject2
    # Invalid index/key
    assert registry.get(2) is None
    assert registry.get("invalid_key") is None

    # Remove an object
    registry.remove(0)
    assert len(registry.shnifterjects) == 1
    assert registry.get("key2") is None

    # Validate the 'all' property
    all_shnifterjects = registry.all
    assert "command" in all_shnifterjects[0]
    assert all_shnifterjects[0]["command"] == "cmd1"

    # Clean up by removing all objects
    registry.remove()
    assert len(registry.shnifterjects) == 0
    assert registry.get("key1") is None
