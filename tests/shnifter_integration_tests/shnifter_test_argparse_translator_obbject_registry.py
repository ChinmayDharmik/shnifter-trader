import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test Shnifterject Registry."""

from unittest.mock import Mock

import pytest
from shnifter_interface.argparse_translator.shnifterject_registry import Registry
from shnifter_core.app.model.shnifterject import Shnifterject

# pylint: disable=redefined-outer-name, protected-access


@pytest.fixture
def registry():
    """Fixture to create a Registry instance for testing."""
    return Registry()


@pytest.fixture
def mock_shnifterject():
    """Fixture to create a mock Shnifterject for testing."""

    class MockModel:
        """Mock model for testing."""

        def __init__(self, value):
            self.mock_value = value
            self._model_json_schema = "mock_json_schema"

        def model_json_schema(self):
            return self._model_json_schema

    shnifter = Mock(spec=Shnifterject)
    shnifter.id = "123"
    shnifter.provider = "test_provider"
    shnifter.extra = {"command": "test_command"}
    shnifter._route = "/test/route"
    shnifter._standard_params = Mock()
    shnifter._standard_params = {}
    shnifter.results = [MockModel(1), MockModel(2)]
    return shnifter


def test_listing_all_shnifterjects(registry, mock_shnifterject):
    """Test listing all shnifterjects with additional properties."""
    registry.register(mock_shnifterject)

    all_shnifterjects = registry.all
    assert len(all_shnifterjects) == 1
    assert all_shnifterjects[0]["command"] == "test_command"
    assert all_shnifterjects[0]["provider"] == "test_provider"


def test_registry_initialization(registry):
    """Test the Registry is initialized correctly."""
    assert registry.shnifterjects == []


def test_register_new_shnifterject(registry, mock_shnifterject):
    """Test registering a new Shnifterject."""
    registry.register(mock_shnifterject)
    assert mock_shnifterject in registry.shnifterjects


def test_register_duplicate_shnifterject(registry, mock_shnifterject):
    """Test that duplicate Shnifterjects are not added."""
    registry.register(mock_shnifterject)
    registry.register(mock_shnifterject)
    assert len(registry.shnifterjects) == 1


def test_get_shnifterject_by_index(registry, mock_shnifterject):
    """Test retrieving an shnifterject by its index."""
    registry.register(mock_shnifterject)
    retrieved = registry.get(0)
    assert retrieved == mock_shnifterject


def test_remove_shnifterject_by_index(registry, mock_shnifterject):
    """Test removing an shnifterject by index."""
    registry.register(mock_shnifterject)
    registry.remove(0)
    assert mock_shnifterject not in registry.shnifterjects


def test_remove_last_shnifterject_by_default(registry, mock_shnifterject):
    """Test removing the last shnifterject by default."""
    registry.register(mock_shnifterject)
    registry.remove()
    assert not registry.shnifterjects
