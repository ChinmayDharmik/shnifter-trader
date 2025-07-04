import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the Controller Factory."""

from unittest.mock import MagicMock, patch

import pytest
from shnifter_interface.controllers.engine_controller_factory import (
    EngineControllerFactory,
)

# pylint: disable=redefined-outer-name, unused-argument


@pytest.fixture
def mock_processor():
    """Fixture to mock ArgparseClassProcessor."""
    with patch(
        "shnifter_interface.controllers.engine_controller_factory.ArgparseClassProcessor"
    ) as mock:
        instance = mock.return_value
        instance.paths = {"settings": "menu"}
        instance.translators = {"test_router_settings": MagicMock()}
        yield instance


@pytest.fixture
def engine_router():
    """Fixture to provide a mock engine_router class."""

    class MockRouter:
        pass

    return MockRouter


@pytest.fixture
def factory(engine_router, mock_processor):
    """Fixture to create a EngineControllerFactory with mocked dependencies."""
    return EngineControllerFactory(
        engine_router=engine_router, reference={"test": "ref"}
    )


def test_init(mock_processor):
    """Test the initialization of the EngineControllerFactory."""
    factory = EngineControllerFactory(
        engine_router=MagicMock(), reference={"test": "ref"}
    )
    assert factory.router_name.lower() == "magicmock"
    assert factory.controller_name == "MagicmockController"


def test_create_controller(factory):
    """Test the creation of a controller class."""
    ControllerClass = factory.create()

    assert "EngineController" in [base.__name__ for base in ControllerClass.__bases__]
    assert ControllerClass.CHOICES_GENERATION
    assert "settings" in ControllerClass.CHOICES_MENUS
    assert "test_router_settings" not in [
        cmd.replace("test_router_", "") for cmd in ControllerClass.CHOICES_COMMANDS
    ]
