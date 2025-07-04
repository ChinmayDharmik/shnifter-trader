import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the base engine controller."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from shnifter_interface.controllers.base_engine_controller import (
    EngineController,
    Session,
)

# pylint: disable=protected-access, unused-variable, redefined-outer-name


@pytest.fixture
def engine_controller():
    """Return a engine controller."""
    session = Session()  # noqa: F841
    translators = {"test_command": MagicMock(), "test_menu": MagicMock()}  # noqa: F841
    translators["test_command"]._parser = Mock(
        _actions=[Mock(dest="data", choices=[], type=str, nargs=None)]
    )
    translators["test_command"].execute_func = Mock(return_value=Mock())
    translators["test_menu"]._parser = Mock(
        _actions=[Mock(dest="data", choices=[], type=str, nargs=None)]
    )
    translators["test_menu"].execute_func = Mock(return_value=Mock())

    controller = EngineController(
        name="test", parent_path=["engine"], translators=translators
    )
    return controller


@pytest.mark.integration
def test_engine_controller_initialization(engine_controller):
    """Test the initialization of the engine controller."""
    expected_path = "/engine/test/"
    assert (
        expected_path == engine_controller.PATH
    ), "Controller path was not set correctly"


@pytest.mark.integration
def test_command_generation(engine_controller):
    """Test the generation of commands."""
    command_name = "test_command"
    mock_execute_func = Mock(return_value=(Mock(), None))
    engine_controller.translators[command_name].execute_func = mock_execute_func

    engine_controller._generate_command_call(
        name=command_name, translator=engine_controller.translators[command_name]
    )
    command_method_name = f"call_{command_name}"
    assert hasattr(
        engine_controller, command_method_name
    ), "Command method was not created"


@patch(
    "shnifter_interface.controllers.base_engine_controller.EngineController._link_shnifterject_to_data_processing_commands"
)
@patch(
    "shnifter_interface.controllers.base_engine_controller.EngineController._generate_commands"
)
@patch(
    "shnifter_interface.controllers.base_engine_controller.EngineController._generate_sub_controllers"
)
@pytest.mark.integration
def test_engine_controller_calls(
    mock_sub_controllers, mock_commands, mock_link_commands
):
    """Test the calls of the engine controller."""
    translators = {"test_command": Mock()}
    translators["test_command"].parser = Mock()
    translators["test_command"].execute_func = Mock()
    _ = EngineController(
        name="test", parent_path=["engine"], translators=translators
    )
    mock_sub_controllers.assert_called_once()
    mock_commands.assert_called_once()
    mock_link_commands.assert_called_once()
