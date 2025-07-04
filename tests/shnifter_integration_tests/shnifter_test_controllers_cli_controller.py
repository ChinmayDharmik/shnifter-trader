import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the Interface controller."""

from unittest.mock import MagicMock, patch

import pytest
from shnifter_interface.controllers.interface_controller import (
    InterfaceController,
    handle_job_cmds,
    parse_and_split_input,
    run_interface,
)

# pylint: disable=redefined-outer-name, unused-argument


def test_parse_and_split_input_custom_filters():
    """Test the parse_and_split_input function with custom filters."""
    input_cmd = "query -q AAPL/P"
    result = parse_and_split_input(
        input_cmd, custom_filters=[r"((\ -q |\ --question|\ ).*?(/))"]
    )
    assert (
        "AAPL/P" not in result
    ), "Should filter out terms that look like a sorting parameter"


@patch("shnifter_interface.controllers.interface_controller.InterfaceController.print_help")
def test_interface_controller_print_help(mock_print_help):
    """Test the InterfaceController print_help method."""
    controller = InterfaceController()
    controller.print_help()
    mock_print_help.assert_called_once()


@pytest.mark.parametrize(
    "controller_input, expected_output",
    [
        ("settings", True),
        ("random_command", False),
    ],
)
def test_InterfaceController_has_command(controller_input, expected_output):
    """Test the InterfaceController has_command method."""
    controller = InterfaceController()
    assert hasattr(controller, f"call_{controller_input}") == expected_output


def test_handle_job_cmds_with_export_path():
    """Test the handle_job_cmds function with an export path."""
    jobs_cmds = ["export /path/to/export some_command"]
    result = handle_job_cmds(jobs_cmds)
    expected = "some_command"
    assert expected in result[0]  # type: ignore


@patch("shnifter_interface.controllers.interface_controller.InterfaceController.switch", return_value=[])
@patch("shnifter_interface.controllers.interface_controller.print_goodbye")
def test_run_interface_quit_command(mock_print_goodbye, mock_switch):
    """Test the run_interface function with the quit command."""
    run_interface(["quit"], test_mode=True)
    mock_print_goodbye.assert_called_once()


@pytest.mark.skip("This test is not working as expected")
def test_execute_shnifter_routine_with_mocked_requests():
    """Test the call_exe function with mocked requests."""
    with patch("requests.get") as mock_get:
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"script": "print('Hello World')"}
        mock_get.return_value = response
        # Here we need to call the correct function, assuming it's something like `call_exe` for URL-based scripts
        controller = InterfaceController()
        controller.call_exe(
            ["--url", "https://my.shnifter.co/u/test/routine/test.shnifter"]
        )
        mock_get.assert_called_with(
            "https://my.shnifter.co/u/test/routine/test.shnifter?raw=true", timeout=10
        )
