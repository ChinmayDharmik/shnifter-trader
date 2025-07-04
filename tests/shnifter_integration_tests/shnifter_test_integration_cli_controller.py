import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the Interface controller integration."""

from shnifter_interface.controllers.interface_controller import (
    InterfaceController,
)


def test_parse_input_valid_commands():
    """Test parse_input method."""
    controller = InterfaceController()
    input_string = "exe --file test.shnifter"
    expected_output = [
        "exe --file test.shnifter"
    ]  # Adjust based on actual expected behavior
    assert controller.parse_input(input_string) == expected_output


def test_parse_input_invalid_commands():
    """Test parse_input method."""
    controller = InterfaceController()
    input_string = "nonexistentcommand args"
    expected_output = ["nonexistentcommand args"]
    actual_output = controller.parse_input(input_string)
    assert (
        actual_output == expected_output
    ), f"Expected {expected_output}, got {actual_output}"
