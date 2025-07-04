import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Test the Interface module."""

from unittest.mock import patch

from shnifter_interface.interface import main


@patch("shnifter_interface.config.setup.bootstrap")
@patch("shnifter_interface.controllers.interface_controller.launch")
@patch("sys.argv", ["shnifter", "--dev", "--debug"])
def test_main_with_dev_and_debug(mock_launch, mock_bootstrap):
    """Test the main function with dev and debug flags."""
    main()
    mock_bootstrap.assert_called_once()
    mock_launch.assert_called_once_with(True, True)


@patch("shnifter_interface.config.setup.bootstrap")
@patch("shnifter_interface.controllers.interface_controller.launch")
@patch("sys.argv", ["shnifter"])
def test_main_without_arguments(mock_launch, mock_bootstrap):
    """Test the main function without arguments."""
    main()
    mock_bootstrap.assert_called_once()
    mock_launch.assert_called_once_with(False, False)


@patch("shnifter_interface.config.setup.bootstrap")
@patch("shnifter_interface.controllers.interface_controller.launch")
@patch("sys.argv", ["shnifter", "--dev"])
def test_main_with_dev_only(mock_launch, mock_bootstrap):
    """Test the main function with dev flag only."""
    main()
    mock_bootstrap.assert_called_once()
    mock_launch.assert_called_once_with(True, False)


@patch("shnifter_interface.config.setup.bootstrap")
@patch("shnifter_interface.controllers.interface_controller.launch")
@patch("sys.argv", ["shnifter", "--debug"])
def test_main_with_debug_only(mock_launch, mock_bootstrap):
    """Test the main function with debug flag only."""
    main()
    mock_bootstrap.assert_called_once()
    mock_launch.assert_called_once_with(False, True)
