import unittest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime

"""Integration tests for the core_service module."""

from unittest.mock import create_autospec, patch

import pytest
import requests
from shnifter_interface.controllers.core_service import upload_routine
from shnifter_core.app.model.core.core_session import CoreSession

# pylint: disable=unused-argument, redefined-outer-name, unused-variable


@pytest.fixture
def auth_header():
    """Return a fake auth header."""
    return "Bearer fake_token"


@pytest.fixture
def core_session_mock():
    """Return a mock CoreSession."""
    mock = create_autospec(CoreSession, instance=True)
    mock.username = "TestUser"
    return mock


# Fixture for routine data
@pytest.fixture
def routine_data():
    """Return a dictionary with routine data."""
    return {
        "name": "Test Routine",
        "description": "A test routine",
        "routine": "print('Hello World')",
        "override": False,
        "tags": "test",
        "public": True,
    }


@pytest.mark.integration
def test_upload_routine_timeout(auth_header, routine_data):
    """Test upload_routine with a timeout exception."""
    with patch(
        "requests.post", side_effect=requests.exceptions.Timeout
    ) as mocked_post:  # noqa: F841

        response = upload_routine(auth_header, **routine_data)

        assert response is None


@pytest.mark.integration
def test_upload_routine_connection_error(auth_header, routine_data):
    """Test upload_routine with a connection error."""
    with patch(
        "requests.post", side_effect=requests.exceptions.ConnectionError
    ) as mocked_post:  # noqa: F841

        response = upload_routine(auth_header, **routine_data)

        assert response is None
