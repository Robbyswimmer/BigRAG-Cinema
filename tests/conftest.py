"""Shared test fixtures for BigRAG Cinema tests."""

import os

import pytest


@pytest.fixture
def test_data_dir():
    """Return the absolute path to the test_data directory."""
    return os.path.join(os.path.dirname(__file__), "test_data")
