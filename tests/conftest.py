# tests/conftest.py
import os
import pytest
import sqlite3
import sys

# 1) Adjust sys.path to include the scripts folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts/database_management'))

from create_schema import create_schema

@pytest.fixture
def test_db_path(tmp_path):
    """
    Creates a temporary directory and database path for testing,
    runs create_schema, and yields the path.
    """
    db_dir = tmp_path / "database"
    db_dir.mkdir()
    db_path = db_dir / "test_experiments.db"

    # Create the schema in the test database
    create_schema(str(db_path))

    yield str(db_path)

    # Optionally clean up if desired
    # (tmp_path is auto-removed by pytest after test session)
