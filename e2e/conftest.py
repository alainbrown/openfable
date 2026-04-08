"""Shared fixtures for e2e tests."""

import os

import psycopg2
import pytest

DB_URL = os.environ.get(
    "OPENFABLE_DATABASE_URL",
    "postgresql://openfable:openfable@db:5432/openfable",
)


@pytest.fixture()
def clean_db():
    """Truncate all application tables for a clean test run."""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("TRUNCATE nodes, chunks, documents CASCADE")
    conn.close()
