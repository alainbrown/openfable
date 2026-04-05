"""Add content column to documents table.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("content", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("documents", "content")
