"""Add start_idx and end_idx to chunks table for character offset tracking (D-05).

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-04
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("chunks", sa.Column("start_idx", sa.Integer(), nullable=True))
    op.add_column("chunks", sa.Column("end_idx", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("chunks", "end_idx")
    op.drop_column("chunks", "start_idx")
