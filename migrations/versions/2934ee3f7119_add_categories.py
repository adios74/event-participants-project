"""Add categories

Revision ID: 2934ee3f7119
Revises: 7b7e8cdd1d4c
Create Date: 2025-05-15 01:21:12.805960

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2934ee3f7119'
down_revision = '7b7e8cdd1d4c'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('event', schema=None) as batch_op:
        batch_op.add_column(sa.Column('category', sa.String(length=50), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('event', schema=None) as batch_op:
        batch_op.drop_column('category')

    # ### end Alembic commands ###
