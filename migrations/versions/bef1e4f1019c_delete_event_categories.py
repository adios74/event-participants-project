"""delete event categories

Revision ID: bef1e4f1019c
Revises: bae9399c7a57
Create Date: 2025-05-16 04:15:05.385245

"""
from alembic import op
import sqlalchemy as sa



from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = 'bef1e4f1019c'
down_revision = 'bae9399c7a57'
branch_labels = None
depends_on = None


def upgrade():
    # Сначала проверяем существование таблицы event_category
    conn = op.get_bind()
    inspector = inspect(conn)

    if 'event_category' in inspector.get_table_names():
        # Проверяем существование столбца category_id в event
        if 'category_id' in [col['name'] for col in inspector.get_columns('event')]:
            with op.batch_alter_table('event') as batch_op:
                # Пытаемся удалить foreign key, если он существует
                for fk in inspector.get_foreign_keys('event'):
                    if fk['referred_table'] == 'event_category':
                        batch_op.drop_constraint(fk['name'], type_='foreignkey')
                        break
                batch_op.drop_column('category_id')

        # Удаляем таблицу категорий
        op.drop_table('event_category')


def downgrade():
    # Создаем таблицу категорий заново
    op.create_table('event_category',
                    sa.Column('id', sa.Integer(), nullable=False),
                    sa.Column('name', sa.String(length=50), nullable=True),
                    sa.PrimaryKeyConstraint('id')
                    )

    # Добавляем столбец и связь
    with op.batch_alter_table('event') as batch_op:
        batch_op.add_column(sa.Column('category_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            'fk_event_category',
            'event_category',
            ['category_id'],
            ['id']
        )