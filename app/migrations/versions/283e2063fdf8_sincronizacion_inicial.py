"""sincronizacion inicial

Revision ID: 283e2063fdf8
Revises:
Create Date: 2026-06-22 09:45:05.568694

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '283e2063fdf8'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'usuarios',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('username', sa.String(length=80), nullable=False),
        sa.Column('email', sa.String(length=120), nullable=False),
        sa.Column('password', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_admin', sa.Boolean(), nullable=False),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('email'),
    )
    op.create_table(
        'historial_analisis',
        sa.Column('id', sa.Integer(), primary_key=True, nullable=False),
        sa.Column('usuario_id', sa.Integer(), nullable=False),
        sa.Column('modulo', sa.String(length=20), nullable=False),
        sa.Column('algoritmo', sa.String(length=50), nullable=False),
        sa.Column('metricas', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('n_registros', sa.Integer(), nullable=False),
        sa.Column('ejecutado_en', sa.DateTime(), nullable=True),
        sa.Column('duracion_seg', sa.Numeric(precision=6, scale=2), nullable=True),
        sa.ForeignKeyConstraint(['usuario_id'], ['usuarios.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_historial_analisis_usuario_id', 'historial_analisis', ['usuario_id'])


def downgrade():
    op.drop_table('historial_analisis')
    op.drop_table('usuarios')
