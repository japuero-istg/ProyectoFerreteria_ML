from flask import Blueprint, render_template, redirect, url_for, flash
from flask_login import login_required, current_user
from app.services.data_loader import check_processed_datasets_exist  # <--- GUARDIA FASE 3

clustering_bp = Blueprint("clustering", __name__)


# =====================================================================
# GUARDIA DE INTEGRIDAD DE DATOS (FASE 3 -> FASE 5)
# =====================================================================
# Mismo patrón que prediction.py: bloquea TODAS las rutas de este blueprint
# si el Paso 2 (preprocesamiento) de la Fase 3 no se ha completado.
# Se deja integrado desde ya para que, al programar las vistas reales de
# Clustering, el bloqueo ya esté garantizado sin tener que recordarlo
# ruta por ruta.
@clustering_bp.before_request
def _bloquear_si_datos_no_listos():
    if not current_user.is_authenticated:
        return  # @login_required se encargará de redirigir al login

    if not check_processed_datasets_exist():
        if current_user.is_admin:
            flash(
                "⚠️ Los datos de la Fase 3 (Paso 2: Preprocesamiento) no han sido "
                "generados. Debes completarlos en el Panel Admin antes de usar "
                "el módulo de Clustering.",
                "warning",
            )
            return redirect(url_for("data_manager.admin_panel"))

        flash(
            "⚠️ El módulo de Clustering aún no está disponible. El administrador "
            "debe completar la preparación de datos (Fase 3) primero.",
            "warning",
        )
        return redirect(url_for("dashboard.index"))


@clustering_bp.route("/clustering")
@login_required
def index():
    # TODO (Fase 5): reemplazar por render_template real cuando se programe el módulo.
    return "Módulo de Clustering - En construcción (Fase 5)"
