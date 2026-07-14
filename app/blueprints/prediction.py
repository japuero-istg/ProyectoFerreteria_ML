import time
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from flask_login import login_required, current_user
import os

from app.models import db
from app.models.ml_result import MLResult
from app.models.models_pool import (
    proyectar_demanda_semanal,
    ejecutar_entrenamiento_regresion,
    predecir_interactivo,
    obtener_todas_las_metricas_modulo,
    ALGORITMOS_VALIDOS,
)
from app.services.data_loader import check_processed_datasets_exist, TIENDAS_INFO

prediction_bp = Blueprint('prediction', __name__, url_prefix='/models')


# =====================================================================
# GUARDIA DE INTEGRIDAD DE DATOS (FASE 3 → FASE 4)
# =====================================================================
@prediction_bp.before_request
def _bloquear_si_datos_no_listos():
    if not current_user.is_authenticated:
        return
    if not check_processed_datasets_exist():
        if current_user.is_admin:
            flash(
                "⚠️ Los datos de la Fase 3 (Paso 2: Preprocesamiento) no han sido "
                "generados. Debes completarlos en el Panel Admin antes de usar "
                "la Suite de Regresión.",
                "warning",
            )
            return redirect(url_for("data_manager.admin_panel"))
        flash(
            "⚠️ El módulo de Regresión aún no está disponible. El administrador "
            "debe completar la preparación de datos (Fase 3) primero.",
            "warning",
        )
        return redirect(url_for("dashboard.index"))


# =====================================================================
# RUTAS
# =====================================================================
@prediction_bp.route('/prediction')
@login_required
def index():
    return redirect(url_for('prediction.dashboard', modulo_id=1))


@prediction_bp.route('/dashboard/<int:modulo_id>', methods=['GET'])
@login_required
def dashboard(modulo_id):
    if modulo_id not in [1, 2, 5]:
        flash("Módulo inválido para la suite de regresión.", "danger")
        return redirect(url_for('prediction.dashboard', modulo_id=1))

    _templates = {
        1: ('prediction/transaccional.html', "Módulo 1: Unidades por Transacción (Grano Fino - Combinación #1)"),
        2: ('prediction/demanda.html',       "Módulo 2: Demanda Semanal por Categoría (Grano Agregado - Combinación #2)"),
        5: ('prediction/margen.html',        "Módulo 5: Margen Unitario de Ganancia (Rentabilidad - Combinación #5)"),
    }
    plantilla, nombre_modulo = _templates[modulo_id]

    # Carga TODAS las métricas disponibles para este módulo (dict algoritmo→metricas)
    todas_metricas = obtener_todas_las_metricas_modulo(modulo_id)

    return render_template(
        plantilla,
        modulo_id=modulo_id,
        nombre_modulo=nombre_modulo,
        todas_metricas=todas_metricas,          # dict {alg: {...}} para la tabla comparativa
        algoritmos_validos=ALGORITMOS_VALIDOS,  # lista para el selector del simulador
        tiendas_info=TIENDAS_INFO,               # dict {id: {ciudad, ...}} para mostrar ciudad en vez de "Tienda #"
    )


@prediction_bp.route('/train/<int:modulo_id>', methods=['POST'])
@login_required
def entrenar_modelo(modulo_id):
    if modulo_id not in [1, 2, 5]:
        flash("Módulo inválido para la pipeline de entrenamiento.", "danger")
        return redirect(url_for('prediction.dashboard', modulo_id=1))

    try:
        marcador_inicio = time.time()
        algoritmo       = request.form.get("algoritmo", "LinearRegression")
        n_estimators    = int(request.form.get("n_estimators", 100))
        try:
            test_size = float(request.form.get("test_size", 0.20))
        except (ValueError, TypeError):
            test_size = 0.20

        params = {"n_estimators": n_estimators, "test_size": test_size}

        metricas_resultado = ejecutar_entrenamiento_regresion(modulo_id, algoritmo, params)
        duracion           = time.time() - marcador_inicio

        historial_registro = MLResult(
            usuario_id=current_user.id,
            modulo=f"regresion_m{modulo_id}",
            algoritmo=algoritmo,
            metricas=metricas_resultado,
            n_registros=metricas_resultado.get("samples_entrenamiento", 0),
            duracion_seg=round(duracion, 2),
        )
        db.session.add(historial_registro)
        db.session.commit()

        flash(
            f"🎯 ¡{algoritmo} entrenado con split {int(test_size*100)}% y guardado "
            f"como archivo independiente en app/models/!",
            "success",
        )

    except Exception as e:
        db.session.rollback()
        flash(f"⚠️ Error en la pipeline de entrenamiento: {str(e)}", "danger")

    return redirect(url_for('prediction.dashboard', modulo_id=modulo_id))


@prediction_bp.route('/simulate/<int:modulo_id>', methods=['POST'])
@login_required
def simular_prediccion(modulo_id):
    if modulo_id not in [1, 2, 5]:
        return jsonify({"status": "error", "message": "ID de módulo no válido"}), 400

    try:
        inputs = request.json
        if not inputs:
            return jsonify({"status": "error", "message": "No se recibieron parámetros"}), 400

        # El frontend envía qué algoritmo usar en la inferencia
        algoritmo = inputs.pop("algoritmo_inferencia", "LinearRegression")
        if algoritmo not in ALGORITMOS_VALIDOS:
            return jsonify({"status": "error", "message": f"Algoritmo '{algoritmo}' no reconocido"}), 400

        resultado = predecir_interactivo(modulo_id, inputs, algoritmo)
        return jsonify({"status": "success", "resultado": round(resultado, 4), "algoritmo": algoritmo})

    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Fallo en inferencia: {str(e)}"}), 500


@prediction_bp.route('/proyeccion/demanda', methods=['POST'])
@login_required
def proyeccion_demanda():
    """
    Endpoint AJAX que devuelve la proyección de demanda para las próximas
    6 semanas de una tienda+categoría dada, usando el algoritmo indicado del Módulo 2.
    Consumido desde demanda.html vía fetch().
    M2 REDISEÑADO: requiere categoria_id porque el modelo ahora predice a nivel
    tienda+categoría (no tienda+producto), lo que elevó el R² de 0.13 a 0.70.
    """
    try:
        data        = request.json or {}
        tienda_id   = int(data.get("tienda_id", 1))
        categoria_id = data.get("categoria_id", "Herramientas")
        algoritmo   = data.get("algoritmo", "RandomForestRegressor")
        n_semanas   = int(data.get("n_semanas", 6))

        if algoritmo not in ALGORITMOS_VALIDOS:
            return jsonify({"status": "error", "message": f"Algoritmo '{algoritmo}' no válido"}), 400

        resultado = proyectar_demanda_semanal(algoritmo, tienda_id, n_semanas, categoria_id)

        if resultado.get("error"):
            return jsonify({"status": "error", "message": resultado["error"]}), 400

        return jsonify({"status": "success", **resultado})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
