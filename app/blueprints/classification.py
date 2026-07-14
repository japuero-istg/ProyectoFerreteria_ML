import time
import threading
from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from flask_login import login_required, current_user

from app.models import db
from app.models.ml_result import MLResult
from app.models.models_poolCL import (
    ejecutar_entrenamiento_clasificacion,
    predecir_interactivo_clasificacion,
    obtener_todas_las_metricas_modulo_cl,
    marcar_entrenamiento_iniciado,
    marcar_entrenamiento_finalizado,
    esta_entrenando,
    ALGORITMOS_VALIDOS_CL,
    NOMBRES_MODULO_CL,
)
from app.services.data_loader import check_processed_datasets_exist, TIENDAS_INFO

# DECISIÓN DE RUTAS: el Blueprint NO define url_prefix propio — se registra
# sin prefijo aquí a propósito. El prefijo real ("/clasificacion") lo asigna
# app.py en el momento del registro. Ver notas de versiones anteriores.
#
# NOTA (v7.0/v9.0): el Módulo 6 (Tipo de Cliente) fue retirado del alcance
# del proyecto. Este blueprint ahora solo sirve el Módulo 3.
classification_bp = Blueprint("classification", __name__)


# =====================================================================
# GUARDIA DE INTEGRIDAD DE DATOS (FASE 3 -> FASE 4)
# =====================================================================
@classification_bp.before_request
def _bloquear_si_datos_no_listos():
    if not current_user.is_authenticated:
        return  # @login_required se encargará de redirigir al login

    if not check_processed_datasets_exist():
        if current_user.is_admin:
            flash(
                "⚠️ Los datos de la Fase 3 (Paso 2: Preprocesamiento) no han sido "
                "generados. Debes completarlos en el Panel Admin antes de usar "
                "el módulo de Clasificación.",
                "warning",
            )
            return redirect(url_for("data_manager.admin_panel"))

        flash(
            "⚠️ El módulo de Clasificación aún no está disponible. El administrador "
            "debe completar la preparación de datos (Fase 3) primero.",
            "warning",
        )
        return redirect(url_for("dashboard.index"))


# =====================================================================
# RUTAS (solo Módulo 3 — Módulo 6 retirado del alcance, ver v9.0)
# =====================================================================
@classification_bp.route("/")
@login_required
def index():
    return redirect(url_for("classification.dashboard", modulo_id=3))


@classification_bp.route("/dashboard/<int:modulo_id>", methods=["GET"])
@login_required
def dashboard(modulo_id):
    if modulo_id != 3:
        flash("Módulo inválido para la suite de clasificación.", "danger")
        return redirect(url_for("classification.dashboard", modulo_id=3))

    todas_metricas = obtener_todas_las_metricas_modulo_cl(modulo_id)

    # Opción B (entrenamiento asíncrono): algoritmos con un .flag activo se
    # muestran "Entrenando…" en la tabla comparativa en vez de bloquear la
    # página completa esperando la respuesta HTTP del POST /train.
    algoritmos_en_progreso = [
        alg for alg in ALGORITMOS_VALIDOS_CL[modulo_id] if esta_entrenando(modulo_id, alg)
    ]

    return render_template(
        "classification/categoria.html",
        modulo_id=modulo_id,
        nombre_modulo=NOMBRES_MODULO_CL[modulo_id],
        todas_metricas=todas_metricas,
        algoritmos_validos=ALGORITMOS_VALIDOS_CL[modulo_id],
        tiendas_info=TIENDAS_INFO,
        algoritmos_en_progreso=algoritmos_en_progreso,
    )


@classification_bp.route("/train/<int:modulo_id>", methods=["POST"])
@login_required
def entrenar_modelo(modulo_id):
    if modulo_id != 3:
        flash("Módulo inválido para la pipeline de entrenamiento.", "danger")
        return redirect(url_for("classification.dashboard", modulo_id=3))

    algoritmo = request.form.get("algoritmo", ALGORITMOS_VALIDOS_CL[modulo_id][0])
    if algoritmo not in ALGORITMOS_VALIDOS_CL[modulo_id]:
        flash(f"⚠️ Algoritmo '{algoritmo}' no válido para el Módulo {modulo_id}.", "danger")
        return redirect(url_for("classification.dashboard", modulo_id=modulo_id))

    if esta_entrenando(modulo_id, algoritmo):
        flash(
            f"⏳ {algoritmo} ya se está entrenando en segundo plano — espera a que termine "
            f"antes de lanzarlo de nuevo.",
            "warning",
        )
        return redirect(url_for("classification.dashboard", modulo_id=modulo_id))

    n_estimators = int(request.form.get("n_estimators", 100))
    try:
        test_size = float(request.form.get("test_size", 0.20))
    except (ValueError, TypeError):
        test_size = 0.20
    params = {"n_estimators": n_estimators, "test_size": test_size}

    # ── Opción B: ENTRENAMIENTO ASÍNCRONO ──────────────────────────────────
    # Fix del error 524 (Cloudflare) / timeouts de proxy: en vez de bloquear
    # esta petición HTTP hasta que termine el pipeline completo (que con
    # 231,772 filas y algoritmos secuenciales como GBC puede tardar más que
    # el timeout del proxy), se marca el flag de "entrenando", se lanza el
    # trabajo real en un hilo daemon en segundo plano, y esta ruta responde
    # de inmediato con un redirect. El resultado se recoge al refrescar el
    # dashboard más tarde (o automáticamente, ver auto-refresh en el template).
    #
    # Nota técnica: gunicorn corre con workers "sync" (un request a la vez
    # por proceso) — lanzar un hilo Python dentro del request handler y
    # retornar de inmediato es seguro: el worker queda libre para atender
    # nuevas peticiones en cuanto esta vista retorna, mientras el hilo sigue
    # entrenando en segundo plano dentro del mismo proceso.
    usuario_id = current_user.id
    app_obj = current_app._get_current_object()

    marcar_entrenamiento_iniciado(modulo_id, algoritmo)
    hilo = threading.Thread(
        target=_entrenar_en_segundo_plano,
        args=(app_obj, modulo_id, algoritmo, params, usuario_id),
        daemon=True,
    )
    hilo.start()

    flash(
        f"🕒 Entrenamiento de {algoritmo} iniciado en segundo plano "
        f"(dataset grande — puede tardar varios minutos). Esta página se actualiza sola "
        f"cuando termina; también puedes navegar a otro módulo y volver.",
        "info",
    )
    return redirect(url_for("classification.dashboard", modulo_id=modulo_id))


def _entrenar_en_segundo_plano(app_obj, modulo_id, algoritmo, params, usuario_id):
    """
    Ejecuta el pipeline de entrenamiento real FUERA del ciclo request/response
    de Flask (ver comentario en entrenar_modelo() de más arriba). Corre en un
    hilo daemon separado del hilo que atendió el POST original — por eso
    necesita su propio app_context() para poder usar `db.session` de forma
    segura (Flask-SQLAlchemy liga la sesión al contexto de aplicación activo
    en el hilo actual, no al hilo que originó el request HTTP).
    """
    with app_obj.app_context():
        try:
            marcador_inicio = time.time()
            metricas_resultado = ejecutar_entrenamiento_clasificacion(modulo_id, algoritmo, params)
            duracion = time.time() - marcador_inicio

            historial_registro = MLResult(
                usuario_id=usuario_id,
                modulo=f"clasificacion_m{modulo_id}",
                algoritmo=algoritmo,
                metricas=metricas_resultado,
                n_registros=metricas_resultado.get("samples_entrenamiento", 0),
                duracion_seg=round(duracion, 2),
            )
            db.session.add(historial_registro)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            # No hay usuario esperando esta excepción en vivo (el request ya
            # respondió hace rato) — se deja constancia en el log del
            # contenedor para poder diagnosticar fallos de entrenamiento
            # asíncrono sin depender de que el usuario reporte el error.
            print(f"⚠️ [ENTRENAMIENTO EN SEGUNDO PLANO] Error entrenando {algoritmo} "
                  f"(Módulo {modulo_id}, usuario {usuario_id}): {e}")
        finally:
            marcar_entrenamiento_finalizado(modulo_id, algoritmo)


@classification_bp.route("/simulate/<int:modulo_id>", methods=["POST"])
@login_required
def simular_prediccion(modulo_id):
    if modulo_id != 3:
        return jsonify({"status": "error", "message": "ID de módulo no válido"}), 400

    try:
        inputs = request.json
        if not inputs:
            return jsonify({"status": "error", "message": "No se recibieron parámetros"}), 400

        algoritmo = inputs.pop("algoritmo_inferencia", ALGORITMOS_VALIDOS_CL[modulo_id][0])
        if algoritmo not in ALGORITMOS_VALIDOS_CL[modulo_id]:
            return jsonify({"status": "error", "message": f"Algoritmo '{algoritmo}' no reconocido para este módulo"}), 400

        resultado = predecir_interactivo_clasificacion(modulo_id, inputs, algoritmo)
        return jsonify({"status": "success", **resultado, "algoritmo": algoritmo})

    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Fallo en inferencia: {str(e)}"}), 500
