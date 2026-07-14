from flask import Blueprint, render_template, redirect, url_for, flash, request, abort
from flask_login import login_required, current_user
import pandas as pd
import redis
import json
import os       # <--- MEJORA DE CONTROL: Requerido para operaciones atómicas de borrado físico sobre el disco duro
import glob     # <--- MEJORA DE CONTROL: Requerido para el mapeo dinámico por patrones de archivos (*.pkl, *.json)
from app.services.data_loader import (
    generate_and_save_datasets, 
    preprocesar_datasets_maestros,
    check_raw_datasets_exist,
    check_processed_datasets_exist,
    load_current_datasets,
    FILE_TRANSACCIONAL, FILE_SEMANAL,
    FILE_TRANSACCIONAL_PROC, FILE_SEMANAL_PROC
)

data_manager_bp = Blueprint('data_manager', __name__, url_prefix='/data')

# =====================================================================
# CONFIGURACIÓN E INICIALIZACIÓN DE REDIS CON EL PREFIJO REQUERIDO
# =====================================================================
# Usamos 'redis' como host ya que es el nombre del servicio en la red de Docker.
# Ajusta el puerto (6379 o el expuesto 6373) según la configuración interna del contenedor.
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
REDIS_KEY_PREFIX = "ml_proyecto2p_v1"
CACHE_TTL = 900  # Tiempo de vida de la caché: 15 minutos

def obtener_de_cache(clave_interna):
    """Recupera datos serializados desde Redis utilizando el prefijo institucional."""
    clave_maestra = f"{REDIS_KEY_PREFIX}:{clave_interna}"
    try:
        datos = redis_client.get(clave_maestra)
        if datos:
            print(f"⚡ [REDIS CACHE HIT] Recuperada exitosamente: {clave_maestra}")
            return json.loads(datos)
    except Exception as e:
        print(f"⚠️ [REDIS ERROR] Fallo al leer de la caché: {str(e)}")
    return None

def guardar_en_cache(clave_interna, valor_dict):
    """Persiste datos en la RAM de Redis formateados en JSON estructurado."""
    clave_maestra = f"{REDIS_KEY_PREFIX}:{clave_interna}"
    try:
        redis_client.setex(
            name=clave_maestra,
            time=CACHE_TTL,
            value=json.dumps(valor_dict)
        )
        print(f"💾 [REDIS CACHE SET] Datos indexados bajo la clave: {clave_maestra}")
    except Exception as e:
        print(f"⚠️ [REDIS ERROR] Fallo al escribir en la caché: {str(e)}")

def invalidar_cache_completa():
    """Borra todas las llaves asociadas al prefijo cuando se regeneran o preprocesan los datos."""
    try:
        patron = f"{REDIS_KEY_PREFIX}:*"
        llaves = redis_client.keys(patron)
        if llaves:
            redis_client.delete(*llaves)
            print(f"🧹 [REDIS CACHE FLUSH] Se eliminaron {len(llaves)} llaves de caché obsoletas.")
    except Exception as e:
        print(f"⚠️ [REDIS ERROR] Fallo al limpiar la caché: {str(e)}")


# =====================================================================
# SECCIÓN INYECTADA: PURGA COMPLETA DE ARTEFACTOS ANALÍTICOS (.PKL Y .JSON)
# =====================================================================
def limpiar_artefactos_regresion_viejos():
    """
    Busca de forma exhaustiva y remueve físicamente del almacenamiento tanto los 
    modelos binarios (.pkl) como sus metadatos de métricas e historial asociados (.json). 
    Al gatillarse junto a una nueva simulación, obliga a la Suite de Regresión a 
    reiniciar su estado visual y operativo, mitigando el Data Drift de manera automatizada.
    """
    # Mapeo estricto basado en la estructura real verificada en el entorno Docker del servidor
    patrones = [
        "app/static/data/*.pkl",   # Ruta real de artefactos en el contenedor (v6.0)
        "app/static/data/*.json",  # Métricas y gráficos base64 en la ruta real (v6.0)
        "app/models/data/*.pkl",        # Resguardo: versiones anteriores al cambio de ruta
        "app/models/data/*.json",       # Resguardo: métricas de versiones anteriores
        "models/*.pkl",            # Resguardo por ejecuciones de consola
        "models/*.json",           # Resguardo metadatos de consola
        "*.pkl",                   # Red de seguridad en la raíz del espacio de trabajo
    ]
    
    archivos_encontrados = []
    for patron in patrones:
        archivos_encontrados.extend(glob.glob(patron))
        
    for archivo in archivos_encontrados:
        # Exclusión de protección estricta para evitar remover archivos de configuración del proyecto
        if "package.json" in archivo or "composer.json" in archivo or "manifest.json" in archivo:
            continue
        try:
            os.remove(archivo)
            print(f"🧹 [FACTORY RESET] Purgado archivo obsoleto del servidor: {archivo}")
        except OSError as e:
            print(f"⚠️ [FILE REMOVE ERROR] No se pudo eliminar el artefacto {archivo}: {str(e)}")


# =====================================================================
# CONTROLADOR DEL EXPLORADOR CON ARQUITECTURA DE CACHÉ INTERNA
# =====================================================================
@data_manager_bp.route('/explorer', methods=['GET'])
@login_required
def explorer():
    """VISTA PÚBLICA DE AUDITORÍA: Permite ver tablas brutas vs pulidas con aceleración Redis."""
    raw_ok = check_raw_datasets_exist()
    proc_ok = check_processed_datasets_exist()
    
    active_tab = request.args.get('tab', 'raw')
    active_subtab = request.args.get('subtab', 'tx')
    
    # Parámetros de paginación individuales
    page_tx = request.args.get('page_tx', 1, type=int)
    page_sem = request.args.get('page_sem', 1, type=int)
    page_tx_p = request.args.get('page_tx_p', 1, type=int)
    page_sem_p = request.args.get('page_sem_p', 1, type=int)
    
    per_page = 12
    slices = {"tx": None, "sem": None, "tx_p": None, "sem_p": None}
    stats = {}

    # Generamos una llave de caché única que contenga la pestaña, subpestaña y todas las páginas activas
    cache_key = f"explorer:tab_{active_tab}:sub_{active_subtab}:p_tx_{page_tx}:p_sem_{page_sem}:p_txp_{page_tx_p}:p_semp_{page_sem_p}"
    
    # Intento 1: Buscar si la vista completa ya fue procesada y guardada en la RAM de Redis
    datos_cache = obtener_de_cache(cache_key)
    if datos_cache:
        return render_template(
            'data_manager/explorer.html',
            raw_initialized=raw_ok,
            proc_initialized=proc_ok,
            active_tab=active_tab,
            active_subtab=active_subtab,
            df_tx=datos_cache.get("df_tx"),
            df_sem=datos_cache.get("df_sem"),
            df_tx_p=datos_cache.get("df_tx_p"),
            df_sem_p=datos_cache.get("df_sem_p"),
            stats=datos_cache.get("stats"),
            p_tx=page_tx, p_sem=page_sem,
            p_tx_p=page_tx_p, p_sem_p=page_sem_p
        )

    # [REDIS CACHE MISS] Si no está en memoria, realizamos la lectura física normal desde el disco duro
    print("💾 [REDIS CACHE MISS] Leyendo matrices desde el sistema de archivos local de la VM...")
    
    if raw_ok:
        try:
            df_tx, df_sem = load_current_datasets()
            if df_tx is not None and df_sem is not None:
                stats.update({
                    "tx_rows": f"{len(df_tx):,}",
                    "sem_rows": f"{len(df_sem):,}",
                    "ingresos": f"${df_tx['total_venta'].sum():,.2f}",
                    "tx_max": (len(df_tx) // per_page) + 1,
                    "sem_max": (len(df_sem) // per_page) + 1
                })
                slices["tx"] = df_tx.iloc[(page_tx-1)*per_page : page_tx*per_page].to_dict(orient='records')
                slices["sem"] = df_sem.iloc[(page_sem-1)*per_page : page_sem*per_page].to_dict(orient='records')
        except Exception:
            raw_ok = False

    if proc_ok:
        try:
            df_tx_p = pd.read_csv(FILE_TRANSACCIONAL_PROC)
            df_sem_p = pd.read_csv(FILE_SEMANAL_PROC)
            
            stats.update({
                "tx_p_rows": f"{len(df_tx_p):,}",
                "sem_p_rows": f"{len(df_sem_p):,}",
                "tx_p_cols": len(df_tx_p.columns),
                "sem_p_cols": len(df_sem_p.columns),
                "tx_p_max": (len(df_tx_p) // per_page) + 1,
                "sem_p_max": (len(df_sem_p) // per_page) + 1
            })
            slices["tx_p"] = df_tx_p.iloc[(page_tx_p-1)*per_page : page_tx_p*per_page].to_dict(orient='records')
            slices["sem_p"] = df_sem_p.iloc[(page_sem_p-1)*per_page : page_sem_p*per_page].to_dict(orient='records')
        except Exception:
            proc_ok = False

    # Empaquetamos los resultados calculados
    payload_a_guardar = {
        "df_tx": slices["tx"],
        "df_sem": slices["sem"],
        "df_tx_p": slices["tx_p"],
        "df_sem_p": slices["sem_p"],
        "stats": stats
    }
    
    # Almacenamos el payload estructurado en Redis para acelerar las futuras consultas
    guardar_en_cache(cache_key, payload_a_guardar)

    return render_template(
        'data_manager/explorer.html',
        raw_initialized=raw_ok,
        proc_initialized=proc_ok,
        active_tab=active_tab,
        active_subtab=active_subtab,
        df_tx=slices["tx"],
        df_sem=slices["sem"],
        df_tx_p=slices["tx_p"],
        df_sem_p=slices["sem_p"],
        stats=stats,
        p_tx=page_tx, p_sem=page_sem,
        p_tx_p=page_tx_p, p_sem_p=page_sem_p
    )


@data_manager_bp.route('/admin-panel', methods=['GET'])
@login_required
def admin_panel():
    if not current_user.is_admin:
        abort(403)
    return render_template(
        'data_manager/admin_panel.html', 
        raw_initialized=check_raw_datasets_exist(), 
        proc_initialized=check_processed_datasets_exist()
    )


@data_manager_bp.route('/admin-panel/generate', methods=['POST'])
@login_required
def generate():
    if not current_user.is_admin:
        abort(403)
    try:
        # ─── ACCIÓN AUTOMÁTICA INCORPORADA Y REFORZADA ──────────────────────
        # Antes de sobreescribir los archivos de datos masivos .csv, limpiamos por completo
        # tanto los estimadores .pkl antiguos como los archivos de métricas .json para 
        # forzar un factory reset absoluto en el ecosistema analítico.
        limpiar_artefactos_regresion_viejos()
        # ────────────────────────────────────────────────────────────────────

        if generate_and_save_datasets():
            # Limpiamos la caché inmediatamente para asegurar consistencia de los nuevos datos
            invalidar_cache_completa()
            flash("✅ Paso 1 Completado: Se simuló el bloque comercial en bruto de 5 años. Artefactos anteriores purgados de 'app/models/' y caché de Redis sincronizada.", "success")
    except Exception as e:
        flash(f"Error crítico en simulación: {str(e)}", "danger")
    return redirect(url_for('data_manager.admin_panel'))


@data_manager_bp.route('/admin-panel/preprocess', methods=['POST'])
@login_required
def preprocess():
    if not current_user.is_admin:
        abort(403)
    try:
        if preprocesar_datasets_maestros():
            # Limpiamos la caché para que el explorador refleje las nuevas columnas e ingeniería de características
            invalidar_cache_completa()
            flash("🔄 Paso 2 Completado: Pipeline analítico aplicado de forma exitosa. Vectores sincronizados en memoria.", "info")
    except Exception as e:
        flash(f"Error crítico en Feature Engineering: {str(e)}", "danger")
    return redirect(url_for('data_manager.admin_panel'))
