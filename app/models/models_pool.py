import os
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Indispensable para entornos Docker / hilos web
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.abspath(os.path.join(BASE_MODELS_DIR, "..", "static", "data"))

FILE_TRANSACCIONAL_PROC = os.path.join(DIR_DATA, "dataset_transacciones_processed.csv")
FILE_SEMANAL_PROC       = os.path.join(DIR_DATA, "dataset_demanda_semanal_processed.csv")

#DIR_MODELS = BASE_MODELS_DIR
DIR_MODELS = os.path.join(BASE_MODELS_DIR, "data")
os.makedirs(DIR_MODELS, exist_ok=True)   # ← crea app/models/data/ si no existe

# Algoritmos soportados en la suite de regresión
ALGORITMOS_VALIDOS = ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"]


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE ARTEFACTOS  (por módulo + por algoritmo)
# ─────────────────────────────────────────────────────────────────────────────
def _ruta_pkl(modulo_id: int, algoritmo: str) -> str:
    return os.path.join(DIR_MODELS, f"modelo_modulo_{modulo_id}_{algoritmo}.pkl")

def _ruta_json(modulo_id: int, algoritmo: str) -> str:
    return os.path.join(DIR_MODELS, f"metricas_modulo_{modulo_id}_{algoritmo}.json")


def obtener_mapeo_variables(modulo_id):
    if modulo_id == 1:
        # AUD-04: costo_unidad y margen_unitario removidos (multicolinealidad exacta).
        features = [
            "año", "semana_iso", "dia_semana", "tienda_id", "factor_tamano",
            "antiguedad_tienda", "densidad_poblacional_zona_enc",
            "precio_unidad", "es_feriado", "promocion_activa"
        ]
        target    = "unidades_vendidas"
        file_path = FILE_TRANSACCIONAL_PROC

    elif modulo_id == 2:
        # M2 REDISEÑADO — Demanda Semanal por Categoría (Opción B).
        # Target: demanda_categoria (unidades agregadas por tienda+categoría+semana).
        # Split: por año (train 2021-2024, test 2025) en vez de shuffle=False global,
        # porque el shuffle=False sobre datos ordenados por tienda+categoría mezclaba
        # series distintas en el mismo corte temporal — el nuevo split es honesto.
        # R² validado empíricamente: 0.59-0.62 en test año 2025 vs 0.13-0.17 anterior.
        features = [
            "año", "semana_iso", "semana_sin", "semana_cos",
            "tienda_id", "factor_tamano", "antiguedad_tienda", "densidad_poblacional_zona_enc",
            "precio_mediano", "promedio_margen", "dias_con_promocion", "dias_con_feriado",
            "demanda_lag_1", "demanda_lag_2", "demanda_lag_4", "media_movil_4",
            "num_productos_distintos", "num_transacciones"
        ]
        target    = "demanda_categoria"
        file_path = FILE_SEMANAL_PROC

    elif modulo_id == 5:
        # costo_unidad excluido para evitar identidad trivial margen = precio - costo.
        # AUD-11: cat_* incluidas como OHE — la categoría de producto es la feature
        # más legítima para explicar diferencias de margen entre familias de producto
        # (herramientas vs materiales vs eléctrico, etc.), y su ausencia generaba
        # correlación espuria de escala entre precio_unidad y margen_unitario.
        features = [
            "año", "semana_iso", "dia_semana", "tienda_id", "factor_tamano",
            "antiguedad_tienda", "densidad_poblacional_zona_enc",
            "precio_unidad", "unidades_vendidas", "es_feriado", "promocion_activa"
        ]
        target    = "margen_unitario"
        file_path = FILE_TRANSACCIONAL_PROC

    else:
        raise ValueError("ID de módulo no soportado en la pipeline analítica de regresión.")

    return features, target, file_path


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO  (guarda artefactos POR ALGORITMO, no sobreescribe los demás)
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_entrenamiento_regresion(modulo_id, algoritmo_nombre, params=None):
    if params is None:
        params = {}

    n_estimators = int(params.get("n_estimators", 100))
    test_size    = float(params.get("test_size", 0.20))

    features_base, target, file_path = obtener_mapeo_variables(modulo_id)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos procesados en: {file_path}")

    df = pd.read_csv(file_path)

    # AUD-05: orden temporal antes del split para evitar data leakage futuro→pasado.
    df = df.sort_values(by=["año", "semana_iso"]).reset_index(drop=True)

    # Columnas OHE dinámicas: M2 usa reg_*+cat_*, M1 usa reg_+cli_*, M5 usa reg_+cli_+cat_*
    if modulo_id == 2:
        prefix_filter = ("reg_", "cat_")
    elif modulo_id == 5:
        prefix_filter = ("reg_", "cli_", "cat_")
    else:
        prefix_filter = ("reg_", "cli_")
    columnas_extra = [col for col in df.columns if col.startswith(prefix_filter)]
    X_cols         = [c for c in (features_base + columnas_extra) if c in df.columns]

    X = df[X_cols]
    y = df[target]

    # M2: split por año — test = 2025 (año completo más reciente), train = 2021-2024.
    # No se usa df["año"].max() porque el dataset incluye semanas parciales de 2026
    # (primeras semanas del año siguiente al último año completo simulado) que generarían
    # un test set incompleto con R² colapsado por distribución truncada.
    if modulo_id == 2:
        AÑO_TEST_M2 = 2025
        mask_test   = df["año"] == AÑO_TEST_M2
        X_train, X_test = X[~mask_test], X[mask_test]
        y_train, y_test = y[~mask_test], y[mask_test]
        test_size   = round(mask_test.sum() / len(df), 4)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

    # Instanciación del algoritmo
    if algoritmo_nombre in ["RandomForestRegressor", "RandomForest"]:
        modelo        = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        algoritmo_label = "RandomForestRegressor"
    elif algoritmo_nombre in ["GradientBoostingRegressor", "GradientBoosting"]:
        modelo        = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        algoritmo_label = "GradientBoostingRegressor"
    else:
        modelo        = LinearRegression()
        algoritmo_label = "LinearRegression"

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas estándar
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metricas_dicc = {
        "algoritmo":            algoritmo_label,
        "samples_entrenamiento": int(len(df)),
        "test_size_utilizado":  test_size,
        "r2":                   round(r2,   4),
        "mse":                  round(mse,  4),
        "mae":                  round(mae,  4),
        "rmse":                 round(rmse, 4),
        "cv_r2_folds":          None,
        "cv_r2_promedio":       None,
        "mape":                 None,
    }

    # ── Baseline "ingenuo" + Mejora % (criterio del profesor)
    # M1/M5: baseline = predecir siempre el promedio del target en train.
    # M2: baseline = demanda_lag_1 ("la semana pasada"), más honesto para
    #     series temporales que un promedio global constante.
    if modulo_id == 2 and "demanda_lag_1" in X_test.columns:
        baseline_pred = X_test["demanda_lag_1"].to_numpy()
    else:
        baseline_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))

    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    if baseline_mae > 0:
        mejora_vs_baseline = (baseline_mae - mae) / baseline_mae * 100
    else:
        mejora_vs_baseline = None

    metricas_dicc["baseline_mae"] = round(float(baseline_mae), 4)
    metricas_dicc["mejora_vs_baseline"] = (
        round(float(mejora_vs_baseline), 2) if mejora_vs_baseline is not None else None
    )

    # ── MAPE: Módulos 1 y 5 (valores continuos con denominador > 0)
    if modulo_id in [1, 5]:
        y_t_np = np.array(y_test)
        y_p_np = np.clip(np.array(y_pred), 0, None)
        mask   = y_t_np > 0
        if np.any(mask):
            mape = np.mean(np.abs((y_t_np[mask] - y_p_np[mask]) / y_t_np[mask])) * 100
            metricas_dicc["mape"] = round(mape, 2)
        else:
            metricas_dicc["mape"] = 0.0

    # ── Validación cruzada temporal: solo Módulo 2
    # Walk-forward por año: train en años 1..N, test en año N+1
    # Más honesto que TimeSeriesSplit global que mezclaba series de distintas tiendas-categorías
    if modulo_id == 2:
        años_unicos = sorted(df["año"].unique())
        scores_wf   = []
        for i in range(1, len(años_unicos)):
            años_train = años_unicos[:i]
            año_test_wf = años_unicos[i]
            mask_tr = df["año"].isin(años_train)
            mask_te = df["año"] == año_test_wf
            if mask_te.sum() == 0:
                continue
            modelo.fit(X[mask_tr], y[mask_tr])
            s = r2_score(y[mask_te], modelo.predict(X[mask_te]))
            scores_wf.append(round(float(s), 4))
        metricas_dicc["cv_r2_folds"]    = scores_wf
        metricas_dicc["cv_r2_promedio"] = round(float(np.mean(scores_wf)), 4)

    # ── Gráficos persistidos en el JSON como base64 (mismo patrón que grafico_residuos)
    metricas_dicc["grafico_residuos"] = generar_grafico_residuos_base64(
        modulo_id, y_test, y_pred, modelo, X_test.columns
    )
    metricas_dicc["grafico_importancia"] = generar_grafico_importancia_base64(
        modelo, list(X_test.columns), algoritmo_label, modulo_id
    )
    metricas_dicc["grafico_real_vs_pred"] = generar_grafico_real_vs_pred_base64(
        y_test, y_pred, algoritmo_label, modulo_id
    )
    metricas_dicc["grafico_histograma_residuos"] = generar_grafico_histograma_residuos_base64(
        y_test, y_pred, algoritmo_label, modulo_id
    )
    metricas_dicc["grafico_curva_aprendizaje"] = generar_grafico_curva_aprendizaje_base64(
        modelo, X, y, algoritmo_label, modulo_id
    )

    # ── Persistencia POR ALGORITMO (no sobreescribe los demás algoritmos)
    with open(_ruta_pkl(modulo_id, algoritmo_label), "wb") as f:
        pickle.dump(modelo, f)

    with open(_ruta_json(modulo_id, algoritmo_label), "w", encoding="utf-8") as f:
        json.dump(metricas_dicc, f, indent=4, ensure_ascii=False)

    return metricas_dicc


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def obtener_metricas_locales(modulo_id, algoritmo):
    """Devuelve las métricas de UN algoritmo específico (o None si no existe)."""
    ruta = _ruta_json(modulo_id, algoritmo)
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def obtener_todas_las_metricas_modulo(modulo_id):
    """
    Devuelve un dict {algoritmo: metricas} con todos los algoritmos
    que ya han sido entrenados para el módulo indicado.
    El orden es fijo: Linear → RandomForest → GradientBoosting.
    """
    resultado = {}
    for alg in ALGORITMOS_VALIDOS:
        datos = obtener_metricas_locales(modulo_id, alg)
        if datos is not None:
            resultado[alg] = datos
    return resultado  # {} si ninguno ha sido entrenado todavía


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA INTERACTIVA
# ─────────────────────────────────────────────────────────────────────────────
def predecir_interactivo(modulo_id, inputs_dict, algoritmo):
    """Carga el .pkl del algoritmo indicado y devuelve la predicción escalar."""
    ruta = _ruta_pkl(modulo_id, algoritmo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"El algoritmo '{algoritmo}' del Módulo {modulo_id} "
            f"aún no ha sido entrenado. Ejecute primero su pipeline."
        )

    with open(ruta, "rb") as f:
        modelo = pickle.load(f)

    features, _, file_path = obtener_mapeo_variables(modulo_id)
    df_ejemplo = pd.read_csv(file_path, nrows=1)
    if modulo_id == 2:
        prefix_filter = ("reg_", "cat_")
    elif modulo_id == 5:
        prefix_filter = ("reg_", "cli_", "cat_")
    else:
        prefix_filter = ("reg_", "cli_")
    cols_extra = [c for c in df_ejemplo.columns if c.startswith(prefix_filter)]
    X_cols     = [c for c in (features + cols_extra) if c in df_ejemplo.columns]

    valores = []
    for col in X_cols:
        if col in inputs_dict:
            valores.append(float(inputs_dict[col]))
        elif col.startswith("reg_") and inputs_dict.get("region_seleccionada") == col.replace("reg_", ""):
            valores.append(1.0)
        elif col.startswith("cli_") and inputs_dict.get("cliente_tipo_seleccionado") == col.replace("cli_", ""):
            valores.append(1.0)
        elif col.startswith("cat_") and inputs_dict.get("categoria_seleccionada") == col.replace("cat_", ""):
            valores.append(1.0)
        else:
            valores.append(0.0)

    prediccion = modelo.predict(np.array(valores).reshape(1, -1))
    return float(prediccion[0])


# ─────────────────────────────────────────────────────────────────────────────
# PROYECCIÓN DE DEMANDA — RECURSIVE FORECASTING (Módulo 2)
# ─────────────────────────────────────────────────────────────────────────────
def proyectar_demanda_semanal(algoritmo, tienda_id, n_semanas, categoria_id):
    """
    Proyección de demanda a n_semanas hacia adelante para una combinación
    tienda×categoría específica, usando el modelo del Módulo 2 ya entrenado
    para el algoritmo indicado.

    Estrategia: RECURSIVE FORECASTING (según definición del proyecto v7.0,
    sección 2.3): se predice t+1 con los lags reales más recientes del
    historial; esa predicción se incorpora al historial y se usa como lag
    para predecir t+2; se itera n_semanas veces. Es el enfoque estándar para
    proyectar modelos de regresión con features autorregresivas cuando no
    hay datos futuros reales disponibles.

    Las variables exógenas que no se pueden proyectar de forma autorregresiva
    (precio_mediano, promedio_margen, dias_con_promocion, dias_con_feriado,
    num_productos_distintos, num_transacciones, perfil de tienda, región,
    categoría) se mantienen constantes en el último valor histórico conocido
    de esa serie — es una simplificación razonable y explícita, ya que el
    objetivo de esta salida es el volumen de reabastecimiento, no simular
    escenarios de precio o promoción futuros.
    """
    modulo_id = 2

    # ── 1. Modelo entrenado del algoritmo solicitado
    ruta_modelo = _ruta_pkl(modulo_id, algoritmo)
    if not os.path.exists(ruta_modelo):
        return {
            "error": f"El algoritmo '{algoritmo}' del Módulo 2 aún no ha sido "
                     f"entrenado. Ejecute primero su pipeline de entrenamiento."
        }

    with open(ruta_modelo, "rb") as f:
        modelo = pickle.load(f)

    # ── 2. Dataset procesado (misma fuente y mismo orden de columnas que en
    #      el entrenamiento, para que el vector de features sea compatible)
    features_base, target, file_path = obtener_mapeo_variables(modulo_id)
    if not os.path.exists(file_path):
        return {"error": "No se encontró el dataset procesado del Módulo 2 (Fase 3)."}

    df = pd.read_csv(file_path)
    df = df.sort_values(by=["año", "semana_iso"]).reset_index(drop=True)

    col_categoria = f"cat_{categoria_id}"
    if col_categoria not in df.columns:
        return {"error": f"La categoría '{categoria_id}' no existe en el dataset procesado."}

    # ── 3. Serie histórica de la combinación tienda + categoría seleccionada
    df_serie = df[(df["tienda_id"] == tienda_id) & (df[col_categoria] == 1)].copy()
    if df_serie.empty:
        return {
            "error": f"No hay historial de demanda para la tienda {tienda_id} "
                     f"y la categoría '{categoria_id}'."
        }
    df_serie = df_serie.sort_values(by=["año", "semana_iso"]).reset_index(drop=True)

    # Misma lista de columnas de entrada (orden) que usó ejecutar_entrenamiento_regresion
    prefix_filter  = ("reg_", "cat_")
    columnas_extra = [c for c in df.columns if c.startswith(prefix_filter)]
    X_cols         = [c for c in (features_base + columnas_extra) if c in df.columns]

    ultima_fila_historica = df_serie.iloc[-1].copy()
    año_hist    = int(ultima_fila_historica["año"])
    semana_hist = int(ultima_fila_historica["semana_iso"])

    # Historial reciente de demanda REAL (orden ascendente: [t-3, t-2, t-1, t])
    # para sembrar los lags recursivos; se rellena con 0 si la serie es corta,
    # igual que el fillna(0) aplicado en el preprocesamiento (Fase 3).
    historial_demanda = [float(v) for v in df_serie[target].tail(4).tolist()]
    while len(historial_demanda) < 4:
        historial_demanda.insert(0, 0.0)

    demanda_promedio_historica = float(df_serie[target].mean())

    # ── 4. Recursive forecasting: n_semanas iteraciones
    año_actual    = año_hist
    semana_actual = semana_hist
    proyeccion    = []

    for paso in range(1, int(n_semanas) + 1):
        semana_actual += 1
        if semana_actual > 52:
            semana_actual = 1
            año_actual   += 1

        fila = ultima_fila_historica.copy()
        fila["año"]        = año_actual
        fila["semana_iso"] = semana_actual
        fila["semana_sin"] = np.sin(2 * np.pi * semana_actual / 52.0)
        fila["semana_cos"] = np.cos(2 * np.pi * semana_actual / 52.0)

        # Lags autorregresivos: usan el historial (real + ya proyectado)
        fila["demanda_lag_1"]  = historial_demanda[-1]
        fila["demanda_lag_2"]  = historial_demanda[-2]
        fila["demanda_lag_4"]  = historial_demanda[-4]
        fila["media_movil_4"]  = round(float(np.mean(historial_demanda[-4:])), 2)

        X_paso = pd.DataFrame([fila])[X_cols]
        pred   = float(modelo.predict(X_paso)[0])
        pred   = max(pred, 0.0)  # la demanda no puede ser negativa

        # NOTA: nombres de claves alineados EXACTAMENTE con lo que consume
        # demanda.html en ejecutarProyeccion() -> d.proyecciones.map(p => ...
        # (usa p.semana, p.año, p.demanda_estimada). No cambiar sin actualizar el JS.
        proyeccion.append({
            "semana_num":       paso,
            "año":              año_actual,
            "semana":           semana_actual,
            "demanda_estimada": round(pred, 2),
        })

        # Desplaza la ventana de historial para la siguiente iteración
        historial_demanda.append(pred)
        historial_demanda.pop(0)

    return {
        "modulo_id":                    modulo_id,
        "tienda_id":                    tienda_id,
        "categoria_id":                 categoria_id,
        "algoritmo":                    algoritmo,
        "ultima_semana_historica":      {"año": año_hist, "semana_iso": semana_hist},
        "demanda_promedio_historica":   round(demanda_promedio_historica, 2),
        "proyecciones":                 proyeccion,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE RESIDUOS
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_residuos_base64(modulo_id, y_real, y_pred, modelo=None, features_list=None):
    residuos = np.array(y_real) - np.array(y_pred)
    plt.figure(figsize=(10, 4.5))
    plt.scatter(y_pred, residuos, alpha=0.5, color="#002855", edgecolors="none", s=25)
    plt.axhline(y=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Valores Predichos por el Estimador", fontweight="bold")
    plt.ylabel("Residuos (Magnitud del Error)", fontweight="bold")
    plt.title(
        "Distribución Estadística de Residuos (Análisis de Homocedasticidad)",
        fontsize=12, fontweight="bold", color="#002855"
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close("all")
    return f"data:image/png;base64,{img_b64}"


# ─────────────────────────────────────────────────────────────────────────────
# HISTOGRAMA DE RESIDUOS (normalidad de errores)
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_histograma_residuos_base64(y_real, y_pred, algoritmo_label, modulo_id):
    """
    Histograma de (y_real - y_pred) con curva de densidad Normal superpuesta,
    para evaluar visualmente el supuesto de normalidad de los errores.
    Complementa (no reemplaza) el scatter de homocedasticidad ya existente
    en generar_grafico_residuos_base64().
    """
    try:
        residuos = np.array(y_real) - np.array(y_pred)
        mu, sigma = float(np.mean(residuos)), float(np.std(residuos))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(
            residuos, bins=30, density=True, alpha=0.65,
            color="#0d6efd", edgecolor="white", label="Residuos observados"
        )

        # Curva de densidad Normal(μ, σ) superpuesta, calculada manualmente
        # (evita dependencia extra de scipy solo para la PDF gaussiana).
        if sigma > 0:
            x = np.linspace(residuos.min(), residuos.max(), 300)
            pdf = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            ax.plot(x, pdf, color="#dc3545", lw=2.2, label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")

        ax.axvline(0, color="#002855", linestyle="--", lw=1.5, alpha=0.8)
        ax.set_xlabel("Residuo (Real − Predicho)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Densidad", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Histograma de Residuos · {algoritmo_label}\nEvaluación Visual de Normalidad",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
        return f"data:image/png;base64,{img_b64}"

    except Exception:
        plt.close("all")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CURVAS DE APRENDIZAJE (diagnóstico de Overfitting / Underfitting)
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_curva_aprendizaje_base64(modelo, X, y, algoritmo_label, modulo_id):
    """
    Score de entrenamiento vs. score de validación (R²) en función del tamaño
    de la muestra de entrenamiento, usando sklearn.model_selection.learning_curve
    sobre el mismo algoritmo/params ya entrenado (se clona internamente, no
    afecta al modelo ya ajustado que se persiste en el .pkl).

    Lectura: curvas que divergen (train alto, validación baja y estancada) =
    Overfitting. Ambas curvas bajas y muy juntas = Underfitting.
    """
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            modelo, X, y,
            train_sizes=np.linspace(0.15, 1.0, 6),
            cv=3,
            scoring="r2",
            n_jobs=-1,
            random_state=42,
            shuffle=False,
        )

        train_mean = train_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_mean   = val_scores.mean(axis=1)
        val_std    = val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(train_sizes, train_mean, "o-", color="#002855", lw=2, label="Score Entrenamiento")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                         alpha=0.15, color="#002855")

        ax.plot(train_sizes, val_mean, "o-", color="#dc3545", lw=2, label="Score Validación")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                         alpha=0.15, color="#dc3545")

        ax.set_xlabel("Tamaño de la Muestra de Entrenamiento", fontweight="bold", fontsize=10)
        ax.set_ylabel("R² Score", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Curvas de Aprendizaje · {algoritmo_label}\nDiagnóstico de Overfitting / Underfitting",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
        return f"data:image/png;base64,{img_b64}"

    except Exception:
        plt.close("all")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE IMPORTANCIA DE VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_importancia_base64(modelo, feature_names, algoritmo_label, modulo_id):
    """
    Genera el ranking de importancia de features como gráfico de barras horizontales.
    - RF y GBR: usa model.feature_importances_ (nativo, basado en impureza).
    - LinearRegression: usa coeficientes absolutos normalizados (interpretación de
      magnitud relativa, no de importancia probabilística — se indica en el título).
    Siempre muestra las Top-15 para que el gráfico sea legible aunque haya muchas OHE.
    """
    try:
        if hasattr(modelo, "feature_importances_"):
            importancias = modelo.feature_importances_
            metodo_label = "Importancia por Reducción de Impureza (Gini)"
        else:
            # Regresión Lineal: |coef| normalizado al rango [0,1]
            coefs = np.abs(modelo.coef_)
            importancias = coefs / (coefs.sum() + 1e-10)
            metodo_label = "Importancia Relativa — |Coeficiente| Normalizado"

        # Ordenar y tomar Top-15
        indices     = np.argsort(importancias)[::-1][:15]
        top_vals    = importancias[indices]
        top_labels  = [feature_names[i] for i in indices]

        # Etiquetas más legibles: quitar prefijos OHE (reg_, cli_, cat_)
        def limpiar(lbl):
            for pfx in ("reg_", "cli_", "cat_"):
                if lbl.startswith(pfx):
                    return lbl[len(pfx):]
            return lbl

        top_labels_clean = [limpiar(l) for l in top_labels]

        # Paleta degradada — más oscuro = más importante
        colores = plt.cm.Blues(np.linspace(0.35, 0.85, len(top_vals)))[::-1]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        bars = ax.barh(range(len(top_vals)), top_vals[::-1], color=colores[::-1],
                       edgecolor="white", linewidth=0.6)
        ax.set_yticks(range(len(top_vals)))
        ax.set_yticklabels(top_labels_clean[::-1], fontsize=9, fontweight="bold")
        ax.set_xlabel("Importancia Relativa", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Top Features · {algoritmo_label}\n{metodo_label}",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Etiquetas de valor dentro de cada barra
        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(
                bar.get_width() * 0.98, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="right",
                fontsize=8, color="white", fontweight="bold"
            )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
        return f"data:image/png;base64,{img_b64}"

    except Exception:
        plt.close("all")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO REAL vs. PREDICHO
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_real_vs_pred_base64(y_real, y_pred, algoritmo_label, modulo_id):
    """
    Scatter de valores reales vs. predichos con línea de referencia perfecta (45°).
    Muestra una muestra aleatoria de hasta 2,500 puntos para evitar over-plotting
    en datasets grandes (el modelo se entrenó sobre todos — esto es solo visualización).
    El R² anotado corresponde al test set completo, no a la muestra visualizada.
    """
    try:
        y_r = np.array(y_real)
        y_p = np.array(y_pred)

        # Muestra representativa para visualización (evita over-plotting)
        N_MAX = 2500
        if len(y_r) > N_MAX:
            idx  = np.random.default_rng(42).choice(len(y_r), N_MAX, replace=False)
            y_r_plot = y_r[idx]
            y_p_plot = y_p[idx]
        else:
            y_r_plot, y_p_plot = y_r, y_p

        r2_completo = r2_score(y_r, y_p)

        # Límites del gráfico con 5% de margen
        lim_min = min(y_r_plot.min(), y_p_plot.min())
        lim_max = max(y_r_plot.max(), y_p_plot.max())
        margen  = (lim_max - lim_min) * 0.05
        lim_min -= margen
        lim_max += margen

        fig, ax = plt.subplots(figsize=(7, 6))

        # Puntos con densidad coloreada por distancia a la diagonal
        distancias = np.abs(y_r_plot - y_p_plot)
        sc = ax.scatter(
            y_r_plot, y_p_plot,
            c=distancias, cmap="RdYlGn_r",
            alpha=0.55, s=20, edgecolors="none"
        )
        plt.colorbar(sc, ax=ax, label="Error absoluto", shrink=0.8)

        # Línea de predicción perfecta (referencia 45°)
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color="#002855", lw=2, linestyle="--", label="Predicción perfecta")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Valor Real (Test Set)", fontweight="bold", fontsize=10)
        ax.set_ylabel("Valor Predicho por el Modelo", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Real vs. Predicho · {algoritmo_label}",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )

        # Anotación del R² del test set completo
        ax.annotate(
            f"R² (test completo) = {r2_completo:.4f}",
            xy=(0.05, 0.93), xycoords="axes fraction",
            fontsize=10, fontweight="bold",
            color="#002855",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#002855", lw=1.2)
        )

        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=110)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
        return f"data:image/png;base64,{img_b64}"

    except Exception:
        plt.close("all")
        return None
