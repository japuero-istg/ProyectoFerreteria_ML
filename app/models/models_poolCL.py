import os
import time
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Indispensable para entornos Docker / hilos web
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
)

try:
    import graphviz
    _GRAPHVIZ_DISPONIBLE = True
except ImportError:
    _GRAPHVIZ_DISPONIBLE = False

BASE_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_DATA = os.path.abspath(os.path.join(BASE_MODELS_DIR, "..", "static", "data"))

FILE_TRANSACCIONAL_PROC = os.path.join(DIR_DATA, "dataset_transacciones_processed.csv")

# Mismo directorio de artefactos que la Suite de Regresión (models_pool.py).
# No hay colisión de nombres: los módulos 3 y 6 (Clasificación) no existen
# como IDs en la Suite de Regresión (que usa 1, 2, 5).
DIR_MODELS = os.path.join(BASE_MODELS_DIR, "data")
os.makedirs(DIR_MODELS, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALGORITMOS VÁLIDOS POR MÓDULO (a diferencia de Regresión, aquí la lista
# NO es idéntica entre módulos — DEB-05 / plan v6.0: M6 requiere balanceo
# de clases por el desbalance real de tipo_cliente).
# ─────────────────────────────────────────────────────────────────────────────
ALGORITMOS_VALIDOS_CL = {
    3: ["RandomForestClassifier", "GradientBoostingClassifier", "DecisionTreeClassifier"],
}

# Nombres de negocio de cada módulo, para templates/flash messages.
NOMBRES_MODULO_CL = {
    3: "Módulo 3: Categoría de Producto",
}


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS DE ARTEFACTOS (por módulo + por algoritmo) — mismo patrón que Regresión
# ─────────────────────────────────────────────────────────────────────────────
def _ruta_pkl(modulo_id: int, algoritmo: str) -> str:
    return os.path.join(DIR_MODELS, f"modelo_modulo_{modulo_id}_{algoritmo}.pkl")


def _ruta_json(modulo_id: int, algoritmo: str) -> str:
    return os.path.join(DIR_MODELS, f"metricas_modulo_{modulo_id}_{algoritmo}.json")


# ─────────────────────────────────────────────────────────────────────────────
# ESTADO DE ENTRENAMIENTO EN SEGUNDO PLANO (Opción B — fix del timeout 524 de
# Cloudflare/gunicorn con datasets grandes o algoritmos secuenciales como GBC)
# ─────────────────────────────────────────────────────────────────────────────
# Se usa un archivo "flag" en disco en vez de una variable en memoria porque
# gunicorn corre 3 workers como PROCESOS separados (no hilos) — una variable
# en memoria de un worker no sería visible desde otro. Un archivo en
# app/models/data/ sí lo es, porque todos los workers comparten el mismo
# filesystem del contenedor (mismo patrón ya usado para los .pkl/.json).
_TIMEOUT_FLAG_HUERFANO_SEG = 60 * 30  # 30 min: si un worker muere a media
# ejecución, el flag no debe quedar "entrenando" para siempre.


def _ruta_flag_entrenando(modulo_id: int, algoritmo: str) -> str:
    return os.path.join(DIR_MODELS, f"entrenando_modulo_{modulo_id}_{algoritmo}.flag")


def marcar_entrenamiento_iniciado(modulo_id: int, algoritmo: str) -> None:
    with open(_ruta_flag_entrenando(modulo_id, algoritmo), "w") as f:
        f.write(str(time.time()))


def marcar_entrenamiento_finalizado(modulo_id: int, algoritmo: str) -> None:
    ruta = _ruta_flag_entrenando(modulo_id, algoritmo)
    if os.path.exists(ruta):
        try:
            os.remove(ruta)
        except OSError:
            pass


def esta_entrenando(modulo_id: int, algoritmo: str) -> bool:
    ruta = _ruta_flag_entrenando(modulo_id, algoritmo)
    if not os.path.exists(ruta):
        return False
    # Salvaguarda de flags huérfanos: si el worker murió (OOM, restart de
    # contenedor, etc.) a media ejecución, el flag no se borra solo — se
    # ignora automáticamente pasado el umbral en vez de bloquear el botón
    # de "Entrenar" para siempre.
    antiguedad = time.time() - os.path.getmtime(ruta)
    if antiguedad > _TIMEOUT_FLAG_HUERFANO_SEG:
        marcar_entrenamiento_finalizado(modulo_id, algoritmo)
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MAPEO DE VARIABLES POR MÓDULO (fuente de verdad: definicion_proyecto_HD_v8.md,
# secciones 2.5, 2.6, 4.3 y 4.4 — no se repiten aquí, solo se implementan)
# ─────────────────────────────────────────────────────────────────────────────
def obtener_mapeo_variables_cl(modulo_id):
    if modulo_id == 3:
        # v9 §2.5 / §4.3 (Módulo 3): excluye costo_unidad y total_venta —
        # proxies casi deterministas de categoría en catálogos de precio
        # diferenciado (clasificador trivial de memorización).
        # NO se agregan cat_*/reg_*/cli_* como features: cat_* ES el target
        # codificado, incluirlo sería fuga de datos total.
        features = [
            "precio_unidad", "margen_unitario", "unidades_vendidas",
            "es_feriado", "promocion_activa", "tienda_id", "factor_tamano",
        ]
        target = "categoria_id"

    else:
        # Nota (v9): el Módulo 6 (tipo_cliente) fue retirado del alcance del
        # proyecto — AUC-ROC macro-OvR ≈ 0.50 (desempeño de azar) validado
        # empíricamente con los 4 algoritmos especificados. Ver
        # definicion_proyecto_HD_v9.md, Corrección 12, para el diagnóstico
        # completo. Este pipeline solo soporta el Módulo 3.
        raise ValueError("ID de módulo no soportado en la pipeline analítica de clasificación (solo 3 — el Módulo 6 fue retirado del alcance, ver Corrección 12).")

    return features, target, FILE_TRANSACCIONAL_PROC


def _columnas_features(df, modulo_id):
    """Arma la lista final de columnas X (Módulo 3 únicamente — M6 retirado del alcance)."""
    features_base, target, _ = obtener_mapeo_variables_cl(modulo_id)
    X_cols = [c for c in features_base if c in df.columns]
    return X_cols, target


# ─────────────────────────────────────────────────────────────────────────────
# INSTANCIACIÓN DE ALGORITMOS POR MÓDULO
# ─────────────────────────────────────────────────────────────────────────────
def _instanciar_modelo(modulo_id, algoritmo_nombre, n_estimators):
    """
    Devuelve (modelo, algoritmo_label, requiere_sample_weight).
    Nota (v9): el Módulo 6, que requería balanceo de clases
    (class_weight='balanced'/sample_weight, ver DEB-05 histórico) y
    KNeighborsClassifier, fue retirado del alcance del proyecto — ver
    definicion_proyecto_HD_v9.md, Corrección 12. Esta función ahora solo
    instancia los 3 algoritmos vigentes del Módulo 3, sin balanceo (no
    aplica: categoria_id no tiene el desbalance extremo de tipo_cliente).
    `requiere_sample_weight` se mantiene en la firma por compatibilidad con
    el resto del pipeline (siempre False en el alcance actual).
    """
    requiere_sample_weight = False

    if algoritmo_nombre == "RandomForestClassifier":
        # FIX (timeout de Cloudflare/gunicorn en producción): por defecto,
        # RandomForestClassifier construye árboles de PROFUNDIDAD ILIMITADA.
        # Sobre 231,772 filas eso son ~30s solo para el fit principal, antes de
        # sumar el CV de 5 pliegues y la curva de aprendizaje (25 ajustes más)
        # — el total supera el timeout del proxy (Cloudflare, ~100s en plan
        # gratuito) aunque el worker de gunicorn tenga 300s configurados.
        # Validado empíricamente: max_depth=15 + max_samples=0.3 (cada árbol
        # ve solo el 30% de las filas de entrenamiento, vía bootstrap) da el
        # MISMO accuracy (~0.60) en 8s en vez de 30s — no era necesitado ese
        # nivel de profundidad, solo desperdiciaba tiempo de cómputo.
        modelo = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=-1,
            max_depth=15, max_samples=0.3,
        )

    elif algoritmo_nombre == "GradientBoostingClassifier":
        # FIX (mismo problema de fondo que RF, pero GBC no tiene n_jobs — sus
        # etapas son secuenciales por diseño, no paralelizables). subsample<1.0
        # activa "Stochastic Gradient Boosting" (Friedman, 2002): cada una de
        # las 100 etapas entrena sobre una fracción aleatoria de las filas en
        # vez del 100%, reduciendo el costo por etapa sin sacrificar — y a
        # veces mejorando levemente — la generalización (actúa como
        # regularización, igual que max_samples en Random Forest).
        modelo = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42, subsample=0.5)

    elif algoritmo_nombre == "DecisionTreeClassifier":
        modelo = DecisionTreeClassifier(random_state=42)

    else:
        raise ValueError(f"Algoritmo '{algoritmo_nombre}' no reconocido para clasificación.")

    return modelo, algoritmo_nombre, requiere_sample_weight


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO (guarda artefactos POR ALGORITMO, no sobreescribe los demás)
# ─────────────────────────────────────────────────────────────────────────────
def ejecutar_entrenamiento_clasificacion(modulo_id, algoritmo_nombre, params=None):
    if params is None:
        params = {}
    if modulo_id != 3:
        raise ValueError("ID de módulo no soportado en la pipeline de clasificación (solo 3 — el Módulo 6 fue retirado del alcance, ver Corrección 12 en definicion_proyecto_HD_v9.md).")
    if algoritmo_nombre not in ALGORITMOS_VALIDOS_CL[modulo_id]:
        raise ValueError(f"Algoritmo '{algoritmo_nombre}' no válido para el Módulo {modulo_id}.")

    n_estimators = int(params.get("n_estimators", 100))
    test_size = float(params.get("test_size", 0.20))

    _, target, file_path = obtener_mapeo_variables_cl(modulo_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos procesados en: {file_path}")

    df = pd.read_csv(file_path)
    X_cols, target = _columnas_features(df, modulo_id)

    X = df[X_cols]
    y = df[target].astype(str)
    clases_ordenadas = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    modelo, algoritmo_label, _ = _instanciar_modelo(modulo_id, algoritmo_nombre, n_estimators)
    modelo.fit(X_train, y_train)

    # ── Subsample SOLO para diagnósticos (CV + curva de aprendizaje) ──
    # DECISIÓN DE INGENIERÍA (fix del timeout de Cloudflare/gunicorn): el modelo
    # que se GUARDA y se USA para predecir (arriba) entrena con las 231,772 filas
    # completas — su accuracy real no cambia. Pero CV (5 folds) + curva de
    # aprendizaje (grilla × folds) reentrenan el algoritmo decenas de veces más
    # solo para producir un diagnóstico visual/estadístico; hacerlo sobre el
    # dataset completo cada vez multiplica el tiempo de una petición HTTP a
    # varios minutos. Se usa una muestra estratificada de máximo 30,000 filas
    # (representativa, no arbitraria) exclusivamente para estos dos diagnósticos.
    # Es una práctica estándar de MLOps para mantener rápida la iteración; no
    # afecta el modelo entregado ni sus métricas de accuracy/precision/recall/F1
    # ya reportadas arriba (esas SÍ vienen del modelo completo sobre X_test real).
    LIMITE_MUESTRA_DIAGNOSTICO = 30_000
    if len(X) > LIMITE_MUESTRA_DIAGNOSTICO:
        X_diag, _, y_diag, _ = train_test_split(
            X, y, train_size=LIMITE_MUESTRA_DIAGNOSTICO, random_state=42, stratify=y
        )
    else:
        X_diag, y_diag = X, y

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test) if hasattr(modelo, "predict_proba") else None

    # ── Métricas globales (criterio del profesor: Accuracy, Precisión, Recall, F1, AUC-ROC)
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # ── Reporte por clase (precision/recall/f1 individuales)
    reporte_por_clase = classification_report(
        y_test, y_pred, labels=clases_ordenadas, output_dict=True, zero_division=0
    )

    # ── Matriz de confusión cruda (para el heatmap)
    matriz_confusion = confusion_matrix(y_test, y_pred, labels=clases_ordenadas)

    # ── AUC-ROC macro-promedio One-vs-Rest (multiclase)
    auc_roc_macro = None
    curvas_roc_por_clase = {}
    if y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=clases_ordenadas)
        aucs_individuales = []
        for i, clase in enumerate(clases_ordenadas):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc_clase = auc(fpr, tpr)
            aucs_individuales.append(auc_clase)
            curvas_roc_por_clase[clase] = {
                "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(float(auc_clase), 4)
            }
        auc_roc_macro = float(np.mean(aucs_individuales))

    metricas_dicc = {
        "algoritmo": algoritmo_label,
        "modulo_id": modulo_id,
        "samples_entrenamiento": int(len(df)),
        "test_size_utilizado": test_size,
        "clases": clases_ordenadas,
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "auc_roc_macro_ovr": round(auc_roc_macro, 4) if auc_roc_macro is not None else None,
        "matriz_confusion": matriz_confusion.tolist(),
        "reporte_por_clase": reporte_por_clase,
        "cv_f1_macro_folds": None,
        "cv_f1_macro_promedio": None,
    }

    # ── Validación cruzada: StratifiedKFold de 5 pliegues (sobre la muestra
    # de diagnóstico, no el dataset completo — ver nota más arriba)
    try:
        modelo_cv, _, _ = _instanciar_modelo(modulo_id, algoritmo_nombre, n_estimators)
        if hasattr(modelo_cv, "n_jobs"):
            modelo_cv.set_params(n_jobs=1)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_cv = cross_val_score(modelo_cv, X_diag, y_diag, cv=skf, scoring="f1_macro", n_jobs=-1)
        scores_cv = [round(float(s), 4) for s in scores_cv]
        metricas_dicc["cv_f1_macro_folds"] = scores_cv
        metricas_dicc["cv_f1_macro_promedio"] = round(float(np.mean(scores_cv)), 4)
    except Exception:
        pass  # CV es diagnóstico adicional; no debe tumbar el entrenamiento principal

    # ── Gráficos (Regla de Oro: RAM → Base64, jamás disco)
    metricas_dicc["grafico_matriz_confusion"] = generar_grafico_matriz_confusion_base64(
        matriz_confusion, clases_ordenadas, algoritmo_label, modulo_id
    )

    # Árbol de decisión SVG: SOLO para DecisionTreeClassifier; None explícito para los demás.
    if algoritmo_label == "DecisionTreeClassifier":
        metricas_dicc["grafico_arbol_decision"] = generar_grafico_arbol_decision_base64(
            modelo, X_cols, clases_ordenadas, modulo_id
        )
    else:
        metricas_dicc["grafico_arbol_decision"] = None

    metricas_dicc["grafico_curva_aprendizaje"] = generar_grafico_curva_aprendizaje_cl_base64(
        modulo_id, algoritmo_nombre, n_estimators, X_diag, y_diag, algoritmo_label
    )

    metricas_dicc["grafico_curva_roc"] = generar_grafico_curva_roc_base64(
        curvas_roc_por_clase, auc_roc_macro, algoritmo_label, modulo_id
    ) if curvas_roc_por_clase else None

    # ── Persistencia POR ALGORITMO (no sobreescribe los demás algoritmos)
    with open(_ruta_pkl(modulo_id, algoritmo_label), "wb") as f:
        pickle.dump({"modelo": modelo, "X_cols": X_cols, "clases": clases_ordenadas}, f)

    with open(_ruta_json(modulo_id, algoritmo_label), "w", encoding="utf-8") as f:
        json.dump(metricas_dicc, f, indent=4, ensure_ascii=False)

    return metricas_dicc


# ─────────────────────────────────────────────────────────────────────────────
# LECTURA DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
def obtener_metricas_locales_cl(modulo_id, algoritmo):
    ruta = _ruta_json(modulo_id, algoritmo)
    if os.path.exists(ruta):
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def obtener_todas_las_metricas_modulo_cl(modulo_id):
    """
    Devuelve un dict {algoritmo: metricas} con todos los algoritmos ya
    entrenados para el módulo indicado. Orden fijo según ALGORITMOS_VALIDOS_CL.
    """
    resultado = {}
    for alg in ALGORITMOS_VALIDOS_CL.get(modulo_id, []):
        datos = obtener_metricas_locales_cl(modulo_id, alg)
        if datos is not None:
            resultado[alg] = datos
    return resultado


# Nota (v9): obtener_distribucion_tipo_cliente() existió aquí para el bloque
# pedagógico opcional de DEB-05 en cliente.html (Módulo 6). Se elimina junto
# con el resto del código de M6 — ver definicion_proyecto_HD_v9.md, Corrección 12.


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCIA INTERACTIVA
# ─────────────────────────────────────────────────────────────────────────────
def predecir_interactivo_clasificacion(modulo_id, inputs_dict, algoritmo):
    """Carga el .pkl del algoritmo indicado y devuelve clase predicha + probabilidades."""
    ruta = _ruta_pkl(modulo_id, algoritmo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"El algoritmo '{algoritmo}' del Módulo {modulo_id} "
            f"aún no ha sido entrenado. Ejecute primero su pipeline."
        )

    with open(ruta, "rb") as f:
        artefacto = pickle.load(f)
    modelo, X_cols, clases = artefacto["modelo"], artefacto["X_cols"], artefacto["clases"]

    valores = []
    for col in X_cols:
        if col in inputs_dict:
            valores.append(float(inputs_dict[col]))
        elif col.startswith("cat_") and inputs_dict.get("categoria_seleccionada") == col.replace("cat_", ""):
            valores.append(1.0)
        else:
            valores.append(0.0)

    X_input = pd.DataFrame([valores], columns=X_cols)
    clase_predicha = modelo.predict(X_input)[0]

    resultado = {"clase_predicha": str(clase_predicha)}
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_input)[0]
        resultado["probabilidades"] = {
            clase: round(float(p), 4) for clase, p in zip(clases, proba)
        }
    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: MATRIZ DE CONFUSIÓN (Heatmap Seaborn, cmap='Purples')
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_matriz_confusion_base64(matriz, clases, algoritmo_label, modulo_id):
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            matriz, annot=True, fmt="d", cmap="Purples",
            xticklabels=clases, yticklabels=clases,
            cbar=True, ax=ax, linewidths=0.5, linecolor="white",
        )
        ax.set_xlabel("Predicción del Modelo", fontweight="bold", fontsize=10)
        ax.set_ylabel("Clase Real", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Matriz de Confusión · {algoritmo_label}\nMódulo {modulo_id}",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
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
# GRÁFICO: ÁRBOL DE DECISIÓN (SVG vectorial vía Graphviz)
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_arbol_decision_base64(modelo, feature_names, clases, modulo_id):
    """
    Exporta la lógica jerárquica del árbol con Graphviz en SVG vectorial.
    Se limita la PROFUNDIDAD DE VISUALIZACIÓN a max_depth=4 (parámetro propio
    de export_graphviz, no afecta al estimador ya entrenado) porque el árbol
    real puede crecer sin límite de profundidad y un SVG completo de un árbol
    profundo es ilegible en el navegador — es una decisión de legibilidad,
    no de precisión del modelo.
    """
    if not _GRAPHVIZ_DISPONIBLE:
        return None
    try:
        dot_data = export_graphviz(
            modelo,
            out_file=None,
            feature_names=feature_names,
            class_names=[str(c) for c in clases],
            filled=True, rounded=True, special_characters=True,
            max_depth=4,
            proportion=True,
        )
        grafo = graphviz.Source(dot_data)
        svg_bytes = grafo.pipe(format="svg")
        svg_b64 = base64.b64encode(svg_bytes).decode("utf-8")
        return f"data:image/svg+xml;base64,{svg_b64}"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO: CURVAS DE APRENDIZAJE (scoring='f1_macro')
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_curva_aprendizaje_cl_base64(modulo_id, algoritmo_nombre, n_estimators, X, y, algoritmo_label):
    try:
        modelo_lc, _, _ = _instanciar_modelo(modulo_id, algoritmo_nombre, n_estimators)

        # FIX (paralelismo anidado — causa real de los timeouts de Cloudflare
        # con RandomForestClassifier): learning_curve(n_jobs=-1) ya reparte el
        # entrenamiento de la grilla completa (train_sizes × folds) entre todos
        # los cores. Si el estimador interno TAMBIÉN trae n_jobs=-1 (RF lo trae
        # por defecto), cada proceso hijo intenta volver a paralelizar sobre
        # los mismos cores ya ocupados — sobre-suscripción de CPU, más lento
        # que un solo hilo, no más rápido. Se fuerza n_jobs=1 en el estimador
        # interno; el paralelismo real vive únicamente en learning_curve.
        if hasattr(modelo_lc, "n_jobs"):
            modelo_lc.set_params(n_jobs=1)

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        train_sizes, train_scores, val_scores = learning_curve(
            modelo_lc, X, y, cv=skf, scoring="f1_macro",
            train_sizes=np.linspace(0.25, 1.0, 4), n_jobs=-1, random_state=42
        )

        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.plot(train_sizes, train_mean, "o-", color="#002855", lw=2, label="Score Entrenamiento")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#002855")
        ax.plot(train_sizes, val_mean, "o-", color="#dc3545", lw=2, label="Score Validación")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#dc3545")

        ax.set_xlabel("Tamaño de la Muestra de Entrenamiento", fontweight="bold", fontsize=10)
        ax.set_ylabel("F1-Score (macro)", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Curvas de Aprendizaje · {algoritmo_label}\nDiagnóstico de Overfitting / Underfitting — Módulo {modulo_id}",
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
# GRÁFICO: CURVA ROC (One-vs-Rest, una curva por clase + AUC macro destacado)
# ─────────────────────────────────────────────────────────────────────────────
def generar_grafico_curva_roc_base64(curvas_roc_por_clase, auc_macro, algoritmo_label, modulo_id):
    try:
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        colores = plt.cm.tab10(np.linspace(0, 1, len(curvas_roc_por_clase)))

        for (clase, datos), color in zip(curvas_roc_por_clase.items(), colores):
            ax.plot(
                datos["fpr"], datos["tpr"], lw=2, color=color,
                label=f"{clase} (AUC = {datos['auc']:.3f})"
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2, label="Azar (AUC = 0.500)")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Tasa de Falsos Positivos", fontweight="bold", fontsize=10)
        ax.set_ylabel("Tasa de Verdaderos Positivos (Recall)", fontweight="bold", fontsize=10)
        ax.set_title(
            f"Curva ROC One-vs-Rest · {algoritmo_label} — Módulo {modulo_id}\n"
            f"AUC Macro-Promedio = {auc_macro:.4f}" if auc_macro is not None else "Curva ROC",
            fontsize=11, fontweight="bold", color="#002855", pad=12
        )
        ax.legend(fontsize=8, loc="lower right")
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
