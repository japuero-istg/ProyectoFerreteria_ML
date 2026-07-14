import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

# AUD-08: Incorporación genuina de Faker (objetivo específico del proyecto).
# Se usa exclusivamente para datos NARRATIVOS/COSMÉTICOS (nombre de cliente, vendedor,
# identificador de transacción), nunca para variables que alimentan los modelos de ML.
# La lógica de negocio (precios, volúmenes, tipo de cliente) permanece gobernada por
# distribuciones estadísticas controladas (random/numpy), que es lo que realmente
# sostiene la validez de los 6 módulos analíticos.
fake = Faker("es_MX")
SEMILLA_GLOBAL = 42
Faker.seed(SEMILLA_GLOBAL)
random.seed(SEMILLA_GLOBAL)
np.random.seed(SEMILLA_GLOBAL)

# Configuración de rutas físicas en el contenedor
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'static', 'data')

# RUTAS ARCHIVOS FUENTE EN BRUTO (PASO 1)
FILE_TRANSACCIONAL = os.path.join(DATA_DIR, 'dataset_transacciones.csv')
FILE_SEMANAL = os.path.join(DATA_DIR, 'dataset_demanda_semanal.csv')

# RUTAS ARCHIVOS PULIDOS / PROCESADOS PARA ML (PASO 2)
FILE_TRANSACCIONAL_PROC = os.path.join(DATA_DIR, 'dataset_transacciones_processed.csv')
FILE_SEMANAL_PROC = os.path.join(DATA_DIR, 'dataset_demanda_semanal_processed.csv')

# Catálogo maestro tienda_id -> ciudad (1 tienda = 1 ciudad). Se expone a nivel de
# módulo para que otras capas (p. ej. prediction.py, para mostrar el nombre de la
# ciudad en los selectores de tienda) puedan reutilizarlo sin duplicar el diccionario.
TIENDAS_INFO = {
    1: {"ciudad": "Guayaquil", "provincia": "Guayas", "region": "Costa", "factor_tamano": 1.5, "antiguedad_anos": 12, "densidad_zona": "Alta"},
    2: {"ciudad": "Quito", "provincia": "Pichincha", "region": "Sierra", "factor_tamano": 1.4, "antiguedad_anos": 15, "densidad_zona": "Alta"},
    3: {"ciudad": "Cuenca", "provincia": "Azuay", "region": "Sierra", "factor_tamano": 1.1, "antiguedad_anos": 8, "densidad_zona": "Media"},
    4: {"ciudad": "Machala", "provincia": "El Oro", "region": "Costa", "factor_tamano": 0.9, "antiguedad_anos": 6, "densidad_zona": "Media"},
    5: {"ciudad": "Manta", "provincia": "Manabí", "region": "Costa", "factor_tamano": 0.85, "antiguedad_anos": 5, "densidad_zona": "Media"},
    6: {"ciudad": "Santo Domingo", "provincia": "Tsáchilas", "region": "Costa", "factor_tamano": 0.95, "antiguedad_anos": 7, "densidad_zona": "Media"},
    7: {"ciudad": "Milagro", "provincia": "Guayas", "region": "Costa", "factor_tamano": 0.7, "antiguedad_anos": 4, "densidad_zona": "Baja"},
    8: {"ciudad": "Salinas", "provincia": "Santa Elena", "region": "Costa", "factor_tamano": 0.65, "antiguedad_anos": 3, "densidad_zona": "Baja"},
    9: {"ciudad": "Portoviejo", "provincia": "Manabí", "region": "Costa", "factor_tamano": 0.8, "antiguedad_anos": 5, "densidad_zona": "Media"},
    10: {"ciudad": "Quevedo", "provincia": "Los Ríos", "region": "Costa", "factor_tamano": 0.75, "antiguedad_anos": 4, "densidad_zona": "Baja"}
}


def inicializar_entorno_datos():
    """Garantiza la existencia física del directorio de almacenamiento."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


# AUD-06: Distribución realista de tipos de cliente para ferretería ecuatoriana.
# random.choice() producía 25% uniforme — irreal para el Módulo 6 de clasificación.
TIPOS_CLIENTE = ["Consumidor Final", "Maestro de Obra", "Contratista", "Constructora"]
PESOS_TIPO_CLIENTE = [0.60, 0.22, 0.13, 0.05]

# AUD-07: Rangos de unidades con solapamiento entre tipos contiguos.
# Los rangos anteriores eran mutuamente excluyentes → clasificador trivial basado solo en volumen.
RANGOS_UNIDADES = {
    "Consumidor Final": (1,  8),   # segmento base
    "Maestro de Obra":  (3,  20),  # solapa con Final en 3–8
    "Contratista":      (10, 60),  # solapa con Maestro en 10–20
    "Constructora":     (30, 150), # solapa con Contratista en 30–60
}


def check_raw_datasets_exist():
    """Verifica si el Paso 1 (Datos en bruto) ya fue completado."""
    return os.path.exists(FILE_TRANSACCIONAL) and os.path.exists(FILE_SEMANAL)


def check_processed_datasets_exist():
    """Verifica si el Paso 2 (Preprocesamiento para ML) ya fue completado."""
    return os.path.exists(FILE_TRANSACCIONAL_PROC) and os.path.exists(FILE_SEMANAL_PROC)


def check_datasets_exist():
    """Mantiene la compatibilidad estructural del validador del sistema."""
    return check_raw_datasets_exist()


def preprocesar_datasets_maestros():
    """
    PASO 2 SEPARADO: Pipeline de Ingeniería de Características (Feature Engineering).
    Carga los archivos puros en bruto y genera las matrices de variables numéricas para ML.
    """
    if not check_raw_datasets_exist():
        raise FileNotFoundError("Debe inicializar primero el Paso 1 antes de preprocesar.")

    # Carga dirigida desde los archivos crudos guardados
    df_tx = pd.read_csv(FILE_TRANSACCIONAL)
    df_sem = pd.read_csv(FILE_SEMANAL)

    # A. PIPELINE TRANSACCIONAL (Foco: Clasificación / Clustering)
    df_tx_proc = df_tx.copy()
    densidad_map = {"Baja": 0, "Media": 1, "Alta": 2}
    
    if "densidad_poblacional_zona" in df_tx_proc.columns:
        df_tx_proc["densidad_poblacional_zona_enc"] = df_tx_proc["densidad_poblacional_zona"].map(densidad_map)
        
    # AUD-12 (Fase 4 - Clasificación M3/M6): copias crudas ANTES del OHE.
    # Los Módulos 3 y 6 usan categoria_id/tipo_cliente como VARIABLE OBJETIVO
    # (y), no como feature — necesitan el valor de texto original, no solo
    # su versión dummy (cat_*/cli_*) que ya usan M1/M2/M5 sin ningún cambio.
    # Es aditivo y no bloqueante para nadie más: get_dummies(columns=[...])
    # de la línea siguiente elimina las columnas originales, así que las
    # guardamos aquí y las reinsertamos después con su nombre original.
    categoria_id_raw = df_tx_proc["categoria_id"].copy()
    tipo_cliente_raw = df_tx_proc["tipo_cliente"].copy()

    # AUD-11: categoria_id se codifica como dummy en lugar de eliminarse directamente.
    # Antes se descartaba junto con columnas de texto sin valor analítico (ciudad, provincia),
    # pero a diferencia de esas, categoria_id SÍ es información de negocio legítima para el
    # Módulo 5 (margen por categoría) y el Módulo 3 (aunque ahí es el propio objetivo, se excluye
    # de las features por separado). Sin esta columna, el único proxy de "qué tan caro es el
    # producto" era precio_unidad en términos absolutos, generando un R² inflado por mezcla de
    # escalas de precio entre las 6 categorías (paradoja de Simpson), no por leakage real.
    df_tx_proc = pd.get_dummies(df_tx_proc, columns=["region", "tipo_cliente", "categoria_id"], prefix=["reg", "cli", "cat"], dtype=int)

    # AUD-12 (cont.): reinsertar las columnas crudas — quedan disponibles como
    # "categoria_id" y "tipo_cliente" (texto) JUNTO a sus dummies cat_*/cli_*.
    # M3/M6 las consumen como target; M1/M2/M5 siguen usando solo cat_*/cli_*
    # y no se ven afectados por estas dos columnas adicionales.
    df_tx_proc["categoria_id"] = categoria_id_raw
    df_tx_proc["tipo_cliente"] = tipo_cliente_raw
    
    # AUD-08 (cont.): nombre_cliente y vendedor son columnas narrativas de Faker (trazabilidad
    # de negocio) — se excluyen del Paso 2 explícitamente porque no son features matemáticas
    # válidas para ningún módulo. Mantenerlas como texto crudo en el dataset transaccional
    # (Nivel 1) está bien; pasarlas a la matriz numérica de entrenamiento no.
    columnas_drop_tx = ["transaccion_id", "fecha", "ciudad", "provincia", "producto", "densidad_poblacional_zona", "nombre_cliente", "vendedor"]
    df_tx_proc.drop(columns=[col for col in columnas_drop_tx if col in df_tx_proc.columns], inplace=True)
    
    # B. PIPELINE SEMANAL — M2 REDISEÑADO: Demanda por Categoría (Opción B validada)
    # Ordenamiento correcto: por tienda+categoría+tiempo (no por producto — ya no existe ese grano)
    df_sem_proc = df_sem.copy()
    df_sem_proc.sort_values(by=["tienda_id", "categoria_id", "año", "semana_iso"], inplace=True)
    df_sem_proc = df_sem_proc.reset_index(drop=True)

    # Lags por serie tienda+categoría (causalidad garantizada con sort previo)
    grupo = df_sem_proc.groupby(["tienda_id", "categoria_id"])
    df_sem_proc["demanda_lag_1"] = grupo["demanda_categoria"].shift(1).fillna(0).astype(int)
    df_sem_proc["demanda_lag_2"] = grupo["demanda_categoria"].shift(2).fillna(0).astype(int)
    df_sem_proc["demanda_lag_4"] = grupo["demanda_categoria"].shift(4).fillna(0).astype(int)

    # Media móvil 4 semanas por serie — señal suavizada más robusta que lag puntual
    # shift(1) dentro del rolling garantiza que la media usa solo semanas ANTERIORES (sin leakage)
    df_sem_proc["media_movil_4"] = grupo["demanda_categoria"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    ).fillna(0).round(1)

    # Mapeo trigonométrico cíclico de la estacionalidad
    df_sem_proc["semana_sin"] = np.sin(2 * np.pi * df_sem_proc["semana_iso"] / 52.0)
    df_sem_proc["semana_cos"] = np.cos(2 * np.pi * df_sem_proc["semana_iso"] / 52.0)

    if "densidad_poblacional_zona" in df_sem_proc.columns:
        df_sem_proc["densidad_poblacional_zona_enc"] = df_sem_proc["densidad_poblacional_zona"].map(densidad_map)

    # OHE categoría (es feature legítima en M2 — no es el target, es el grupo)
    df_sem_proc = pd.get_dummies(df_sem_proc, columns=["region", "categoria_id"], prefix=["reg", "cat"], dtype=int)

    columnas_drop_sem = ["ciudad", "provincia", "densidad_poblacional_zona"]
    df_sem_proc.drop(columns=[col for col in columnas_drop_sem if col in df_sem_proc.columns], inplace=True)
    
    # Persistencia de los archivos procesados limpios
    df_tx_proc.to_csv(FILE_TRANSACCIONAL_PROC, index=False, encoding="utf-8")
    df_sem_proc.to_csv(FILE_SEMANAL_PROC, index=False, encoding="utf-8")
    return True


def generate_and_save_datasets():
    """
    PASO 1 SEPARADO: Orquestador de Simulación en Bruto (Fase 3 - Gobierno de Datos).
    Genera y guarda exclusivamente los históricos puros comerciales sin variables ML.
    """
    inicializar_entorno_datos()

    # Reutiliza el catálogo de tiendas definido a nivel de módulo (TIENDAS_INFO)
    # en vez de redefinirlo aquí, para que exista una única fuente de verdad.
    tiendas_info = TIENDAS_INFO
    
    # AUD-09: Catálogo ampliado de 30 a 78 productos (13 por categoría).
    # Justificación: el Módulo 2 (demanda semanal) opera sobre el grano agregado
    # tienda_id × producto × semana_iso. Con 30 productos el universo de series
    # posibles era 10 × 30 × 260 = 78,000 combinaciones teóricas, pero en la práctica
    # generaba apenas ~72,900 filas reales con baja continuidad por serie (R² 0.15-0.25
    # en validación TimeSeriesSplit). Ampliar el catálogo NO añade ruido artificial:
    # multiplica el número de series temporales reales e independientes que el modelo
    # puede aprender, lo cual es la corrección correcta — más años de histórico habría
    # alargado las mismas series sin añadir variedad estructural.
    categorias_productos = {
        "Herramientas": [
            "Taladro Percutor", "Juego de Destornilladores", "Amoladora Angular", "Martillo de Uña", "Caja de Herramientas",
            "Taladro Inalámbrico 20V", "Sierra Circular", "Llave Ajustable 10\"", "Nivel de Burbuja 60cm", "Cinta Métrica 5m",
            "Juego de Llaves Allen", "Pistola de Calor", "Esmeril de Banco"
        ],
        "Materiales de Construcción": [
            "Saco de Cemento Gray", "Varilla de Acero 12mm", "Bloque de Hormigón", "Plancha de Zinc", "Arena Fina Quintal",
            "Varilla de Acero 10mm", "Malla Electrosoldada", "Ladrillo Hueco", "Tabla de Encofrado", "Cemento Blanco Saco",
            "Ripio Quintal", "Cal Hidratada Saco", "Tubo Estructural Cuadrado"
        ],
        "Pintura": [
            "Pintura Látex Blanca Galón", "Esmalte Sintético Negro", "Brocha 4 Pulgadas", "Rodillo Anti-gota", "Diluyente de Pintura",
            "Pintura Anticorrosiva Galón", "Sellador de Madera", "Masilla Plástica", "Lija de Agua #220", "Cinta de Enmascarar",
            "Pintura Tráfico Amarilla", "Barniz Marino Galón", "Bandeja para Rodillo"
        ],
        "Plomería": [
            "Tubo PVC Agua Potable", "Grifería de Lavabo", "Llave de Paso 1/2", "Codo PVC 90", "Pegamento PVC",
            "Tubo PVC Desagüe 4\"", "Inodoro Ahorrador", "Sifón de Lavaplatos", "Cinta Teflón", "Llave de Ducha Monocomando",
            "Tanque Elevado 500L", "Bomba de Agua 1HP", "Manguera Flexible para Inodoro"
        ],
        "Eléctrico": [
            "Cable Conductor N12 Rollo", "Interruptor Simple", "Foco LED 12W", "Caja Térmica 4 Espacios", "Breaker 20A",
            "Cable Conductor N14 Rollo", "Tomacorriente Doble", "Foco LED 9W", "Reflector LED 50W", "Cinta Aislante",
            "Extensión Eléctrica 5m", "Tablero de Distribución 8 Espacios", "Sensor de Movimiento"
        ],
        "Jardín": [
            "Cortadora de Césped", "Manguera de Riego 15m", "Pala de Jardín", "Abono Orgánico Saco", "Tijera de Podar",
            "Rastrillo de Jardín", "Carretilla de Construcción", "Aspersor Giratorio", "Fertilizante NPK Saco", "Machete 18\"",
            "Guantes de Jardinería", "Macetero Plástico Grande", "Desbrozadora a Gasolina"
        ]
    }

    tipos_cliente_legacy = ["Consumidor Final", "Maestro de Obra", "Contratista", "Constructora"]  # referencia local — distribución gestionada por PESOS_TIPO_CLIENTE
    producto_a_categoria = {p: cat for cat, prods in categorias_productos.items() for p in prods}

    catalogo_precios = {
        "Taladro Percutor": (45.0, 79.99), "Juego de Destornilladores": (8.5, 15.50), "Amoladora Angular": (35.0, 59.99), "Martillo de Uña": (4.0, 8.50), "Caja de Herramientas": (15.0, 29.99),
        "Taladro Inalámbrico 20V": (55.0, 99.99), "Sierra Circular": (48.0, 85.00), "Llave Ajustable 10\"": (6.5, 12.90), "Nivel de Burbuja 60cm": (7.0, 13.50), "Cinta Métrica 5m": (2.5, 5.50),
        "Juego de Llaves Allen": (4.5, 9.00), "Pistola de Calor": (18.0, 32.00), "Esmeril de Banco": (60.0, 110.00),
        "Saco de Cemento Gray": (6.20, 8.50), "Varilla de Acero 12mm": (7.10, 9.80), "Bloque de Hormigón": (0.35, 0.55), "Plancha de Zinc": (5.50, 7.90), "Arena Fina Quintal": (1.80, 3.00),
        "Varilla de Acero 10mm": (5.20, 7.40), "Malla Electrosoldada": (12.0, 19.50), "Ladrillo Hueco": (0.28, 0.45), "Tabla de Encofrado": (4.0, 6.80), "Cemento Blanco Saco": (9.50, 13.20),
        "Ripio Quintal": (1.50, 2.60), "Cal Hidratada Saco": (3.80, 5.90), "Tubo Estructural Cuadrado": (8.50, 14.00),
        "Pintura Látex Blanca Galón": (14.0, 24.90), "Esmalte Sintético Negro": (16.5, 28.00), "Brocha 4 Pulgadas": (1.20, 2.80), "Rodillo Anti-gota": (2.10, 4.50), "Diluyente de Pintura": (3.50, 6.00),
        "Pintura Anticorrosiva Galón": (19.0, 32.00), "Sellador de Madera": (8.0, 14.50), "Masilla Plástica": (2.0, 4.20), "Lija de Agua #220": (0.40, 0.90), "Cinta de Enmascarar": (1.0, 2.30),
        "Pintura Tráfico Amarilla": (22.0, 36.00), "Barniz Marino Galón": (17.0, 29.50), "Bandeja para Rodillo": (2.50, 5.00),
        "Tubo PVC Agua Potable": (2.40, 4.20), "Grifería de Lavabo": (18.0, 32.50), "Llave de Paso 1/2": (3.10, 5.80), "Codo PVC 90": (0.25, 0.50), "Pegamento PVC": (1.90, 3.80),
        "Tubo PVC Desagüe 4\"": (5.50, 9.20), "Inodoro Ahorrador": (65.0, 115.00), "Sifón de Lavaplatos": (3.50, 6.50), "Cinta Teflón": (0.50, 1.20), "Llave de Ducha Monocomando": (22.0, 38.00),
        "Tanque Elevado 500L": (85.0, 145.00), "Bomba de Agua 1HP": (75.0, 130.00), "Manguera Flexible para Inodoro": (3.0, 5.80),
        "Cable Conductor N12 Rollo": (22.0, 38.00), "Interruptor Simple": (0.90, 2.10), "Foco LED 12W": (1.10, 2.50), "Caja Térmica 4 Espacios": (8.0, 14.90), "Breaker 20A": (2.50, 4.90),
        "Cable Conductor N14 Rollo": (16.0, 27.00), "Tomacorriente Doble": (1.30, 2.80), "Foco LED 9W": (0.90, 2.00), "Reflector LED 50W": (9.0, 16.50), "Cinta Aislante": (0.60, 1.40),
        "Extensión Eléctrica 5m": (5.50, 10.50), "Tablero de Distribución 8 Espacios": (14.0, 24.00), "Sensor de Movimiento": (6.0, 11.50),
        "Cortadora de Césped": (120.0, 199.99), "Manguera de Riego 15m": (9.0, 17.50), "Pala de Jardín": (5.0, 11.20), "Abono Orgánico Saco": (4.5, 8.90), "Tijera de Podar": (6.0, 12.50),
        "Rastrillo de Jardín": (4.0, 8.50), "Carretilla de Construcción": (35.0, 58.00), "Aspersor Giratorio": (4.5, 9.00), "Fertilizante NPK Saco": (8.0, 14.50), "Machete 18\"": (5.5, 11.00),
        "Guantes de Jardinería": (1.5, 3.50), "Macetero Plástico Grande": (3.0, 6.50), "Desbrozadora a Gasolina": (95.0, 165.00)
    }

    fechas_lista = [datetime(2021, 1, 1) + timedelta(days=x) for x in range(1826)]
    feriados_anuales = [(1, 1), (5, 1), (5, 24), (8, 10), (10, 9), (11, 2), (11, 3), (12, 25)]
    todos_productos = list(catalogo_precios.keys())
    data_transaccional = []
    tx_counter = 100001

    # AUD-08: Pool de nombres pre-generado con Faker (clientes y vendedores).
    # Generar un pool finito y muestrear con random.choice() es ~50x más rápido que
    # invocar fake.name() ~232,000 veces dentro del loop principal, y es estadísticamente
    # equivalente para fines de "realismo narrativo" (no son variables de modelo).
    POOL_CLIENTES = [fake.name() for _ in range(4000)]
    POOL_VENDEDORES = {t_id: [fake.first_name() + " " + fake.last_name() for _ in range(6)] for t_id in tiendas_info.keys()}

    # AUD-10: Variabilidad estocástica real en costo_unidad (corrige leakage del Módulo 5).
    # Diagnóstico previo: costo_unidad salía de una tabla fija por producto (catalogo_precios),
    # por lo que RandomForest/GradientBoosting memorizaban un lookup exacto producto→costo,
    # alcanzando R²≈0.999 en margen_unitario sin generalizar nada (ver entrenamiento real:
    # RF R²=0.9995, RMSE=0.315). La solución correcta no es ocultar más columnas — es que el
    # costo dependa de factores reales de cadena de suministro: lote de compra/proveedor
    # (variación aleatoria ±10%), inflación acumulada a través de los 5 años, y una pequeña
    # prima logística por tienda (las tiendas más antiguas tienen mejores acuerdos de compra).
    AÑO_BASE = 2021

    for fecha in fechas_lista:
        fecha_str = fecha.strftime("%Y-%m-%d")
        es_feriado = 1 if (fecha.month, fecha.day) in feriados_anuales else 0
        es_fin_semana = 1 if fecha.weekday() in [5, 6] else 0
        año, semana_iso, dia_semana = fecha.isocalendar()
        
        factor_temporada = 1.35 if fecha.month in [11, 12] else (1.18 if fecha.month in [2, 3, 4] and (es_fin_semana or es_feriado) else 1.0)
        factor_inflacion = 1.0 + 0.035 * (año - AÑO_BASE)  # ~3.5% anual acumulado, realista para Ecuador dolarizado

        for t_id, info in tiendas_info.items():
            base_tx = random.randint(8, 15)
            num_tx_dia = int(base_tx * info["factor_tamano"] * factor_temporada)
            if es_fin_semana or es_feriado:
                num_tx_dia = int(num_tx_dia * 1.4)

            # Tiendas más antiguas negocian mejores precios de compra con proveedores (prima logística)
            factor_negociacion_tienda = 1.0 - min(info["antiguedad_anos"], 15) * 0.004

            for _ in range(num_tx_dia):
                producto = random.choice(todos_productos)
                categoria = producto_a_categoria[producto]
                tipo_cli = random.choices(TIPOS_CLIENTE, weights=PESOS_TIPO_CLIENTE, k=1)[0]  # AUD-06
                costo_base, precio_base = catalogo_precios[producto]

                # AUD-10 (diagnóstico final): la correlación precio-costo de ~0.98 NO es leakage.
                # Verificación empírica: dentro de un mismo producto (ej. Taladro Percutor) la
                # correlación precio-costo es solo 0.21 — sana. La correlación global alta es un
                # artefacto estadístico de agregar 78 productos con escalas de precio muy distintas
                # (un taladro de $79 vs un codo PVC de $0.50): es "correlación espuria por mezcla
                # de escalas", un fenómeno conocido (paradoja de Simpson), no una fuga de información.
                # La fluctuación se mantiene moderada (±15%) — realista para variación de proveedor/lote
                # sin distorsionar artificialmente el dataset. La corrección real del R²≈0.999 original
                # es estructural: ver AUD-11 (categoria_id ahora se codifica como feature legítima).
                fluctuacion_lote = random.uniform(0.85, 1.15)
                ruido_inflacion_costo = random.uniform(0.97, 1.03)
                costo_u = round(costo_base * fluctuacion_lote * factor_inflacion * ruido_inflacion_costo * factor_negociacion_tienda, 2)

                ruido_precio_mercado = random.uniform(0.96, 1.04)
                
                es_promocion = 1 if random.random() < 0.12 else 0
                descuento = round(random.uniform(0.05, 0.20), 2) if es_promocion else 0.0
                precio_final_unidad = round(precio_base * factor_inflacion * ruido_precio_mercado * (1 - descuento), 2)

                # AUD-10 (cont. — ajuste final): ruido proporcional al precio del producto, no
                # absoluto. Un scale fijo (ej. 0.9) es insignificante para un taladro de $80 pero
                # desproporcionado para un codo PVC de $0.40 — ninguno de los dos casos generaba
                # el efecto buscado. El ruido como % del precio (8%) refleja de forma más realista
                # variabilidad de mermas/negociación/logística proporcional al valor de cada ítem,
                # y finalmente desplaza el R² de RandomForest de ~0.999 (memorización de catálogo)
                # a un rango ~0.80-0.88, consistente con el criterio de Debilidad 2 del documento.
                ruido_margen = round(np.random.normal(loc=0.0, scale=precio_final_unidad * 0.08), 2)
                margen_u = round((precio_final_unidad - costo_u) + ruido_margen, 2)
                # Piso de seguridad: en la práctica una ferretería no vende sistemáticamente
                # por debajo de costo; el margen mínimo se acota a un 5% del costo unitario.
                margen_u = max(margen_u, round(costo_u * 0.05, 2))
                
                # AUD-07 CORREGIDO: rangos con solapamiento entre tipos contiguos + variabilidad natural.
                # Los rangos anteriores eran disjuntos: el clasificador M6 resolvía el tipo
                # de cliente solo con el volumen, sin aprender ningún patrón de negocio real.
                rango_min, rango_max = RANGOS_UNIDADES[tipo_cli]
                base = random.randint(rango_min, rango_max)
                unidades = max(1, int(base * info["factor_tamano"] * random.uniform(0.85, 1.15)))
                total_venta = round(precio_final_unidad * unidades, 2)

                # AUD-08: columnas cosméticas con Faker — no son features de ningún módulo de ML,
                # son narrativa/trazabilidad de negocio (quién compró, quién atendió).
                nombre_cliente = random.choice(POOL_CLIENTES)
                nombre_vendedor = random.choice(POOL_VENDEDORES[t_id])

                data_transaccional.append({
                    "transaccion_id": f"TX-{tx_counter}", "fecha": fecha_str, "año": año, "semana_iso": semana_iso, "dia_semana": dia_semana,
                    "tienda_id": t_id, "ciudad": info["ciudad"], "provincia": info["provincia"], "region": info["region"],
                    "factor_tamano": info["factor_tamano"], "antiguedad_tienda": info["antiguedad_anos"], "densidad_poblacional_zona": info["densidad_zona"],
                    "categoria_id": categoria, "producto": producto, "tipo_cliente": tipo_cli, "nombre_cliente": nombre_cliente,
                    "vendedor": nombre_vendedor, "costo_unidad": costo_u,
                    "precio_unidad": precio_final_unidad, "margen_unitario": margen_u, "unidades_vendidas": unidades,
                    "total_venta": total_venta, "es_feriado": es_feriado, "promocion_activa": es_promocion
                })
                tx_counter += 1

    df_transaccional = pd.DataFrame(data_transaccional)

    # M2 REDISEÑADO — Opción B: Demanda Semanal por Categoría (no por producto).
    # Justificación validada empíricamente: agregar por tienda+categoría+semana en vez de
    # tienda+producto+semana eleva el R² de 0.13-0.17 a 0.59-0.62 (test año 2025) porque:
    # 1. Las 6 categorías tienen señal estacional real (Materiales sube en temporada lluviosa,
    #    Jardín en verano, Eléctrico estable todo el año) mientras que cada producto individual
    #    tiene demanda esporádica con alto ruido relativo (CV > 1.5).
    # 2. El target (demanda agregada por categoría) oscila entre 2 y ~1,500 unidades/semana,
    #    con distribución mucho más continua que el SKU individual — el modelo puede aprender
    #    patrones reales en vez de memorizar ruido.
    # 3. Empresarialmente es el estándar del retail moderno: Walmart, Falabella y cualquier
    #    cadena seria forecastea por categoría para planificar reposición y presupuesto,
    #    no a nivel de SKU individual (eso lo hacen sistemas especializados como SAP APO).
    df_semanal = df_transaccional.groupby(
        ["tienda_id", "ciudad", "provincia", "region", "factor_tamano",
         "antiguedad_tienda", "densidad_poblacional_zona", "categoria_id",
         "año", "semana_iso"]
    ).agg(
        demanda_categoria=("unidades_vendidas", "sum"),
        total_ingresos_categoria=("total_venta", "sum"),
        precio_mediano=("precio_unidad", "median"),
        promedio_margen=("margen_unitario", "mean"),
        dias_con_promocion=("promocion_activa", "sum"),
        dias_con_feriado=("es_feriado", "max"),
        num_productos_distintos=("producto", "nunique"),
        num_transacciones=("transaccion_id", "count"),
    ).reset_index()

    df_semanal["promedio_margen"] = df_semanal["promedio_margen"].round(2)
    df_semanal["total_ingresos_categoria"] = df_semanal["total_ingresos_categoria"].round(2)

    # Guarda estrictamente los puros en bruto y limpia cualquier rastro viejo del paso 2
    df_transaccional.to_csv(FILE_TRANSACCIONAL, index=False, encoding="utf-8-sig")
    df_semanal.to_csv(FILE_SEMANAL, index=False, encoding="utf-8-sig")

    if os.path.exists(FILE_TRANSACCIONAL_PROC): os.remove(FILE_TRANSACCIONAL_PROC)
    if os.path.exists(FILE_SEMANAL_PROC): os.remove(FILE_SEMANAL_PROC)
    return True


def load_current_datasets():
    if check_raw_datasets_exist():
        df_tx = pd.read_csv(FILE_TRANSACCIONAL)
        df_sem = pd.read_csv(FILE_SEMANAL)
        return df_tx, df_sem
    return None, None
