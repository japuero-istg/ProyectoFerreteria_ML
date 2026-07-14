# Proyecto Ferretería ML

Aplicación Flask de inteligencia artificial para una ferretería: regresión
(predicción de ventas), clasificación y clustering, con autenticación de
usuarios y almacenamiento de resultados en PostgreSQL (columna `JSONB`) y
caché de modelos en Redis.

## Requisitos

- Python 3.12
- PostgreSQL (la app usa el tipo `JSONB` de Postgres, no compatible con SQLite)
- Redis
- Git

## 1. Clonar e instalar dependencias

```bash
git clone git@github.com:japuero-istg/ProyectoFerreteria_ML.git
cd ProyectoFerreteria_ML

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configurar variables de entorno

Copia el archivo de ejemplo y completa tus credenciales (el `.env` **no** se
sube al repositorio):

```bash
cp .env.example .env
```

Edita `.env` con los valores de tu PostgreSQL y Redis:

```
FLASK_ENV=development
SECRET_KEY=...
DATABASE_URL=postgresql://USUARIO:PASSWORD@HOST:5432/NOMBRE_DB
REDIS_URL=redis://HOST:6379/0
```

> Si ya tienes PostgreSQL y Redis corriendo en el host (puertos 5432 / 6379),
> usa esas credenciales. En caso contrario, levanta los servicios con Docker:
>
> ```bash
> docker compose up -d        # crea istg-postgresql e istg-redis
> ```
>
> Recuerda crear la base de datos antes de continuar, p. ej.:
> `CREATE DATABASE ferreteria_ml;`

## 3. Crear la base de datos y aplicar migraciones

```bash
# Crea la BD si aún no existe (en psql / tu cliente Postgres)
# CREATE DATABASE ferreteria_ml;

flask db upgrade --directory app/migrations
```

## 4. Ejecutar la aplicación

Opción A — con el CLI de Flask (usa `.flaskenv` y `.env` automáticamente):

```bash
flask run
```

Opción B — ejecutando el módulo directamente (carga `.env` vía `load_dotenv`):

```bash
python app/app.py
```

La app queda disponible en http://localhost:5000 y expone el health check en
http://localhost:5000/health.

## Estructura

```
app/
  app.py                 # Factory de la aplicación Flask
  config.py              # Configuración por entorno
  models/                # User, MLResult + instancias de extensiones
  blueprints/            # auth, dashboard, prediction, classification, clustering, data_manager
  services/              # data_loader (generación/carga de datos)
  migrations/            # Migraciones Alembic
  static/  templates/    # Frontend
docker-compose.yml       # Postgres + Redis para desarrollo local
requirements.txt
.env.example
```

## Notas

- El modelo `MLResult` usa `JSONB` de PostgreSQL; no se soporta SQLite.
- `.flaskenv` define `FLASK_APP=app.app:create_app` y `FLASK_ENV=development`.
- No commitear el `.env` (contiene credenciales); usa `.env.example` como plantilla.
