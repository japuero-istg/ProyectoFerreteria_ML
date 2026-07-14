import os  # <-- CORRECCIÓN 1: Faltaba importar la librería 'os' nativa de Python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()
import redis
from flask import Flask, jsonify
from app.config import config_by_name
from app.models import bcrypt, db, login_manager, migrate


def create_app(config_name: str | None = None) -> Flask:
    """
    Factory function de la aplicación.
    config_name: "development" | "production" | "testing".
    Si no se pasa, se toma de la variable de entorno FLASK_ENV
    (definida en .env / docker-compose.yml).
    """
    # Aquí se usa 'os.environ', por eso necesitábamos el 'import os' arriba
    config_name = config_name or os.environ.get("FLASK_ENV", "production")
    app = Flask(__name__)
    app.config.from_object(config_by_name[config_name])
    _init_extensions(app)
    _register_blueprints(app)
    _register_health_check(app)
    _register_context_processors(app)  # <-- NUEVO: inyecta 'datos_listos' a todas las plantillas
    return app


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    migrate.init_app(app, db)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"
    login_manager.login_message = "Debes iniciar sesión para acceder a esta página."

    @login_manager.user_loader
    def load_user(user_id: str):
        # CORRECCIÓN 2: Se agregó el prefijo 'app.' al módulo de usuarios
        from app.models.user import User
        return User.query.get(int(user_id))

    # Cliente Redis centralizado (istg-redis), disponible para cache_service.py
    # vía app.extensions["redis"]. decode_responses=False porque los modelos
    # entrenados se guardan serializados con pickle (bytes), no texto.
    app.extensions["redis"] = redis.from_url(
        app.config["REDIS_URL"],
        decode_responses=False,
    )


def _register_blueprints(app: Flask) -> None:
    # CORRECCIÓN 3: Se agregó el prefijo 'app.' a TODOS los blueprints internos
    from app.blueprints.auth import auth_bp
    from app.blueprints.classification import classification_bp
    from app.blueprints.clustering import clustering_bp
    from app.blueprints.dashboard import dashboard_bp
    from app.blueprints.prediction import prediction_bp
    from app.blueprints.data_manager import data_manager_bp
    
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(prediction_bp, url_prefix="/prediccion")
    app.register_blueprint(classification_bp, url_prefix="/clasificacion")
    app.register_blueprint(clustering_bp, url_prefix="/clustering")
    app.register_blueprint(data_manager_bp)

def _register_health_check(app: Flask) -> None:
    @app.route("/health")
    def health():
        """Usado por el healthcheck definido en docker-compose.yml."""
        return jsonify(status="ok"), 200


def _register_context_processors(app: Flask) -> None:
    """
    Inyecta variables globales disponibles en TODAS las plantillas Jinja,
    sin necesidad de pasarlas manualmente en cada render_template().

    'datos_listos' refleja si el Paso 2 (preprocesamiento) de la Fase 3
    ya fue completado por el Admin. base.html la usa para deshabilitar
    visualmente los enlaces a los módulos analíticos (Regresión,
    Clasificación, Clustering) en el sidebar.
    """
    @app.context_processor
    def inject_estado_datos():
        from app.services.data_loader import check_processed_datasets_exist
        return dict(datos_listos=check_processed_datasets_exist())


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=5000, debug=True)
