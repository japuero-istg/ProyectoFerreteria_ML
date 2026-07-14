import os
from datetime import timedelta


class Config:
    """Configuración base, compartida por todos los entornos."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "d2v-s@ecr3T-k4y-ca$mb1aR")

    # --- Base de datos (istg-postgresql centralizado) ---
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # --- Redis (istg-redis centralizado, caché de modelos) ---
    REDIS_URL = os.environ.get("REDIS_URL")
    REDIS_KEY_PREFIX = os.environ.get("REDIS_KEY_PREFIX", "ml_proyecto2p_v1")
    REDIS_CACHE_TTL_SEG = int(os.environ.get("REDIS_CACHE_TTL_SEG", 3600))

    # --- Sesión / cookies (la app se sirve detrás de Traefik con TLS) ---
    REMEMBER_COOKIE_DURATION = timedelta(days=7)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_ECHO = False
    SESSION_COOKIE_SECURE = False


class ProductionConfig(Config):
    DEBUG = False
    SESSION_COOKIE_SECURE = True  # requiere HTTPS; Traefik termina TLS antes del contenedor


class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    # Por defecto, tests contra SQLite en memoria (no contra istg-postgresql)
    SQLALCHEMY_DATABASE_URI = os.environ.get("TEST_DATABASE_URL", "sqlite:///:memory:")
    WTF_CSRF_ENABLED = False
    SESSION_COOKIE_SECURE = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

