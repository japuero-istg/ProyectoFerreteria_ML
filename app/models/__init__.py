from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

# Instancias únicas de las extensiones, importadas tanto por app.py (para
# inicializarlas con init_app) como por los módulos de modelos (user.py,
# ml_result.py) que las usarán para definir tablas y relaciones.
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
bcrypt = Bcrypt()

# Los modelos concretos se importan aquí una vez creados en la FASE 1,
# para que Flask-Migrate los detecte al generar migraciones:
#
# from app.models.user import User
# from app.models.ml_result import MLResult



# Los modelos concretos se importan aquí para que Flask-Migrate los detecte
# al generar migraciones. El import va AL FINAL del archivo porque user.py y
# ml_result.py a su vez importan `db` desde este mismo módulo (import circular
# evitado por orden de ejecución).
from app.models.user import User          # noqa: E402,F401
from app.models.ml_result import MLResult  # noqa: E402,F401
