from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import login_required, login_user, logout_user, current_user

from app.models import bcrypt, db
from app.models.user import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Ruta para el registro de nuevos usuarios."""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        # Validaciones básicas
        if not username or not email or not password:
            flash("Todos los campos son obligatorios.", "danger")
            return redirect(url_for("auth.register"))

        # Verificar si el usuario o email ya existen
        if User.query.filter_by(username=username).first():
            flash("El nombre de usuario ya está registrado.", "danger")
            return redirect(url_for("auth.register"))

        if User.query.filter_by(email=email).first():
            flash("El correo electrónico ya está registrado.", "danger")
            return redirect(url_for("auth.register"))

        # Crear nuevo usuario con contraseña hasheada
        hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")
        new_user = User(
            username=username,
            email=email,
            password=hashed_password
        )

        try:
            db.session.add(new_user)
            db.session.commit()
            flash("¡Registro exitoso! Ahora puedes iniciar sesión.", "success")
            return redirect(url_for("auth.login"))
        except Exception as e:
            db.session.rollback()
            flash("Ocurrió un error al registrar el usuario. Inténtalo de nuevo.", "danger")
            # En producción puedes loggear el error 'str(e)' en tus archivos de log

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Ruta para el inicio de sesión."""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username_or_email = request.form.get("username_or_email", "").strip()
        password = request.form.get("password", "")
        remember = True if request.form.get("remember") else False

        if not username_or_email or not password:
            flash("Por favor, llena todos los campos.", "warning")
            return redirect(url_for("auth.login"))

        # Buscar usuario por username o por email
        user = User.query.filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()

        # Validar credenciales y si el usuario está activo
        if user and bcrypt.check_password_hash(user.password, password):
            if not user.is_active:
                flash("Esta cuenta ha sido desactivada.", "danger")
                return redirect(url_for("auth.login"))
            
            login_user(user, remember=remember)
            
            # Flask-Login maneja automáticamente la redirección segura a la página que intentaba acceder (next)
            next_page = request.args.get("next")
            return redirect(next_page) if next_page else redirect(url_for("dashboard.index"))
        else:
            flash("Credenciales incorrectas. Inténtalo de nuevo.", "danger")
            return redirect(url_for("auth.login"))

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    """Ruta para cerrar la sesión de forma segura."""
    logout_user()
    flash("Has cerrado sesión correctamente.", "info")
    return redirect(url_for("auth.login"))
