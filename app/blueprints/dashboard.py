from flask import Blueprint, render_template
from flask_login import login_required, current_user
from app.models import db
from app.models.ml_result import MLResult
from collections import Counter

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
@login_required
def index():
    """Vista principal del Dashboard con las métricas y el historial real desde la BD."""
    
    # 1. Recuperamos todo el historial del usuario logueado ordenado por la fecha más reciente
    ultimos_analisis = MLResult.query.filter_by(usuario_id=current_user.id)\
                                    .order_by(MLResult.ejecutado_en.desc())\
                                    .all()
    
    # 2. Procesamiento dinámico de KPIs matemáticos sobre la colección activa
    total_ejecuciones = len(ultimos_analisis)
    ultimo_algoritmo = ultimos_analisis[0].algoritmo if total_ejecuciones > 0 else "N/A"
    
    if total_ejecuciones > 0:
        # Extraemos los módulos y determinamos el de mayor frecuencia analítica
        lista_modulos = [item.modulo for item in ultimos_analisis]
        modulo_mas_usado = Counter(lista_modulos).most_common(1)[0][0]
    else:
        modulo_mas_usado = "Ninguno"
    
    # 3. Mapeo de variables JSON esperado por la vista templates/dashboard/index.html
    kpis = {
        "total_ejecuciones": total_ejecuciones,
        "modulo_mas_usado": modulo_mas_usado,
        "ultimo_algoritmo": ultimo_algoritmo
    }
    
    return render_template("dashboard/index.html", kpis=kpis, analisis=ultimos_analisis)
