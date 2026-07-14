from datetime import datetime

from sqlalchemy.dialects.postgresql import JSONB

from app.models import db


class MLResult(db.Model):
    __tablename__ = "historial_analisis"

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False)

    modulo = db.Column(db.String(20), nullable=False)        # 'clasificacion' | 'regresion' | 'clustering'
    algoritmo = db.Column(db.String(50), nullable=False)     # 'RandomForest' | 'KMeans' | etc.
    metricas = db.Column(JSONB, nullable=False)              # {"r2": 0.87, "rmse": 12.3, ...} — indexable con ->> y @>
    n_registros = db.Column(db.Integer, nullable=False)
    ejecutado_en = db.Column(db.DateTime, default=datetime.utcnow)
    duracion_seg = db.Column(db.Numeric(6, 2))

    usuario = db.relationship("User", back_populates="resultados")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "usuario_id": self.usuario_id,
            "modulo": self.modulo,
            "algoritmo": self.algoritmo,
            "metricas": self.metricas,
            "n_registros": self.n_registros,
            "ejecutado_en": self.ejecutado_en.isoformat() if self.ejecutado_en else None,
            "duracion_seg": float(self.duracion_seg) if self.duracion_seg is not None else None,
        }

    def __repr__(self) -> str:
        return f"<MLResult {self.modulo}/{self.algoritmo} usuario={self.usuario_id}>"
