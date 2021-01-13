"""Define app database schema."""

from typing import Dict
from app import db


class BaseModel(db.Model):
    """Base model with default columns and methods."""

    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=db.func.now())
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())

    def to_dict(self: object) -> Dict:  # pragma: no cover
        data = {}
        columns = self.__table__.columns.keys()
        for key in columns:
            data[key] = getattr(self, key)
        return data


class User(BaseModel):
    """API User."""

    name = db.Column(db.String, nullable=False)
    api_key = db.Column(db.String, unique=True, nullable=False)
    last_activity_at = db.Column(db.DateTime)
