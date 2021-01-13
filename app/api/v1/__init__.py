"""Initialize API version by creating Blueprint and importing routes."""

from flask import Blueprint

api_v1 = Blueprint("api", __name__, url_prefix="/api/v1")

from app.api.v1 import routes  # NoQA
