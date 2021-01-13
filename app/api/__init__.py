"""Initialize API and provide decorators."""

from functools import wraps
from flask import request

from app import db
from app.models.models import User


def require_auth(func):
    """Create decorator for authorization with API key."""

    @wraps(func)
    def func_wrapper(*args, **kwargs):
        if "API_KEY" not in request.headers:
            return "Credentials missing", 401
        api_keys = [user[0] for user in User.query.with_entities(User.api_key).all()]
        if request.headers["API_KEY"] not in api_keys:
            return "Credentials not valid", 401
        user = User.query.filter(User.api_key == request.headers["API_KEY"]).one()
        user.last_activity_at = db.func.now()
        db.session.commit()
        return func(*args, **kwargs)

    return func_wrapper
