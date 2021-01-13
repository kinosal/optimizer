"""Flask app configurations."""

import os


class Config:
    """Base configuration for Flask app."""

    TESTING = False
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    """Production config with PostgreSQL database connection."""

    user = os.environ.get("PROD_DB_USER")
    pw = os.environ.get("PROD_DB_PW")
    url = os.environ.get("PROD_DB_URL")
    name = os.environ.get("PROD_DB_NAME")
    SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{pw}@{url}/{name}"


class DevelopmentConfig(Config):
    """Development config with PostgreSQL database connection."""

    DEBUG = True
    user = os.environ.get("DEV_DB_USER")
    pw = os.environ.get("DEV_DB_PW")
    url = os.environ.get("DEV_DB_URL")
    name = os.environ.get("DEV_DB_NAME")
    SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{pw}@{url}/{name}"


class TestingConfig(Config):
    """Testing config with SQLite database connection."""

    TESTING = True
    DEBUG = True
    basedir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(basedir, "tests/test.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{path}"


def set_config():  # pragma: no cover
    """Return appropriate Config class based on Flask environment."""
    if os.environ.get("FLASK_ENV") == "development":
        return DevelopmentConfig
    return ProductionConfig
