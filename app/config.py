import os


class Config:
    """
    Base configuration for Flask app with testing, debug and tracking set to false
    """

    TESTING = False
    DEBUG = False
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    """
    Add production PostgreSQL database connection from environment variables to Config
    """


class DevelopmentConfig(Config):
    """
    Set debug to true and add development PostgreSQL database connection
    from environment variables to Config
    """

    DEBUG = True


class TestingConfig(Config):
    """
    Set testing and debug to true and add testing SQLite database connection to Config
    """

    TESTING = True
    DEBUG = True


def set_config():  # pragma: no cover
    """
    Return appropriate Config class based on Flask environment
    """
    if os.environ.get("FLASK_ENV") == "development":
        return DevelopmentConfig
    return ProductionConfig
