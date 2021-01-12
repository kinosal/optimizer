"""Run Flask app."""

from app import create_app
import app.config as config

app = create_app(config.set_config())
app.app_context().push()

if __name__ == "__main__":
    app.run()
