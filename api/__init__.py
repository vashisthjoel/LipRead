"""FastAPI application package for Lip-Reading REST service.

Exposes the application instance and factory for ASGI deployment.
"""

from api.main import app, create_app

__all__ = ["app", "create_app"]
