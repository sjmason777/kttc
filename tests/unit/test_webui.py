"""Unit tests for WebUI server module.

Tests web server initialization and basic routes.
"""

import pytest

from kttc.webui.server import create_app


@pytest.mark.unit
class TestWebUIServer:
    """Test WebUI server functionality."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        """Test that create_app returns a FastAPI app."""
        # Act
        app = create_app()

        # Assert
        assert app is not None
        assert hasattr(app, "get")  # FastAPI has route decorators
        assert hasattr(app, "post")

    def test_app_has_routes_configured(self) -> None:
        """Test that app has basic routes configured."""
        # Arrange
        app = create_app()

        # Act
        routes = [getattr(route, "path", None) for route in app.routes]

        # Assert
        assert "/" in routes  # Home route
        # Note: More routes would be tested in integration tests


@pytest.mark.unit
class TestWebUIConfiguration:
    """Test WebUI configuration."""

    def test_default_configuration(self) -> None:
        """Test app is created with default configuration."""
        # Act
        app = create_app()

        # Assert
        assert app.title == "KTTC Web UI" or "KTTC" in app.title
        # Basic smoke test - app should be created successfully
