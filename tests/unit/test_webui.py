"""Unit tests for WebUI server module.

Tests web server initialization and basic routes.
"""

import pytest
from pydantic import ValidationError

from kttc.webui.server import EvaluateRequest, EvaluateResponse, create_app


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


@pytest.mark.unit
class TestEvaluateRequest:
    """Test EvaluateRequest model."""

    def test_valid_request(self) -> None:
        """Test creating valid request."""
        request = EvaluateRequest(
            source_text="Hello world",
            translation="Hola mundo",
            source_lang="en",
            target_lang="es",
        )
        assert request.source_text == "Hello world"
        assert request.translation == "Hola mundo"
        assert request.source_lang == "en"
        assert request.target_lang == "es"

    def test_request_with_optional_fields(self) -> None:
        """Test request with optional fields."""
        request = EvaluateRequest(
            source_text="Hello",
            translation="Hola",
            source_lang="en",
            target_lang="es",
            reference="Hola",
            context={"domain": "general"},
            domain="technical",
        )
        assert request.reference == "Hola"
        assert request.context == {"domain": "general"}
        assert request.domain == "technical"

    def test_request_default_optional_fields(self) -> None:
        """Test optional fields default to None."""
        request = EvaluateRequest(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="ru",
        )
        assert request.reference is None
        assert request.context is None
        assert request.domain is None

    def test_request_validates_source_lang_format(self) -> None:
        """Test source language code validation."""
        # Valid 2-letter code should work
        request = EvaluateRequest(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="ru",
        )
        assert request.source_lang == "en"

    def test_request_validates_target_lang_format(self) -> None:
        """Test target language code validation."""
        request = EvaluateRequest(
            source_text="Test",
            translation="Test",
            source_lang="en",
            target_lang="de",
        )
        assert request.target_lang == "de"

    def test_request_rejects_empty_source_text(self) -> None:
        """Test empty source text is rejected."""
        with pytest.raises(ValidationError):
            EvaluateRequest(
                source_text="",
                translation="Test",
                source_lang="en",
                target_lang="ru",
            )

    def test_request_rejects_empty_translation(self) -> None:
        """Test empty translation is rejected."""
        with pytest.raises(ValidationError):
            EvaluateRequest(
                source_text="Test",
                translation="",
                source_lang="en",
                target_lang="ru",
            )


@pytest.mark.unit
class TestEvaluateResponse:
    """Test EvaluateResponse model."""

    def test_valid_response(self) -> None:
        """Test creating valid response."""
        response = EvaluateResponse(
            mqm_score=95.5,
            status="pass",
            errors_count=2,
            errors_by_severity={"minor": 2},
            errors=[],
            processing_time=1.5,
        )
        assert response.mqm_score == 95.5
        assert response.status == "pass"
        assert response.errors_count == 2
        assert response.processing_time == 1.5

    def test_response_with_errors(self) -> None:
        """Test response with error details."""
        response = EvaluateResponse(
            mqm_score=80.0,
            status="fail",
            errors_count=3,
            errors_by_severity={"critical": 1, "major": 1, "minor": 1},
            errors=[
                {"type": "accuracy", "severity": "critical", "description": "Mistranslation"},
                {"type": "fluency", "severity": "major", "description": "Grammar error"},
                {"type": "style", "severity": "minor", "description": "Informal tone"},
            ],
            processing_time=2.3,
        )
        assert len(response.errors) == 3
        assert response.errors[0]["type"] == "accuracy"

    def test_response_pass_status(self) -> None:
        """Test response with pass status."""
        response = EvaluateResponse(
            mqm_score=98.0,
            status="pass",
            errors_count=0,
            errors_by_severity={},
            errors=[],
            processing_time=0.5,
        )
        assert response.status == "pass"
        assert response.errors_count == 0

    def test_response_fail_status(self) -> None:
        """Test response with fail status."""
        response = EvaluateResponse(
            mqm_score=50.0,
            status="fail",
            errors_count=10,
            errors_by_severity={"critical": 5, "major": 5},
            errors=[],
            processing_time=3.2,
        )
        assert response.status == "fail"
