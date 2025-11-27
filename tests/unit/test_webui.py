"""Unit tests for WebUI server module.

Tests web server initialization and basic routes.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from kttc.webui.server import (
    EvaluateRequest,
    EvaluateResponse,
    ServerStats,
    _format_errors_list,
    _get_fallback_html,
    _prepare_error_response,
    create_app,
)


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


@pytest.mark.unit
class TestServerStats:
    """Test ServerStats model."""

    def test_server_stats_defaults(self) -> None:
        """Test ServerStats with default values."""
        stats = ServerStats()
        assert stats.total_evaluations == 0
        assert stats.average_mqm_score == 0.0
        assert stats.uptime_seconds == 0.0

    def test_server_stats_with_values(self) -> None:
        """Test ServerStats with custom values."""
        stats = ServerStats(
            total_evaluations=100,
            average_mqm_score=85.5,
            uptime_seconds=3600.0,
        )
        assert stats.total_evaluations == 100
        assert stats.average_mqm_score == 85.5
        assert stats.uptime_seconds == 3600.0


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions."""

    def test_get_fallback_html(self) -> None:
        """Test fallback HTML generation."""
        html = _get_fallback_html()
        assert isinstance(html, str)
        assert "KTTC Dashboard" in html
        assert "<!DOCTYPE html>" in html
        assert "<form" in html

    def test_prepare_error_response(self) -> None:
        """Test error response preparation."""
        mock_report = MagicMock()
        mock_error1 = MagicMock()
        mock_error1.severity.value = "critical"
        mock_error2 = MagicMock()
        mock_error2.severity.value = "minor"
        mock_error3 = MagicMock()
        mock_error3.severity.value = "minor"
        mock_report.errors = [mock_error1, mock_error2, mock_error3]

        result = _prepare_error_response(mock_report)

        assert result["critical"] == 1
        assert result["major"] == 0
        assert result["minor"] == 2

    def test_format_errors_list(self) -> None:
        """Test errors list formatting."""
        mock_report = MagicMock()
        mock_error = MagicMock()
        mock_error.category = "Accuracy"
        mock_error.subcategory = "Mistranslation"
        mock_error.severity.value = "critical"
        mock_error.description = "Wrong translation"
        mock_error.suggestion = "Use correct term"
        mock_error.location = "line 1"
        mock_report.errors = [mock_error]

        result = _format_errors_list(mock_report)

        assert len(result) == 1
        assert result[0]["category"] == "Accuracy"
        assert result[0]["subcategory"] == "Mistranslation"
        assert result[0]["severity"] == "critical"
        assert result[0]["description"] == "Wrong translation"
        assert result[0]["suggestion"] == "Use correct term"
        assert result[0]["location"] == "line 1"

    def test_format_errors_list_empty(self) -> None:
        """Test formatting empty errors list."""
        mock_report = MagicMock()
        mock_report.errors = []

        result = _format_errors_list(mock_report)

        assert result == []
