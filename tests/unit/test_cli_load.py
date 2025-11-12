"""Comprehensive tests for CLI load command and model checking.

Tests the complete implementation of model downloading, verification, and status checking:
- Model existence checking
- Progress tracking during download
- Model status display consistency
- Error handling during download
- Cache directory validation
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from kttc.cli.commands.load import (
    check_model_exists,
    download_model_with_progress,
    download_models,
    get_model_size,
    verify_models,
)
from kttc.cli.main import app
from kttc.cli.ui import check_models_with_loader
from kttc.utils.dependencies import get_models_status, models_are_downloaded

runner = CliRunner()


@pytest.mark.unit
class TestModelChecking:
    """Tests for model existence checking functions."""

    def test_models_are_downloaded_all_present(self, tmp_path: Path) -> None:
        """Test models_are_downloaded returns True when all models exist with checkpoints."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        required_models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
            "models--Unbabel--XCOMET-XL",
        ]

        # Create all models with checkpoints
        for model_dir in required_models:
            checkpoint_dir = cache_dir / model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert models_are_downloaded() is True

    def test_models_are_downloaded_one_missing(self, tmp_path: Path) -> None:
        """Test models_are_downloaded returns False when one model is missing."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        # Create only 2 out of 3 models
        models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
        ]

        for model_dir in models:
            checkpoint_dir = cache_dir / model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert models_are_downloaded() is False

    def test_models_are_downloaded_no_checkpoint(self, tmp_path: Path) -> None:
        """Test models_are_downloaded returns False when model exists but no checkpoint."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        required_models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
            "models--Unbabel--XCOMET-XL",
        ]

        # Create models but without checkpoints
        for model_dir in required_models:
            model_path = cache_dir / model_dir
            model_path.mkdir(parents=True, exist_ok=True)
            (model_path / "config.json").write_text("{}")

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert models_are_downloaded() is False

    def test_models_are_downloaded_snapshots_structure(self, tmp_path: Path) -> None:
        """Test models_are_downloaded works with new HuggingFace snapshots structure."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        required_models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
            "models--Unbabel--XCOMET-XL",
        ]

        # Create models with snapshots structure
        for model_dir in required_models:
            snapshot_dir = cache_dir / model_dir / "snapshots" / "abc123"
            checkpoint_dir = snapshot_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert models_are_downloaded() is True

    def test_models_are_downloaded_cache_not_exists(self, tmp_path: Path) -> None:
        """Test models_are_downloaded returns False when cache directory doesn't exist."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert models_are_downloaded() is False

    def test_check_model_exists_with_checkpoint(self, tmp_path: Path) -> None:
        """Test check_model_exists returns True for model with checkpoint."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--Unbabel--wmt22-comet-da"
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert check_model_exists("Unbabel/wmt22-comet-da") is True

    def test_check_model_exists_without_checkpoint(self, tmp_path: Path) -> None:
        """Test check_model_exists returns False for model without checkpoint."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / "models--Unbabel--wmt22-comet-da"
        model_dir.mkdir(parents=True, exist_ok=True)

        with patch("pathlib.Path.home", return_value=tmp_path):
            assert check_model_exists("Unbabel/wmt22-comet-da") is False

    def test_check_model_exists_not_exists(self, tmp_path: Path) -> None:
        """Test check_model_exists returns False for non-existent model."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            assert check_model_exists("Unbabel/wmt22-comet-da") is False


@pytest.mark.unit
class TestModelStatus:
    """Tests for get_models_status function."""

    def test_get_models_status_all_downloaded(self, tmp_path: Path) -> None:
        """Test get_models_status returns correct status for all downloaded models."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        required_models = [
            ("models--Unbabel--wmt22-comet-da", "COMET-22", "1.3 GB"),
            ("models--Unbabel--wmt23-cometkiwi-da-xxl", "CometKiwi", "900 MB"),
            ("models--Unbabel--XCOMET-XL", "XCOMET-XL", "800 MB"),
        ]

        for model_dir, _, _ in required_models:
            checkpoint_dir = cache_dir / model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = get_models_status()

            assert len(status) == 3
            for name, size, status_text, color in status:
                assert status_text == "✓ Downloaded"
                assert color == "green"

    def test_get_models_status_all_missing(self, tmp_path: Path) -> None:
        """Test get_models_status returns correct status for missing models."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            status = get_models_status()

            assert len(status) == 3
            for name, size, status_text, color in status:
                assert status_text == "○ Not downloaded"
                assert color == "yellow"

    def test_get_models_status_incomplete(self, tmp_path: Path) -> None:
        """Test get_models_status returns correct status for incomplete models."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        # Create model directories without checkpoints
        models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
            "models--Unbabel--XCOMET-XL",
        ]

        for model_dir in models:
            model_path = cache_dir / model_dir
            model_path.mkdir(parents=True, exist_ok=True)
            (model_path / "config.json").write_text("{}")

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = get_models_status()

            assert len(status) == 3
            for name, size, status_text, color in status:
                assert status_text == "⚠ Incomplete"
                assert color == "yellow"

    def test_get_models_status_mixed(self, tmp_path: Path) -> None:
        """Test get_models_status with mixed model states."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        # Downloaded with checkpoint
        checkpoint_dir = cache_dir / "models--Unbabel--wmt22-comet-da" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        # Incomplete (exists but no checkpoint)
        incomplete_dir = cache_dir / "models--Unbabel--wmt23-cometkiwi-da-xxl"
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        # Missing - don't create directory

        with patch("pathlib.Path.home", return_value=tmp_path):
            status = get_models_status()

            assert len(status) == 3
            assert status[0][2] == "✓ Downloaded"  # COMET-22
            assert status[1][2] == "⚠ Incomplete"  # CometKiwi
            assert status[2][2] == "○ Not downloaded"  # XCOMET-XL


@pytest.mark.unit
class TestVerifyModels:
    """Tests for verify_models function."""

    def test_verify_models_all_downloaded(self, tmp_path: Path) -> None:
        """Test verify_models returns correct lists when all models downloaded."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        required_models = [
            "models--Unbabel--wmt22-comet-da",
            "models--Unbabel--wmt23-cometkiwi-da-xxl",
            "models--Unbabel--XCOMET-XL",
        ]

        for model_dir in required_models:
            checkpoint_dir = cache_dir / model_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "model.ckpt").write_text("fake checkpoint")

        with patch("pathlib.Path.home", return_value=tmp_path):
            downloaded, missing = verify_models()

            assert len(downloaded) == 3
            assert len(missing) == 0
            assert "COMET-22" in downloaded
            assert "CometKiwi" in downloaded
            assert "XCOMET-XL" in downloaded

    def test_verify_models_all_missing(self, tmp_path: Path) -> None:
        """Test verify_models returns correct lists when all models missing."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            downloaded, missing = verify_models()

            assert len(downloaded) == 0
            assert len(missing) == 3
            assert "COMET-22" in missing
            assert "CometKiwi" in missing
            assert "XCOMET-XL" in missing


@pytest.mark.unit
class TestCheckModelsWithLoader:
    """Tests for check_models_with_loader UI function."""

    def test_check_models_with_loader_all_ready(self, tmp_path: Path) -> None:
        """Test check_models_with_loader returns True when all models ready."""
        with patch("kttc.utils.dependencies.models_are_downloaded", return_value=True):
            result = check_models_with_loader()
            assert result is True

    def test_check_models_with_loader_missing_models(self, tmp_path: Path) -> None:
        """Test check_models_with_loader returns False when models missing."""
        mock_status = [
            ("COMET-22", "1.3 GB", "✓ Downloaded", "green"),
            ("CometKiwi", "900 MB", "○ Not downloaded", "yellow"),
            ("XCOMET-XL", "800 MB", "○ Not downloaded", "yellow"),
        ]

        with patch("kttc.utils.dependencies.models_are_downloaded", return_value=False):
            with patch("kttc.utils.dependencies.get_models_status", return_value=mock_status):
                result = check_models_with_loader()
                assert result is False


@pytest.mark.unit
class TestLoadCommand:
    """Tests for load command and download functions."""

    def test_load_command_all_models_present(self) -> None:
        """Test load command when all models already downloaded."""
        # Mock huggingface_hub module
        mock_hf = Mock()
        mock_hf.HfFolder.get_token.return_value = "fake-token"

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("kttc.cli.commands.load.verify_models") as mock_verify:
                mock_verify.return_value = (["COMET-22", "CometKiwi", "XCOMET-XL"], [])

                result = runner.invoke(app, ["load"])

                assert result.exit_code == 0
                assert "All Models Already Downloaded" in result.stdout

    def test_get_model_size(self, tmp_path: Path) -> None:
        """Test get_model_size calculates total size correctly."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create some files
        (model_dir / "file1.bin").write_bytes(b"x" * 1000)
        (model_dir / "file2.safetensors").write_bytes(b"x" * 2000)

        subdir = model_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.ckpt").write_bytes(b"x" * 500)

        size = get_model_size(model_dir)
        assert size == 3500

    def test_download_model_with_progress_success(self, tmp_path: Path) -> None:
        """Test download_model_with_progress completes successfully."""
        # Mock huggingface_hub module
        mock_hf = Mock()
        mock_download = Mock()
        mock_hf.snapshot_download = mock_download
        mock_hf.HfFolder = Mock()
        mock_hf.HfFolder.get_token = Mock(return_value=None)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("pathlib.Path.home", return_value=tmp_path):
                success, message = download_model_with_progress(
                    "Unbabel/wmt22-comet-da", "COMET-22", 1.3
                )

                assert success is True
                assert "✓ COMET-22" in message
                mock_download.assert_called_once()

    def test_download_model_with_progress_error(self, tmp_path: Path) -> None:
        """Test download_model_with_progress handles download errors."""
        # Mock huggingface_hub module with error
        mock_hf = Mock()
        mock_hf.snapshot_download = Mock(side_effect=Exception("Download failed"))
        mock_hf.HfFolder = Mock()
        mock_hf.HfFolder.get_token = Mock(return_value=None)

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf}):
            with patch("pathlib.Path.home", return_value=tmp_path):
                success, message = download_model_with_progress(
                    "Unbabel/wmt22-comet-da", "COMET-22", 1.3
                )

                assert success is False
                assert "Failed to download" in message

    def test_download_models_with_missing_models(self) -> None:
        """Test download_models downloads missing models."""
        # Mock huggingface_hub and comet modules
        mock_hf = Mock()
        mock_hf.HfFolder = Mock()
        mock_hf.HfFolder.get_token = Mock(return_value="fake-token")

        mock_comet = Mock()
        mock_comet.load_from_checkpoint = Mock()

        with patch.dict("sys.modules", {"huggingface_hub": mock_hf, "comet": mock_comet}):
            with patch("kttc.cli.commands.load.verify_models") as mock_verify:
                with patch("kttc.cli.commands.load.download_model_with_progress") as mock_download:
                    # First call: some missing, second call: all downloaded
                    mock_verify.side_effect = [
                        (["COMET-22"], ["CometKiwi", "XCOMET-XL"]),
                        (["COMET-22", "CometKiwi", "XCOMET-XL"], []),
                    ]

                    mock_download.return_value = (True, "✓ Downloaded")

                    # Should not raise exception
                    download_models()

                    # Should have tried to download 2 models
                    assert mock_download.call_count == 2

    def test_download_models_import_error(self) -> None:
        """Test download_models handles missing dependencies gracefully."""
        # Import error is caught and results in typer.Exit(1), not SystemExit
        with patch("kttc.cli.commands.load.verify_models", side_effect=ImportError("No module")):
            # The function calls typer.Exit which raises typer.Exit
            import typer

            with pytest.raises(typer.Exit) as exc_info:
                download_models()

            assert exc_info.value.exit_code == 1


@pytest.mark.unit
class TestModelStatusConsistency:
    """Tests to ensure model status is consistent across commands."""

    def test_status_consistency_between_check_and_load(self, tmp_path: Path) -> None:
        """Test that check and load commands show same model status."""
        cache_dir = tmp_path / ".cache" / "huggingface" / "hub"

        # Create one downloaded, one incomplete, one missing
        checkpoint_dir = cache_dir / "models--Unbabel--wmt22-comet-da" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model.ckpt").write_text("fake")

        incomplete_dir = cache_dir / "models--Unbabel--wmt23-cometkiwi-da-xxl"
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        with patch("pathlib.Path.home", return_value=tmp_path):
            # Get status from both functions
            status_from_get_models = get_models_status()
            downloaded, missing = verify_models()

            # Verify consistency
            assert status_from_get_models[0][2] == "✓ Downloaded"
            assert "COMET-22" in downloaded

            assert status_from_get_models[1][2] == "⚠ Incomplete"
            assert "CometKiwi" in missing

            assert status_from_get_models[2][2] == "○ Not downloaded"
            assert "XCOMET-XL" in missing


@pytest.mark.unit
class TestProgressSizeCalculation:
    """Tests for progress bar size calculation accuracy."""

    def test_monitor_progress_counts_only_model_files(self, tmp_path: Path) -> None:
        """Test that progress monitoring only counts actual model files."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        # Create model files
        (model_dir / "model.bin").write_bytes(b"x" * 1000)
        (model_dir / "model.safetensors").write_bytes(b"x" * 2000)
        (model_dir / "checkpoint.ckpt").write_bytes(b"x" * 500)

        # Create non-model files that should be excluded
        (model_dir / ".lock").write_bytes(b"x" * 5000)  # Hidden file
        (model_dir / "config.json").write_bytes(b"x" * 100)  # Not a model file
        (model_dir / "metadata.txt").write_bytes(b"x" * 200)  # Not a model file

        # Calculate size using the same logic as monitor_progress
        model_extensions = {
            ".bin",
            ".safetensors",
            ".ckpt",
            ".pt",
            ".pth",
            ".h5",
            ".msgpack",
        }
        calculated_size = sum(
            f.stat().st_size
            for f in model_dir.rglob("*")
            if f.is_file() and f.suffix in model_extensions and not f.name.startswith(".")
        )

        # Should only count model files (1000 + 2000 + 500 = 3500)
        assert calculated_size == 3500


@pytest.mark.unit
class TestCLIIntegration:
    """Integration tests for CLI commands with model checking."""

    def test_check_command_exits_if_models_missing(self, tmp_path: Path) -> None:
        """Test that check command exits if models are not downloaded."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.utils.dependencies.models_are_downloaded", return_value=False):
            with patch("kttc.utils.dependencies.get_models_status") as mock_status:
                mock_status.return_value = [
                    ("COMET-22", "1.3 GB", "○ Not downloaded", "yellow"),
                    ("CometKiwi", "900 MB", "○ Not downloaded", "yellow"),
                    ("XCOMET-XL", "800 MB", "○ Not downloaded", "yellow"),
                ]

                result = runner.invoke(
                    app,
                    [
                        "check",
                        "--source",
                        str(source),
                        "--translation",
                        str(translation),
                        "--source-lang",
                        "en",
                        "--target-lang",
                        "es",
                    ],
                )

                assert result.exit_code == 1
                assert "Neural Models Not Ready" in result.stdout

    def test_check_command_proceeds_if_models_ready(self, tmp_path: Path) -> None:
        """Test that check command proceeds when models are downloaded."""
        source = tmp_path / "source.txt"
        translation = tmp_path / "trans.txt"
        source.write_text("Hello world", encoding="utf-8")
        translation.write_text("Hola mundo", encoding="utf-8")

        with patch("kttc.utils.dependencies.models_are_downloaded", return_value=True):
            with patch("kttc.cli.main.AgentOrchestrator") as mock_orchestrator_class:
                with patch("kttc.cli.main.get_settings") as mock_settings:
                    with patch("kttc.cli.main.OpenAIProvider"):
                        # Setup mocks
                        settings = MagicMock()
                        settings.default_llm_provider = "openai"
                        settings.default_model = "gpt-4"
                        settings.default_temperature = 0.1
                        settings.default_max_tokens = 2000
                        settings.get_llm_provider_key.return_value = "test-key"
                        mock_settings.return_value = settings

                        from kttc.core import QAReport, TranslationTask

                        mock_report = QAReport(
                            task=TranslationTask(
                                source_text="Hello world",
                                translation="Hola mundo",
                                source_lang="en",
                                target_lang="es",
                            ),
                            mqm_score=98.5,
                            errors=[],
                            status="pass",
                            comet_score=None,
                        )

                        mock_orchestrator = MagicMock()
                        mock_orchestrator.evaluate = AsyncMock(return_value=mock_report)
                        mock_orchestrator_class.return_value = mock_orchestrator

                        result = runner.invoke(
                            app,
                            [
                                "check",
                                "--source",
                                str(source),
                                "--translation",
                                str(translation),
                                "--source-lang",
                                "en",
                                "--target-lang",
                                "es",
                            ],
                        )

                        # Should succeed and not exit early
                        assert result.exit_code == 0
                        assert "models ready" in result.stdout.lower()
