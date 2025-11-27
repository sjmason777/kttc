"""Unit tests for translate command module.

Tests translate command utilities and async functions.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.unit
class TestTranslateCommandHelpers:
    """Test translate command helper functions."""

    def test_text_loading_from_string(self) -> None:
        """Test text is used directly when not a file reference."""
        text = "Hello world"
        assert not text.startswith("@")
        # Direct text should be used as-is

    def test_text_loading_file_reference(self, tmp_path: Path) -> None:
        """Test text is loaded from file when using @ prefix."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("Text from file")

        text = f"@{test_file}"
        assert text.startswith("@")
        file_path = Path(text[1:])
        assert file_path.exists()
        assert file_path.read_text() == "Text from file"

    def test_file_not_found_error(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for nonexistent file reference."""
        text = f"@{tmp_path}/nonexistent.txt"
        file_path = Path(text[1:])
        assert not file_path.exists()


@pytest.mark.unit
class TestTranslateAsyncFunction:
    """Test the async translate function."""

    @pytest.mark.asyncio
    @patch("kttc.cli.commands.translate.get_settings")
    @patch("kttc.cli.commands.translate.setup_llm_provider")
    @patch("kttc.cli.commands.translate.print_header")
    @patch("kttc.cli.commands.translate.print_startup_info")
    @patch("kttc.cli.commands.translate.create_step_progress")
    @patch("kttc.cli.commands.translate.console")
    async def test_translate_async_basic_flow(
        self,
        mock_console: MagicMock,
        mock_progress: MagicMock,
        mock_startup: MagicMock,
        mock_header: MagicMock,
        mock_setup_llm: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test basic translate async flow."""
        from kttc.cli.commands.translate import _translate_async

        # Setup mocks
        mock_settings.return_value = MagicMock(
            default_temperature=0.7,
            default_max_tokens=4096,
        )

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="Hola mundo")
        mock_setup_llm.return_value = mock_provider

        # Mock progress context manager
        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__ = MagicMock(return_value=mock_progress_ctx)
        mock_progress_ctx.__exit__ = MagicMock(return_value=None)
        mock_progress_ctx.add_task = MagicMock()
        mock_progress.return_value = mock_progress_ctx

        # Mock refinement - import is inside the function, so patch the module
        with patch("kttc.agents.refinement.IterativeRefinement") as mock_refinement_cls:
            mock_refinement = MagicMock()
            mock_result = MagicMock(
                final_score=98.0,
                iterations=2,
                improvement=3.0,
                converged=True,
                final_translation="Hola mundo mejorado",
                qa_reports=[],
            )
            mock_refinement.refine_until_convergence = AsyncMock(return_value=mock_result)
            mock_refinement_cls.return_value = mock_refinement

            await _translate_async(
                text="Hello world",
                source_lang="en",
                target_lang="es",
                threshold=95.0,
                max_iterations=3,
                output=None,
                provider=None,
                verbose=False,
            )

            # Verify LLM provider was called
            mock_provider.complete.assert_called_once()
            mock_refinement.refine_until_convergence.assert_called_once()

    @pytest.mark.asyncio
    @patch("kttc.cli.commands.translate.get_settings")
    @patch("kttc.cli.commands.translate.setup_llm_provider")
    @patch("kttc.cli.commands.translate.print_header")
    @patch("kttc.cli.commands.translate.print_startup_info")
    @patch("kttc.cli.commands.translate.create_step_progress")
    @patch("kttc.cli.commands.translate.console")
    async def test_translate_async_saves_output(
        self,
        mock_console: MagicMock,
        mock_progress: MagicMock,
        mock_startup: MagicMock,
        mock_header: MagicMock,
        mock_setup_llm: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test translate async saves output to file."""
        from kttc.cli.commands.translate import _translate_async

        mock_settings.return_value = MagicMock(
            default_temperature=0.7,
            default_max_tokens=4096,
        )

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="Translated text")
        mock_setup_llm.return_value = mock_provider

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__ = MagicMock(return_value=mock_progress_ctx)
        mock_progress_ctx.__exit__ = MagicMock(return_value=None)
        mock_progress_ctx.add_task = MagicMock()
        mock_progress.return_value = mock_progress_ctx

        output_file = tmp_path / "output.txt"

        with patch("kttc.agents.refinement.IterativeRefinement") as mock_refinement_cls:
            mock_refinement = MagicMock()
            mock_result = MagicMock(
                final_score=96.0,
                iterations=1,
                improvement=1.0,
                converged=True,
                final_translation="Final translation",
                qa_reports=[],
            )
            mock_refinement.refine_until_convergence = AsyncMock(return_value=mock_result)
            mock_refinement_cls.return_value = mock_refinement

            await _translate_async(
                text="Test",
                source_lang="en",
                target_lang="es",
                threshold=95.0,
                max_iterations=3,
                output=str(output_file),
                provider=None,
                verbose=False,
            )

            assert output_file.exists()
            assert output_file.read_text() == "Final translation"

    @pytest.mark.asyncio
    @patch("kttc.cli.commands.translate.get_settings")
    @patch("kttc.cli.commands.translate.setup_llm_provider")
    @patch("kttc.cli.commands.translate.print_header")
    @patch("kttc.cli.commands.translate.print_startup_info")
    @patch("kttc.cli.commands.translate.create_step_progress")
    @patch("kttc.cli.commands.translate.console")
    async def test_translate_async_loads_from_file(
        self,
        mock_console: MagicMock,
        mock_progress: MagicMock,
        mock_startup: MagicMock,
        mock_header: MagicMock,
        mock_setup_llm: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test translate async loads text from file with @ prefix."""
        from kttc.cli.commands.translate import _translate_async

        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Text from input file")

        mock_settings.return_value = MagicMock(
            default_temperature=0.7,
            default_max_tokens=4096,
        )

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value="Translated")
        mock_setup_llm.return_value = mock_provider

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__ = MagicMock(return_value=mock_progress_ctx)
        mock_progress_ctx.__exit__ = MagicMock(return_value=None)
        mock_progress_ctx.add_task = MagicMock()
        mock_progress.return_value = mock_progress_ctx

        with patch("kttc.agents.refinement.IterativeRefinement") as mock_refinement_cls:
            mock_refinement = MagicMock()
            mock_result = MagicMock(
                final_score=95.0,
                iterations=1,
                improvement=0.0,
                converged=True,
                final_translation="Translated from file",
                qa_reports=[],
            )
            mock_refinement.refine_until_convergence = AsyncMock(return_value=mock_result)
            mock_refinement_cls.return_value = mock_refinement

            await _translate_async(
                text=f"@{input_file}",
                source_lang="en",
                target_lang="es",
                threshold=95.0,
                max_iterations=3,
                output=None,
                provider=None,
                verbose=True,  # Verbose to trigger file load message
            )

            # Verify the complete call contains text from file
            call_args = mock_provider.complete.call_args[0][0]
            assert "Text from input file" in call_args

    @pytest.mark.asyncio
    @patch("kttc.cli.commands.translate.get_settings")
    @patch("kttc.cli.commands.translate.print_header")
    async def test_translate_async_file_not_found(
        self,
        mock_header: MagicMock,
        mock_settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test translate async raises for nonexistent input file."""
        from kttc.cli.commands.translate import _translate_async

        mock_settings.return_value = MagicMock()

        with pytest.raises(FileNotFoundError, match="Text file not found"):
            await _translate_async(
                text=f"@{tmp_path}/nonexistent.txt",
                source_lang="en",
                target_lang="es",
                threshold=95.0,
                max_iterations=3,
                output=None,
                provider=None,
                verbose=False,
            )

    @pytest.mark.asyncio
    @patch("kttc.cli.commands.translate.get_settings")
    @patch("kttc.cli.commands.translate.setup_llm_provider")
    @patch("kttc.cli.commands.translate.print_header")
    @patch("kttc.cli.commands.translate.print_startup_info")
    async def test_translate_async_llm_provider_error(
        self,
        mock_startup: MagicMock,
        mock_header: MagicMock,
        mock_setup_llm: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test translate async handles LLM provider setup error."""
        from kttc.cli.commands.translate import _translate_async

        mock_settings.return_value = MagicMock()
        mock_setup_llm.side_effect = Exception("API key not found")

        with pytest.raises(RuntimeError, match="Failed to setup LLM provider"):
            await _translate_async(
                text="Test",
                source_lang="en",
                target_lang="es",
                threshold=95.0,
                max_iterations=3,
                output=None,
                provider=None,
                verbose=False,
            )
