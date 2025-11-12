# Copyright 2025 KTTC AI (https://github.com/kttc-ai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load neural models command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

from kttc.cli.ui import console, print_error, print_info, print_success, print_warning


class SmartDownloadColumn(ProgressColumn):
    """Custom download column that shows units correctly (bytes vs files vs items).

    Unlike Rich's DownloadColumn which always shows 'bytes', this column
    adapts based on the task's unit field stored in our RichProgressTqdm class.
    """

    def render(self, task: Task) -> Text:
        """Render the download progress.

        Args:
            task: Rich Task object

        Returns:
            Formatted text showing progress with correct units
        """
        # Check if this is a file counting task (0-10 range, "it" unit)
        # or a byte download task (large numbers, "B" unit)
        completed = int(task.completed)
        total = int(task.total) if task.total else 0

        # Heuristic: if total < 100 and desc contains "Fetching", it's file counting
        is_file_count = total > 0 and total < 100 and "Fetching" in (task.description or "")

        if is_file_count:
            # Show as "N/M files"
            return Text(f"{completed}/{total} files", style="progress.download")
        elif total > 0:
            # Show as bytes with proper formatting
            download_col = DownloadColumn()
            return download_col.render(task)
        else:
            # Indeterminate progress
            return Text("", style="progress.download")


class SmartSpeedColumn(ProgressColumn):
    """Custom speed column that only shows for byte downloads, not file counting."""

    def render(self, task: Task) -> Text:
        """Render the transfer speed.

        Args:
            task: Rich Task object

        Returns:
            Formatted text showing speed or empty for file counting
        """
        total = int(task.total) if task.total else 0
        is_file_count = total > 0 and total < 100 and "Fetching" in (task.description or "")

        if is_file_count:
            # Don't show speed for file counting
            return Text("")
        else:
            # Show speed for byte downloads
            speed_col = TransferSpeedColumn()
            return speed_col.render(task)


class RichProgressTqdm:
    """Custom tqdm-compatible class that integrates HuggingFace downloads with Rich progress.

    This class mimics tqdm's interface but uses Rich's Progress for better visual display.
    HuggingFace's snapshot_download will call this class for each file being downloaded.
    """

    _progress: Progress | None = None
    _active_tasks: dict[str, TaskID] = {}
    _lock: Any | None = None  # For tqdm compatibility

    def __init__(
        self,
        iterable: Any = None,
        desc: str | None = None,
        total: int | float | None = None,
        leave: bool = True,
        file: Any = None,
        ncols: int | None = None,
        mininterval: float = 0.1,
        maxinterval: float = 10.0,
        miniters: int | float | None = None,
        ascii: bool | str | None = None,
        disable: bool = False,
        unit: str = "it",
        unit_scale: bool | int | float = False,
        dynamic_ncols: bool = False,
        smoothing: float = 0.3,
        bar_format: str | None = None,
        initial: int | float = 0,
        position: int | None = None,
        postfix: dict[str, Any] | None = None,
        unit_divisor: int = 1000,
        write_bytes: bool = False,
        lock_args: tuple[bool | None, float | None] | None = None,
        nrows: int | None = None,
        colour: str | None = None,
        delay: float = 0,
        gui: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize tqdm-compatible progress tracker.

        Args:
            desc: Description to display (e.g., "Downloading config.json")
            total: Total iterations/bytes expected
            disable: Whether to disable this progress bar
            unit: Unit of measurement (e.g., "B" for bytes)
            unit_scale: Whether to auto-scale the unit
            initial: Initial counter value
            **kwargs: Other tqdm parameters (ignored for compatibility)
        """
        self.desc = desc or ""
        self.total = total
        self.disable = disable
        self.n = initial  # Current progress value
        self.unit = unit
        self.unit_scale = unit_scale
        self.task_id: TaskID | None = None

        # Only create progress bar if not disabled and we have a Rich Progress instance
        if not disable and RichProgressTqdm._progress is not None:
            # Format description - handle file downloads vs file counting
            if self.desc:
                display_desc = f"[cyan]{self.desc}"
            else:
                display_desc = "[cyan]Downloading..."

            # Create a unique task for this download
            self.task_id = RichProgressTqdm._progress.add_task(
                display_desc,
                total=total if total and total > 0 else None,
                completed=int(initial),
            )

            # Track task by description (if we have one)
            task_key = self.desc if self.desc else f"task_{id(self)}"
            RichProgressTqdm._active_tasks[task_key] = self.task_id

    def update(self, n: int | float = 1) -> None:
        """Update progress by n units.

        Args:
            n: Number of units to advance (default: 1)
        """
        if not self.disable and self.task_id is not None and RichProgressTqdm._progress is not None:
            self.n += n
            RichProgressTqdm._progress.update(self.task_id, advance=n)

    def close(self) -> None:
        """Close the progress bar."""
        if self.task_id is not None and RichProgressTqdm._progress is not None:
            # Mark task as completed
            if self.total is not None and self.total > 0:
                RichProgressTqdm._progress.update(self.task_id, completed=self.total)

            # Remove from active tasks - find and remove by task_id
            task_key_to_remove = None
            for key, task_id in RichProgressTqdm._active_tasks.items():
                if task_id == self.task_id:
                    task_key_to_remove = key
                    break
            if task_key_to_remove:
                del RichProgressTqdm._active_tasks[task_key_to_remove]

    def set_description(self, desc: str, refresh: bool = True) -> None:
        """Update the description.

        Args:
            desc: New description
            refresh: Whether to refresh display (ignored, Rich auto-refreshes)
        """
        self.desc = desc
        if self.task_id is not None and RichProgressTqdm._progress is not None:
            RichProgressTqdm._progress.update(self.task_id, description=f"[cyan]{desc}")

    def set_postfix(
        self, ordered_dict: dict[str, Any] | None = None, refresh: bool = True, **kwargs: Any
    ) -> None:
        """Set postfix metadata (tqdm compatibility - not displayed in Rich).

        Args:
            ordered_dict: Dictionary of postfix items
            refresh: Whether to refresh display (ignored)
            **kwargs: Additional postfix items
        """
        pass  # Rich doesn't support postfix, but we need this for tqdm compatibility

    def __enter__(self) -> RichProgressTqdm:
        """Context manager entry."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Context manager exit."""
        self.close()

    def __iter__(self) -> Any:
        """Iterate over wrapped iterable (not used by HuggingFace)."""
        return iter([])

    @classmethod
    def set_progress(cls, progress: Progress) -> None:
        """Set the Rich Progress instance to use for all progress bars.

        Args:
            progress: Rich Progress instance
        """
        cls._progress = progress
        cls._active_tasks = {}

    @classmethod
    def clear_progress(cls) -> None:
        """Clear the Rich Progress instance."""
        cls._progress = None
        cls._active_tasks = {}

    @classmethod
    def get_lock(cls) -> Any:
        """Get lock for tqdm compatibility.

        Returns:
            A lock-like object (we use a dummy since Rich handles thread safety)
        """
        if cls._lock is None:
            import threading

            cls._lock = threading.RLock()
        return cls._lock

    @classmethod
    def set_lock(cls, lock: Any) -> None:
        """Set lock for tqdm compatibility.

        Args:
            lock: Lock object to use
        """
        cls._lock = lock

    @classmethod
    def write(cls, s: str, file: Any = None, end: str = "\n", nolock: bool = False) -> None:
        """Write message (tqdm compatibility).

        Args:
            s: String to write
            file: File to write to (ignored, we use console)
            end: Line ending
            nolock: Whether to skip locking
        """
        # Use Rich console for output
        if cls._progress is not None:
            console.print(s, end=end)

    @staticmethod
    def format_sizeof(num: int | float, suffix: str = "B", divisor: float = 1024.0) -> str:
        """Format bytes as human-readable size (tqdm compatibility).

        Args:
            num: Number of bytes
            suffix: Suffix to append (default: "B")
            divisor: Divisor for unit conversion (default: 1024)

        Returns:
            Formatted string like "1.5 MB"
        """
        for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if abs(num) < divisor:
                return f"{num:3.1f} {unit}{suffix}"
            num /= divisor
        return f"{num:.1f} Y{suffix}"


def check_model_exists(model_name: str) -> bool:
    """Check if model is already downloaded and valid.

    Args:
        model_name: HuggingFace model name (e.g., "Unbabel/wmt22-comet-da")

    Returns:
        True if model exists and has checkpoint file, False otherwise
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # HuggingFace converts / to -- and adds "models--" prefix
    model_dir_name = f"models--{model_name.replace('/', '--')}"
    model_path = cache_dir / model_dir_name

    if not model_path.exists():
        return False

    # Check if checkpoint file exists (validates model is complete)
    checkpoint_path = model_path / "checkpoints" / "model.ckpt"
    if not checkpoint_path.exists():
        # Try snapshots directory structure (newer HF format)
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists():
            # Check if any snapshot has the checkpoint
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    ckpt = snapshot / "checkpoints" / "model.ckpt"
                    if ckpt.exists():
                        return True
        return False

    return True


def get_model_size(model_path: Path) -> int:
    """Get total size of downloaded model in bytes.

    Args:
        model_path: Path to model directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    for item in model_path.rglob("*"):
        if item.is_file():
            total_size += item.stat().st_size
    return total_size


def download_model_with_progress(
    model_name: str, description: str, expected_size_gb: float
) -> tuple[bool, str]:
    """Download a single model with progress tracking.

    Args:
        model_name: HuggingFace model name
        description: Human-readable description
        expected_size_gb: Expected size in GB for informational purposes

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Monkey-patch tqdm.auto.tqdm to use our Rich progress bars
        # This is necessary because HuggingFace doesn't pass tqdm_class to individual file downloads
        import tqdm.auto

        original_tqdm = tqdm.auto.tqdm

        # Temporarily replace tqdm with our custom class
        tqdm.auto.tqdm = RichProgressTqdm  # type: ignore

        try:
            from huggingface_hub import HfFolder, snapshot_download

            # Get HF token if available (required for gated models)
            hf_token = HfFolder.get_token()

            # Download - will use our RichProgressTqdm for ALL progress bars
            # IMPORTANT: allow_patterns=None ensures ALL files are downloaded including large checkpoints
            snapshot_download(
                repo_id=model_name,
                local_files_only=False,
                allow_patterns=None,  # Download all files (don't skip large ones)
                ignore_patterns=None,  # Don't ignore any files
                token=hf_token,  # Use token for gated models
            )
        finally:
            # Restore original tqdm
            tqdm.auto.tqdm = original_tqdm

        return True, f"✓ {description}"

    except Exception as e:
        return False, f"Failed to download {model_name}: {e}"


def verify_models() -> tuple[list[str], list[str]]:
    """Verify which models are downloaded and which are missing.

    Returns:
        Tuple of (downloaded_models, missing_models)
    """
    from kttc.utils.dependencies import get_models_status

    models_status = get_models_status()

    downloaded = []
    missing = []

    for display_name, _, status_text, _ in models_status:
        if "✓ Downloaded" in status_text:
            downloaded.append(display_name)
        else:
            missing.append(display_name)

    return downloaded, missing


def download_models() -> None:
    """Download all required neural models for metrics and benchmarks."""
    try:
        # Check if user is authenticated with HuggingFace
        # Required for gated models (CometKiwi, XCOMET-XL)
        try:
            from huggingface_hub import HfFolder

            hf_token = HfFolder.get_token()
            if not hf_token:
                console.print()
                console.print(
                    Panel(
                        "[bold yellow]⚠ HuggingFace Authentication Required[/bold yellow]\n\n"
                        "Some models (CometKiwi, XCOMET-XL) are gated and require authentication.\n\n"
                        "[bold]To authenticate:[/bold]\n"
                        "1. Create account at https://huggingface.co/\n"
                        "2. Accept model licenses:\n"
                        "   • https://huggingface.co/Unbabel/wmt23-cometkiwi-da-xxl\n"
                        "   • https://huggingface.co/Unbabel/XCOMET-XL\n"
                        "3. Get access token: https://huggingface.co/settings/tokens\n"
                        "4. Run: [cyan]huggingface-cli login[/cyan]\n\n"
                        "[dim]Note: COMET-22 can be downloaded without authentication[/dim]",
                        title="Authentication Required",
                        border_style="yellow",
                    )
                )
                console.print()
                print_warning("Please authenticate and try again.")
                raise typer.Exit(1)
        except ImportError:
            pass  # huggingface_hub not installed, will fail later anyway

        # Check current status
        downloaded, missing = verify_models()

        if not missing:
            console.print()
            console.print(
                Panel(
                    "[bold green]✓ All Models Already Downloaded[/bold green]\n\n"
                    "The following models are ready:\n"
                    "• COMET-22 (~1.3GB)\n"
                    "• CometKiwi (~900MB)\n"
                    "• XCOMET-XL (~800MB)\n\n"
                    "[dim]Models are cached in ~/.cache/huggingface/[/dim]",
                    title="Models Ready",
                    border_style="green",
                )
            )
            console.print()
            console.print("[bold green]Ready to use![/bold green] You can now run:")
            console.print("  • kttc check <file>", style="cyan")
            console.print("  • kttc compare <file1> <file2>", style="cyan")
            console.print("  • kttc benchmark", style="cyan")
            console.print()
            return

        # Show what will be downloaded
        console.print()
        status_table = Table(show_header=True, header_style="bold cyan", box=None)
        status_table.add_column("Model", style="bold")
        status_table.add_column("Status", justify="center")

        for model in ["COMET-22", "CometKiwi", "XCOMET-XL"]:
            status = (
                "[green]✓ Downloaded[/green]"
                if model in downloaded
                else "[yellow]○ Missing[/yellow]"
            )
            status_table.add_row(model, status)

        console.print(status_table)
        console.print()

        console.print(
            Panel(
                "[bold cyan]🔽 Downloading Neural Quality Models[/bold cyan]\n\n"
                f"Will download {len(missing)} model(s):\n"
                + "\n".join(f"• {m}" for m in missing)
                + "\n\n[bold]Total: ~3GB[/bold]\n\n"
                "[dim]Models will be cached in ~/.cache/huggingface/\n"
                "Downloads can be resumed if interrupted.[/dim]",
                title="Download Required",
                border_style="cyan",
            )
        )
        console.print()

        # Download models with progress
        models_to_download = []
        if "COMET-22" in missing:
            models_to_download.append(("Unbabel/wmt22-comet-da", "COMET-22", 1.3))
        if "CometKiwi" in missing:
            models_to_download.append(("Unbabel/wmt23-cometkiwi-da-xxl", "CometKiwi", 0.9))
        if "XCOMET-XL" in missing:
            models_to_download.append(("Unbabel/XCOMET-XL", "XCOMET-XL", 0.8))

        # Create Rich progress context and inject it into our custom tqdm class
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            SmartDownloadColumn(),  # Our custom column that handles files vs bytes
            SmartSpeedColumn(),  # Our custom column that hides speed for file counting
            TimeRemainingColumn(),
            console=console,
            transient=False,  # Keep progress bars visible after completion
        ) as progress:
            # Set the progress instance for our custom tqdm class
            RichProgressTqdm.set_progress(progress)

            try:
                results = []
                for model_name, description, size_gb in models_to_download:
                    console.print(f"\n[bold cyan]Downloading {description}...[/bold cyan]")
                    success, message = download_model_with_progress(
                        model_name, description, size_gb
                    )
                    results.append((success, message))
            finally:
                # Clean up the progress reference
                RichProgressTqdm.clear_progress()

        console.print()

        # Show results
        all_success = all(success for success, _ in results)

        if all_success:
            print_success("All models downloaded successfully!")
            console.print()

            # Validate by loading models
            console.print("[dim]Validating models...[/dim]")
            try:
                # Try to load models to verify they work
                from comet import load_from_checkpoint

                for model_name, desc, _ in models_to_download:
                    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                    model_dir = f"models--{model_name.replace('/', '--')}"
                    model_path = cache_dir / model_dir

                    # Find checkpoint - try both old and new HuggingFace formats
                    checkpoint_path: Path | None = None

                    # Try old format: checkpoints/model.ckpt
                    old_format = model_path / "checkpoints" / "model.ckpt"
                    if old_format.exists():
                        checkpoint_path = old_format
                    else:
                        # Try new format: snapshots/<hash>/checkpoints/model.ckpt
                        snapshots_dir = model_path / "snapshots"
                        if snapshots_dir.exists():
                            for snapshot in snapshots_dir.iterdir():
                                if snapshot.is_dir():
                                    ckpt = snapshot / "checkpoints" / "model.ckpt"
                                    if ckpt.exists():
                                        checkpoint_path = ckpt
                                        break

                    # Skip validation if checkpoint not found
                    if checkpoint_path is None or not checkpoint_path.exists():
                        console.print(
                            f"[yellow]⚠[/yellow] [dim]Skipping validation for {desc} "
                            f"(checkpoint not found at expected location)[/dim]"
                        )
                        continue

                    # Verify can load
                    load_from_checkpoint(str(checkpoint_path))

                print_success("Models validated successfully!")
            except Exception as e:
                print_warning(f"Models downloaded but validation failed: {e}")
                print_info("You can still try using them.")

            console.print()
            console.print("[bold green]Ready to use![/bold green] You can now run:")
            console.print("  • kttc check <file>", style="cyan")
            console.print("  • kttc compare <file1> <file2>", style="cyan")
            console.print("  • kttc benchmark", style="cyan")
            console.print()
        else:
            print_error("Some models failed to download:")
            for success, message in results:
                if not success:
                    console.print(f"  [red]{message}[/red]")
            console.print()
            print_info("You can run 'kttc load' again to retry.")
            raise typer.Exit(1)

    except ImportError:
        print_error("Neural metrics dependencies not installed.")
        console.print()
        console.print("This should not happen. Please report this issue.")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Failed to download models: {e}")
        console.print()
        console.print_exception()
        raise typer.Exit(1)


def run_load() -> None:
    """Run the load command synchronously."""
    download_models()
