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

"""Dependency management and lazy loading for optional packages.

This module handles:
- Checking if optional dependencies are installed
- Prompting user to install missing packages
- Auto-installation with progress tracking
- Graceful degradation when packages unavailable
"""

from __future__ import annotations

import subprocess  # nosec B404 - required for pip install functionality
import sys
from typing import Literal

from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm

from kttc.utils.console import console, print_error, print_info, print_success, print_warning

# Optional dependency groups
DependencyGroup = Literal["metrics", "webui", "benchmark"]

DEPENDENCY_GROUPS: dict[DependencyGroup, dict[str, str]] = {
    "metrics": {
        "sentence-transformers": "Sentence embeddings for semantic similarity (~500MB)",
        "sacrebleu": "BLEU and other classical metrics (lightweight)",
    },
    "webui": {
        "fastapi": "Web framework for UI server",
        "uvicorn": "ASGI server for FastAPI",
        "websockets": "WebSocket support for real-time updates",
    },
    "benchmark": {
        "sentence-transformers": "Sentence embeddings (~500MB)",
        "sacrebleu": "Classical metrics (lightweight)",
        "datasets": "Hugging Face datasets for benchmarking",
        "numpy": "Numerical computing for statistics",
    },
}


def check_package_installed(package_name: str) -> bool:
    """Check if a Python package is installed.

    Args:
        package_name: Name of the package (e.g., 'unbabel-comet')

    Returns:
        True if installed, False otherwise
    """
    try:
        # Handle package name mapping (pip name vs import name)
        import_name = package_name.replace("-", "_")
        __import__(import_name)
        return True
    except ImportError:
        return False


def check_dependency_group(group: DependencyGroup) -> tuple[bool, list[str]]:
    """Check if all packages in a dependency group are installed.

    Args:
        group: Dependency group name

    Returns:
        Tuple of (all_installed, missing_packages)
    """
    packages = DEPENDENCY_GROUPS.get(group, {})
    missing = []

    for package in packages:
        if not check_package_installed(package):
            missing.append(package)

    return len(missing) == 0, missing


def show_missing_dependencies_prompt(
    group: DependencyGroup, missing: list[str], command: str
) -> bool:
    """Show beautiful prompt about missing dependencies.

    Args:
        group: Dependency group name
        missing: List of missing package names
        command: Command that requires these dependencies

    Returns:
        True if user wants to install, False otherwise
    """
    packages_info = DEPENDENCY_GROUPS[group]

    # Build description lines
    lines = [
        f"The '[cyan]{command}[/cyan]' command requires additional packages.",
        "",
        "[bold]Missing packages:[/bold]",
    ]

    total_size = 0
    for pkg in missing:
        desc = packages_info.get(pkg, "Required dependency")
        lines.append(f"  • [cyan]{pkg}[/cyan] - {desc}")

        # Estimate size
        if "2GB" in desc:
            total_size += 2000
        elif "500MB" in desc:
            total_size += 500
        elif "lightweight" in desc or pkg in ["sacrebleu", "numpy"]:
            total_size += 50

    lines.extend(
        [
            "",
            f"[bold]Estimated download size:[/bold] ~{total_size}MB",
            "",
            "[bold]Installation options:[/bold]",
            "  1. Auto-install now (recommended)",
            f"  2. Manual: [dim]pip install kttc[{group}][/dim]",
            "  3. Skip (limited functionality)",
        ]
    )

    # Show panel
    panel = Panel(
        "\n".join(lines),
        title="📦 Missing Dependencies",
        border_style="yellow",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()

    # Ask user
    return bool(Confirm.ask("Install missing dependencies now?", default=True))


def install_dependency_group(group: DependencyGroup) -> bool:
    """Install a dependency group with progress tracking.

    Args:
        group: Dependency group to install

    Returns:
        True if installation succeeded, False otherwise
    """
    print_info(f"Installing '{group}' dependencies...")
    console.print()

    try:
        # Build pip command
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"kttc[{group}]",
            "--quiet",  # Reduce pip output
        ]

        # Run installation with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Installing {group} packages...",
                total=None,  # Indeterminate progress
            )

            # Run pip install (cmd uses sys.executable + enum-controlled group)
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            check=True)

            progress.update(task, completed=True)

        if result.returncode == 0:
            console.print()
            print_success(f"Successfully installed {group} dependencies!")
            return True
        console.print()
        print_error(f"Installation failed: {result.stderr}")
        return False

    except subprocess.TimeoutExpired:
        console.print()
        print_error("Installation timed out. Please check your internet connection.")
        return False
    except Exception as e:
        console.print()
        print_error(f"Installation error: {e}")
        return False


def ensure_dependency_group(group: DependencyGroup, command: str, required: bool = True) -> bool:
    """Ensure a dependency group is installed, prompting user if needed.

    Args:
        group: Dependency group name
        command: Command that requires these dependencies
        required: If True, exit if dependencies not installed

    Returns:
        True if dependencies available, False otherwise
    """
    # Check if already installed
    all_installed, missing = check_dependency_group(group)

    if all_installed:
        return True

    # Show prompt
    if show_missing_dependencies_prompt(group, missing, command):
        # User wants to install
        success = install_dependency_group(group)

        if success:
            return True
        if required:
            print_error("Cannot proceed without required dependencies.")
            sys.exit(1)
        return False
    # User declined installation
    if required:
        print_warning("Command requires additional dependencies.")
        print_info(f"Install manually: pip install kttc[{group}]")
        sys.exit(1)
    else:
        print_warning("Proceeding with limited functionality...")
        return False


def show_optimization_tips() -> None:
    """Show tips for faster installation and better performance."""
    tips = [
        "[bold]💡 Installation Tips:[/bold]",
        "",
        "• [cyan]Slow install?[/cyan] Models are large (~2-3GB total)",
        "• [cyan]Use cache:[/cyan] Models download once, then cached",
        "• [cyan]Offline mode:[/cyan] Models work offline after first download",
        "• [cyan]Faster install:[/cyan] Use pip with --no-cache-dir if space limited",
        "",
        "[bold]Performance:[/bold]",
        "• First run downloads models (~5-10 min on slow connection)",
        "• Subsequent runs use cached models (instant)",
        "• Models stored in: ~/.cache/huggingface/",
    ]

    panel = Panel(
        "\n".join(tips),
        title="ℹ️ Tips",
        border_style="blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


# Convenience functions for specific features


def has_webui() -> bool:
    """Check if webui dependencies are available."""
    all_installed, _ = check_dependency_group("webui")
    return all_installed


def has_benchmark() -> bool:
    """Check if benchmark dependencies are available."""
    all_installed, _ = check_dependency_group("benchmark")
    return all_installed


def require_webui(command: str) -> None:
    """Ensure webui dependencies are installed or exit.

    Args:
        command: Name of the command requiring webui
    """
    ensure_dependency_group("webui", command, required=True)


def require_benchmark(command: str) -> None:
    """Ensure benchmark dependencies are installed or exit.

    Args:
        command: Name of the command requiring benchmark features
    """
    ensure_dependency_group("benchmark", command, required=True)
