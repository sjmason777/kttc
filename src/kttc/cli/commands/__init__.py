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

"""CLI subcommands for KTTC.

This module exports all command modules for registration with the main app.
"""

from kttc.cli.commands.benchmark import run_benchmark
from kttc.cli.commands.compare import run_compare
from kttc.cli.commands.glossary import glossary_app
from kttc.cli.commands.terminology import terminology_app

# Lazy imports to avoid circular dependencies
# These are imported when needed by main.py directly

__all__ = [
    # Command functions (imported lazily in main.py)
    "run_benchmark",
    "run_compare",
    # Typer apps for sub-commands
    "glossary_app",
    "terminology_app",
]
