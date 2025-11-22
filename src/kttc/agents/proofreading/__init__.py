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

"""Proofreading agents for monolingual text checking.

This module provides agents for checking text quality without translation,
focusing on grammar, spelling, punctuation, and style in a single language.

Used by the `kttc check --self` and `kttc proofread` commands.
"""

from .grammar_agent import GrammarAgent
from .spelling_agent import SpellingAgent

__all__ = ["GrammarAgent", "SpellingAgent"]
