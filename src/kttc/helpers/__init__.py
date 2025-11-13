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

"""Language-specific helpers for enhanced quality checking.

Provides NLP-powered analysis for specific languages to:
- Verify LLM error reports (anti-hallucination)
- Perform deterministic grammar checks
- Detect language-specific issues with high accuracy
"""

from kttc.helpers.base import LanguageHelper
from kttc.helpers.detection import detect_language, get_helper_for_language

__all__ = [
    "LanguageHelper",
    "detect_language",
    "get_helper_for_language",
]
