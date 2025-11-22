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

"""Style analysis module for literary translation quality assessment.

This module provides automatic detection and analysis of stylistic features
in source texts, enabling style-aware translation quality evaluation.

Key components:
- StyleFingerprint: Automatic style analysis of source texts
- StyleProfile: Data model for style characteristics
- ComparativeStyleAnalyzer: Compare style preservation in translations

The system automatically detects:
- Intentional grammatical deviations (Platanov-style)
- Folk speech patterns (Leskov-style skaz)
- Stream of consciousness (Erofeev, Joyce)
- Register mixing and other literary devices

Example:
    >>> from kttc.style import StyleFingerprint, ComparativeStyleAnalyzer
    >>>
    >>> fingerprint = StyleFingerprint()
    >>> source_profile = fingerprint.analyze(source_text, lang="ru")
    >>>
    >>> if source_profile.deviation_score > 0.3:
    ...     print("Literary text detected - adjusting evaluation")
    ...     # Agent weights will be automatically adjusted
"""

from .analyzer import StyleFingerprint
from .comparative import ComparativeStyleAnalyzer
from .models import (
    StyleDeviation,
    StyleDeviationType,
    StylePattern,
    StyleProfile,
)

__all__ = [
    "StyleFingerprint",
    "ComparativeStyleAnalyzer",
    "StyleProfile",
    "StyleDeviation",
    "StyleDeviationType",
    "StylePattern",
]
