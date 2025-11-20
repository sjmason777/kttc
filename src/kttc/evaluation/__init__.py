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

"""Lightweight evaluation metrics and error detection for translation quality.

This module provides CPU-based metrics (chrF, BLEU, TER) and rule-based
error detection that don't require GPU or heavy neural models.
"""

from kttc.evaluation.error_detection import ErrorDetector, RuleBasedError
from kttc.evaluation.metrics import LightweightMetrics, MetricScores

__all__ = [
    "LightweightMetrics",
    "MetricScores",
    "ErrorDetector",
    "RuleBasedError",
]
