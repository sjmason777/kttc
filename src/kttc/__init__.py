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

"""
KTTC - Knowledge Translation Transmutation Core

Transforming translations into gold-standard quality.

An autonomous multi-agent platform that transmutes raw translations into
certified quality through AI-powered quality assurance.
"""

__version__ = "0.4.1"
__author__ = "KTTC Development"
__email__ = "dev@kt.tc"

from kttc.core.models import (
    ErrorAnnotation,
    ErrorSeverity,
    QAReport,
    TranslationTask,
)
from kttc.core.mqm import MQMScorer

__all__ = [
    "ErrorAnnotation",
    "ErrorSeverity",
    "MQMScorer",
    "QAReport",
    "TranslationTask",
    "__version__",
]
