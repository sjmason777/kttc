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

"""Core business logic and data models."""

from kttc.core.batch_processor import BatchFileParser, BatchGrouper, BatchTranslation
from kttc.core.glossary import Glossary, GlossaryManager, GlossaryMetadata, TermEntry
from kttc.core.models import ErrorAnnotation, ErrorSeverity, QAReport, TranslationTask
from kttc.core.mqm import MQMScorer
from kttc.core.profiles import (
    BUILTIN_PROFILES,
    MQMProfile,
    get_profile_info,
    list_available_profiles,
    load_profile,
)

__all__ = [
    "BatchFileParser",
    "BatchGrouper",
    "BatchTranslation",
    "BUILTIN_PROFILES",
    "ErrorAnnotation",
    "ErrorSeverity",
    "Glossary",
    "GlossaryManager",
    "GlossaryMetadata",
    "MQMProfile",
    "MQMScorer",
    "QAReport",
    "TermEntry",
    "TranslationTask",
    "get_profile_info",
    "list_available_profiles",
    "load_profile",
]
