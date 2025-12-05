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

"""Automatic QA triggers for proactive translation quality checks.

Implements proactive QA (vs reactive) - automatically trigger checks based on:
- File changes (CI/CD integration)
- Threshold violations
- Scheduled checks
- Custom conditions

Similar to Janus Smart LQA's proactive trigger system.
"""

from .conditions import (
    BaseCondition,
    CompositeCondition,
    ErrorCountCondition,
    FilePatternCondition,
    LanguagePairCondition,
    ScoreThresholdCondition,
    TimeBasedCondition,
)
from .manager import TriggerManager
from .models import Trigger, TriggerAction, TriggerEvent, TriggerResult

__all__ = [
    # Models
    "Trigger",
    "TriggerAction",
    "TriggerEvent",
    "TriggerResult",
    # Manager
    "TriggerManager",
    # Conditions
    "BaseCondition",
    "CompositeCondition",
    "ErrorCountCondition",
    "FilePatternCondition",
    "LanguagePairCondition",
    "ScoreThresholdCondition",
    "TimeBasedCondition",
]
