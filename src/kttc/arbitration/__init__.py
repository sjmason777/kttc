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

"""Arbitration cycle for translation QA disputes.

Implements the full arbitration workflow:
1. QA reviewer flags errors
2. Translator submits objections
3. Arbiter reviews and makes final decision
4. Results are logged and tracked

This module enables professional translation QA workflows
similar to Logrus Perfectionist and Janus Smart LQA.
"""

from .models import (
    ArbitrationDecision,
    ArbitrationResult,
    ArbitrationStatus,
    Objection,
    ObjectionStatus,
)
from .workflow import ArbitrationWorkflow

__all__ = [
    "ArbitrationWorkflow",
    "ArbitrationDecision",
    "ArbitrationResult",
    "ArbitrationStatus",
    "Objection",
    "ObjectionStatus",
]
