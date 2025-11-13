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

"""QA agents for translation quality evaluation.

Agents evaluate different quality dimensions following the MQM framework:
- AccuracyAgent: Checks translation accuracy (mistranslation, omission, etc.)
- FluencyAgent: Checks grammar and fluency
- TerminologyAgent: Checks terminology consistency
- HallucinationAgent: Detects hallucinated content and factual errors
- ContextAgent: Checks document-level consistency and coherence
- AgentOrchestrator: Coordinates multiple agents in parallel
- WeightedConsensus: Weighted consensus mechanism for multi-agent evaluation
"""

from .accuracy import AccuracyAgent
from .base import AgentError, AgentEvaluationError, AgentParsingError, BaseAgent
from .consensus import WeightedConsensus
from .context import ContextAgent
from .fluency import FluencyAgent
from .hallucination import HallucinationAgent
from .orchestrator import AgentOrchestrator
from .parser import ErrorParser
from .terminology import TerminologyAgent

__all__ = [
    "BaseAgent",
    "AccuracyAgent",
    "FluencyAgent",
    "TerminologyAgent",
    "HallucinationAgent",
    "ContextAgent",
    "AgentOrchestrator",
    "WeightedConsensus",
    "ErrorParser",
    "AgentError",
    "AgentEvaluationError",
    "AgentParsingError",
]
