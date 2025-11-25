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
- FluencyAgent: Checks grammar and fluency (base class)
  - EnglishFluencyAgent: English-specific fluency with LanguageTool integration
  - ChineseFluencyAgent: Chinese-specific fluency with HanLP integration
  - RussianFluencyAgent: Russian-specific fluency with MAWO NLP integration
- TerminologyAgent: Checks terminology consistency
- HallucinationAgent: Detects hallucinated content and factual errors
- ContextAgent: Checks document-level consistency and coherence
- StylePreservationAgent: Checks authorial voice/style preservation in literary texts
- AgentOrchestrator: Coordinates multiple agents in parallel
- WeightedConsensus: Weighted consensus mechanism for multi-agent evaluation
- DomainProfile: Domain-specific agent configurations
- DomainDetector: Automatic domain detection for adaptive agent selection
- DynamicAgentSelector: Budget-aware agent selection for cost optimization

Literary text support:
The system automatically detects literary texts with stylistic deviations
(Leskov-style skaz, Platanov-style modernism, stream of consciousness)
and adjusts agent weights accordingly. No --literary flag needed.
"""

from .accuracy import AccuracyAgent
from .base import (
    AgentError,
    AgentEvaluationError,
    AgentParsingError,
    BaseAgent,
    SelfAssessmentResult,
)
from .consensus import WeightedConsensus
from .context import ContextAgent
from .debate import DebateOrchestrator, DebateResult, DebateRound
from .domain_profiles import (
    DOMAIN_PROFILES,
    DomainDetector,
    DomainProfile,
    get_domain_profile,
    list_available_domains,
)
from .dynamic_selector import DynamicAgentSelector
from .fluency import FluencyAgent
from .fluency_chinese import ChineseFluencyAgent
from .fluency_english import EnglishFluencyAgent
from .fluency_russian import RussianFluencyAgent
from .hallucination import HallucinationAgent
from .orchestrator import AgentOrchestrator
from .parser import ErrorParser
from .style_preservation import StylePreservationAgent
from .terminology import TerminologyAgent

__all__ = [
    "BaseAgent",
    "SelfAssessmentResult",
    "AccuracyAgent",
    "FluencyAgent",
    "EnglishFluencyAgent",
    "ChineseFluencyAgent",
    "RussianFluencyAgent",
    "TerminologyAgent",
    "HallucinationAgent",
    "ContextAgent",
    "StylePreservationAgent",
    "AgentOrchestrator",
    "WeightedConsensus",
    "DomainProfile",
    "DomainDetector",
    "DOMAIN_PROFILES",
    "get_domain_profile",
    "list_available_domains",
    "DynamicAgentSelector",
    "DebateOrchestrator",
    "DebateResult",
    "DebateRound",
    "ErrorParser",
    "AgentError",
    "AgentEvaluationError",
    "AgentParsingError",
]
