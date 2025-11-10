"""QA agents for translation quality evaluation.

Agents evaluate different quality dimensions following the MQM framework:
- AccuracyAgent: Checks translation accuracy (mistranslation, omission, etc.)
- FluencyAgent: Checks grammar and fluency
- TerminologyAgent: Checks terminology consistency
- HallucinationAgent: Detects hallucinated content and factual errors
- ContextAgent: Checks document-level consistency and coherence
- AgentOrchestrator: Coordinates multiple agents in parallel
"""

from .accuracy import AccuracyAgent
from .base import AgentError, AgentEvaluationError, AgentParsingError, BaseAgent
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
    "ErrorParser",
    "AgentError",
    "AgentEvaluationError",
    "AgentParsingError",
]
