"""QA agents for translation quality evaluation.

Agents evaluate different quality dimensions following the MQM framework:
- AccuracyAgent: Checks translation accuracy (mistranslation, omission, etc.)
- FluencyAgent: Checks grammar and fluency
- TerminologyAgent: Checks terminology consistency
- AgentOrchestrator: Coordinates multiple agents in parallel
"""

from .accuracy import AccuracyAgent
from .base import AgentError, AgentEvaluationError, AgentParsingError, BaseAgent
from .fluency import FluencyAgent
from .orchestrator import AgentOrchestrator
from .parser import ErrorParser
from .terminology import TerminologyAgent

__all__ = [
    "BaseAgent",
    "AccuracyAgent",
    "FluencyAgent",
    "TerminologyAgent",
    "AgentOrchestrator",
    "ErrorParser",
    "AgentError",
    "AgentEvaluationError",
    "AgentParsingError",
]
