"""Base abstract class for QA agents.

All QA agents (accuracy, fluency, terminology) must implement this interface.
"""

from abc import ABC, abstractmethod

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider


class BaseAgent(ABC):
    """Abstract base class for QA agents.

    Each agent is responsible for checking a specific quality dimension
    (accuracy, fluency, terminology) using an LLM provider.
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        """Initialize agent with LLM provider.

        Args:
            llm_provider: LLM provider for generating evaluations
        """
        self.llm_provider = llm_provider

    @abstractmethod
    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate a translation task and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of error annotations found by the agent

        Raises:
            AgentError: If evaluation fails

        Example:
            >>> agent = AccuracyAgent(provider)
            >>> errors = await agent.evaluate(task)
            >>> print(f"Found {len(errors)} accuracy errors")
        """
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Get the error category this agent checks.

        Returns:
            Error category name (e.g., 'accuracy', 'fluency', 'terminology')
        """
        pass


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class AgentEvaluationError(AgentError):
    """Raised when agent evaluation fails."""

    pass


class AgentParsingError(AgentError):
    """Raised when parsing LLM response fails."""

    pass
