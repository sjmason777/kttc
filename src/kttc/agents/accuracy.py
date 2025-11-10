"""Accuracy Agent for translation quality evaluation.

Checks translation accuracy using MQM framework categories:
- Mistranslation: Incorrect meaning
- Omission: Missing source information
- Addition: Extra information not in source
- Untranslated: Source language words in translation
"""

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider, LLMError, PromptTemplate

from .base import AgentEvaluationError, BaseAgent
from .parser import ErrorParser


class AccuracyAgent(BaseAgent):
    """Agent for evaluating translation accuracy.

    Uses LLM to identify accuracy errors following MQM framework.
    Checks for mistranslations, omissions, additions, and untranslated text.

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> agent = AccuracyAgent(provider)
        >>> task = TranslationTask(
        ...     source_text="The cat is on the mat",
        ...     translation="El gato está en la alfombra",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
        >>> errors = await agent.evaluate(task)
        >>> print(f"Found {len(errors)} accuracy errors")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize AccuracyAgent.

        Args:
            llm_provider: LLM provider for generating evaluations
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._prompt_template = PromptTemplate.load("accuracy")

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "accuracy"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation accuracy and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of accuracy error annotations

        Raises:
            AgentEvaluationError: If LLM evaluation fails
        """
        try:
            # Format prompt with task data
            prompt = self._prompt_template.format(
                source_text=task.source_text,
                translation=task.translation,
                source_lang=task.source_lang,
                target_lang=task.target_lang,
            )

            # Get LLM evaluation
            llm_response = await self.llm_provider.complete(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse errors from response
            errors = ErrorParser.parse_errors(llm_response)

            # Validate that all errors match this agent's category
            validated_errors = [
                error for error in errors if error.category.lower() == self.category.lower()
            ]

            return validated_errors

        except LLMError as e:
            raise AgentEvaluationError(f"LLM evaluation failed for accuracy check: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(f"Unexpected error during accuracy evaluation: {e}") from e
