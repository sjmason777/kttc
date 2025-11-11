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

"""Terminology Agent for translation quality evaluation.

Checks terminology accuracy using MQM framework categories:
- Inconsistency: Same term translated differently
- Misuse: Wrong technical term chosen
- Untranslated: Terms left in source language
- Formatting: Incorrect capitalization, singular/plural, notation
"""

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider, LLMError, PromptTemplate

from .base import AgentEvaluationError, BaseAgent
from .parser import ErrorParser


class TerminologyAgent(BaseAgent):
    """Agent for evaluating translation terminology.

    Uses LLM to identify terminology errors following MQM framework.
    Checks for inconsistency, misuse, untranslated terms, and formatting issues.

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> agent = TerminologyAgent(provider)
        >>> task = TranslationTask(
        ...     source_text="The API uses REST architecture",
        ...     translation="La interfaz usa arquitectura REST",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
        >>> errors = await agent.evaluate(task)
        >>> print(f"Found {len(errors)} terminology errors")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize TerminologyAgent.

        Args:
            llm_provider: LLM provider for generating evaluations
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._prompt_template = PromptTemplate.load("terminology")

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "terminology"

    def get_base_prompt(self) -> str:
        """Get the base prompt template for terminology evaluation.

        Returns:
            The raw prompt template string
        """
        return self._prompt_template.template

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation terminology and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of terminology error annotations

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
            raise AgentEvaluationError(f"LLM evaluation failed for terminology check: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(
                f"Unexpected error during terminology evaluation: {e}"
            ) from e
