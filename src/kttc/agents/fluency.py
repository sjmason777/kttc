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

"""Fluency Agent for translation quality evaluation.

Checks translation fluency using MQM framework categories:
- Grammar: Verb tense, word order, articles, prepositions
- Spelling: Spelling mistakes, capitalization, punctuation
- Inconsistency: Terminology, formatting, number formats
- Register: Formality level, tone consistency
- Readability: Natural phrasing, sentence flow, idioms
"""

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider, LLMError, PromptTemplate

from .base import AgentEvaluationError, BaseAgent
from .parser import ErrorParser


class FluencyAgent(BaseAgent):
    """Agent for evaluating translation fluency.

    Uses LLM to identify fluency errors following MQM framework.
    Checks for grammar, spelling, inconsistency, register, and readability issues.

    Example:
        >>> provider = OpenAIProvider(api_key="...")
        >>> agent = FluencyAgent(provider)
        >>> task = TranslationTask(
        ...     source_text="The cat is on the mat",
        ...     translation="El gato está en la alfombra",
        ...     source_lang="en",
        ...     target_lang="es"
        ... )
        >>> errors = await agent.evaluate(task)
        >>> print(f"Found {len(errors)} fluency errors")
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize FluencyAgent.

        Args:
            llm_provider: LLM provider for generating evaluations
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        super().__init__(llm_provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._prompt_template = PromptTemplate.load("fluency")

    @property
    def category(self) -> str:
        """Get error category this agent checks."""
        return "fluency"

    def get_base_prompt(self) -> str:
        """Get the base prompt template for fluency evaluation.

        Returns:
            The raw prompt template string
        """
        return self._prompt_template.template

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate translation fluency and return found errors.

        Args:
            task: Translation task to evaluate

        Returns:
            List of fluency error annotations

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
            raise AgentEvaluationError(f"LLM evaluation failed for fluency check: {e}") from e
        except Exception as e:
            raise AgentEvaluationError(f"Unexpected error during fluency evaluation: {e}") from e
