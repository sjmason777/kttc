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

"""Russian-specific Fluency Agent.

Specialized fluency checking for Russian language with support for:
- Case agreement (падежное согласование)
- Aspect usage (совершенный/несовершенный вид)
- Word order validation
- Particle usage (же, ли, бы)
- Register/formality checking

Based on Russian Language Translation Quality 2025 research.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, cast

from kttc.core import ErrorAnnotation, ErrorSeverity, TranslationTask
from kttc.llm import BaseLLMProvider

from .fluency import FluencyAgent

logger = logging.getLogger(__name__)


class RussianFluencyAgent(FluencyAgent):
    """Specialized fluency agent for Russian language.

    Extends base FluencyAgent with Russian-specific checks:
    - Case agreement (6 cases: Nominative, Genitive, Dative, Accusative, Instrumental, Prepositional)
    - Verb aspect (perfective/imperfective)
    - Particle usage
    - Register consistency (ты/вы)

    Example:
        >>> agent = RussianFluencyAgent(llm_provider)
        >>> task = TranslationTask(
        ...     source_text="Hello",
        ...     translation="Привет",
        ...     source_lang="en",
        ...     target_lang="ru"
        ... )
        >>> errors = await agent.evaluate(task)
    """

    RUSSIAN_CHECKS = {
        "case_agreement": "Case agreement validation (падежное согласование)",
        "aspect_usage": "Verb aspect (perfective/imperfective) correctness",
        "word_order": "Natural word order for Russian",
        "particle_usage": "Particle correctness (же, ли, бы, etc.)",
        "register": "Formality register (ты/вы consistency)",
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        """Initialize Russian fluency agent.

        Args:
            llm_provider: LLM provider for evaluations
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        super().__init__(llm_provider, temperature, max_tokens)
        # Store Russian-specific prompt template
        self._russian_prompt_base = """You are a native Russian speaker and linguistic expert.

Your task: Identify Russian-specific linguistic errors in the translation.

## TRANSLATION (Russian):
{translation}

## RUSSIAN-SPECIFIC CHECKS:

1. **Case Agreement (Падежное согласование)**
   - Check noun-adjective agreement
   - Check numeral-noun agreement
   - Check pronoun-noun agreement

2. **Verb Aspect (Вид глагола)**
   - Perfective (совершенный): completed action
   - Imperfective (несовершенный): ongoing/repeated action
   - Check if aspect matches context

3. **Word Order**
   - Russian has flexible word order but some orders sound more natural
   - Check if word order is natural for native speakers

4. **Particle Usage**
   - Particles: же, ли, бы, ведь, вот, etc.
   - Check if particles are used correctly

5. **Register Consistency**
   - Formal (вы, Вы) vs informal (ты)
   - Check consistency throughout text

6. **Diminutives (Уменьшительно-ласкательные)**
   - Check if diminutives are appropriate for context
   - Overuse can sound childish

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "case_agreement|aspect_usage|word_order|particle_usage|register|diminutive",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Russian linguistic issue",
      "suggestion": "Corrected version in Russian"
    }}
  ]
}}

Rules:
- Be strict - native Russians should find text natural
- Focus ONLY on Russian-specific issues
- If no Russian-specific errors, return empty errors array
- Provide character positions (0-indexed)

Output only valid JSON, no explanation."""

    def get_base_prompt(self) -> str:
        """Get the combined base prompt for Russian fluency evaluation.

        Returns:
            The combined base fluency prompt + Russian-specific prompt
        """
        base_fluency = super().get_base_prompt()
        return f"{base_fluency}\n\n---\n\nRUSSIAN-SPECIFIC CHECKS:\n{self._russian_prompt_base}"

    async def evaluate(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Evaluate Russian fluency with specialized checks.

        Args:
            task: Translation task (target_lang must be 'ru')

        Returns:
            List of fluency error annotations

        Raises:
            AgentEvaluationError: If evaluation fails
        """
        # Run standard fluency checks first
        base_errors = await super().evaluate(task)

        # Add Russian-specific checks if target language is Russian
        if task.target_lang == "ru":
            try:
                russian_errors = await self._check_russian_specifics(task)
                base_errors.extend(russian_errors)
                logger.info(
                    f"RussianFluencyAgent found {len(russian_errors)} Russian-specific issues"
                )
            except Exception as e:
                logger.warning(f"Russian-specific checks failed: {e}")

        return base_errors

    async def _check_russian_specifics(self, task: TranslationTask) -> list[ErrorAnnotation]:
        """Perform Russian-specific fluency checks.

        Args:
            task: Translation task

        Returns:
            List of Russian-specific errors
        """
        prompt = f"""You are a native Russian speaker and linguistic expert.

Your task: Identify Russian-specific linguistic errors in the translation.

## TRANSLATION (Russian):
{task.translation}

## RUSSIAN-SPECIFIC CHECKS:

1. **Case Agreement (Падежное согласование)**
   - Check noun-adjective agreement
   - Check numeral-noun agreement
   - Check pronoun-noun agreement

2. **Verb Aspect (Вид глагола)**
   - Perfective (совершенный): completed action
   - Imperfective (несовершенный): ongoing/repeated action
   - Check if aspect matches context

3. **Word Order**
   - Russian has flexible word order but some orders sound more natural
   - Check if word order is natural for native speakers

4. **Particle Usage**
   - Particles: же, ли, бы, ведь, вот, etc.
   - Check if particles are used correctly

5. **Register Consistency**
   - Formal (вы, Вы) vs informal (ты)
   - Check consistency throughout text

6. **Diminutives (Уменьшительно-ласкательные)**
   - Check if diminutives are appropriate for context
   - Overuse can sound childish

Output JSON format:
{{
  "errors": [
    {{
      "subcategory": "case_agreement|aspect_usage|word_order|particle_usage|register|diminutive",
      "severity": "critical|major|minor",
      "location": [start_char, end_char],
      "description": "Specific Russian linguistic issue",
      "suggestion": "Corrected version in Russian"
    }}
  ]
}}

Rules:
- Be strict - native Russians should find text natural
- Focus ONLY on Russian-specific issues
- If no Russian-specific errors, return empty errors array
- Provide character positions (0-indexed)

Output only valid JSON, no explanation."""

        try:
            response = await self.llm_provider.complete(
                prompt, temperature=self.temperature, max_tokens=self.max_tokens
            )

            # Parse response
            response_data = self._parse_json_response(response)
            errors_data = response_data.get("errors", [])

            errors = []
            for error_dict in errors_data:
                location = error_dict.get("location", [0, 10])
                if isinstance(location, list) and len(location) == 2:
                    location_tuple = (location[0], location[1])
                else:
                    location_tuple = (0, 10)

                errors.append(
                    ErrorAnnotation(
                        category="fluency",
                        subcategory=f"russian_{error_dict.get('subcategory', 'specific')}",
                        severity=ErrorSeverity(error_dict.get("severity", "minor")),
                        location=location_tuple,
                        description=error_dict.get("description", "Russian linguistic issue"),
                        suggestion=error_dict.get("suggestion"),
                    )
                )

            return errors

        except Exception as e:
            logger.error(f"Russian-specific check failed: {e}")
            return []

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM.

        Args:
            response: Raw response text

        Returns:
            Parsed JSON dictionary
        """
        try:
            return cast(dict[str, Any], json.loads(response))
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(1)))
                except json.JSONDecodeError:
                    pass

            # Try to find JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return cast(dict[str, Any], json.loads(json_match.group(0)))
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON response: {response[:200]}")
            return {"errors": []}
