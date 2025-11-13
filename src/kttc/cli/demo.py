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

"""Demo/Mock LLM provider for testing CLI without API calls."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from kttc.llm import BaseLLMProvider


class DemoLLMProvider(BaseLLMProvider):
    """Mock LLM provider that simulates responses without making API calls.

    This provider is used for testing and demonstrations to avoid spending tokens.
    It returns realistic-looking responses that mimic the structure of real LLM outputs.
    """

    def __init__(self, model: str = "demo-model", **kwargs: Any) -> None:
        """Initialize demo provider."""
        self.model = model
        self.kwargs = kwargs

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> str:
        """Simulate LLM completion with demo response.

        Args:
            prompt: The prompt (analyzed to provide relevant response)
            temperature: Ignored in demo mode
            max_tokens: Ignored in demo mode
            **kwargs: Additional arguments (ignored)

        Returns:
            Simulated LLM response based on prompt content
        """
        # Simulate API latency
        await asyncio.sleep(0.5)

        # Parse prompt to determine which agent is calling
        prompt_lower = prompt.lower()

        # Russian-specific fluency agent (expects JSON)
        if "russian-specific" in prompt_lower or (
            "russian" in prompt_lower and "output only valid json" in prompt_lower
        ):
            return """{
  "errors": [
    {
      "subcategory": "case_agreement",
      "severity": "minor",
      "location": [15, 35],
      "description": "Case agreement issue: adjective-noun agreement incorrect",
      "suggestion": "Correct case agreement"
    }
  ]
}"""

        # Hallucination/Entity preservation agent (expects JSON)
        elif "entity preservation" in prompt_lower or "hallucination" in prompt_lower:
            return """{
  "errors": []
}"""

        # Context/Coherence agent (expects JSON)
        elif "coherence" in prompt_lower:
            return """{
  "errors": []
}"""

        # Accuracy agent response (expects ERROR_START format)
        elif "accuracy" in prompt_lower or "meaning" in prompt_lower or "omission" in prompt_lower:
            return """**Errors Found:**

1. **Minor Mistranslation** (Line 1)
   - Category: accuracy
   - Severity: minor
   - Source: "The quick brown fox"
   - Translation: "Быстрая коричневая лиса"
   - Issue: "quick" typically translates to "быстрый" not "быстрая" - gender agreement error
   - Suggestion: "Быстрый коричневый лис"

2. **Addition** (Line 2)
   - Category: accuracy
   - Severity: minor
   - Source: "jumps over"
   - Translation: "перепрыгивает через высокий"
   - Issue: Added "высокий" (high) which is not in source
   - Suggestion: Remove "высокий"

**Overall Assessment:**
Translation captures the main meaning but has 2 minor accuracy issues affecting precision.

**MQM Deductions:** -3.5 points"""

        # Fluency agent response (expects ERROR_START format)
        elif "fluency" in prompt_lower or "grammar" in prompt_lower or "spelling" in prompt_lower:
            return """**Errors Found:**

1. **Grammar Error** (Line 1)
   - Category: fluency
   - Severity: minor
   - Text: "собакой ленивый"
   - Issue: Adjective should come before noun in Russian
   - Suggestion: "ленивой собакой"

**Overall Assessment:**
Text is mostly fluent with natural Russian phrasing. One minor grammatical ordering issue.

**MQM Deductions:** -1.5 points"""

        # Terminology agent response (expects ERROR_START format)
        elif (
            "terminology" in prompt_lower or "domain" in prompt_lower or "technical" in prompt_lower
        ):
            return """**Errors Found:**

None detected.

**Overall Assessment:**
No terminology issues found. General content without specialized domain terms.

**MQM Deductions:** 0 points"""

        # Default orchestrator response
        else:
            return """**Translation Quality Report**

**Accuracy:** 2 minor errors (-3.5 points)
**Fluency:** 1 minor error (-1.5 points)
**Terminology:** No errors (0 points)

**Total MQM Deductions:** -5.0 points
**Final MQM Score:** 95.0/100

**Status:** PASS (threshold: 95.0)

**Summary:**
Good quality translation with minor accuracy and fluency issues. Main meaning preserved but could be more precise."""

    async def stream(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Simulate streaming completion.

        Args:
            prompt: The prompt
            temperature: Ignored in demo mode
            max_tokens: Ignored in demo mode
            **kwargs: Additional arguments (ignored)

        Yields:
            Text chunks simulating streaming response
        """
        # Get full response
        full_response = await self.complete(prompt, temperature, max_tokens, **kwargs)

        # Stream it word by word
        words = full_response.split()
        for i, word in enumerate(words):
            await asyncio.sleep(0.02)  # Simulate streaming delay
            if i < len(words) - 1:
                yield word + " "
            else:
                yield word

    async def complete_with_response_model(
        self,
        prompt: str,
        response_model: type,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        **kwargs: Any,
    ) -> Any:
        """Not implemented for demo provider.

        Args:
            prompt: The prompt
            response_model: Expected response model type
            temperature: Ignored
            max_tokens: Ignored
            **kwargs: Additional arguments

        Raises:
            NotImplementedError: This method is not supported in demo mode
        """
        raise NotImplementedError("Structured outputs not supported in demo mode")

    def get_token_count(self, text: str) -> int:
        """Return approximate token count.

        Args:
            text: Text to count tokens for

        Returns:
            Approximate token count (words * 1.3)
        """
        return int(len(text.split()) * 1.3)
