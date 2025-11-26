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

"""Debate mode for multi-agent error verification.

Implements agent debate mechanism where errors found by one agent
are verified by another agent, improving precision and reducing false positives.

Based on adversarial debate protocols from AI safety research.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from kttc.core import ErrorAnnotation, TranslationTask
from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class DebateResult:
    """Result of debate between agents about an error.

    Attributes:
        error: The error being debated
        original_agent: Agent that found the error
        verifier_agent: Agent that verified/challenged the error
        verdict: Final verdict: 'confirmed', 'rejected', or 'modified'
        confidence: Confidence in the verdict (0.0-1.0)
        reasoning: Explanation of the verdict
        modified_error: Modified error if verdict is 'modified'
    """

    error: ErrorAnnotation
    original_agent: str
    verifier_agent: str
    verdict: str  # 'confirmed', 'rejected', 'modified'
    confidence: float
    reasoning: str
    modified_error: ErrorAnnotation | None = None


@dataclass
class DebateRound:
    """A single round of debate between agents.

    Attributes:
        round_number: Current round number
        errors_debated: Number of errors debated in this round
        results: Debate results for each error
        summary: Summary statistics
    """

    round_number: int
    errors_debated: int
    results: list[DebateResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


class DebateOrchestrator:
    """Orchestrates debate between agents about detected errors.

    Debate mode improves precision by having agents cross-verify errors:
    1. One agent finds errors
    2. Another agent verifies each error
    3. Only errors agreed upon by both agents are kept

    Example:
        >>> debate = DebateOrchestrator(llm_provider)
        >>> verified_errors = await debate.run_debate(errors, task)
    """

    # Verifier assignments: which agent verifies which agent's errors
    VERIFIER_ASSIGNMENTS = {
        "accuracy": "fluency",  # Fluency verifies accuracy errors
        "fluency": "accuracy",  # Accuracy verifies fluency errors
        "terminology": "accuracy",  # Accuracy verifies terminology errors
        "hallucination": "accuracy",  # Accuracy verifies hallucination errors
        "context": "fluency",  # Fluency verifies context errors
    }

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        temperature: float = 0.1,
        max_tokens: int = 1500,
        confidence_threshold: float = 0.6,
    ):
        """Initialize debate orchestrator.

        Args:
            llm_provider: LLM provider for verification calls
            temperature: Sampling temperature for debate
            max_tokens: Maximum tokens in debate responses
            confidence_threshold: Minimum confidence to confirm error
        """
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.confidence_threshold = confidence_threshold

    async def run_debate(
        self,
        errors: list[ErrorAnnotation],
        task: TranslationTask,
        max_rounds: int = 1,
    ) -> tuple[list[ErrorAnnotation], list[DebateRound]]:
        """Run debate to verify errors.

        Args:
            errors: List of errors to verify
            task: Translation task context
            max_rounds: Maximum debate rounds

        Returns:
            Tuple of (verified_errors, debate_rounds)
        """
        if not errors:
            return [], []

        verified_errors: list[ErrorAnnotation] = []
        debate_rounds: list[DebateRound] = []

        for round_num in range(1, max_rounds + 1):
            round_result = await self._run_debate_round(errors, task, round_num)
            debate_rounds.append(round_result)

            # Collect confirmed/modified errors
            for result in round_result.results:
                if result.verdict == "confirmed":
                    verified_errors.append(result.error)
                elif result.verdict == "modified" and result.modified_error:
                    verified_errors.append(result.modified_error)

            # Update summary
            round_result.summary = {
                "confirmed": sum(1 for r in round_result.results if r.verdict == "confirmed"),
                "rejected": sum(1 for r in round_result.results if r.verdict == "rejected"),
                "modified": sum(1 for r in round_result.results if r.verdict == "modified"),
                "total": len(round_result.results),
            }

            logger.info(
                f"Debate round {round_num}: "
                f"{round_result.summary['confirmed']} confirmed, "
                f"{round_result.summary['rejected']} rejected, "
                f"{round_result.summary['modified']} modified"
            )

        return verified_errors, debate_rounds

    async def _run_debate_round(
        self,
        errors: list[ErrorAnnotation],
        task: TranslationTask,
        round_number: int,
    ) -> DebateRound:
        """Run a single debate round.

        Args:
            errors: Errors to debate
            task: Translation task
            round_number: Current round number

        Returns:
            DebateRound with results
        """
        results: list[DebateResult] = []

        for error in errors:
            # Get verifier for this error type
            verifier_agent = self.VERIFIER_ASSIGNMENTS.get(error.category, "accuracy")

            # Run verification
            result = await self._verify_error(error, task, verifier_agent)
            results.append(result)

        return DebateRound(
            round_number=round_number,
            errors_debated=len(errors),
            results=results,
        )

    async def _verify_error(
        self,
        error: ErrorAnnotation,
        task: TranslationTask,
        verifier_agent: str,
    ) -> DebateResult:
        """Have verifier agent check an error.

        Args:
            error: Error to verify
            task: Translation task
            verifier_agent: Agent ID to verify

        Returns:
            DebateResult with verdict
        """
        prompt = self._build_verification_prompt(error, task, verifier_agent)

        try:
            response = await self.llm_provider.complete(
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return self._parse_verification_response(
                response, error, error.category, verifier_agent
            )

        except Exception as e:
            logger.warning(f"Verification failed for error: {e}")
            # On failure, default to confirming the error
            return DebateResult(
                error=error,
                original_agent=error.category,
                verifier_agent=verifier_agent,
                verdict="confirmed",
                confidence=0.5,
                reasoning=f"Verification failed: {e}. Defaulting to confirmed.",
            )

    def _build_verification_prompt(
        self,
        error: ErrorAnnotation,
        task: TranslationTask,
        verifier_agent: str,
    ) -> str:
        """Build verification prompt for debate.

        Args:
            error: Error to verify
            task: Translation task
            verifier_agent: Verifier agent type

        Returns:
            Verification prompt string
        """
        agent_focus = {
            "accuracy": "semantic accuracy and meaning preservation",
            "fluency": "grammatical correctness and natural expression",
            "terminology": "terminology consistency and domain accuracy",
        }

        focus = agent_focus.get(verifier_agent, "translation quality")

        return f"""You are an expert translation reviewer focusing on {focus}.

Another agent has identified the following error. Your task is to verify if this error is valid.

## SOURCE TEXT ({task.source_lang}):
{task.source_text}

## TRANSLATION ({task.target_lang}):
{task.translation}

## ERROR REPORTED:
- Category: {error.category}
- Subcategory: {error.subcategory}
- Severity: {error.severity.value}
- Location: characters {error.location[0]}-{error.location[1]}
- Description: {error.description}
- Suggestion: {error.suggestion or "None"}

## YOUR TASK:
Analyze this error and provide your verdict:
1. Is this a valid error? Or is it a false positive?
2. If valid, is the severity appropriate?
3. Should the error be confirmed, rejected, or modified?

## OUTPUT FORMAT (JSON):
{{
    "verdict": "confirmed" | "rejected" | "modified",
    "confidence": 0.0-1.0,
    "reasoning": "Your explanation",
    "modified_severity": "critical|major|minor|neutral" (only if modified),
    "modified_description": "Updated description" (only if modified)
}}

Output only valid JSON, no additional text."""

    def _parse_verification_response(
        self,
        response: str,
        error: ErrorAnnotation,
        original_agent: str,
        verifier_agent: str,
    ) -> DebateResult:
        """Parse verification response.

        Args:
            response: LLM response
            error: Original error
            original_agent: Agent that found error
            verifier_agent: Agent that verified

        Returns:
            DebateResult
        """
        try:
            # Try to extract JSON
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            data = json.loads(json_match.group(0))

            verdict = data.get("verdict", "confirmed")
            confidence = float(data.get("confidence", 0.7))
            reasoning = data.get("reasoning", "")

            # Handle modified errors
            modified_error = None
            if verdict == "modified":
                from kttc.core import ErrorSeverity

                modified_severity = data.get("modified_severity", error.severity.value)
                modified_description = data.get("modified_description", error.description)

                modified_error = ErrorAnnotation(
                    category=error.category,
                    subcategory=error.subcategory,
                    severity=ErrorSeverity(modified_severity),
                    location=error.location,
                    description=modified_description,
                    suggestion=error.suggestion,
                )

            return DebateResult(
                error=error,
                original_agent=original_agent,
                verifier_agent=verifier_agent,
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                modified_error=modified_error,
            )

        except Exception as e:
            logger.warning(f"Failed to parse verification response: {e}")
            return DebateResult(
                error=error,
                original_agent=original_agent,
                verifier_agent=verifier_agent,
                verdict="confirmed",
                confidence=0.5,
                reasoning=f"Parse error: {e}",
            )

    def get_debate_summary(self, rounds: list[DebateRound]) -> dict[str, Any]:
        """Get summary of all debate rounds.

        Args:
            rounds: List of debate rounds

        Returns:
            Summary statistics
        """
        total_confirmed = 0
        total_rejected = 0
        total_modified = 0
        total_errors = 0

        for round_result in rounds:
            total_confirmed += round_result.summary.get("confirmed", 0)
            total_rejected += round_result.summary.get("rejected", 0)
            total_modified += round_result.summary.get("modified", 0)
            total_errors += round_result.summary.get("total", 0)

        return {
            "rounds": len(rounds),
            "total_errors_debated": total_errors,
            "confirmed": total_confirmed,
            "rejected": total_rejected,
            "modified": total_modified,
            "precision_improvement": (
                f"{(total_rejected / total_errors * 100):.1f}%" if total_errors > 0 else "N/A"
            ),
        }
