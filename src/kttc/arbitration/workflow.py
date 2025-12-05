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

"""Arbitration workflow orchestration.

Manages the complete arbitration cycle from QA report to final decision.
Can use LLM as AI arbiter or support human-in-the-loop decisions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .models import (
    ArbitrationDecision,
    ArbitrationResult,
    ArbitrationStatus,
    Objection,
    ObjectionStatus,
)

if TYPE_CHECKING:
    from kttc.core import QAReport
    from kttc.llm import BaseLLMProvider

logger = logging.getLogger(__name__)

# Prompt template for AI arbiter
ARBITER_PROMPT = """You are an expert translation quality arbiter. Your role is to fairly evaluate
disputes between translators and QA reviewers.

## Context
- Source language: {source_lang}
- Target language: {target_lang}
- Domain: {domain}

## Original Error
- Error ID: {error_id}
- Category: {category}
- Severity: {severity}
- Description: {description}
- Error location: {location}

## Translator's Objection
- Reason: {objection_reason}
- Evidence provided: {evidence}
- Proposed resolution: {proposed_resolution}

## Your Task
Evaluate this objection and make a fair decision. Consider:
1. Is the original error assessment accurate?
2. Does the translator's evidence support their objection?
3. Are there valid stylistic or cultural considerations?
4. Would a native speaker find this acceptable?

## Response Format (JSON)
{{
    "decision": "upheld" | "dismissed" | "reduced" | "escalated",
    "reason": "Clear explanation of your decision",
    "confidence": 0.0-1.0,
    "suggested_severity": "critical" | "major" | "minor" | null
}}
"""


class ArbitrationWorkflow:
    """Orchestrates the complete arbitration cycle.

    Supports both AI-assisted and human-in-the-loop arbitration.

    Example:
        >>> workflow = ArbitrationWorkflow()
        >>> result = workflow.start_arbitration(qa_report)
        >>> result.add_objection(Objection(
        ...     error_id="err_001",
        ...     reason="This is acceptable in formal Russian"
        ... ))
        >>> await workflow.process_with_ai(result, llm_provider)
    """

    def __init__(self, auto_approve_threshold: float = 0.9):
        """Initialize arbitration workflow.

        Args:
            auto_approve_threshold: Confidence threshold for auto-approval
        """
        self.auto_approve_threshold = auto_approve_threshold

    def start_arbitration(self, report: QAReport) -> ArbitrationResult:
        """Start arbitration cycle for a QA report.

        Args:
            report: The QA report to arbitrate

        Returns:
            New ArbitrationResult in DRAFT status
        """
        return ArbitrationResult(
            report_id=report.id if hasattr(report, "id") else str(id(report)),
            status=ArbitrationStatus.DRAFT,
            original_error_count=len(report.errors),
            final_error_count=len(report.errors),
            original_mqm_score=report.mqm_score,
            final_mqm_score=report.mqm_score,
            metadata={
                "source_lang": report.source_lang if hasattr(report, "source_lang") else "",
                "target_lang": report.target_lang if hasattr(report, "target_lang") else "",
            },
        )

    def submit_objection(
        self,
        result: ArbitrationResult,
        error_id: str,
        reason: str,
        evidence: list[str] | None = None,
        proposed_resolution: str | None = None,
        translator_id: str = "",
    ) -> Objection:
        """Submit an objection to a flagged error.

        Args:
            result: The arbitration result to add objection to
            error_id: ID of the error being contested
            reason: Detailed reason for objection
            evidence: Supporting evidence (optional)
            proposed_resolution: Suggested resolution (optional)
            translator_id: Identifier of the translator

        Returns:
            The created Objection
        """
        objection = Objection(
            error_id=error_id,
            translator_id=translator_id,
            reason=reason,
            evidence=evidence or [],
            proposed_resolution=proposed_resolution,
        )

        result.add_objection(objection)
        result.status = ArbitrationStatus.PENDING_ARBITRATION

        logger.info(
            "Objection submitted for error %s by translator %s",
            error_id,
            translator_id or "unknown",
        )

        return objection

    async def process_with_ai(
        self,
        result: ArbitrationResult,
        llm_provider: BaseLLMProvider,
        qa_report: QAReport,
    ) -> ArbitrationResult:
        """Process all pending objections with AI arbiter.

        Args:
            result: Arbitration result with objections
            llm_provider: LLM provider for AI arbitration
            qa_report: Original QA report for error details

        Returns:
            Updated ArbitrationResult with decisions
        """
        # Build error lookup
        errors_by_id = {getattr(e, "id", str(i)): e for i, e in enumerate(qa_report.errors)}

        pending_objections = [o for o in result.objections if o.status == ObjectionStatus.PENDING]

        for objection in pending_objections:
            error = errors_by_id.get(objection.error_id)
            if not error:
                logger.warning("Error %s not found in report", objection.error_id)
                continue

            try:
                decision = await self._get_ai_decision(
                    objection=objection,
                    error=error,
                    result=result,
                    llm_provider=llm_provider,
                )
                self._apply_decision(objection, decision)

            except Exception as e:
                logger.error("AI arbitration failed for %s: %s", objection.error_id, e)
                # Mark for manual review
                objection.decision = ArbitrationDecision.ESCALATED
                objection.decision_reason = f"AI review failed: {e}"

        # Finalize
        result.finalize()
        self._update_final_counts(result, qa_report)

        return result

    async def _get_ai_decision(
        self,
        objection: Objection,
        error: object,
        result: ArbitrationResult,
        llm_provider: BaseLLMProvider,
    ) -> dict[str, Any]:
        """Get AI arbiter decision for an objection.

        Args:
            objection: The objection to evaluate
            error: The original error
            result: Arbitration result with context
            llm_provider: LLM provider

        Returns:
            Decision dictionary
        """
        prompt = ARBITER_PROMPT.format(
            source_lang=result.metadata.get("source_lang", "unknown"),
            target_lang=result.metadata.get("target_lang", "unknown"),
            domain=result.metadata.get("domain", "general"),
            error_id=objection.error_id,
            category=getattr(error, "category", "unknown"),
            severity=getattr(error, "severity", "unknown"),
            description=getattr(error, "description", ""),
            location=getattr(error, "location", ""),
            objection_reason=objection.reason,
            evidence=", ".join(objection.evidence) if objection.evidence else "None",
            proposed_resolution=objection.proposed_resolution or "None",
        )

        response = await llm_provider.complete(prompt, temperature=0.1)

        # Parse JSON response
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if 0 <= start < end:
                decision: dict[str, Any] = json.loads(response[start:end])
                return decision
            raise ValueError("No JSON found in response")
        except json.JSONDecodeError as e:
            logger.error("Failed to parse AI decision: %s", e)
            return {
                "decision": "escalated",
                "reason": "Failed to parse AI response",
                "confidence": 0.0,
            }

    def _apply_decision(self, objection: Objection, decision: dict[str, Any]) -> None:
        """Apply AI decision to objection.

        Args:
            objection: The objection to update
            decision: Decision dictionary from AI
        """
        decision_type = decision.get("decision", "escalated").lower()
        reason = decision.get("reason", "")
        confidence = decision.get("confidence", 0.5)

        # Check confidence threshold for auto-approval
        if confidence < self.auto_approve_threshold:
            decision_type = "escalated"
            reason = f"Low confidence ({confidence:.2f}): {reason}"

        if decision_type == "dismissed":
            objection.accept(reason)
        elif decision_type == "upheld":
            objection.reject(reason)
        elif decision_type == "reduced":
            objection.reduce_severity(reason)
        else:
            objection.decision = ArbitrationDecision.ESCALATED
            objection.decision_reason = reason or "Requires manual review"
            objection.resolved_at = datetime.now()

    def _update_final_counts(self, result: ArbitrationResult, qa_report: QAReport) -> None:
        """Update final error counts based on decisions.

        Args:
            result: Arbitration result to update
            qa_report: Original QA report
        """
        dismissed_count = sum(
            1 for o in result.objections if o.decision == ArbitrationDecision.DISMISSED
        )

        result.final_error_count = result.original_error_count - dismissed_count

        # Recalculate MQM score (simplified)
        if result.original_error_count > 0:
            error_ratio = result.final_error_count / result.original_error_count
            score_improvement = (1 - error_ratio) * (100 - result.original_mqm_score)
            result.final_mqm_score = min(100.0, result.original_mqm_score + score_improvement)

    def decide_manually(
        self,
        objection: Objection,
        decision: ArbitrationDecision,
        reason: str = "",
    ) -> None:
        """Apply manual decision to an objection.

        Args:
            objection: The objection to decide
            decision: The decision to apply
            reason: Reason for decision
        """
        if decision == ArbitrationDecision.DISMISSED:
            objection.accept(reason)
        elif decision == ArbitrationDecision.UPHELD:
            objection.reject(reason)
        elif decision == ArbitrationDecision.REDUCED:
            objection.reduce_severity(reason)
        else:
            objection.decision = decision
            objection.decision_reason = reason
            objection.resolved_at = datetime.now()

        logger.info(
            "Manual decision applied to %s: %s",
            objection.error_id,
            decision.value,
        )

    def export_summary(self, result: ArbitrationResult) -> dict[str, Any]:
        """Export arbitration summary for reporting.

        Args:
            result: Completed arbitration result

        Returns:
            Summary dictionary
        """
        return {
            "report_id": result.report_id,
            "status": result.status.value,
            "original_errors": result.original_error_count,
            "final_errors": result.final_error_count,
            "errors_dismissed": result.original_error_count - result.final_error_count,
            "objections_total": len(result.objections),
            "objections_accepted": result.objections_accepted,
            "objections_rejected": result.objections_rejected,
            "acceptance_rate": result.objection_acceptance_rate,
            "original_mqm_score": result.original_mqm_score,
            "final_mqm_score": result.final_mqm_score,
            "score_improvement": result.score_improvement,
            "created_at": result.created_at.isoformat(),
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        }
